"""Independent replay of the course's public straight-approach controller.

Recompute paths, feedback, stopping and handover from recorded public inputs.
No controller, trajectory generator, braking predictor or simulator truth is
imported. Sensor-to-grid projection is shared; grid traversal is independent.
Physical contacts, ONNX outputs and raw sensor projection have separate audits.
"""
from pathlib import Path
import argparse
import ast
import bisect
from collections import Counter
import hashlib
import json
import math
import sys
import traceback

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from onboard_geometry.map_contract import snapshot, WIDTH, HEIGHT, RESOLUTION
from audit_epochs import validate as validate_epochs
from audit_blind_arrival_actual import audit as audit_blind, number, vector, zero, fresh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def close(a, b, tolerance=1e-8):
    if number(a) and number(b):
        return abs(a-b) <= tolerance
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y, tolerance) for x, y in zip(a, b))
    return a == b


def angle(x):
    return math.atan2(math.sin(x), math.cos(x))


def course_parameters():
    """Read only the declared literals, without executing the production module."""
    path = Path(__file__).with_name('approach_control.py')
    def value(node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == 'math' and node.func.attr == 'radians' and len(node.args) == 1:
                return math.radians(ast.literal_eval(node.args[0]))
        return ast.literal_eval(node)
    result = None
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'PARAMETERS' for t in node.targets):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'dict':
                result = {k.arg: value(k.value) for k in node.value.keywords}
            else:
                result = ast.literal_eval(node.value)
    if result is None:
        raise ValueError('Missing literal course approach parameters')
    fixed = dict(coarse_duration_s=30., coarse_stop_m=.115, track_duration_s=30.,
                 prealign_rad=math.radians(10), goal_age_s=2., memory_age_s=6.5,
                 parking_angle_rad=math.radians(20), parking_vx=-.20,
                 parking_wz=.5, parking_duration_s=.8, transfer_x_m=[.13,.28])
    for name, expected in fixed.items():
        if not close(result.get(name), expected):
            raise ValueError('Unreviewed course parameter: '+name)
    if result.get('transfer_y_m') != [-.10,.08]:
        raise ValueError('Unreviewed continuous-transfer lateral region')
    return result, path


class Checks:
    def __init__(self):
        self.counts = Counter()
        self.errors = []
        self.failure_count = 0

    def __call__(self, name, passed, t=None, detail=None):
        self.counts[name] += 1
        if not passed:
            self.failure_count += 1
            if len(self.errors) < 300:
                self.errors.append(dict(check=name, t=t, detail=detail))


def path_gate(points, grid):
    """Walk the full 10 cm envelope; do not call the production check_path."""
    if not points:
        return False, 'no_path'
    if not all(vector(p, 2) for p in points):
        return False, 'nonfinite_path'
    if math.dist(points[0], grid['pose'][:2]) > .06 or math.dist(points[-1], grid['goal']) > .06:
        return False, 'endpoint_mismatch'
    dense = []
    for a, b in zip(points, points[1:]):
        count = max(1, math.ceil(math.dist(a,b)/(RESOLUTION/2)))
        dense.extend([[a[j]+(b[j]-a[j])*i/count for j in range(2)] for i in range(count)])
    dense.append(points[-1])
    radius = .1+RESOLUTION*math.sqrt(2)/2
    cells = math.ceil(radius/RESOLUTION)
    for x, y in dense:
        i0 = math.floor((x-grid['origin'][0])/RESOLUTION)
        j0 = math.floor((y-grid['origin'][1])/RESOLUTION)
        if not 0 <= i0 < WIDTH or not 0 <= j0 < HEIGHT:
            return False, 'outside_map'
        for j in range(j0-cells, j0+cells+1):
            for i in range(i0-cells, i0+cells+1):
                cx, cy = grid['origin'][0]+(i+.5)*RESOLUTION, grid['origin'][1]+(j+.5)*RESOLUTION
                if math.hypot(cx-x,cy-y) > radius:
                    continue
                if not 0 <= i < WIDTH or not 0 <= j < HEIGHT:
                    return False, 'envelope_outside_map'
                cell = grid['data'][j*WIDTH+i]
                if cell:
                    return False, 'unknown_in_envelope' if cell == -1 else 'occupied_in_envelope'
    return True, 'known_fixture_or_projected_free_envelope'


def map_target(pose, point, behind, side):
    x,y,yaw = pose
    ball = [x+math.cos(yaw)*point[0]-math.sin(yaw)*point[1],
            y+math.sin(yaw)*point[0]+math.cos(yaw)*point[1]]
    dx,dy = .68-ball[0], -ball[1]
    length = math.hypot(dx,dy)
    if length < .1:
        raise ValueError('Degenerate ball-to-goal direction')
    ux,uy = dx/length,dy/length
    return [ball[0]-behind*ux-side*uy,ball[1]-behind*uy+side*ux]


def public_request(request, capture, check, label):
    case, prior = request['case'], request['provenance']
    t = case['query_s']
    check(label+'_same_capture', close(t,capture['timestamp_s']) and close(case['odom']['capture_s'],t),t)
    check(label+'_public_pose', close(case['odom']['pose_xyyaw'],capture['estimated_pose_map_xyyaw']),t)
    check(label+'_public_image', prior['source_file_sha256']['rgb_pixels']==capture['source_rgb_sha256'],t)
    check(label+'_no_truth', prior.get('layout')=='empty' and prior.get('odometry_origin')=='sensor_only'
          and case.get('recorded_evidence',{}).get('truth_read') is False and capture.get('truth_inputs') is False,t)
    return snapshot(case,prior)


def audit_paths(run, captures, check):
    bytime = {round(c['timestamp_s'],8): c for c in captures}
    queries = rows(run/'direct-path/queries.jsonl')
    query_count = len(queries)
    summary=json.loads((run/'direct-path/summary.json').read_text())
    check('direct_query_archive_complete',summary.get('queries')==query_count)
    check('direct_query_summary_identity',summary.get('planner')=='direct_straight_line')
    for index, query in enumerate(queries):
        t=query['case']['query_s'];c=bytime.get(round(t,8))
        check('direct_query_sequence',query.get('sequence')==index,t)
        check('direct_query_has_capture',c is not None,t)
        if c is None:
            continue
        grid=public_request(query,c,check,'direct')
        observed=next((item for item in reversed(captures) if item['timestamp_s']<=t+1e-9
                       and item.get('navigation_epoch')==c.get('navigation_epoch') and item.get('public_metric_ball')),None)
        point=None if observed is None else observed['public_metric_ball'].get('point_level_m')
        check('direct_target_has_recent_public_ball',point is not None and fresh(t,observed['timestamp_s'],.25),t)
        if point is not None:
            check('direct_target_from_observed_ball',close(grid['goal'],map_target(observed['estimated_pose_map_xyyaw'],point,.22,.035)),t)
        a,b=grid['pose'][:2],grid['goal'];distance=math.dist(a,b)
        count=max(1,math.ceil(distance/.02))
        expected=[[a[j]+(b[j]-a[j])*k/count for j in range(2)] for k in range(count+1)]
        allowed,reason=path_gate(expected,grid);response=query['response']
        check('direct_complete_line_recomputed',close(response.get('path_xy'),expected if allowed else []),t)
        check('direct_line_length_recomputed',close(response.get('path_length_m'),distance),t)
        envelope=response.get('envelope') or {}
        check('direct_full_envelope_recomputed',envelope.get('accepted') is allowed and envelope.get('reason')==reason
              and (not allowed or close(envelope.get('test_radius_m'),.1)),t)
        check('direct_response_public_only',response.get('planner')=='direct_straight_line'
              and response.get('online_hidden_state') is False and close(response.get('query_s'),t),t)
    validated={};map_count=0
    for c in captures:
        t=c['timestamp_s']
        for key in ('coarse_track_path','coarse_active_path'):
            route=c.get(key) or {};request=route.get('map_request')
            if not request:
                continue
            grid=public_request(request,c,check,'coarse')
            point=(c.get('public_metric_ball') or {}).get('point_level_m') if key=='coarse_track_path' else c['public_neutral_fine_step']['memory'].get('point_level_m')
            check('coarse_point_from_public_estimate',close(route.get('point_level_m'),point),t)
            check('coarse_target_from_observed_ball',point is not None and close(grid['goal'],map_target(grid['pose'],point,.28,.028)),t)
            allowed,reason=path_gate([grid['pose'][:2],grid['goal']],grid)
            check('coarse_complete_prefix_recomputed',route.get('allowed') is allowed and (route.get('envelope') or {}).get('reason')==reason,t)
            check('coarse_capture_binding',close(route.get('capture_s'),t) and close(route.get('source_capture_s'),t) and route.get('truth_used') is False,t)
            validated[(round(t,8),key)]=allowed;map_count+=1
        if (c.get('public_fine_entry') or {}).get('reason')=='settled_coarse_track_current_path':
            check('coarse_entry_has_current_path',validated.get((round(t,8),'coarse_track_path')) is True,t)
    return dict(queries=query_count,coarse_maps=map_count),validated


def audit_transfers(policy,captures,parameters,check):
    times=[r['timestamp_s'] for r in policy];details=[]
    for c in captures:
        tr=c.get('continuous_transfer')
        if not tr:
            continue
        t=tr['capture_s'];idx=bisect.bisect_left(times,t)
        check('transfer_has_previous_policy',0<idx<len(policy),t)
        if not 0<idx<len(policy):
            continue
        before=policy[idx-1]['public_neutral_fine_step'];current=c['public_neutral_fine_step']
        p=current['memory'].get('point_level_m');heading=current.get('goal_heading_rad');goal=current.get('goal_fix_s')
        domain=bool(vector(p,3) and parameters['transfer_x_m'][0]<=p[0]<=parameters['transfer_x_m'][1]
                    and math.hypot(*p[:2])<=.4 and parameters['transfer_y_m'][0]<=p[1]<=parameters['transfer_y_m'][1])
        expected=dict(current_guard=current.get('capture_safe') is True,memory=current['memory'].get('memory_available') is True,
                      near_domain=domain,heading=number(heading) and abs(heading)<=math.radians(10),goal=fresh(t,goal,2.))
        for name,value in expected.items():
            check('transfer_'+name+'_recomputed',tr['checks'].get(name) is value,t)
        check('transfer_point_binding',close(tr.get('point_level_m'),p),t)
        allowed=all(expected.values());check('transfer_result_recomputed',tr.get('transferred') is allowed,t)
        if allowed:
            check('transfer_ownership',before['coarse_track'] and before['state']=='TRACK' and not current['coarse_track'] and current['state']=='TRACK' and not c.get('coarse_active_path'),t)
            check('transfer_clocks_preserved',before['step_start_s']==current['step_start_s']==tr['step_start_s']
                  and before['anchor_capture_s']==current['anchor_capture_s']==tr['anchor_s'],t)
            check('transfer_memory_anchor_preserved',before['memory']['ball_capture_timestamp_s']==current['memory']['ball_capture_timestamp_s'],t)
            check('transfer_head_preserved',before['head_offsets']==current['head_offsets'] and before['return_head_target_native']==current['return_head_target_native'],t)
            check('transfer_current_guard',current['capture_safe'] and close(current['capture_guard_s'],t),t)
        else:
            check('transfer_rejection_stops',current['state']=='SETTLE' and zero(current['velocity'],3),t)
        details.append(dict(t=t,transferred=allowed,point=p,checks=expected))
    return details


def audit_motion(policy,captures,readbacks,parameters,validated_paths,check):
    reads={round(r['timestamp_s'],8):(r.get('readback') or {}).get('actual_packet') for r in readbacks}
    captures_bytime={round(c['timestamp_s'],8):c for c in captures}
    active_key=None;filter_epoch=None;heading=None;heading_anchor=None;last_read=None
    lateral_lp=heading_lp=None;rate_lp=0.;brake=None;brake_command=None
    previous=None;history=[];history_anchor=None;previous_prediction=None
    maximum_error=0.;track_ticks=turn_ticks=macro_ticks=step_ticks=0;onsets=[]
    macro_primitive=None;macro_onsets=[]
    for r in policy:
        t=r['timestamp_s'];f=r['public_neutral_fine_step'];state=f['state'];epoch=r.get('navigation_epoch')
        key=(epoch,f.get('started_s'),f.get('step_start_s'),f.get('anchor_capture_s'))
        dt=.02 if previous is None else t-previous['timestamp_s']
        packet=reads.get(round(t,8))
        if state in ('RETURN','STEP','TURN','TRACK','MACRO') and any(abs(v)>1e-9 for v in r['actual_velocity_command']):
            memory=r['public_handoff']['memory']
            check('moving_public_guards',f.get('capture_safe') is True and fresh(t,f.get('capture_guard_s'),.2)
                  and memory.get('memory_available') is True and number(memory.get('ball_age_s')) and 0<=memory['ball_age_s']<=6.5
                  and r['public_handoff']['state']=='APPROACH' and f.get('override') is True,t)
            check('moving_public_packet',packet is not None and packet.get('orientation_valid') is True and fresh(t,packet.get('read_timestamp_s'),.08),t)
            check('moving_no_kick',not r.get('kick_requested'),t)
        if state in ('RETURN','SETTLE','REACQUIRE'):
            check('observation_body_stopped',zero(r['actual_velocity_command'],3),t)
        if state=='REACQUIRE' and previous and previous['public_neutral_fine_step']['state']=='SETTLE':
            check('reacquire_requires_measured_quiet',f.get('capture_safe') is True
                  and number(f.get('quiet_started_s')) and t-f['quiet_started_s']>=.5-1e-9,t)
        if f.get('reason') in ('public_guard_stop_requires_fresh_reacquisition','public_goal_stop_requires_fresh_reacquisition') and state=='SETTLE':
            check('guard_stop_zero_before_new_observation',zero(r['actual_velocity_command'],3) and not r.get('kick_requested'),t)
            if previous and previous.get('navigation_epoch')==epoch and previous['public_neutral_fine_step']['state']=='TRACK':
                pf=previous['public_neutral_fine_step']
                check('guard_stop_preserves_movement_clock',f['started_s']==pf['started_s'] and f['anchor_capture_s']==pf['anchor_capture_s'],t)
                invalid=(f.get('capture_safe') is False if f['reason'].startswith('public_guard') else not fresh(t,f.get('goal_fix_s'),2.))
                check('guard_stop_withdrawal_reason_recomputed',invalid,t)

        if f.get('coarse_track') and state in ('RETURN','TRACK','TURN'):
            c=captures_bytime.get(round(f['capture_guard_s'],8))
            route_key='coarse_active_path' if c and c.get('coarse_active_path') else 'coarse_track_path'
            check('coarse_motion_current_path',c is not None and validated_paths.get((round(c['timestamp_s'],8),route_key)) is True,t)
            if state=='TRACK':
                p=f['memory']['point_level_m']
                check('coarse_motion_bounds',p[0]>.115 and t-f['step_start_s']<30.+1e-8 and t-f['anchor_capture_s']<=6.5+1e-8,t)
                check('coarse_no_brake_or_kick',f.get('track_brake_started_s') is None and not r['kick_requested'],t)
        if previous and previous.get('navigation_epoch')==epoch and previous['public_neutral_fine_step']['state']=='TURN' and state in ('TURN','SETTLE'):
            pf=previous['public_neutral_fine_step']
            if f.get('reason') not in ('public_guard_stop_requires_fresh_reacquisition','public_goal_stop_requires_fresh_reacquisition','parking_path_requires_reobservation') and packet is not None:
                start=f['turn_start_s'];startrow=next(q for q in policy if close(q['timestamp_s'],start))
                sf=startrow['public_neutral_fine_step'];initial=sf['goal_heading_rad']
                def bearing(point,h):
                    foot=math.atan2(point[1],point[0])+math.asin(min(1.,.035/max(math.hypot(*point[:2]),.035)))
                    return h-max(-.06,min(.06,angle(h-foot)))
                sign=math.copysign(1,initial if f.get('turn_goal_only') else bearing(sf['memory']['point_level_m'],initial))
                h=f['goal_heading_rad'] if f.get('turn_goal_only') else bearing(f['memory']['point_level_m'],f['goal_heading_rad'])
                g,w=packet['gravity_body'],packet['gyro_body_rad_s'];rate=-sum(x*y for x,y in zip(w,g))/math.sqrt(sum(v*v for v in g))
                elapsed=t-start;predictive=rate*sign>.1 and (h-.12*rate)*sign<=.015
                if f.get('turn_goal_only'):
                    check('turn_initial_heading_immutable',close(f['turn_initial_goal_heading_rad'],initial),t)
                    predictive=predictive and elapsed>=.16-1e-9 and f['goal_fix_s']>start and (initial-h)*sign>=.03
                expected_stop=(elapsed>=.08-1e-9 and (h*sign<=.015 or predictive)) or elapsed>=.8-1e-9
                check('turn_stop_from_public_progress', (state=='SETTLE') is bool(expected_stop),t)
        if state=='STEP':
            step_ticks+=1
            final=f.get('final_stride') is True
            point=f['memory'].get('point_level_m');start=f.get('step_start_s')
            maximum=1.2 if final else .6;limit=.112 if final else .105
            expected=[.25 if final else .2,0.,0.]
            check('step_actual_fixed_command',close(r['actual_velocity_command'],expected)
                  and close(f.get('velocity'),expected) and f.get('override') is True,t)
            check('step_declared_duration',close(f.get('maximum_step_s'),maximum),t)
            check('step_bounded_duration_and_distance',number(start) and 0<=t-start<maximum-1e-9
                  and vector(point,3) and limit<point[0]<=(.4 if f.get('motion_kind')=='FAR_STEP' else .25)
                  and -.10<=point[1]<=.08,t)
            first=close(t,start)
            check('step_heading_current',number(f.get('step_goal_heading_rad'))
                  and number(f.get('goal_heading_rad'))
                  and abs(angle(f['goal_heading_rad']-f['step_goal_heading_rad']))<=.15
                  and fresh(t,f.get('goal_fix_s'),2. if first else 1.1),t)
            if first:
                check('step_heading_anchor_from_camera',close(f.get('step_goal_heading_rad'),f.get('goal_heading_rad')),t)
            elif previous and previous.get('navigation_epoch')==epoch:
                pf=previous['public_neutral_fine_step']
                check('step_start_and_heading_immutable',pf.get('step_start_s')==start
                      and pf.get('step_goal_heading_rad')==f.get('step_goal_heading_rad')
                      and pf.get('final_stride')==f.get('final_stride'),t)
        if previous and previous.get('navigation_epoch')==epoch and previous['public_neutral_fine_step']['state']=='STEP' and state!='STEP':
            pf=previous['public_neutral_fine_step'];point=f['memory'].get('point_level_m')
            check('step_exit_stops_body',zero(r['actual_velocity_command'],3),t)
            if state in ('SETTLE','IDLE') and f.get('reason') in (None,'released_to_kick_handoff'):
                maximum=1.2 if pf.get('final_stride') else .6;limit=.112 if pf.get('final_stride') else .105
                check('step_stop_from_duration_or_public_range',number(pf.get('step_start_s'))
                      and (t-pf['step_start_s']>=maximum-1e-9 or vector(point,3) and point[0]<=limit)
                      and close(f.get('step_end_s'),t),t)
        if previous and previous.get('navigation_epoch')==epoch and previous['public_neutral_fine_step']['state']=='MACRO' and state!='MACRO':
            pf=previous['public_neutral_fine_step']
            check('macro_exit_stops_body',zero(r['actual_velocity_command'],3),t)
            if state=='SETTLE' and f.get('reason') is None:
                duration=pf['primitive']['forward_duration_s']
                check('macro_duration_stop_clock',t-pf['step_start_s']>=duration-1e-9
                      and t-pf['step_start_s']<duration+.021
                      and close(f.get('step_end_s'),t),t)
        if state=='MACRO':
            macro_ticks+=1
            point=f['memory'].get('point_level_m')
            check('macro_point_domain',vector(point,3) and .075<point[0]<=.25
                  and -.10<=point[1]<=.08,t)
            primitive=f.get('primitive') or {}
            command=primitive.get('velocity_command');duration=primitive.get('forward_duration_s')
            candidate_valid=bool(vector(command,3) and close(command[0],.3)
                                 and command[1] in (-.2,-.1,0.) and close(command[2],0.)
                                 and duration in (.6,.8))
            check('macro_declared_sampled_command',candidate_valid,t)
            check('macro_public_goal_current',fresh(t,f.get('goal_fix_s'),2.)
                  and number(f.get('goal_heading_rad')) and abs(f['goal_heading_rad'])<=.4,t)
            check('macro_bounded_duration',candidate_valid and number(f.get('step_start_s'))
                  and 0<=t-f['step_start_s']<duration-1e-9,t)
            check('macro_actual_owner',f.get('motion_kind')=='MACRO' and f.get('override') is True
                  and not f.get('coarse_track') and close(f.get('velocity'),r['actual_velocity_command']),t)
            if packet is None or not candidate_valid:
                check('macro_feedback_inputs_available',False,t);previous=r;continue
            if key!=active_key:
                active_key=key;lateral_lp=heading_lp=None;brake=None;brake_command=None
                # The production filter belongs to the navigation object, not
                # a maneuver: retain its TRACK/MACRO history within this epoch.
                if epoch!=filter_epoch:
                    filter_epoch=epoch;rate_lp=0.
                heading_anchor=f.get('motion_heading_anchor') or {}
                heading=heading_anchor.get('heading_rad')
                check('macro_anchor_from_current_camera',number(heading)
                      and close(heading,f.get('goal_heading_rad'))
                      and close(heading_anchor.get('public_goal_fix_s'),f.get('goal_fix_s'))
                      and heading_anchor.get('source')=='public_camera_goal_at_movement_start'
                      and heading_anchor.get('seq')==packet['seq']
                      and close(heading_anchor.get('read_s'),packet['read_timestamp_s']),t)
                if not number(heading):
                    previous=r;continue
                last_read=(heading_anchor['seq'],heading_anchor['read_s'])
                point=f['memory'].get('point_level_m')
                check('macro_initial_selection_domain',vector(point,3) and .13<=point[0]<=.24
                      and -.10<=point[1]<=.06 and abs(heading)<=.08
                      and close(t,f['step_start_s']),t)
                macro_primitive=primitive
                macro_onsets.append(dict(t=t,epoch=epoch,duration_s=duration,
                                         nominal_command=command,heading=heading))
            else:
                check('macro_anchor_immutable',f.get('motion_heading_anchor')==heading_anchor,t)
                check('macro_primitive_immutable',primitive==macro_primitive,t)
            seq,read=packet['seq'],packet['read_timestamp_s']
            g,w=packet['gravity_body'],packet['gyro_body_rad_s'];norm=math.sqrt(sum(v*v for v in g))
            rate=-sum(x*y for x,y in zip(w,g))/norm
            check('macro_heading_read_monotone',seq>=last_read[0] and read>=last_read[1],t)
            if seq>last_read[0]:
                delta=read-last_read[1];check('macro_heading_read_gap',0<delta<=.08,t)
                heading=angle(heading-rate*delta);last_read=(seq,read)
            else:
                check('macro_heading_duplicate_clock',read==last_read[1],t)
            check('macro_heading_independently_integrated',number(f.get('motion_heading_prediction_rad'))
                  and abs(angle(heading-f['motion_heading_prediction_rad']))<1e-9
                  and f.get('motion_heading_prediction_is_measurement') is False,t)
            rate_lp+=(1-math.exp(-dt/.1))*(rate-rate_lp)
            expected=[command[0],command[1],max(-.5,min(.5,2.5*heading-.25*rate_lp))]
            point=f['memory']['point_level_m']
            error=-math.sin(heading)*point[0]+math.cos(heading)*point[1]+.035
            check('macro_lateral_and_shot_error_recomputed',close(f.get('macro_lateral_command'),command[1])
                  and close(f.get('macro_shot_error_m'),error),t)
            command_error=max(abs(a-b)for a,b in zip(expected,r['actual_velocity_command']))
            maximum_error=max(maximum_error,command_error)
            check('macro_actual_command_recomputed',command_error<1e-9,t,
                  dict(expected=expected,actual=r['actual_velocity_command'],filtered_rate_rad_s=rate_lp))
            previous=r
            continue
        if state!='TRACK':
            if brake is not None and key==active_key:
                check('braking_ends_at_zero',zero(r['actual_velocity_command'],3) and t-brake<=.18+1e-8,t)
                brake=None;brake_command=None
            if state=='TURN':
                turn_ticks+=1
                turn_sign=math.copysign(1,f['turn_initial_goal_heading_rad']) if f.get('turn_goal_only') else None
                if turn_sign is None:
                    # The direction was fixed on the first TURN tick and is
                    # independently recoverable from its initial public point.
                    start=f['turn_start_s'];startrow=next(q for q in policy if close(q['timestamp_s'],start))
                    sf=startrow['public_neutral_fine_step'];point=sf['memory']['point_level_m'];h=sf['goal_heading_rad']
                    foot=math.atan2(point[1],point[0])+math.asin(min(1.,.035/max(math.hypot(*point[:2]),.035)))
                    bearing=h-max(-.06,min(.06,angle(h-foot)))
                    turn_sign=math.copysign(1,bearing)
                expected=[-.2,0.,.5*turn_sign] if f.get('parking_arc') else [0.,0.,.8*turn_sign]
                check('turn_actual_command',close(r['actual_velocity_command'],expected),t)
                check('turn_bounded',fresh(t,f.get('turn_start_s'),.8) and fresh(t,f.get('goal_fix_s'),8. if f.get('parking_arc') else 2.),t)
                if f.get('parking_arc'):
                    route=f.get('parking_path') or {};request=route.get('map_request')
                    check('parking_current_path',route.get('allowed') is True and fresh(t,route.get('capture_s'),.2) and request is not None,t)
                    if request:
                        capture=captures_bytime.get(round(route['capture_s'],8))
                        check('parking_has_capture',capture is not None,t)
                        if capture:
                            grid=public_request(request,capture,check,'parking');pose=grid['pose']
                            target=[pose[0]-.16*math.cos(pose[2]),pose[1]-.16*math.sin(pose[2])]
                            check('parking_declared_rear_target',close(grid['goal'],target) and close(route.get('distance_m'),.16)
                                  and route.get('rear_sensor_coverage') is False,t)
                            check('parking_full_rear_envelope',path_gate([pose[:2],target],grid)[0],t)
            previous=r
            continue
        track_ticks+=1
        check('track_command_owner',close(r['actual_velocity_command'],f['velocity']) and f.get('override') is True,t)
        check('track_fresh_goal',fresh(t,f.get('goal_fix_s'),2.) and abs(f['goal_heading_rad'])<=.45,t)
        check('track_bounded',0<=t-f['step_start_s']<30.+1e-8 and f['memory']['point_level_m'][0]>.09,t)
        if packet is None:
            check('track_packet_available',False,t);previous=r;continue
        if key!=active_key:
            active_key=key;lateral_lp=heading_lp=None;brake=None;brake_command=None
            if epoch!=filter_epoch:
                filter_epoch=epoch;rate_lp=0.
            heading_anchor=f['motion_heading_anchor'];heading=heading_anchor['heading_rad']
            last_read=(heading_anchor['seq'],heading_anchor['read_s'])
            check('motion_anchor_from_camera',close(heading,f['goal_heading_rad']) and close(heading_anchor['public_goal_fix_s'],f['goal_fix_s'])
                  and fresh(t,heading_anchor['public_goal_fix_s'],2.) and heading_anchor['source']=='public_camera_goal_at_movement_start',t)
        else:
            check('motion_anchor_immutable',f['motion_heading_anchor']==heading_anchor,t)
        seq,read=packet['seq'],packet['read_timestamp_s']
        g,w=packet['gravity_body'],packet['gyro_body_rad_s'];norm=math.sqrt(sum(v*v for v in g));rate=-sum(x*y for x,y in zip(w,g))/norm
        check('motion_heading_read_monotone',seq>=last_read[0] and read>=last_read[1],t)
        if seq>last_read[0]:
            delta=read-last_read[1];check('motion_heading_read_gap',0<delta<=.08,t)
            heading=angle(heading-rate*delta);last_read=(seq,read)
        else:
            check('motion_heading_duplicate_clock',read==last_read[1],t)
        check('motion_heading_independently_integrated',abs(angle(heading-f['motion_heading_prediction_rad']))<1e-9
              and f['motion_heading_prediction_is_measurement'] is False,t)
        point=f['memory']['point_level_m'];prediction=f.get('predictive_braking');declared_brake=f.get('track_brake_started_s')
        if brake is None:
            anchor=f['anchor_capture_s']
            if anchor!=history_anchor or history and not 0<t-history[-1][0]<=.025:
                history=[]
            history_anchor=anchor;history.append((t,point[0]));history=[q for q in history if t-q[0]<=.24+1e-9]
            span=t-history[0][0];valid=len(history)>=8 and span>=.16-1e-9;threshold=.125
            if valid:
                mt=sum(q[0] for q in history)/len(history);mx=sum(q[1] for q in history)/len(history)
                slope=sum((tt-mt)*(xx-mx) for tt,xx in history)/sum((tt-mt)**2 for tt,xx in history)
                speed=max(0.,min(.5,-slope));threshold=max(.125,min(.17,.11+.28*speed))
                check('predictor_speed_recomputed',prediction is not None and close(prediction.get('closing_speed_m_s'),speed)
                      and prediction.get('anchor_s')==anchor and prediction.get('truth_used') is False,t)
            check('predictor_same_anchor_OLS',prediction is not None and prediction.get('valid') is valid
                  and close(prediction.get('threshold_m'),threshold) and prediction.get('sample_count')==len(history) and close(prediction.get('span_s'),span),t)
            lateral=-math.sin(heading)*point[0]+math.cos(heading)*point[1]+(.028 if f['coarse_track'] else .035)
            if lateral_lp is None:
                lateral_lp,heading_lp=lateral,heading
            else:
                lateral_lp+=(1-math.exp(-dt/.20))*(lateral-lateral_lp)
                heading_lp+=(1-math.exp(-dt/.12))*(heading-heading_lp)
            rate_lp+=(1-math.exp(-dt/.1))*(rate-rate_lp)
            expected=[.3 if point[0]>.17 else .25,max(-.2,min(.2,.65*math.sin(heading_lp)+10*lateral_lp)),max(-.5,min(.5,2.5*heading_lp-.25*rate_lp))]
            check('track_filter_values_recomputed',close(f['track_lateral_filtered_m'],lateral_lp) and close(f['track_goal_heading_filtered_rad'],heading_lp),t)
            if declared_brake is not None:
                check('brake_onset_threshold_and_clock',close(declared_brake,t) and .115<point[0]<=threshold+1e-12,t)
                check('brake_initial_command_recomputed',close(f['track_brake_initial_command'],expected,1e-9),t)
                brake=t;brake_command=list(expected);onsets.append(dict(t=t,point=point,threshold_m=threshold))
        else:
            check('brake_clock_and_command_unchanged',close(declared_brake,brake) and close(f['track_brake_initial_command'],brake_command),t)
            check('predictor_not_updated_during_brake',prediction==previous_prediction,t)
            expected=[v*max(0.,1-(t-brake)/.16) for v in brake_command]
        error=max(abs(a-b) for a,b in zip(expected,r['actual_velocity_command']));maximum_error=max(maximum_error,error)
        check('actual_track_command_recomputed',error<1e-9,t,dict(expected=expected,actual=r['actual_velocity_command']))
        previous_prediction=prediction;previous=r
    return dict(track_ticks=track_ticks,turn_ticks=turn_ticks,macro_ticks=macro_ticks,step_ticks=step_ticks,
                maximum_command_error=maximum_error,brake_onsets=onsets,macro_onsets=macro_onsets)


def audit_kicks(policy,captures,manifest,check):
    verified,blind=audit_blind(policy,captures,manifest,check);kicks=[]
    for index,r in enumerate(policy):
        if not r.get('kick_requested') or index and policy[index-1].get('kick_requested'):
            continue
        t=r['timestamp_s'];h=r['public_handoff'];f=r['public_neutral_fine_step'];memory=h['memory'];p=memory['point_level_m']
        domain=(.07<=p[0]<=.15 and -.10<=p[1]<=.01) if t in verified else (.07<=p[0]<=.125 and -.065<=p[1]<=-.005)
        checks=dict(owner=h['state']=='KICK' and h['kick_runtime_authorized'] and not f['override'],point=domain,
                    memory=memory['memory_available'] and 0<=memory['ball_age_s']<=6.5,heading=abs(h['last_goal_heading_rad'])<=math.radians(15),
                    goal_age=fresh(t,h['last_goal_fix_s'],2.),quiet=t-h['headreturn_quiet_start_s']>=.5-1e-9,
                    body_zero=zero(r['actual_velocity_command'],3),head_neutral=zero(r['actual_head_command_rad'],4))
        window=[q for q in policy if h['headreturn_quiet_start_s']-1e-9<=q['timestamp_s']<=t+1e-9 and q.get('navigation_epoch')==r.get('navigation_epoch')]
        check('kick_quiet_window_coverage',len(window)>=26 and close(window[0]['timestamp_s'],h['headreturn_quiet_start_s'])
              and all(close(b['timestamp_s']-a['timestamp_s'],.02) for a,b in zip(window,window[1:])),t)
        for q in window:
            packet=q['IMU_prediction'];g=packet['gravity_body'];gyro=packet['gyro_body_rad_s'];qd=packet['encoder_qd_rad_s'];tilt=math.atan2(math.hypot(*g[:2]),-g[2])
            check('kick_quiet_actual_packets',zero(q['actual_head_command_rad'],4) and zero(q['actual_velocity_command'],3)
                  and math.sqrt(sum(v*v for v in gyro))<=.1 and math.sqrt(sum(v*v for v in qd)/len(qd))<=.25 and tilt<=.35,q['timestamp_s'])
        for name,passed in checks.items():
            check('kick_'+name,passed,t)
        kicks.append(dict(t=t,checks=checks,blind_domain_verified=t in verified))
    return dict(blind_arrival=blind,kicks=kicks)


def audit_records(run,manifest,policy,captures,readbacks,parameters,check):
    paths,validated=audit_paths(run,captures,check)
    if any(r['public_neutral_fine_step'].get('coarse_track') and r['public_neutral_fine_step'].get('state')=='TRACK' for r in policy):
        check('actual_approach_has_archived_queries',paths['queries']>0)
    transfers=audit_transfers(policy,captures,parameters,check)
    motion=audit_motion(policy,captures,readbacks,parameters,validated,check)
    kicks=audit_kicks(policy,captures,manifest,check)
    return dict(paths=paths,transfers=transfers,motion=motion,**kicks)


def audit(run):
    check=Checks();parameters,source=course_parameters()
    epoch_report=validate_epochs(run)
    manifest=json.loads((run/'manifest.json').read_text());profile=manifest.get('approach_control') or {}
    check('source_unchanged',not manifest.get('source_changed_during_run'))
    check('course_parameters_declared',all(close(profile.get('parameters',{}).get(k),v) for k,v in parameters.items()))
    check('continuous_predictive_profile',profile.get('continuous_fine_transfer') is True and profile.get('predictive_braking') is True)
    check('straight_path_selected',(manifest.get('path_planner') or {}).get('kind')=='direct')
    summary=json.loads((run/'direct-path/summary.json').read_text())
    cleanup=manifest.get('path_cleanup') or {}
    check('direct_cleanup_count_binding',cleanup.get('queries')==summary.get('queries'))
    check('direct_cleanup_source_binding',cleanup.get('source_sha256')==summary.get('source_sha256')==sha(Path(__file__).with_name('direct_path.py')))
    check('direct_no_external_service',cleanup.get('external_service_started') is False and summary.get('external_service_started') is False)
    check('blind_arrival_enabled',(manifest.get('blind_arrival') or {}).get('profile')=='near-blind-v1')
    selected=['approach_control.py','direct_path.py','predictive_brake.py','blind_arrival.py','kick_handoff.py','executor.py']
    snapshots=manifest.get('source_snapshot',{})
    for name in selected:
        # The recorder also has an executor.py; bind the actual course module.
        entries=[rec for rec in snapshots.values()
                 if Path(rec.get('snapshot',''))==Path('source-snapshot/onboard')/name]
        check('source_snapshot_present',len(entries)==1,detail=name)
        if len(entries)==1:
            record=entries[0];saved=run/record['snapshot'];current=Path(__file__).with_name(name)
            check('source_snapshot_integrity',saved.is_file() and sha(saved)==record['sha256'],detail=name)
            check('source_snapshot_matches_course',current.is_file() and sha(current)==record['sha256'],detail=name)
    files=['policy-trace.jsonl','control.jsonl','public-memory-readback.jsonl']
    policy,captures,readbacks=(rows(run/name) for name in files)
    details=audit_records(run,manifest,policy,captures,readbacks,parameters,check)
    return dict(schema='microduck.direct-navigation-actual-audit.v1',passed=not check.failure_count,
                check_counts=dict(check.counts),failure_count=check.failure_count,errors=check.errors,details=details,
                navigation_epoch_validation=epoch_report,controlled_parameters=parameters,source_sha256={str(p):sha(p) for p in [source,Path(__file__),
                Path(__file__).with_name('audit_blind_arrival_actual.py'),LAB/'onboard_geometry/map_contract.py',
                LAB/'onboard_geometry/obstacle_contract.py',LAB/'audit_epochs.py',run/'manifest.json',*(run/name for name in files),run/'direct-path/queries.jsonl',run/'direct-path/summary.json']},
                scope='Independent public paths, same-read heading/velocity feedback, predictive braking, continuous transfer and quiet kick authority. No truth, actuator or ONNX substitution; physical/model/sensor audits are separate.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a new audit output; retain earlier evidence')
    try:
        result=audit(args.run)
    except Exception as error:
        result=dict(schema='microduck.direct-navigation-actual-audit.v1',passed=False,error=repr(error),traceback=traceback.format_exc())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','details')},indent=2,allow_nan=False))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
