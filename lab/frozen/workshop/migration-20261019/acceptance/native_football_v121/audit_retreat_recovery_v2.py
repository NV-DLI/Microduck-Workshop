#!/usr/bin/env python3
"""Independent audit of v121 bounded pre-kick retreat recovery in one completed run.

v121 motion: exact fixed-duration (1.6 s = 80 ticks) public-heading-feedback helper.
Trusts no self-reported 'safe' flag. It does NOT import retreat_recovery.py,
recovery_handoff.py or heading_feedback.py. It:
  * re-derives the helper law (K 2.5/.25, LP .12/.10 s, clip .5, start h/LP at the
    approved +offset, rate LP at the entry rate) and every latched fault from the RAW
    delivered packets in policy-trace IMU_prediction, and compares it with the command
    actually sent on every motion tick; checks motion index/count/window (first motion
    tick = first policy tick after the authorizing capture = stored retreat start),
    vx exact, vy 0, no post-stop authority, early stop => terminal FAILED, normal stop
    => public backward >= .10 before SETTLE, approved offset/helper/duration source;
  * replays raw same-read public packets (public-memory-readback.jsonl) through
    the pinned SameReadMemoryAdapter + the controller's SoleContact odometry to
    recompute relative odometry and every memory invalidation;
  * recomputes the original 10 cm check_path from each archived map request
    (case + provenance) and checks the fixed rear target and surveyed prior;
  * reconstructs which body command was admissible on every 50 Hz tick
    (approved command only in RETREAT, fresh guarded capture <=.2 s, continuous
    odometry, neutral sent head, stop rule not yet met, original clock) and
    compares it with the command actually sent;
  * checks entry whitelist/budget/reserve/memory clocks, REACQUIRE-only current
    complete re-anchoring, release domain, cumulative counters, scorer
    monotonicity, no kick and no foot-ball force during recovery, single kick;
  * evaluates actual stopping from privileged physics (evaluation only).
Exit 0 only if every check passes. No physics/scorer/state is modified.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

M = Path('@COURSE@/workshop/migration-20261019')
WHITELIST = {'propagated_point_outside_training_box', 'goal_heading_outside_development_gate'}
NEUTRAL = {'HOLD', 'AUTHORIZE', 'RETREAT', 'SETTLE'}
LIMITS = dict(max_recoveries=3, reserve_s=30., capture_age_s=.2, policy_gap_s=.025, ttl_s=4.5, read_age_s=.02000001,
              entry_x=(.06, .14), entry_abs_y=.10, release_x=(.20, .38), release_abs_y=.10, hold_s=3., authorize_s=.5,
              settle_s=3., reacquire_s=8., forward_m=.01, yaw_rad=.25, lateral_m=.05, onset_min_m=.01, goal_fix_s=8.,
              radius_m=.10, contact_n=.001, odometry_tol=1e-9, physical_odometry_error_m=.03,
              vx=-.35, duration_s=1.6, move_steps=80, offsets_rad=(.10, .20), plan_distance_m=.28, max_backward_m=.24,
              min_backward_m=.10, onset_s=1.0, stage_a_backward_m=(.12, .24), post_stop_drift_m=.04, command_tol=1e-9,
              k_heading=2.5, k_rate=.25, tau_heading_s=.12, tau_rate_s=.10, wz_max=.5, read_dt_max=.025, packet_age_max=.08,
              policy_dt_max=.025, min_gravity_norm=.5)


class FeedbackReplay:
    """Independent re-implementation of the approved public heading-feedback law (one recovery)."""
    def __init__(self, vx, offset):
        self.vx = vx; self.offset = offset; self.started = False; self.h = self.hl = self.rl = 0.
        self.last_seq = self.last_read = self.last_payload = self.last_now = None; self.fault = None

    def step(self, now, p):
        if self.fault:
            return [0., 0., 0.]
        dt = None if self.last_now is None else now-self.last_now
        if self.last_now is None or now > self.last_now:
            self.last_now = now
        L = LIMITS
        if dt is not None and not 0 < dt <= L['policy_dt_max']+1e-12:
            self.fault = 'policy_clock_gap'; return [0., 0., 0.]
        p = p or {}
        if p.get('orientation_valid') is not True:
            self.fault = 'invalid_packet'; return [0., 0., 0.]
        try:
            g = np.asarray(p.get('gravity_body'), float); w = np.asarray(p.get('gyro_body_rad_s'), float)
        except (TypeError, ValueError):
            g = w = np.full(1, np.nan)
        if g.shape != (3,) or w.shape != (3,) or not np.isfinite(g).all() or not np.isfinite(w).all() or np.linalg.norm(g) < L['min_gravity_norm']:
            self.fault = 'gravity_or_gyro'; return [0., 0., 0.]
        seq, sample, read, enc = p.get('seq'), p.get('timestamp_s'), p.get('read_timestamp_s'), p.get('encoder_read_timestamp_s')
        if not (isinstance(seq, int) and not isinstance(seq, bool) and seq >= 0):
            self.fault = 'sequence'; return [0., 0., 0.]
        if not (finite(sample, read, enc) and min(sample, read, enc) >= 0 and enc == read and sample <= read+1e-9
                and 0 <= now-read <= L['packet_age_max'] and 0 <= now-sample <= L['packet_age_max']):
            self.fault = 'packet_clock'; return [0., 0., 0.]
        rate = float(np.dot(w, -g/np.linalg.norm(g))); payload = (float(sample), float(read), tuple(w.tolist()), tuple(g.tolist()))
        if not self.started:
            self.started = True; self.h = self.hl = self.offset; self.rl = rate
        else:
            if seq == self.last_seq:
                if payload != self.last_payload:
                    self.fault = 'changed_duplicate'; return [0., 0., 0.]
            elif seq < self.last_seq or read <= self.last_read:
                self.fault = 'nonmonotone'; return [0., 0., 0.]
            elif read-self.last_read > L['read_dt_max']+1e-12:
                self.fault = 'read_gap'; return [0., 0., 0.]
            else:
                a = self.h-rate*(read-self.last_read); self.h = math.atan2(math.sin(a), math.cos(a))
            self.hl += (1.-math.exp(-dt/L['tau_heading_s']))*(self.h-self.hl)
            self.rl += (1.-math.exp(-dt/L['tau_rate_s']))*(rate-self.rl)
        self.last_seq, self.last_read, self.last_payload = seq, float(read), payload
        return [self.vx, 0., max(-L['wz_max'], min(L['wz_max'], L['k_heading']*self.hl-L['k_rate']*self.rl))]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def finite(*xs):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in xs)


def key(t):
    return round(float(t), 6)


class Audit:
    def __init__(self):
        self.checks = {}

    def check(self, name, ok, **details):
        entry = self.checks.setdefault(name, {'passed': True, 'violations': [], 'count': 0})
        entry['count'] += 1
        if not ok:
            entry['passed'] = False
            if len(entry['violations']) < 25:
                entry['violations'].append(details)

    def note(self, name, **details):
        self.checks.setdefault(name, {'passed': True, 'violations': [], 'count': 0}).update(details)


def load_leg(controller, fixture_leg):
    if fixture_leg is not None:
        spec = importlib.util.spec_from_file_location('audit_fixture_leg', fixture_leg)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module.make_leg(), 'TEST_FIXTURE_LEG:'+str(fixture_leg)
    sys.path[:0] = [str(controller), str(M/'football'), str(M/'sensors')]
    from leg_odometry import LegFK
    spec = importlib.util.spec_from_file_location('audit_contact_odometry', Path(controller)/'contact_odometry.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.SoleContactLegOdometry(LegFK(M/'sensors/sources/robot_allcollisions.xml', require_pinned=False)), 'public_sole_contact'


class Replay:
    """Raw same-read packet replay; pose and invalidation are recomputed, never copied."""
    def __init__(self, reads, leg):
        sys.path[:0] = [str(M/'acceptance'), str(M/'football')]
        from public_ball_read_adapter_v2 import SameReadMemoryAdapter
        first = next(r['readback']['actual_packet'] for r in reads if (r.get('readback') or {}).get('actual_packet'))
        self.adapter = SameReadMemoryAdapter(leg, first['joint_names'])
        self.epoch = 0; self.invalidations = []
        memory = self.adapter.memory; original = memory.invalidate

        def counted(reason, _o=original):
            self.epoch += 1; self.invalidations.append(reason); return _o(reason)
        memory.invalidate = counted
        self.reads = reads; self.index = 0; self.errors = []

    def advance(self, t, strict=False):
        while self.index < len(self.reads) and (self.reads[self.index]['timestamp_s'] < t-1e-9 if strict else self.reads[self.index]['timestamp_s'] <= t+1e-9):
            row = self.reads[self.index]; self.index += 1
            rb = row.get('readback') or {}; q = rb.get('actual_packet')
            packet = None
            if rb.get('reason') != 'no_new_delivered_read' and q is not None:
                packet = dict(source='body_powerboard_control_imu', axes='trunk FLU', seq=q['seq'], timestamp_s=q['sample_timestamp_s'],
                              read_timestamp_s=q['read_timestamp_s'], delivered_timestamp_s=q['delivered_timestamp_s'],
                              encoder_read_timestamp_s=q['encoder_timestamp_s'], orientation_valid=q['orientation_valid'],
                              encoder_q_rad=q['q_rad'], encoder_qd_rad_s=q['qd_rad_s'], gyro=q['gyro_body_rad_s'], gravity=q['gravity_body'],
                              gyro_clipped=rb.get('source_gyro_clipped'), accel_clipped=rb.get('source_accel_clipped'))
            try:
                self.adapter.consume(packet, now_s=row['timestamp_s'])
            except ValueError as error:
                self.errors.append({'t': row['timestamp_s'], 'error': str(error)})

    def pose(self, t):
        m = self.adapter.memory
        if m.previous is None or not 0 <= t-m.previous[0] <= LIMITS['read_age_s']:
            return None
        p = m.predicted_pose(t)
        return [float(p[0]), float(p[1]), float(p[3])]


def relative(origin, pose):
    d = np.asarray(pose[:2])-np.asarray(origin[:2]); c, s = math.cos(origin[2]), math.sin(origin[2])
    dyaw = math.atan2(math.sin(pose[2]-origin[2]), math.cos(pose[2]-origin[2]))
    return float(c*d[0]+s*d[1]), float(-s*d[0]+c*d[1]), dyaw


def recheck_path(path, audit, t, fixed_target, expected_target=None):
    """Recompute the original check_path from the archived map request."""
    sys.path[:0] = [str(M/'football')]
    from nav2_service.map_contract import snapshot, check_path
    from nav2_service.obstacle_contract import prior
    req = path.get('map_request') or {}
    case, prov = req.get('case'), req.get('provenance')
    ok = isinstance(case, dict) and isinstance(prov, dict)
    audit.check('rear_path_archived_request', ok, t=t)
    if not ok:
        return False
    pose = path.get('pose_map_xyyaw'); target = path.get('target_map_xy')
    audit.check('rear_path_surveyed_empty_prior', prov.get('layout') == 'empty' and prov.get('static_prior') == prior('empty')
                and prov.get('odometry_origin') == 'sensor_only' and prov.get('kind') == 'isaac_recorded_tof_approximation_with_RGB_field_localization'
                and path.get('layout') == 'empty' and path.get('rear_sensor_coverage') is False, t=t)
    audit.check('rear_path_case_matches_pose_and_clock', case['odom']['pose_xyyaw'] == pose and case['query_s'] == t and path.get('capture_s') == t
                and case['odom']['capture_s'] == t and finite(path.get('goal_fix_s')) and 0 <= t-path['goal_fix_s'] <= LIMITS['goal_fix_s']
                and case['recorded_evidence']['RGB_localization_capture_end_s'] == path['goal_fix_s'], t=t)
    x, y, yaw = pose; c, s = math.cos(yaw), math.sin(yaw); gx, gy = case['goal_body_xy']
    goal_map = [x+c*gx-s*gy, y+s*gx+c*gy]
    audit.check('rear_path_goal_is_archived_target', target is not None and math.dist(goal_map, target) <= 1e-9, t=t)
    if fixed_target is not None:
        audit.check('rear_path_fixed_target', target == fixed_target, t=t, target=target, fixed=fixed_target)
    if expected_target is not None:
        audit.check('rear_path_authorized_target_from_entry_distance', target is not None and math.dist(target, expected_target) <= 1e-9, t=t)
    try:
        gate = check_path([pose[:2], target], snapshot(case, prov), radius=LIMITS['radius_m'])
    except (ValueError, KeyError, TypeError) as error:
        gate = {'accepted': False, 'reason': 'replay_error:'+str(error)}
    audit.check('rear_path_recomputed_10cm_accepted', gate['accepted'] is True and path.get('radius_m') == LIMITS['radius_m'], t=t, reason=gate['reason'])
    return gate['accepted'] is True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run', type=Path)
    p.add_argument('--controller', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fixture-leg', type=Path, help='CPU fixtures only; production audits never pass this')
    p.add_argument('--allow-test-only-approval', action='store_true', help='CPU fixtures only')
    a = p.parse_args(argv)
    if a.output.exists():
        raise FileExistsError('Retain the existing audit: '+str(a.output))
    run = a.run; audit = Audit()
    names = ['recovery-trace.jsonl', 'policy-trace.jsonl', 'public-memory-readback.jsonl', 'control.jsonl', 'physics-trace.jsonl', 'manifest.json']
    missing = [n for n in names if not (run/n).is_file()]
    audit.check('inputs_present', not missing, missing=missing)
    if missing:
        return finish(a, audit, {})
    manifest = json.loads((run/'manifest.json').read_text())
    rec = rows(run/'recovery-trace.jsonl'); pol = rows(run/'policy-trace.jsonl'); reads = rows(run/'public-memory-readback.jsonl')
    ctrl = rows(run/'control.jsonl'); phys = rows(run/'physics-trace.jsonl')
    approval = manifest.get('retreat_recovery_approval') or {}
    params = approval.get('parameters') or {}
    audit.check('approval_present_and_production', bool(params) and (approval.get('test_only') is False or a.allow_test_only_approval),
                test_only=approval.get('test_only'))
    if not params:
        return finish(a, audit, {})
    helper_sha = sha(a.controller/'heading_feedback.py') if (a.controller/'heading_feedback.py').is_file() else None
    audit.check('approved_fixed_motion_parameters', params.get('retreat_vx') == LIMITS['vx'] and params.get('duration_s') == LIMITS['duration_s']
                and params.get('move_steps') == LIMITS['move_steps'] and params.get('heading_offset_rad') in LIMITS['offsets_rad']
                and params.get('helper_sha256') == helper_sha, params=params, controller_helper=helper_sha)
    if approval.get('test_only') is False:
        sa, sb = approval.get('stage_a_binding') or {}, approval.get('stage_b_binding') or {}
        motion = [params.get('helper_sha256'), params.get('duration_s'), params.get('heading_offset_rad'), params.get('retreat_vx')]
        audit.check('offset_source_is_stage_a_smaller_eligible_and_stage_b_exact', sa.get('selected_heading_offset_rad') == params.get('heading_offset_rad')
                    and sa.get('selected_heading_offset_rad') == min(sa.get('eligible_offsets_rad') or [math.inf])
                    and sa.get('motion') == motion and sb.get('motion') == motion and (sb.get('tests') or 0) >= 2)
    offset = float(params['heading_offset_rad']); vx = float(params['retreat_vx'])
    scope = manifest.get('retreat_recovery_scope') or {}
    task_limit = min(180., float(scope.get('task_limit_s', 180.)))
    leg, leg_model = load_leg(a.controller, a.fixture_leg)
    replay = Replay(reads, leg)

    # -- every 50 Hz tick: recovery-trace policy rows must mirror policy-trace -----------
    rpol = {key(r['timestamp_s']): r for r in rec if r.get('kind') == 'policy'}
    rcap = [r for r in rec if r.get('kind') == 'capture']
    audit.check('truth_never_used', all(r.get('truth_used') is False for r in rec))
    audit.check('recovery_never_kick_authorized', all(r.get('kick_authorized') is False for r in rec))
    events = [e for r in rec for e in r.get('events', [])]
    entries = [e for e in events if e.get('event') == 'recovery_entry']
    report = manifest.get('retreat_recovery') or {}
    rejections = report.get('rejections') or []

    # -- entry whitelist / budget / reserve / memory clocks ---------------------------
    audit.check('entry_count_within_budget', len(entries) <= LIMITS['max_recoveries'], entries=len(entries))
    for e in entries:
        mem = e.get('memory') or {}; t = e['timestamp_s']; pnt = mem.get('point_level_m') or [math.nan]*3
        ok = (e.get('original_reason') in WHITELIST and e.get('kick_ever_started') is False and e.get('recoveries_before', 9) < LIMITS['max_recoveries']
              and finite(e.get('task_s')) and task_limit-e['task_s'] >= LIMITS['reserve_s'] and mem.get('memory_available') is True
              and finite(mem.get('ball_capture_timestamp_s'), mem.get('sensor_read_timestamp_s'))
              and 0 <= t-mem['ball_capture_timestamp_s'] <= LIMITS['ttl_s'] and 0 <= t-mem['sensor_read_timestamp_s'] <= LIMITS['read_age_s']
              and LIMITS['entry_x'][0] < pnt[0] <= LIMITS['entry_x'][1] and abs(pnt[1]) <= LIMITS['entry_abs_y'])
        audit.check('entry_whitelist_budget_reserve_memory_clock', ok, t=t, reason=e.get('original_reason'))
        audit.check('entry_fixed_plan_distance', e.get('plan_distance_m') == LIMITS['plan_distance_m'], t=t)
    for r in rejections:
        if r.get('admitted'):
            audit.check('admitted_rejection_whitelisted', r.get('original_reason') in WHITELIST, t=r.get('timestamp_s'))
    audit.check('rejection_history_matches_events', len([r for r in rejections if r.get('admitted')]) == len(entries),
                admitted=len([r for r in rejections if r.get('admitted')]), entries=len(entries))

    # -- per tick command reconstruction -------------------------------------------------
    # Executor order: a capture stamped T runs before the policy tick at T; memory
    # reads integrate only at policy ticks.
    segments = []; seg = None; last_t = None; prev_head = None; prev_kick = False; kick_segments = 0
    caps = sorted(rcap, key=lambda r: r['timestamp_s']); ci = 0; last_cap = None; await_release = None
    for r in pol:
        t = r['timestamp_s']; h = r.get('public_handoff') or {}
        while ci < len(caps) and caps[ci]['timestamp_s'] <= t+1e-9:
            c = caps[ci]; ci += 1
            replay.advance(c['timestamp_s'], strict=True)
            if seg is not None and seg['open']:
                handle_capture(c, seg, replay, audit, params)
                if c.get('released'):
                    seg['release_t'] = c['timestamp_s']; seg['open'] = False; await_release = seg
                elif c.get('handoff_state') == 'FAILED':
                    seg['fail_t'] = c['timestamp_s']; seg['open'] = False
            last_cap = c
        replay.advance(t)
        v = [float(x) for x in r.get('actual_velocity_command') or [0, 0, 0]]
        kick = bool(r.get('kick_requested'))
        if kick and not prev_kick:
            kick_segments += 1
        prev_kick = kick
        if await_release is not None:
            audit.check('release_no_stale_command', not any(v) and h.get('state') == 'APPROACH', t=t)
            await_release = None
        rp = rpol.get(key(t))
        audit.check('no_kick_during_recovery', not (kick and (h.get('state') == 'RECOVERY' or (rp or {}).get('handoff_state') == 'RECOVERY')), t=t)
        if rp is None:
            audit.check('recovery_tick_archived', h.get('state') != 'RECOVERY', t=t)
            prev_head = r.get('actual_head_command_rad')
            continue
        audit.check('recovery_row_mirrors_sent_command', rp.get('actual_velocity_command') == v, t=t)
        audit.check('recovery_row_mirrors_sent_head', rp.get('actual_head_command') == r.get('actual_head_command_rad'), t=t)
        audit.check('recovery_row_mirrors_handoff_state', rp.get('handoff_state') == h.get('state'), t=t)
        for e in rp.get('events', []):
            if e.get('event') == 'recovery_entry':
                audit.check('entry_only_from_closed_segment', seg is None or not seg['open'], t=t)
                seg = {'entry': e, 'start': t, 'open': True, 'epoch_replay': replay.epoch, 'origin': None, 'fixed_target': None,
                       'authorize_t': None, 'authorized_capture_t': None, 'retreat_t': None, 'settle_t': None, 'reacquire_t': None,
                       'release_t': None, 'fail_t': None, 'reacquire_rel': None, 'nonzero_ticks': 0, 'counters': None,
                       'feedback': FeedbackReplay(vx, offset), 'motion_count': 0, 'stop_t': None, 'stop_rel': None, 'feedback_fault': None}
                segments.append(seg); last_t = None
        if seg is None:
            audit.check('recovery_row_without_entry_is_terminal_refusal', not any(v) and rp.get('handoff_state') != 'RECOVERY', t=t)
            prev_head = r.get('actual_head_command_rad')
            continue
        if not seg['open']:
            audit.check('no_command_after_recovery_closed', not any(v), t=t)
            prev_head = r.get('actual_head_command_rad')
            continue
        rs = rp.get('recovery_state'); st = rp.get('recovery') or {}
        if rp.get('handoff_state') == 'FAILED':
            seg['fail_t'] = t; seg['open'] = False
            audit.check('failure_tick_zero_command', not any(v), t=t)
            audit.check('early_stop_is_terminal_not_motion_complete', seg['stop_t'] is None or seg['motion_count'] == LIMITS['move_steps'], t=t)
            prev_head = r.get('actual_head_command_rad')
            continue
        if last_t is not None:
            audit.check('recovery_policy_clock_gap', 0 < t-last_t <= LIMITS['policy_gap_s']+1e-9, t=t, dt=t-last_t)
        last_t = t
        task_s = rp.get('task_s')
        audit.check('original_task_clock_and_reserve', finite(task_s) and task_s < task_limit, t=t)
        for phase, field in (('AUTHORIZE', 'authorize_t'), ('SETTLE', 'settle_t'), ('REACQUIRE', 'reacquire_t')):
            if rs == phase and seg[field] is None:
                seg[field] = st.get({'authorize_t': 'authorize_start_s', 'settle_t': 'settle_start_s', 'reacquire_t': 'reacquire_start_s'}[field])
        if rs == 'HOLD': audit.check('hold_deadline', t-seg['start'] <= LIMITS['hold_s']+1e-9, t=t)
        if rs == 'AUTHORIZE': audit.check('authorize_deadline', seg['authorize_t'] is not None and t-seg['authorize_t'] <= LIMITS['authorize_s']+1e-9, t=t)
        if rs == 'SETTLE': audit.check('settle_deadline', seg['settle_t'] is not None and t-seg['settle_t'] <= LIMITS['settle_s']+1e-9, t=t)
        if rs == 'REACQUIRE': audit.check('reacquire_deadline', seg['reacquire_t'] is not None and t-seg['reacquire_t'] <= LIMITS['reacquire_s']+1e-9, t=t)
        pose = replay.pose(t); od = rp.get('odometry') or {}
        continuous = replay.epoch == seg['epoch_replay']
        if od.get('valid'):
            audit.check('odometry_recomputed_from_raw_packets', pose is not None and od.get('pose_xyyaw') is not None
                        and max(abs(x-y) for x, y in zip(pose, od['pose_xyyaw'])) <= LIMITS['odometry_tol'], t=t)
            audit.check('odometry_continuity_since_entry', continuous, t=t, replay_invalidations=replay.invalidations[-3:])
        elif rs != 'FAILED' and rp.get('decision') != 'entry_hold':
            # Every recovery tick after the entry tick must have used continuous fresh odometry.
            audit.check('recovery_continues_only_with_valid_odometry', False, t=t)
        if rs == 'REACQUIRE' and seg['reacquire_rel'] is None and seg['origin'] is not None and pose is not None and continuous:
            seg['reacquire_rel'] = relative(seg['origin'], pose)
        head = r.get('actual_head_command_rad') or [math.nan]*4
        if rs in NEUTRAL and rs != 'HOLD':
            audit.check('neutral_sent_head_in_owned_phase', all(abs(x) <= 1e-9 for x in head), t=t, state=rs)
        if rs == 'REACQUIRE':
            perm = rp.get('head_permission') or {}
            if not perm.get('allowed') and prev_head is not None:
                audit.check('reacquire_head_holds_without_current_permission', rp.get('requested_head_nominal') == prev_head, t=t)
        prev_head = r.get('actual_head_command_rad')
        motion_due = (seg['authorized_capture_t'] is not None and seg['stop_t'] is None and rs in ('RETREAT', 'SETTLE', 'FAILED'))
        if motion_due:
            k = seg['motion_count']
            if k == 0:
                seg['retreat_t'] = t
                audit.check('motion_window_starts_first_tick_after_authorization', st.get('retreat_start_s') == t
                            and t >= seg['authorized_capture_t'], t=t, stored=st.get('retreat_start_s'))
            rel = relative(seg['origin'], pose) if seg['origin'] is not None and pose is not None and continuous else None
            elapsed = t-seg['retreat_t']
            unsafe = (rel is None or rel[0] > LIMITS['forward_m'] or -rel[0] > LIMITS['max_backward_m'] or abs(rel[2]) > LIMITS['yaw_rad']
                      or abs(rel[1]) > LIMITS['lateral_m'] or (elapsed >= LIMITS['onset_s']-1e-9 and -rel[0] < LIMITS['onset_min_m']))
            if unsafe:
                audit.check('safety_stop_due_zero_and_terminal', not any(v) and rs == 'FAILED', t=t, rel=rel, elapsed=elapsed)
            elif k >= LIMITS['move_steps']:
                seg['stop_t'] = t; seg['stop_rel'] = rel
                audit.check('no_post_stop_authority', not any(v), t=t)
                audit.check('normal_stop_requires_public_backward_0p10', (rs == 'SETTLE') == (-rel[0] >= LIMITS['min_backward_m']), t=t, backward=-rel[0])
                audit.check('settle_only_after_exact_80_motion_ticks', rs != 'SETTLE' or k == LIMITS['move_steps'], t=t)
            else:
                audit.check('early_stop_is_terminal_not_motion_complete', rs != 'SETTLE', t=t, motion_count=k)
                packet = r.get('IMU_prediction')
                expected = seg['feedback'].step(t, packet)
                helper = rp.get('helper') or {}
                audit.check('helper_index_contiguous', helper.get('index') == k and rp.get('motion_tick') is True, t=t, index=helper.get('index'), expected=k)
                raw = (helper.get('inputs') or {})
                audit.check('helper_inputs_are_current_raw_packet', packet is not None and raw.get('seq') == [packet.get('seq')]
                            and raw.get('read_s') == [packet.get('read_timestamp_s')] and raw.get('gyro_body') == [packet.get('gyro_body_rad_s')]
                            and raw.get('gravity_body') == [packet.get('gravity_body')], t=t)
                seg['motion_count'] = k+1
                if seg['feedback'].fault is not None:
                    seg['feedback_fault'] = seg['feedback'].fault
                    audit.check('latched_feedback_fault_zero_and_terminal', not any(v) and rs == 'FAILED', t=t, fault=seg['feedback'].fault)
                else:
                    audit.check('vx_exact_vy_zero_on_motion_ticks', rs == 'FAILED' or (v[0] == vx and v[1] == 0.), t=t, sent=v)
                    audit.check('wz_independent_rederivation', rs == 'FAILED' or (any(v) and max(abs(x-y) for x, y in zip(v, expected)) <= LIMITS['command_tol']),
                                t=t, sent=v, expected=expected)
                    audit.check('command_requested_equals_expected_helper', rs == 'FAILED' or rp.get('expected_helper_command') == v, t=t)
        elif any(v):
            audit.check('no_command_outside_motion_window', False, t=t, state=rs)
        if not any(v):
            continue
        seg['nonzero_ticks'] += 1
        audit.check('nonzero_only_in_retreat', rs == 'RETREAT' and rp.get('decision') == 'retreat_command', t=t, state=rs)
        audit.check('neutral_sent_head_while_moving', all(abs(x) <= 1e-9 for x in head), t=t)
        fresh_cap = last_cap is not None and 0 <= t-last_cap['timestamp_s'] <= LIMITS['capture_age_s']
        audit.check('fresh_guarded_capture_before_command', fresh_cap and last_cap.get('guard_safe') is True
                    and last_cap.get('recovery_after') == 'RETREAT' and last_cap.get('_path_ok') is True, t=t,
                    capture=None if last_cap is None else last_cap['timestamp_s'])
        audit.check('odometry_valid_while_moving', pose is not None and continuous and od.get('valid') is True, t=t)
        audit.check('task_time_remaining_while_moving', finite(task_s) and task_s < task_limit, t=t)
    while ci < len(caps):
        c = caps[ci]; ci += 1
        if seg is not None and seg['open']:
            handle_capture(c, seg, replay, audit, params)
    audit.check('single_kick_segment', kick_segments <= 1, kick_segments=kick_segments)
    for s_ in segments:
        audit.check('motion_count_le_80_and_settle_only_after_80', s_['motion_count'] <= LIMITS['move_steps']
                    and (s_['settle_t'] is None or s_['motion_count'] == LIMITS['move_steps']), start=s_['start'], count=s_['motion_count'])
    audit.note('replay', leg_model=leg_model, invalidations=len(replay.invalidations), replay_errors=replay.errors[:5])

    # -- counters: original approach budgets cumulative, held during recovery ----------
    prev = {}
    for c in ctrl:
        fa = c.get('final_approach') or {}; h = c.get('public_handoff') or {}
        for name, value in (('final_turns', fa.get('turn_attempts')), ('final_advances', fa.get('advance_attempts')), ('handoff_attempts', h.get('attempts'))):
            if value is not None:
                audit.check('cumulative_counters_never_reset', prev.get(name) is None or value >= prev[name], t=c['timestamp_s'], counter=name)
                prev[name] = value
    for s in segments:
        rows_in = [c for c in rcap if c['timestamp_s'] >= s['start'] and c['timestamp_s'] <= (s.get('release_t') or s.get('fail_t') or math.inf)]
        counters = [c.get('approach_counters') for c in rows_in if c.get('approach_counters') is not None]
        audit.check('approach_counters_archived', bool(counters) or not rows_in, start=s['start'])
        audit.check('final_approach_budget_not_consumed_during_recovery', all(x == counters[0] for x in counters), start=s['start'])
        before = [c for c in ctrl if c['timestamp_s'] < s['start'] and (c.get('final_approach') or {}).get('turn_attempts') is not None]
        if before and counters:
            fa = before[-1]['final_approach']
            audit.check('counters_carried_into_recovery', counters[0]['final_turns'] >= fa['turn_attempts']
                        and counters[0]['final_advances'] >= fa['advance_attempts'], start=s['start'])
        attempts = [c.get('attempts') for c in rows_in if c.get('attempts') is not None]
        audit.check('handoff_attempts_preserved', all(x == attempts[0] for x in attempts) and (not attempts or attempts[0] >= s['entry'].get('attempt', 0)), start=s['start'])
    # -- reanchor only in REACQUIRE with current complete grounded same capture ----------
    capture_index = {key(c['timestamp_s']): c for c in rcap}
    last_anchor = None
    for r in reads:
        h = r.get('handoff') or {}; m = h.get('memory') or {}
        anchor = m.get('ball_capture_timestamp_s')
        if h.get('state') == 'RECOVERY' and anchor is not None and anchor != last_anchor and last_anchor is not None:
            c = capture_index.get(key(anchor))
            audit.check('reanchor_only_reacquire_current_complete', c is not None and c.get('recovery_before') == 'REACQUIRE'
                        and (c.get('geometry') or {}).get('complete_grounded_same_capture') is True
                        and (c.get('anchor_attempt') or {}).get('anchored') is True, t=r['timestamp_s'], anchor=anchor)
        if m.get('memory_available'):
            audit.check('memory_ttl_unchanged', finite(m.get('ball_age_s')) and 0 <= m['ball_age_s'] <= LIMITS['ttl_s'], t=r['timestamp_s'])
        if anchor is not None:
            last_anchor = anchor
    for c in rcap:
        if c.get('recovery_before') in NEUTRAL:
            audit.check('no_anchor_outside_reacquire', not (c.get('anchor_attempt') or {}).get('anchored'), t=c['timestamp_s'])
    # -- scorer monotone, no foot-ball force during recovery, physical stopping ----------
    prev_flags = {}
    windows = [(s['start'], s.get('release_t') or s.get('fail_t') or math.inf) for s in segments]
    for r in phys:
        for flag in ('kick_contact_observed', 'qualified_goal_seen', 'fell'):
            if flag in r:
                audit.check('scorer_flags_monotone_no_reset', not (prev_flags.get(flag) and not r[flag]), t=r['timestamp_s'], flag=flag)
                prev_flags[flag] = r[flag]
        if any(s <= r['timestamp_s'] <= e for s, e in windows):
            forces = r.get('raw_paired_foot_ball_forces_n') or []
            audit.check('no_foot_ball_force_during_recovery', all(math.hypot(*f) <= LIMITS['contact_n'] for f in forces), t=r['timestamp_s'])
    contact = ((manifest.get('scorer') or {}).get('contact') or {})
    audit.note('scorer_summary', pre_kick_ball_contact_observed=contact.get('pre_kick_ball_contact_observed'),
               physical_episode_success=(manifest.get('scorer') or {}).get('physical_episode_success'))
    physical = physical_stopping(phys, segments, audit)
    summary = {'segments': [{k: v for k, v in s.items() if k in ('start', 'fail_t', 'authorize_t', 'authorized_capture_t', 'retreat_t', 'stop_t',
                                                              'settle_t', 'reacquire_t', 'release_t', 'nonzero_ticks', 'motion_count',
                                                              'feedback_fault', 'fixed_target', 'origin', 'reacquire_rel', 'stop_rel')}
                            for s in segments], 'physical_stopping': physical, 'approved_motion': {'vx': vx, 'heading_offset_rad': offset,
                                                                                                  'helper_sha256': params.get('helper_sha256')},
               'task_limit_s': task_limit, 'terminal_reason': manifest.get('terminal_reason')}
    return finish(a, audit, summary)


def handle_capture(c, seg, replay, audit, params):
    t = c['timestamp_s']; before = c.get('recovery_before'); after = c.get('recovery_after'); path = c.get('rear_path') or {}
    if before == 'RETREAT' and seg['stop_t'] is not None:
        audit.check('no_retreat_capture_after_stop', False, t=t)
    audit.check('capture_clock_causal', (c.get('memory') or {}).get('sensor_read_timestamp_s') is None
                or c['memory']['sensor_read_timestamp_s'] <= t+1e-9, t=t)
    c['_path_ok'] = False
    if before in ('AUTHORIZE', 'RETREAT') and path.get('allowed') is True:
        expected = None
        if before == 'AUTHORIZE':
            pose = path.get('pose_map_xyyaw'); d = LIMITS['plan_distance_m']
            audit.check('rear_path_fixed_plan_distance_0p28', path.get('distance_m') == d, t=t)
            expected = [pose[0]-d*math.cos(pose[2]), pose[1]-d*math.sin(pose[2])] if pose else None
        c['_path_ok'] = recheck_path(path, audit, t, seg['fixed_target'] if before == 'RETREAT' else None, expected)
    if before == 'AUTHORIZE' and after == 'RETREAT':
        audit.check('authorize_requires_guard_and_recomputed_path', c.get('guard_safe') is True and c['_path_ok'], t=t)
        od = c.get('odometry') or {}; pose = replay.pose(t)
        audit.check('retreat_origin_recomputed', od.get('valid') is True and pose is not None
                    and max(abs(x-y) for x, y in zip(pose, od['pose_xyyaw'])) <= LIMITS['odometry_tol'], t=t)
        seg['origin'] = pose; seg['fixed_target'] = path.get('target_map_xy'); seg['authorized_capture_t'] = t
    if before == 'RETREAT':
        ok = c.get('guard_safe') is True and c['_path_ok'] and 0 <= t-path.get('capture_s', -1e9) <= LIMITS['capture_age_s']
        audit.check('retreat_capture_unsafe_withdraws', ok or after != 'RETREAT', t=t, guard=c.get('guard_safe'), path=path.get('reason'))
    if before == 'REACQUIRE' and after == 'RELEASED':
        m = c.get('memory') or {}; pnt = m.get('point_level_m') or [math.nan]*3
        ok = (c.get('guard_safe') is True and (c.get('geometry') or {}).get('complete_grounded_same_capture') is True
              and m.get('memory_available') is True and m.get('ball_capture_timestamp_s') == t and (c.get('anchor_attempt') or {}).get('anchored') is True
              and LIMITS['release_x'][0] <= pnt[0] <= LIMITS['release_x'][1] and abs(pnt[1]) <= LIMITS['release_abs_y']
              and seg['reacquire_t'] is not None and t-seg['reacquire_t'] <= LIMITS['reacquire_s']+1e-9 and c.get('handoff_state') == 'APPROACH')
        audit.check('release_current_complete_guarded_in_domain', ok, t=t, point=pnt)
    if after == 'FAILED' and seg.get('fail_t') is None:
        seg['fail_t'] = t


def physical_stopping(phys, segments, audit):
    """Evaluation only (privileged physics): original Stage A displacement/yaw/drift after SETTLE."""
    index = {key(r['timestamp_s']): r for r in phys}
    out = []

    def yaw_of(r):
        x, y, z, w = r['root_quaternion_xyzw']; return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    for s in segments:
        if s.get('retreat_t') is None or s.get('reacquire_t') is None or s.get('stop_t') is None:
            continue
        a, b, c0 = index.get(key(s['retreat_t'])), index.get(key(s['reacquire_t'])), index.get(key(s['stop_t']))
        if a is None or b is None or c0 is None:
            audit.check('physical_stopping_evaluable', False, start=s['retreat_t']); continue
        yaw0 = yaw_of(a); c, s_ = math.cos(yaw0), math.sin(yaw0)
        d = np.asarray(b['root_position_world_m'][:2])-np.asarray(a['root_position_world_m'][:2])
        fx, fy = float(c*d[0]+s_*d[1]), float(-s_*d[0]+c*d[1])
        dyaw = math.atan2(math.sin(yaw_of(b)-yaw0), math.cos(yaw_of(b)-yaw0))
        stop_xy = np.asarray(c0['root_position_world_m'][:2])
        drift = max(float(np.linalg.norm(np.asarray(r['root_position_world_m'][:2])-stop_xy)) for r in phys
                    if s['stop_t']-1e-9 <= r['timestamp_s'] <= s['reacquire_t']+1e-9)
        odom = s.get('reacquire_rel')
        item = {'retreat_start_s': s['retreat_t'], 'stop_s': s['stop_t'], 'reacquire_start_s': s['reacquire_t'], 'physical_backward_m': -fx,
                'physical_lateral_m': fy, 'physical_yaw_rad': dyaw, 'post_stop_drift_m': drift,
                'public_odometry_dx_m': None if odom is None else odom[0], 'evaluation_only_privileged_physics': True}
        lo, hi = LIMITS['stage_a_backward_m']
        audit.check('physical_stage_a_displacement_yaw_drift', lo <= -fx <= hi and abs(dyaw) <= LIMITS['yaw_rad']
                    and abs(fy) <= LIMITS['lateral_m'] and drift <= LIMITS['post_stop_drift_m'], **item)
        if odom is not None:
            audit.check('public_odometry_vs_physical_stop', abs(odom[0]-fx) <= LIMITS['physical_odometry_error_m'], **item)
        out.append(item)
    return out


def finish(a, audit, summary):
    passed = bool(audit.checks) and all(c['passed'] for c in audit.checks.values())
    result = {'schema': 'microduck.v121-retreat-recovery-independent-audit.v2', 'passed': passed, 'run': str(a.run),
              'checks': audit.checks, 'summary': summary, 'limits': LIMITS, 'fixture_leg': None if a.fixture_leg is None else str(a.fixture_leg),
              'test_only_approval_allowed': a.allow_test_only_approval, 'auditor_sha256': sha(__file__),
              'scope': 'Recomputed from raw public packets/archived map requests; privileged physics used only for evaluation; no physics success or success-rate claim.',
              'sources': {n: sha(a.run/n) for n in ('recovery-trace.jsonl', 'policy-trace.jsonl', 'public-memory-readback.jsonl', 'control.jsonl', 'manifest.json') if (a.run/n).is_file()}}
    a.output.write_text(json.dumps(result, indent=2, default=str)+'\n')
    print(json.dumps({'passed': passed, 'failed_checks': sorted(k for k, c in audit.checks.items() if not c['passed'])}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
