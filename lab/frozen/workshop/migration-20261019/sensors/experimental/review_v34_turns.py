"""Private offline scoring of executed turns versus separately labelled public estimates."""
from pathlib import Path
import json, hashlib, math
import numpy as np
from scipy.spatial.transform import Rotation

M=Path(__file__).resolve().parents[2];O=M.parents[1]/'outputs/migration-20261019'
R=O/'football-v34-waypoint-position20s'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
wrap=lambda x:float((x+math.pi)%(2*math.pi)-math.pi)

def main():
    controls=[json.loads(x)for x in (R/'control.jsonl').read_text().splitlines()]
    physics=[json.loads(x)for x in (R/'physics-trace.jsonl').read_text().splitlines()]
    def actual(a,b):
        rows=[r for r in physics if a-1e-8<=r['timestamp_s']<=b+1e-8]
        t=np.array([r['timestamp_s']for r in rows]);rot=Rotation.from_quat([r['root_quaternion_xyzw']for r in rows]);e=rot.as_euler('xyz')
        yaw=np.unwrap(e[:,2]);omega=np.array([r['root_angular_velocity_body_rad_s']for r in rows])
        yawrate=(np.sin(e[:,0])*omega[:,1]+np.cos(e[:,0])*omega[:,2])/np.cos(e[:,1])
        return {'start_s':float(t[0]),'end_s':float(t[-1]),'n':len(rows),'actual_yaw_delta_rad':float(yaw[-1]-yaw[0]),
                'actual_yaw_rate_integral_rad':float(np.trapezoid(yawrate,t)),
                'actual_body_gyro_z_integral_rad':float(np.trapezoid(omega[:,2],t)),
                'actual_link_position_delta_world_m':(np.array(rows[-1]['root_position_world_m'])-rows[0]['root_position_world_m']).tolist()}
    def public(r):
        near=r.get('near_ball_phase',{});error=near.get('relative_position_error_m');pose=r.get('estimated_pose_map_xyyaw')
        if error is None or pose is None:return None
        ex,ey=error;c,s=math.cos(pose[2]),math.sin(pose[2]);wp=[pose[0]+c*ex-s*ey,pose[1]+s*ex+c*ey]
        heading=near.get('translation_heading_error_rad')
        return {'timestamp_s':r['timestamp_s'],'translation_heading_rad':heading,'goal_heading_rad':near.get('heading_error_rad'),
                'estimated_pose_map_xyyaw':pose,'position_error_body_xy_m':error,'estimated_waypoint_map_xy_m':wp,
                'heading_equals_atan2_error':heading is not None and abs(wrap(heading-math.atan2(ey,ex)))<1e-12,
                'source_rgb_capture_s':r.get('source_rgb_capture_s')}
    segments=[]
    for i,r in enumerate(controls):
        if r.get('native_command_adapter',{}).get('reason')!='turn_duration_cap':continue
        j=i-1
        while j>=0 and abs(controls[j]['velocity_body_xyw'][2])>.5:j-=1
        start=controls[j+1];s=float(start['timestamp_s']);end=float(r['timestamp_s'])
        if j+1==i:raise ValueError('Cap has no preceding executed turn')
        following=[x for x in controls[i+1:] if x['timestamp_s']<=end+.4+1e-8]
        nextmotion=next((x['timestamp_s']for x in following if any(abs(v)>1e-8 for v in x['velocity_body_xyw'])),None)
        stop_end=min(end+.4,nextmotion if nextmotion is not None else end+.4)
        stop_end=min(stop_end,physics[-1]['timestamp_s'])
        ps,pe=public(start),public(r)
        extra={}
        if ps and pe:
            extra={'public_heading_delta_rad':wrap(pe['translation_heading_rad']-ps['translation_heading_rad']),
                   'public_estimated_body_yaw_delta_rad':wrap(pe['estimated_pose_map_xyyaw'][2]-ps['estimated_pose_map_xyyaw'][2]),
                   'public_estimated_waypoint_shift_m':float(np.linalg.norm(np.array(pe['estimated_waypoint_map_xy_m'])-ps['estimated_waypoint_map_xy_m']))}
        segments.append({'command_wz':start['velocity_body_xyw'][2],'cap_timestamp_s':end,'command_duration_s':end-s,
                         'turn':actual(s,end),'following_zero_command_window':actual(end,stop_end),
                         'stop_window_truncated_by_next_motion':nextmotion is not None,
                         'public_start':ps,'public_cap':pe,**extra})
    source=M/'acceptance/native_football_v34/command_navigation_native.py'
    report={'schema':'microduck.v34-turn-independent-review.v1','scope':'Read-only original20s actual trace; private physics only scores motion, never controller input.',
            'qualification_claimed':False,'duration_cap_count':len(segments),'segments':segments,
            'all_public_translation_headings_match_error_atan2':all(s['public_start']['heading_equals_atan2_error']and s['public_cap']['heading_equals_atan2_error']for s in segments),
            'semantic_notes':['Positive yaw is left/CCW around body up. Euler yaw and raw body gyroZ differ under tilt; the correctly transformed yaw-rate integral is also retained.',
                              'Public estimated waypoint and estimated pose may both change during the turn; bearing change cannot alone establish physical rotation or fixed-map error.',
                              'Zero-command windows are capped at.4s or earlier next nonzero motion; no following active command is labelled recoil.',
                              'These observations separate actual response and public-estimate changes; they do not isolate duration, gaze, or model causality.'],
            'evidence_sha256':{str(p):sha(p)for p in [R/'control.jsonl',R/'physics-trace.jsonl',source,Path(__file__)]}}
    out=O/'football-v34-turn-independent-review-v1.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(out),'caps':len(segments),'rows':[{k:s[k]for k in ['command_duration_s','turn','following_zero_command_window','public_heading_delta_rad','public_estimated_body_yaw_delta_rad','public_estimated_waypoint_shift_m']}for s in segments]},indent=2))

if __name__=='__main__':main()
