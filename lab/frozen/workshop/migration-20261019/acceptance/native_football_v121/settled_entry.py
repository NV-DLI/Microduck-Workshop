"""CPU preparation only: reject transient heading crossings, keep 4.5s baseline."""
import math
import sys
from pathlib import Path
from fine_kick_v65 import standoff_entry


def settled_entry(**args):
    ok,why=standoff_entry(**args)
    if not ok:return False,why
    d=args['final_diag'];t=args['t'];times=d.get('quiet_sample_times',[])
    rms=d.get('gyro_3D_RMS_rad_s');diameter=d.get('ball_XY_diameter_m')
    finite=lambda x:isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
    if d.get('measured_quiet') is not True:return False,'rejected:not_measured_quiet'
    if not finite(rms) or not 0<=rms<=.12 or not finite(diameter) or not 0<=diameter<=.02:
        return False,'rejected:quiet_measurements'
    if (not isinstance(times,list) or len(times)<6 or not all(finite(x)for x in times)
        or any(b<=a or b-a>.2 for a,b in zip(times,times[1:]))
        or abs(times[-1]-t)>1e-8 or not .33<=times[-1]-times[0]<=.45+1e-8):
        return False,'rejected:quiet_capture_window'
    if abs(args['memory']['point_level_m'][1]+.035)>.01:
        return False,'rejected:lateral_entry_margin'
    return True,'settled_standoff_entry_cpu_candidate'


def neutral_turn_entry(**a):
    import numpy as np
    from neutral_fine_step import NeutralFineStep
    d=a['final_diag'];m=a['memory'];t=a['t']
    if not (d.get('phase')=='TURN' and d.get('alignment_mode')=='final_goal_heading' and d.get('measured_quiet') is True):return False,'not_settled_final_yaw'
    if not (a['handoff_state']=='APPROACH' and a['fine_active'] is False and a['fine_state']=='IDLE' and a['complete'] is True and a['guard_safe'] is True):return False,'neutral_yaw_upstream_guard'
    if not (m.get('memory_available') is True and m.get('ball_capture_timestamp_s')==t):return False,'neutral_yaw_fresh_memory_required'
    p=np.asarray(m.get('point_level_m'),float);h=np.asarray(a['actual_head'],float);v=np.asarray(a['velocity_body_xyw'],float)
    if not(p.shape==(3,) and h.shape==(4,) and v.shape==(3,) and np.isfinite(p).all() and np.isfinite(h).all() and np.isfinite(v).all()):return False,'neutral_yaw_nonfinite'
    if not(.13<=p[0]<=.38 and np.linalg.norm(p[:2])<=.4 and abs(p[1])<=.3 and max(abs(h))<=.95 and np.all(abs(h)<=np.array([1.1,1.1,1.4,.31])+1e-9) and v[0]==v[1]==0 and 0<abs(v[2])<=.8):return False,'neutral_yaw_domain'
    gh=a['goal_heading'];gs=a['goal_fix_s']
    if not(math.isfinite(gh) and math.isfinite(gs) and 0<=t-gs<=8):return False,'neutral_yaw_goal_memory'
    times=d.get('quiet_sample_times',[])
    if not isinstance(times,list) or len(times)<6 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in times) or any(b<=a or b-a>.2 for a,b in zip(times,times[1:])) or abs(times[-1]-t)>1e-8 or not .33<=times[-1]-times[0]<=.45+1e-8:return False,'neutral_yaw_window'
    if not(0<=d.get('gyro_3D_RMS_rad_s',math.inf)<=.12 and 0<=d.get('ball_XY_diameter_m',math.inf)<=.02):return False,'neutral_yaw_not_quiet'
    return True,'settled_neutral_head_yaw'


def neutral_forward_entry(**a):
    import numpy as np
    from neutral_fine_step import NeutralFineStep
    d=a['final_diag'];m=a['memory'];t=a['t']
    if not(d.get('phase')=='ADVANCE' and d.get('staging_phase')=='GOAL_ALIGN' and d.get('measured_quiet') is True):return False,'not_settled_final_forward'
    # Reuse the same measured quiet/native-head/current-capture validation as yaw.
    b=dict(a);b['final_diag']={**d,'phase':'TURN','alignment_mode':'final_goal_heading'};b['velocity_body_xyw']=[0.,0.,.6]
    ok,why=neutral_turn_entry(**b)
    if not ok:return False,why
    p=np.asarray(m['point_level_m'],float);v=a['velocity_body_xyw']
    if not(.22<np.linalg.norm(p[:2])<=.36 and -.10<=p[1]<=.08 and v[0]>0 and v[1]==v[2]==0):return False,'neutral_forward_domain'
    if abs(a['goal_heading'])>.065 or abs(NeutralFineStep.waypoint_bearing(p,a['goal_heading']))>.15:return False,'neutral_forward_heading'
    return True,'settled_neutral_head_forward'


def continuous_track_entry(**a):
    import numpy as np
    d=a['final_diag'];m=a['memory']
    if d.get('phase') not in ('OBSERVE','TURN','ADVANCE') or d.get('measured_quiet') is not True:return False,'not_settled_track_entry'
    b=dict(a);b['final_diag']={**d,'phase':'TURN','alignment_mode':'final_goal_heading'};b['velocity_body_xyw']=[0.,0.,.6]
    ok,why=neutral_turn_entry(**b)
    if not ok:return False,why
    p=np.asarray(m['point_level_m'],float);h=a['goal_heading'];lat=-math.sin(h)*p[0]+math.cos(h)*p[1]+.035
    if not(.22<=p[0]<=.38 and np.linalg.norm(p[:2])<=.4 and -.1<=p[1]<=.08 and abs(h)<=.35 and abs(lat)<=.08):return False,'continuous_track_domain'
    return True,'settled_neutral_head_continuous_track'
