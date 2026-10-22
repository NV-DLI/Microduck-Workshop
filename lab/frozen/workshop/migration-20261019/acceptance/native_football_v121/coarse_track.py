"""Empty surveyed field only: fresh path-authorized, quiet coarse TRACK entry.

Same public observations, original 10cm envelope and measured quiet criteria.
No new trained model, no physical velocity claim, no kick authority.
"""
import math
from collections import deque
import numpy as np
from nav2_service.map_contract import snapshot,check_path
from nav2_bridge import make_request
from visual_localization import rotate

class CoarseObserver:
 def __init__(self):self.samples=deque();self.last=None
 def clear(self):self.samples.clear();self.last=None
 def update(self,t,p,gyro,heading,authorized):
  if not authorized or not all(np.isfinite(x).all()for x in (p,gyro)):
   self.clear();return {'phase':'OBSERVE','measured_quiet':False,'reason':'coarse_public_path_guard'}
  if self.last is not None and not 0<t-self.last<=.2:self.clear()
  self.last=t;self.samples.append((t,np.array(p[:2]),float(np.dot(gyro,gyro))))
  while self.samples and t-self.samples[0][0]>.45:self.samples.popleft()
  pts=np.stack([x[1]for x in self.samples]);diam=float(np.linalg.norm(pts[:,None]-pts[None,:],axis=-1).max());rms=float(np.sqrt(np.mean([x[2]for x in self.samples])))
  quiet=len(self.samples)>=6 and t-self.samples[0][0]>=.33 and rms<=.12 and diam<=.02
  return dict(phase='OBSERVE',point_level_m=np.asarray(p).tolist(),goal_heading_rad=float(heading),measured_quiet=quiet,quiet_sample_times=[x[0]for x in self.samples],gyro_3D_RMS_rad_s=rms,ball_XY_diameter_m=diam,coarse_track_observation=True,truth_used=False,kick_authorized=False)

def path_authority(owner,frame,proprio,R,height,point):
 t=proprio['timestamp_s'];p=np.asarray(point,float)
 result=dict(capture_s=t,allowed=False,layout=owner.config.layout,truth_used=False,reason='coarse_path_inputs')
 if owner.config.layout!='empty' or owner.pose is None or not np.isfinite(owner.pose).all() or not 0<=t-owner.goal_time<=8 or p.shape!=(3,)or not np.isfinite(p).all():return result
 sample,hazard=owner._tof_sample(frame,proprio,R,height,p)
 if hazard:result['reason']='current_ToF_hazard';return result
 ball=np.asarray(owner.pose[:2])+np.asarray(rotate(owner.pose[2],p[:2]));delta=owner.goal_map-ball;length=np.linalg.norm(delta)
 if length<.1:return result
 unit=delta/length;side=np.array([-unit[1],unit[0]]);target=ball-.36*unit+.028*side
 # Empty surveyed field is explicit in make_request. This is the same original
 # public map and conservative envelope, independently checked at each capture.
 case,prior=make_request(time_s=t,pose_map_xyyaw=owner.pose.tolist(),velocity_body_xyw=[0.,0.,0.],goal_map_xy=target.tolist(),tof_sample=sample,layout='empty',image_sha256=owner.image_sha,localization_capture_s=owner.goal_time)
 snap=snapshot(case,prior);gate=check_path([owner.pose[:2].tolist(),target.tolist()],snap)
 result.update(allowed=bool(gate['accepted']),reason=gate['reason'],envelope=gate,target_map_xy=target.tolist(),source_capture_s=sample['capture_s'],point_level_m=p.tolist(),map_request={'case':case,'provenance':prior})
 return result

def coarse_entry(*,path,**a):
 d=a['final_diag'];m=a['memory'];t=a['t']
 if not(path and path.get('allowed') is True and path.get('layout')=='empty' and path.get('capture_s')==t):return False,'coarse_current_path_required'
 if not(d.get('coarse_track_observation') is True and d.get('measured_quiet') is True and d.get('phase')=='OBSERVE'):return False,'coarse_measured_quiet_required'
 if not(a['handoff_state']=='APPROACH' and not a['fine_active'] and a['fine_state']=='IDLE' and a['complete'] is True and a['guard_safe'] is True and m.get('memory_available') is True and m.get('ball_capture_timestamp_s')==t):return False,'coarse_fresh_input_required'
 p=np.asarray(m.get('point_level_m'),float);head=np.asarray(a['actual_head'],float);v=np.asarray(a['velocity_body_xyw'],float);h=a['goal_heading'];g=a['goal_fix_s']
 if not(p.shape==(3,) and head.shape==(4,) and v.shape==(3,) and all(np.isfinite(x).all()for x in (p,head,v)) and math.isfinite(h) and math.isfinite(g) and 0<=t-g<=8):return False,'coarse_nonfinite_or_stale'
 if not(.38<p[0]<=.85 and np.linalg.norm(p[:2])<=.9 and abs(p[1])<=.25 and abs(h)<=.35 and abs(-math.sin(h)*p[0]+math.cos(h)*p[1]+.028)<=.18 and max(abs(head))<=.95 and np.all(abs(head)<=np.array([1.1,1.1,1.4,.31])) and not np.any(v)):return False,'coarse_domain'
 times=d.get('quiet_sample_times',[])
 if len(times)<6 or any(not math.isfinite(x)for x in times)or any(not 0<b-a<=.2 for a,b in zip(times,times[1:]))or abs(times[-1]-t)>1e-8 or not .33<=times[-1]-times[0]<=.45+1e-8 or not 0<=d.get('gyro_3D_RMS_rad_s',math.inf)<=.12 or not 0<=d.get('ball_XY_diameter_m',math.inf)<=.02:return False,'coarse_quiet_window'
 return True,'settled_coarse_track_current_path'
