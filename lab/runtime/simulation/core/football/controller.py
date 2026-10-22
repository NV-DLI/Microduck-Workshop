"""Controller: Microduck simulation and control."""
from dataclasses import dataclass,asdict
from collections import deque
import math
import time
import numpy as np
from perception import BetaPerception,validate_observation,level_rotation
from visual_localization import fit_posts,rotate,wrap
from nav2_bridge import make_request,quaternion_xyzw
from gaze_tracking import observed_ball_gaze,GazeSlew

@dataclass(frozen=True)
class ControllerConfig:
 layout:str='empty'
 duration_s:float=180.
 coarse_speed_m_s:float=.06
 fine_speed_m_s:float=.025
 max_yaw_rad_s:float=.2
 preparation_distance_m:float=.22
 kick_ball_body_xy_m:tuple=(.105,-.035)
 kick_offset_calibrated_on_new_stack:bool=False
 gait_id:str='reference'
 ground_mode:str='legacy-imu-sole'

class BetaNavigator:
 def __init__(self,nav2_session,*,camera=None,config=None):
  self.config=config or ControllerConfig()
  if self.config.ground_mode not in ('legacy-imu-sole','tof-virtual-floor'):raise ValueError('Explicit navigation ground model required')
  self.vision=BetaPerception(camera);self.nav2=nav2_session;self.reset()
 def reset(self):
  self.state='SEARCH_GOAL';self.pose=None;self.goal_window=deque(maxlen=3);self.last_time=None;self.first_time=None
  self.goal_time=-math.inf;self.ball=None;self.ball_time=-math.inf;self.goal_map=np.array([.93,0.]);self.image_sha=None
  self.last_plan=None;self.last_plan_time=-math.inf;self.quiet=deque();self.kick_time=None;self.kick_count=0
  self.last_error=None;self.last_output=None;self.last_geometry=None
  self.gaze=GazeSlew();self.last_observed_ball_gaze=None
  self.nav2_wall_seconds=0.;self.nav2_requests=0
 def _output(self,t,command=(0.,0.,0.),reason='',gaze=None):
  requested_gaze=[0.,0.]if gaze is None else gaze
  executed_gaze=self.gaze.update(requested_gaze,t)
  out={'schema':'microduck.beta-control.v1','timestamp_s':t,'runtime_id':'newton-bam-20261019','hardware_profile_id':'beta-gc2093-l8cx-v1',
   'state':self.state,'velocity_body_xyw':list(map(float,command)),
   'velocity_semantics':'desired physical body FLU velocity in m/s, m/s, rad/s; requires independently calibrated current gait adapter; not raw v8 policy command',
   'gaze_body_yaw_pitch_rad':executed_gaze,'gaze_tracking':self.gaze.last,'gaze_convention':'body FLU; yaw positive left, pitch positive downward; adapter must solve named head joints',
   'kick_request':False,'kick_side':'right','kick_duration_s':.5,'reason':reason,'gait_id':self.config.gait_id,
   'old_stop_or_velocity_predictor_used':False,'kick_offset_calibrated_on_new_stack':self.config.kick_offset_calibrated_on_new_stack,
   'source_rgb_capture_s':None if self.last_geometry is None else self.last_geometry['timestamp_s'],'source_rgb_sha256':self.image_sha,
   'navigation_ground_mode':self.config.ground_mode,'virtual_ToF_hardware_model_calibrated':False,
   'estimated_pose_map_xyyaw':None if self.pose is None else self.pose.tolist(),'truth_inputs':False,'football_success':None}
  self.last_output=out;return out
 def _tof_sample(self,frame,o,R,height,observed_ball_level_xyz=None):
  t=frame['timestamp_s']
  if not 0<=o['timestamp_s']-t<=.15:raise ValueError('stale/future ToF')
  cells=[c for row in frame['cells'] for c in row]
  if len(frame['cells'])!=8 or any(len(r)!=8 for r in frame['cells']) or len(cells)!=64:raise ValueError('Exactly8x8 ToF required')
  # Caller must supply same-time FK for each ToF capture. Default synchronous
  # mode permits exact capture only; do not use current q to place an old scan.
  if abs(t-o['timestamp_s'])>1e-6:raise ValueError('ToF requires its own acquisition-time encoder/IMU packet')
  fk=self.vision.head.forward_policy14(o['q_rad'],joint_names=o['joint_names'])['tof_flu'];yaw=self.pose[2]
  c,s=math.cos(yaw),math.sin(yaw);map_R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@R
  loc=map_R@np.asarray(fk.translation_m);valid=[c['status']=='valid' and not c.get('contains_too_close',False) for c in cells]
  sample={'capture_s':t,'fk_capture_s':t,'provenance':'simulation_approximation','status':[None]*64,'ranges_m':[c['range_m'] for c in cells],'isaac_valid_mask':valid,'near_m':.1,'far_m':4.,'unit_rays_sensor':[c['ray_sensor_flu'] for c in cells],
   'sensor_translation_odom':[self.pose[0]+loc[0],self.pose[1]+loc[1],height+loc[2]],'sensor_quaternion_odom_xyzw':quaternion_xyzw(map_R@np.asarray(fk.rotation))}
  # Geometric immediate guard uses only actual valid endpoints, retaining
  # no-return and TooClose separately. A low ground return is not an obstacle.
  hazard=False
  for cell,ok in zip(cells,valid):
   if cell.get('contains_too_close',False):hazard=True;continue
   if not ok or cell['range_m'] is None:continue
   p=np.asarray(fk.translation_m)+np.asarray(fk.rotation)@(np.asarray(cell['ray_sensor_flu'])*cell['range_m']);v=R@p
   # A valid endpoint on the freshly RGB-observed target sphere is expected
   # during the kick approach. This is a sensor-only semantic exception, never
   # a blanket ToF disable. Unknown and TooClose cells still stop above.
   if observed_ball_level_xyz is not None and np.linalg.norm(v-observed_ball_level_xyz)<.047:continue
   if .04<v[2]+height<.45 and 0<v[0]<.105 and abs(v[1])<.09:hazard=True
  return sample,hazard
 def step(self,rgb,proprio,tof_frame,*,now_s,detections=None):
  validate_observation(proprio);t=float(proprio['timestamp_s'])
  if not 0<=now_s-t<=.2:return self._output(now_s,reason='stale_proprioception')
  if self.last_time is not None and t<=self.last_time:raise ValueError('Monotone new control sample required')
  dt=0 if self.last_time is None else t-self.last_time;self.last_time=t
  if self.first_time is None:self.first_time=t
  if dt>.2:
   self.pose=None;self.goal_window.clear();self.ball=None;self.state='SEARCH_GOAL'
   return self._output(t,reason='control_gap_requires_visual_relocalization',gaze=[0.,0.])
  if t-self.first_time>self.config.duration_s:self.state='TIMEOUT';return self._output(t,reason='180s_teaching_limit')
  geo=self.vision.estimate(rgb,proprio,detections,now_s=now_s,tof_frame=tof_frame if self.config.ground_mode=='tof-virtual-floor'else None);self.last_geometry=geo;self.image_sha=geo['image_sha256']
  if self.config.ground_mode=='tof-virtual-floor'and not geo['floor_observation']['accepted']:
   self.state='GROUND_UNAVAILABLE';return self._output(t,reason='actual_ToF_floor_unavailable',gaze=self.gaze.current.tolist())
  navigation_gravity=(-np.asarray(geo['ground']['normal'])).tolist()if self.config.ground_mode=='tof-virtual-floor'else proprio['gravity_body']
  R=level_rotation(navigation_gravity)
  kin=self.vision.leg.estimate(proprio['q_rad'],proprio['qd_rad_s'],proprio['gyro_body_rad_s'],navigation_gravity,proprio['joint_names'])
  v=R@np.asarray(kin['velocity_body_m_s']);omega=R@np.asarray(proprio['gyro_body_rad_s'])
  if self.pose is not None and dt>0:
   delta=rotate(self.pose[2],(v[:2]*dt).tolist());self.pose[:2]+=delta;self.pose[2]=wrap(self.pose[2]+omega[2]*dt)
  goal=geo.get('goal')
  if goal and goal['accepted']:
   points=[(R@np.asarray(p)).tolist() for p in goal['geometry']['post_points_reference_m']]
   try:
    fix=fit_posts(points,float(self.pose[2]) if self.pose is not None else 0.);measured=np.asarray(fix['pose_map_xyyaw'])
    self.goal_window.append((t,measured))
    if len(self.goal_window)==3 and t-self.goal_window[0][0]<.5:
     centre=np.mean([p[:2] for _,p in self.goal_window],axis=0);a=np.array([p[2] for _,p in self.goal_window]);yaw=math.atan2(np.sin(a).mean(),np.cos(a).mean())
     if max(np.linalg.norm(p[:2]-centre) for _,p in self.goal_window)<.04:
      if self.pose is None or np.linalg.norm(self.pose[:2]-centre)<.20:
       self.pose=np.array([*centre,yaw]);self.goal_time=t
   except ValueError:self.goal_window.clear()
  ball=geo.get('ball')
  ball_body=None
  if ball and ball['accepted']:
   ball_body=R@np.asarray(ball['geometry']['point_reference_m'])
   self.last_observed_ball_gaze=observed_ball_gaze(ball['geometry']['point_reference_m'],geo['pose']['translation_m'])
   if self.pose is not None:
    d=rotate(self.pose[2],ball_body[:2]);self.ball=self.pose[:2]+d;self.ball_time=t
  if self.pose is None:
   elapsed=t-self.first_time
   self.state='SEARCH_GOAL';return self._output(t,reason='need_three_observed_goal_base_pairs',gaze=[0. if elapsed<.7 else .4*math.sin((elapsed-.7)*.6),0.])
  # Reobserve the known goal periodically. This arrests encoder-odometry drift
  # without accepting a simulator pose or the old learned velocity predictor.
  if t-self.goal_time>3.:
   self.state='REFRESH_GOAL';return self._output(t,reason='visual_localization_refresh',gaze=[0.,0.])
  if self.ball is None or t-self.ball_time>.25:
   self.state='SEARCH_BALL';return self._output(t,reason='fresh_ball_required',gaze=self.last_observed_ball_gaze or [.25*math.sin((t-self.first_time)*.8),.7])
  try:tof,hazard=self._tof_sample(tof_frame,proprio,R,geo['height']['conditional_root_height_m'],ball_body)
  except (ValueError,KeyError) as e:return self._output(t,reason=str(e))
  if hazard:return self._output(t,reason='actual_ToF_near_obstacle_or_too_close')
  shot=self.goal_map-self.ball;length=np.linalg.norm(shot)
  if length<.1:return self._output(t,reason='ball_goal_direction_degenerate')
  unit=shot/length;shot_yaw=math.atan2(unit[1],unit[0]);yaw_error=wrap(shot_yaw-self.pose[2]);side=np.array([-unit[1],unit[0]])
  desired_root=self.ball-self.config.kick_ball_body_xy_m[0]*unit-self.config.kick_ball_body_xy_m[1]*side
  delta=desired_root-self.pose[:2];local=np.asarray(rotate(-self.pose[2],delta));distance=np.linalg.norm(self.ball-self.pose[:2])
  gaze=self.last_observed_ball_gaze or [0.,.7]
  if self.kick_time is not None:
   self.state='KICK' if t-self.kick_time<.5 else 'OBSERVE_AFTER_KICK'
   if t-self.kick_time>=2. and self.kick_count<3 and np.linalg.norm(self.goal_map-self.ball)>.15:
    self.kick_time=None;self.quiet.clear();self.last_plan=None
   else:return self._output(t,reason='kick_policy_runs_once_then_return_to_walk_zero',gaze=gaze)
  local_blocked=False
  if self.config.layout=='one-obstacle':
   from nav2_service.obstacle_contract import bounds
   a,b,c,d=bounds();pts=np.linspace(self.pose[:2],desired_root,max(2,math.ceil(np.linalg.norm(delta)/.01)))
   local_blocked=any(a-.1<=p[0]<=b+.1 and c-.1<=p[1]<=d+.1 for p in pts)
  if distance>.31 or local_blocked:
   self.state='NAV2_APPROACH';target=self.ball-self.config.preparation_distance_m*unit-self.config.kick_ball_body_xy_m[1]*side
   if self.last_plan is None or t-self.last_plan_time>.12:
    case,provenance=make_request(time_s=t,pose_map_xyyaw=self.pose.tolist(),velocity_body_xyw=[float(v[0]),float(v[1]),float(omega[2])],goal_map_xy=target.tolist(),tof_sample=tof,layout=self.config.layout,image_sha256=self.image_sha,localization_capture_s=self.goal_time)
    try:
     nav_start=time.monotonic()
     try:self.last_plan=self.nav2.query(case,provenance);self.last_plan_time=t
     finally:self.nav2_wall_seconds+=time.monotonic()-nav_start;self.nav2_requests+=1
    except (ValueError,TimeoutError,OSError) as e:self.last_plan=None;return self._output(t,reason='Nav2_failed:'+str(e),gaze=gaze)
   path=self.last_plan['path_xy']
   if not self.last_plan['envelope']['accepted'] or not path:return self._output(t,reason='Nav2_path_not_admitted',gaze=gaze)
   near=min(range(len(path)),key=lambda i:math.dist(path[i],self.pose[:2]));way=next((p for p in path[near:] if math.dist(p,self.pose[:2])>.10),path[-1]);vec=np.asarray(rotate(-self.pose[2],np.asarray(way)-self.pose[:2]));ang=math.atan2(vec[1],vec[0]);norm=max(np.linalg.norm(vec),1e-9)
   cmd=[*(self.config.coarse_speed_m_s*vec/norm),float(np.clip(ang,-self.config.max_yaw_rad_s,self.config.max_yaw_rad_s))]
   self.quiet.clear();return self._output(t,cmd,'actual_Navfn_path',gaze)
  self.state='VISUAL_ALIGN'
  # A local segment must not cut back through the known obstacle after bypass.
  if self.config.layout=='one-obstacle':
   from nav2_service.obstacle_contract import bounds
   lo_x,hi_x,lo_y,hi_y=bounds();n=max(2,math.ceil(np.linalg.norm(delta)/.01));points=np.linspace(self.pose[:2],desired_root,n)
   if any(lo_x-.1<=p[0]<=hi_x+.1 and lo_y-.1<=p[1]<=hi_y+.1 for p in points):
    self.last_plan=None;return self._output(t,reason='local_segment_hits_surveyed_obstacle',gaze=gaze)
  aligned=abs(local[0])<.012 and abs(local[1])<.008 and abs(yaw_error)<.07
  if not aligned:
   self.quiet.clear();cmd=[float(np.clip(.5*local[0],-.018,self.config.fine_speed_m_s)),float(np.clip(.5*local[1],-.018,.018)),float(np.clip(yaw_error,-.12,.12))]
   return self._output(t,cmd,'fresh_RGB_position_error',gaze)
  self.state='SETTLE'
  self.quiet.append((t,self.pose.copy(),self.ball.copy(),float(np.linalg.norm(omega))))
  while self.quiet and t-self.quiet[0][0]>.7:self.quiet.popleft()
  if len(self.quiet)>=3 and t-self.quiet[0][0]>=.5:
   positions=[p for _,p,_,_ in self.quiet];balls=[b for _,_,b,_ in self.quiet]
   quiet=max(np.linalg.norm(p[:2]-positions[0][:2]) for p in positions)<.006 and max(np.linalg.norm(b-balls[0]) for b in balls)<.008 and max(w for *_,w in self.quiet)<.35
   if quiet:
    self.kick_time=t;self.kick_count+=1;self.state='KICK';out=self._output(t,reason='visual_and_IMU_settled_parameters',gaze=gaze);out['kick_request']=True;return out
  return self._output(t,reason='zero_command_observe_real_settling',gaze=gaze)
