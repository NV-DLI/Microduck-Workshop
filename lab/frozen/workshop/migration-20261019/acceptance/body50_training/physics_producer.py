"""Private200Hz batched producer + same-read quantized encoder observations.

No actor/planner receives the world arrays used here for virtual sensor synthesis.
All asynchronous episode resets have independent derivative/fusion/read state.
"""
import math
import torch
from vector_body_imu import VectorBodyImu,VectorBodyProfile,rotate

class Body50PhysicsProducer:
 def __init__(self,num_envs,*,device='cpu',profile=None,seed=38112,sensor_position_trunk_m=(0.,0.,0.)):
  self.sensor=VectorBodyImu(num_envs,device=device,profile=profile,seed=seed);self.n=num_envs;self.device=self.sensor.device;self.offset=torch.tensor(sensor_position_trunk_m,device=self.device,dtype=torch.float64)
  self.ticks=torch.zeros(num_envs,device=self.device,dtype=torch.int64)
  self._position=torch.zeros((num_envs,3),device=self.device,dtype=torch.float64);self._velocity=self._position.clone();self._has_position=torch.zeros(num_envs,device=self.device,dtype=torch.bool);self._has_velocity=self._has_position.clone()
  self.encoder_q=torch.zeros((num_envs,14),device=self.device,dtype=torch.float64);self.encoder_qd=self.encoder_q.clone();self.encoder_read_time=torch.zeros(num_envs,device=self.device,dtype=torch.float64)
 def reset(self,env_ids=None):
  ids=slice(None)if env_ids is None else env_ids;self.sensor.reset(env_ids)
  for value in(self.ticks,self._position,self._velocity,self._has_position,self._has_velocity,self.encoder_q,self.encoder_qd,self.encoder_read_time):value[ids]=0
 def update(self,*,trunk_position_world_m,trunk_quaternion_xyzw,trunk_angular_velocity_world_rad_s,effective_encoder_q_rad,effective_encoder_qd_rad_s,noise_standard_normal=None):
  # Exactly the same two differences as IsaacBodyImu, on a200Hz clock.
  self.ticks+=1;t=self.ticks.to(torch.float64)*.005
  pos=torch.as_tensor(trunk_position_world_m,dtype=torch.float64,device=self.device);q=torch.as_tensor(trunk_quaternion_xyzw,dtype=torch.float64,device=self.device)
  sensor_point=pos+rotate(q,self.offset);velocity=(sensor_point-self._position)/.005;acceleration=(velocity-self._velocity)/.005;enabled=self._has_position&self._has_velocity
  packet=self.sensor.poll(t,q_world_from_trunk_xyzw=q,angular_velocity_world_rad_s=trunk_angular_velocity_world_rad_s,linear_acceleration_world_m_s2=acceleration,noise_standard_normal=noise_standard_normal,enabled_mask=enabled)
  self._velocity.copy_(torch.where(self._has_position[:,None],velocity,self._velocity));self._position.copy_(sensor_point);self._has_velocity|=self._has_position;self._has_position[:]=True
  eq=torch.as_tensor(effective_encoder_q_rad,device=self.device,dtype=torch.float64);ev=torch.as_tensor(effective_encoder_qd_rad_s,device=self.device,dtype=torch.float64)
  if eq.shape!=(self.n,14)or ev.shape!=(self.n,14):raise ValueError('Native14 effective encoders plus declared calibration bias required')
  position_lsb=2*math.pi/4096;velocity_lsb=.229*2*math.pi/60;read=packet['delivered_mask']
  self.sensor._put(self.encoder_q,torch.round(eq/position_lsb)*position_lsb,read);self.sensor._put(self.encoder_qd,torch.round(ev/velocity_lsb)*velocity_lsb,read);self.sensor._put(self.encoder_read_time,t,read)
  return {**self.observe(),'delivered_mask':read}
 def observe(self):
  packet=self.sensor.observe(self.ticks.to(torch.float64)*.005)
  return {**packet,'encoder_q_rad':self.encoder_q.clone(),'encoder_qd_rad_s':self.encoder_qd.clone(),'encoder_read_timestamp_s':self.encoder_read_time.clone()}
 def build_actor_observation(self,base_observation61,*,native_home14,previous_raw_action14):
  """Return(obs,ready). Caller MUST hold physical home when ready is false.

  Invalid rows are all zeros so they cannot accidentally leak the native
  simulator IMU. They are never valid network inputs: use ready to mask actor
  action, and exclude explicit initialization from rewards/episode clocks.
  """
  if base_observation61.shape!=(self.n,61):raise ValueError('Exactly official61D policy input required')
  packet=self.observe();ready=packet['orientation_valid'];out=base_observation61.clone();dtype=out.dtype
  out[:,:3]=packet['gyro_body_rad_s'].to(dtype);out[:,3:6]=packet['gravity_body'].to(dtype);out[:,6:20]=packet['encoder_q_rad'].to(dtype)-native_home14;out[:,20:34]=packet['encoder_qd_rad_s'].to(dtype);out[:,34:48]=previous_raw_action14
  out=torch.where(ready[:,None],out,torch.zeros_like(out));return out,ready
