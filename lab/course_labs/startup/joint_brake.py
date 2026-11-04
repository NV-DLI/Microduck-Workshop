"""Temporary initialization joint constraints; floating root and native motor model retained."""
import numpy as np
import torch
from types import SimpleNamespace

class JointBrake:
 def __init__(self,robot):
  self.robot=robot;self.active=False;self.release_time=None
  self.limits=robot.data.joint_pos_limits.torch.clone();self.defaults=robot.data.default_joint_pos.torch.clone()
  self.model=robot._backend_manager.get_model()
  self.native={n:getattr(self.model,n).numpy().copy() for n in ['joint_limit_lower','joint_limit_upper','joint_armature','body_mass','body_inertia'] if hasattr(self.model,n)}
  if not robot.root_view.is_floating_base:raise ValueError('Startup must retain a free floating base')
 def lock(self):
  q=self.robot.data.joint_pos.torch.clone()
  self.robot.write_joint_position_limit_to_sim_index(limits=torch.stack([q-1e-6,q+1e-6],dim=-1),warn_limit_violation=False)
  self.robot.data.default_joint_pos.torch.copy_(self.defaults);self.active=True
 def release(self,t):
  if not self.active:return
  self.robot.write_joint_position_limit_to_sim_index(limits=self.limits,warn_limit_violation=False)
  self.robot.data.default_joint_pos.torch.copy_(self.defaults)
  for name,expected in self.native.items():
   if not np.array_equal(getattr(self.model,name).numpy(),expected):raise RuntimeError('Original physics not restored: '+name)
  if not torch.equal(self.robot.data.joint_pos_limits.torch,self.limits):raise RuntimeError('Joint limits not restored')
  self.active=False;self.release_time=t
 def metadata(self):return dict(kind='initialization-only physical joint-limit brake',all_28_joint_axes=True,root_free=True,external_body_support=False,pose_projection_during_rollout=False,restored_before_first_policy_action=self.release_time is not None,release_timestamp_s=self.release_time,restored_native_fields=list(self.native),motor_model_retuned=False)

class NoSupportStartup:
 def __init__(self,reference_pose_xyzw,total_mass_kg):
  self.cfg=SimpleNamespace(calibration_timeout_s=10.,supported_stand_s=0.,release_duration_s=0.)
  self.reference=reference_pose_xyzw;self.mass=total_mass_kg;self.valid_at=None;self.task_started_at=None;self.release_at=None;self.failure=None;self.stable_since=None
 def update(self,timestamp_s,*,position_world,quaternion_xyzw,velocity_world,angular_velocity_world,imu_orientation_valid):
  t=float(timestamp_s)
  if imu_orientation_valid and self.valid_at is None:self.valid_at=t
  upright=1-2*(quaternion_xyzw[0]**2+quaternion_xyzw[1]**2)
  stable=imu_orientation_valid and upright>.995 and np.linalg.norm(velocity_world)<.002 and np.linalg.norm(angular_velocity_world)<.03
  if self.task_started_at is None:
   self.stable_since=(self.stable_since if self.stable_since is not None else t) if stable else None
   if t>=3. and self.stable_since is not None and t-self.stable_since>=.5:self.task_started_at=t
   elif t>=10.:self.failure='locked_grounded_initialization_not_stable'
  enabled=self.task_started_at is not None
  return dict(timestamp_s=t,phase='TASK' if enabled else 'JOINT_BRAKE_INITIALIZATION',support_gain=0.,force_world_n=[0.]*3,torque_world_nm=[0.]*3,imu_orientation_valid=bool(imu_orientation_valid),policy_enabled=enabled,navigation_enabled=enabled,score_enabled=enabled,task_started_at_s=self.task_started_at,task_time_s=None if not enabled else t-self.task_started_at,failure=self.failure,physical_state_written=False)
 def metadata(self):return dict(kind='gravity and ground contact with initialization joint brake; no body support',measured_imu_valid_at_s=self.valid_at,task_started_at_s=self.task_started_at,failure=self.failure,minimum_stationary_initialization_s=3.,external_wrench_all_times_zero=True)
