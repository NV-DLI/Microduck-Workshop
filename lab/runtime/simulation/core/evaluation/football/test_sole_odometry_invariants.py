import unittest
from pathlib import Path
import numpy as np
from leg_odometry import LegFK
from sole_kinematics import SoleKinematics
from contact_odometry import SoleContactLegOdometry
M=Path(__file__).resolve().parents[2]
N=('left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle','neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw','right_hip_roll','right_hip_pitch','right_knee','right_ankle')
class SoleInvariants(unittest.TestCase):
 def setUp(self):self.leg=LegFK(M/'sensors/sources/robot_allcollisions.xml',require_pinned=False);self.kin=SoleKinematics(self.leg);self.est=SoleContactLegOdometry(self.leg)
 def test_static_encoders_and_imu_imply_zero_velocity(self):
  for q in [np.zeros(14),np.linspace(-.3,.3,14)]:
   r=self.est.estimate(q,np.zeros(14),[0,0,0],[0,0,-1],N);np.testing.assert_allclose(r['velocity_body_m_s'],0,atol=1e-15);self.assertFalse(r['contact_observed'])
 def test_material_point_jacobian_matches_finite_difference(self):
  q=np.linspace(-.25,.3,14);qd=np.linspace(.2,-.15,14);dt=1e-6
  for name in ['left_foot','right_foot']:
   f=self.kin.foot(name,q,qd,N,np.array([0,0,1]),.0005);R,p,hinges,_=self.kin.frame(name,q,N);local=R.T@(f['contact']-p);R2,p2,_,_=self.kin.frame(name,q+dt*qd,N)
   np.testing.assert_allclose((p2+R2@local-f['contact'])/dt,f['v_contact'],atol=1e-7)
 def test_head_joints_cannot_move_foot_kinematics(self):
  q=np.zeros(14);qd=np.zeros(14);qd[5:9]=[.3,-.2,.1,.4];r=self.est.estimate(q,qd,[0,0,0],[0,0,-1],N);np.testing.assert_allclose(r['velocity_body_m_s'],0,atol=1e-15)
 def test_inputs_are_preserved_and_truth_arguments_rejected(self):
  q=np.linspace(-.2,.2,14);before=q.copy();self.est.estimate(q,np.zeros(14),[0,0,0],[0,0,-1],N);np.testing.assert_array_equal(q,before)
  with self.assertRaises(TypeError):self.est.estimate(q,np.zeros(14),[0,0,0],[0,0,-1],N,root_velocity=[1,0,0])
 def test_invalid_channels_fail_closed(self):
  for q,g in [(np.zeros(13),[0,0,-1]),(np.full(14,np.nan),[0,0,-1]),(np.zeros(14),[0,0,0])]:
   with self.assertRaises(ValueError):self.est.estimate(q,np.zeros(14),[0,0,0],g,N)
