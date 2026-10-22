import math,unittest
import numpy as np
from head_adapter import HeadAdapter,vee
from nav2_bridge import quaternion_xyzw
from test_cpu import NAMES

class HeadAdapterTests(unittest.TestCase):
 def test_native_home_and_named_ik(self):
  adapter=HeadAdapter()
  for target in [(0,0),(.4,0),(-.4,.6),(0,.7)]:
   r=adapter.solve(target);self.assertTrue(r['accepted']);self.assertLess(r['orientation_residual_rad'],1e-4);self.assertFalse(r['world_stabilized'])
 def test_remove_relative_head_motion_and_local_yaw_invariance(self):
  h=HeadAdapter()
  for head_q in [[.3491,.3491,0,0],[.1,.6,-.3,.15]]:
   q=np.zeros(14);qd=np.zeros(14)
   for n,v,w in zip(['neck_pitch','head_pitch','head_yaw','head_roll'],head_q,[.1,-.2,.3,.05]):q[NAMES.index(n)]=v;qd[NAMES.index(n)]=w
   R=np.asarray(h.fk.forward_policy14(q,joint_names=NAMES)['camera_site'].rotation);Rnext=np.asarray(h.fk.forward_policy14(q+qd*1e-5,joint_names=NAMES)['camera_site'].rotation)
   relative_omega=vee((Rnext-R)@R.T/1e-5)
   for local_yaw in [0.,1.2]:
    c,s=math.cos(local_yaw),math.sin(local_yaw);local_R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@R;quat=quaternion_xyzw(local_R)
    packet={'timestamp_s':.2,'orientation_valid':True,'axes':'head FLU','quaternion_order':'wxyz','absolute_world_yaw':False,'gyro':(R.T@relative_omega).tolist(),'quat':[quat[3],*quat[:3]]}
    result=h.body_imu(packet,NAMES,q,qd,capture_s=.2)
    self.assertLess(np.linalg.norm(result['gyro_body_rad_s']),1e-8);self.assertLess(np.linalg.norm(np.asarray(result['gravity_body'])-[0,0,-1]),1e-8)
 def test_stale_imu_rejected(self):
  packet={'timestamp_s':.19,'orientation_valid':True,'axes':'head FLU','quaternion_order':'wxyz','absolute_world_yaw':False,'gyro':[0,0,0],'quat':[1,0,0,0]}
  with self.assertRaises(ValueError):HeadAdapter().body_imu(packet,NAMES,[0]*14,[0]*14,capture_s=.2)
 def test_uninitialized_imu_rejected_before_pose_use(self):
  with self.assertRaisesRegex(ValueError,'initialization'):
   HeadAdapter().body_imu({'orientation_valid':False,'quat':None},NAMES,[0]*14,[0]*14,capture_s=.2)
 def test_15hz_capture_matches_every_fourth_60hz_acquisition(self):
  for n in range(1,2701):self.assertEqual(math.ceil(n*200/15-1e-10),math.ceil(n*4*200/60-1e-10))

if __name__=='__main__':unittest.main(verbosity=2)
