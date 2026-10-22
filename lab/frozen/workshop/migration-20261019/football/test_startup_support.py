import unittest
import numpy as np
from startup_support import StartupSupport

class SupportTests(unittest.TestCase):
 def state(self):return dict(position_world=[0,0,.125],quaternion_xyzw=[0,0,0,1],velocity_world=[0,0,0],angular_velocity_world=[0,0,0])
 def test_no_orientation_fabrication_on_timeout(self):
  s=StartupSupport([0,0,.125,0,0,0,1],.5)
  for t in np.arange(0,5.01,.005):r=s.update(float(t),**self.state(),imu_orientation_valid=False)
  self.assertEqual(r['phase'],'FAILED');self.assertFalse(r['score_enabled']);self.assertIsNone(s.task_started_at)
 def test_only_exact_zero_wrench_task_is_scored(self):
  s=StartupSupport([0,0,.125,0,0,0,1],.5);seen=set();gains=[]
  for t in np.arange(0,3,.005):
   r=s.update(float(t),**self.state(),imu_orientation_valid=t>=.3);seen.add(r['phase'])
   if r['phase']=='RELEASE':gains.append(r['support_gain'])
   if r['score_enabled']:self.assertEqual(r['force_world_n'],[0,0,0]);self.assertEqual(r['torque_world_nm'],[0,0,0])
   else:self.assertFalse(r['navigation_enabled'])
  self.assertEqual(seen,{'CALIBRATE','SUPPORTED_STAND','RELEASE','TASK'});self.assertTrue(all(a>=b for a,b in zip(gains,gains[1:])));self.assertAlmostEqual(s.task_started_at,1.8)
 def test_real_feedback_not_modified_and_capped(self):
  state=self.state();state['position_world']=[10,10,10];copy=list(state['position_world']);s=StartupSupport([0,0,.125,0,0,0,1],.5)
  r=s.update(0,**state,imu_orientation_valid=False);self.assertEqual(state['position_world'],copy);self.assertLessEqual(np.linalg.norm(r['force_world_n']),12.000001)
 def test_invalidated_calibration_does_not_release(self):
  s=StartupSupport([0,0,.125,0,0,0,1],.5);s.update(0,**self.state(),imu_orientation_valid=True);r=s.update(.01,**self.state(),imu_orientation_valid=False)
  self.assertEqual(r['phase'],'FAILED');self.assertIsNone(s.task_started_at)

if __name__=='__main__':unittest.main(verbosity=2)
