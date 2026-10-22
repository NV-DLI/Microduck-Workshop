import unittest
import numpy as np
from neutral_fine_step import NeutralFineStep
class NearRangeSpeed(unittest.TestCase):
 def test_public_range_switch_keeps_feedback_and_model_commands(self):
  packet=dict(seq=1,read_timestamp_s=1.,gyro_body_rad_s=[0.,0.,0.],gravity_body=[0.,0.,-1.])
  for x,expected in ((.3,.3),(.17001,.3),(.17,.25),(.12,.25)):
   f=NeutralFineStep();v=f.track_filtered(np.array([x,-.035,-.08]),0.,.02,packet)
   self.assertEqual(v,[expected,0.,0.])
if __name__=='__main__':unittest.main()
