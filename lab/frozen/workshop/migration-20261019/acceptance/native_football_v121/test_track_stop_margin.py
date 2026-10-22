import unittest
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet
class StopMargin(unittest.TestCase):
 def test_final_track_aims_original_box_center(self):
  f=NeutralFineStep();f.goal_s=1.0
  v=f.track_filtered([.2,-.035,-.08],0.,.02,packet(1.))
  self.assertAlmostEqual(v[1],0.);self.assertEqual(v[0],.3)
 def test_original_kick_box_unchanged(self):
  from final_approach import in_training_box
  self.assertFalse(in_training_box([.1,-.0509,0]))
  self.assertTrue(in_training_box([.1,-.028,0]))
if __name__=='__main__':unittest.main()
