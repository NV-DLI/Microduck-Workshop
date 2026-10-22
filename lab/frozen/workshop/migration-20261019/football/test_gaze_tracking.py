import unittest
import math
from gaze_tracking import GazeSlew,observed_ball_gaze

class GazeTests(unittest.TestCase):
    def test_camera_translation_matters_near_ball(self):
        v=observed_ball_gaze([.105,-.035,-.09],[.074,0,.126])
        self.assertGreater(v[1],1.3)
        self.assertAlmostEqual(v[0],math.atan2(-.035,.031))
    def test_slew_and_bounds_are_explicit(self):
        s=GazeSlew();self.assertEqual(s.update([0,1.4],0),[0,0])
        self.assertAlmostEqual(s.update([0,1.4],.1)[1],.06)
        self.assertEqual(s.update([2,2],10),[.65,1.45]);self.assertTrue(s.last['limit_clipped'])
        self.assertFalse(s.last['world_stabilized'])
    def test_no_backward_clock_or_invalid_target(self):
        s=GazeSlew();s.update([0,0],1)
        with self.assertRaises(ValueError):s.update([0,0],.5)
        with self.assertRaises(ValueError):observed_ball_gaze([0,0,float('nan')],[0,0,0])

if __name__=='__main__':unittest.main()
