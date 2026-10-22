import unittest,math
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet
class ShotFrame(unittest.TestCase):
 def ready(self,h):
  f=NeutralFineStep();f.primitive={'velocity_command':[.3,-.1,0.]};f.goal_heading=h;return f
 def test_rotated_same_goal_frame_point_does_not_command_sideways(self):
  h=.15;c,s=math.cos(h),math.sin(h);p=[c*.2-s*(-.035),s*.2+c*(-.035),-.08]
  f=self.ready(h);v=f.macro_command(p,packet(1.),.02);self.assertAlmostEqual(f.macro_shot_error,0.);self.assertEqual(v[1],-.1)
 def test_declared_hysteresis_and_measured_command_magnitude(self):
  f=self.ready(0.);self.assertEqual(f.macro_command([.2,-.06,-.08],packet(1.),.02)[1],-.1)
  for i in range(30):v=f.macro_command([.15,-.030,-.08],packet(1.),.02)
  self.assertEqual(v[1],-.1);self.assertLess(abs(v[2]),.5)
