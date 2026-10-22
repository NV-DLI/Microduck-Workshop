import unittest
from neutral_fine_step import NeutralFineStep
from short_primitive import select
from test_neutral_fine_step import packet
class Decoupled(unittest.TestCase):
 def test_lateral_sign_does_not_shift_goal_heading_equilibrium(self):
  for y in [-.08,.02]:
   f=NeutralFineStep();f.primitive={'velocity_command':[.3,-.1,0]};f.goal_heading=0.;v=f.macro_command([.18,y,-.08],packet(1.),.02)
   self.assertEqual(v[1],-.1);self.assertAlmostEqual(v[2],0.)
 def test_large_heading_requires_alignment_before_macro(self):
  self.assertIsNone(select([.1867,-.0321,-.08],-.1822))
  self.assertIsNone(select([.2,-.05,-.08],.081))
