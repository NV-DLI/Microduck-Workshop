import unittest
from short_primitive import select
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet
class Fixed(unittest.TestCase):
 def test_original_box_predictions_include_heading_and_four_replicas(self):
  for p in ([.18,-.05,-.08],[.21,-.075,-.08]):
   r=select(p,0.);self.assertIsNotNone(r);self.assertEqual(len(r['predicted_endpoints']),4)
   self.assertTrue(all(.075<=x<=.105 and -.05<=y<=-.02 and abs(h)<=.08 for x,y,h in r['predicted_endpoints']))
 def test_old_unmeasured_intermediate_duration_is_rejected(self):
  self.assertIsNone(select([.196,-.039,-.08],.035))
 def test_selected_lateral_command_is_constant_despite_sway(self):
  f=NeutralFineStep();f.primitive={'velocity_command':[.3,-.1,0]};f.goal_heading=0.
  for y in [-.09,0,.03]:self.assertEqual(f.macro_command([.17,y,-.08],packet(1.),.02)[1],-.1)
