import unittest
from short_primitive import select
class TestPrimitive(unittest.TestCase):
 def test_measured_lateral_step_matches_box(self):
  p=select([.21,-.075,-.08],0.)
  self.assertIsNotNone(p)
  self.assertEqual(p['velocity_command'],[.3,-.2,0])
  self.assertTrue(.4<=p['forward_duration_s']<=.8)
  self.assertTrue(all(.075<=x<=.105 and -.05<=y<=-.02 for x,y,h in p['predicted_endpoints']))
 def test_nonfinite_no_action(self):
  self.assertIsNone(select([float('nan'),0,0],0))
 def test_no_endpoint_fit_no_action(self):
  self.assertIsNone(select([.13,.06,-.08],0.))
