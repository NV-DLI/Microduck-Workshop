import unittest
from short_primitive import select,CANDIDATES
class Interpolated(unittest.TestCase):
 def test_observed_between_grid_start_has_candidate(self):
  r=select([.18,-.05,-.08],0.);self.assertIsNotNone(r)
  self.assertTrue(all(.075<=p[0]<=.105 and -.05<=p[1]<=-.02 for p in r['predicted_endpoints']))
 def test_no_extrapolation_or_larger_source_command(self):
  for r in CANDIDATES:
   self.assertTrue(.4<=r['forward_duration_s']<=.8);self.assertTrue(-.2<=r['velocity_command'][1]<=.2)
   self.assertIn(r['forward_duration_s'],(.6,.8))
   self.assertIn(r['velocity_command'][1],(-.2,-.1,0.))
 def test_original_public_entry_domain(self):
  for p,h in [([.3,-.035,0],0),([.09,-.035,0],0),([.18,-.035,0],.31)]:self.assertIsNone(select(p,h))
