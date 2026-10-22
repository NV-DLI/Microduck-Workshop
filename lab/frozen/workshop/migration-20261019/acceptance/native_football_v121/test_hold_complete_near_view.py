import unittest
from types import SimpleNamespace as S
import numpy as np
from ball_reacquisition import BallAwareGaze
class HeldNearView(unittest.TestCase):
 def make(self,stamp=1.1,near=True):
  o=S(external_fine_step_head_owned=False,ball_recovery_context_valid=False,attention_head_permission=None,attention_owner=S(state='NONE'),state='FINAL_ALIGN',final_entry=near,last_geometry={'timestamp_s':stamp,'ball':{'accepted':True},'floor_observation':{'accepted':True}},ball_recovery_active=False)
  g=BallAwareGaze(o);g.current=np.array([-.9,.7]);g.timestamp=1.;return g,o
 def test_fresh_complete_near_view_is_held_not_recapped(self):
  g,o=self.make();self.assertEqual(g.update([-.4,.7],1.1),[-.9,.7]);self.assertFalse(g.last['new_motion_authority']);self.assertEqual(g.last['capture_s'],1.1)
 def test_stale_or_far_context_cannot_claim_held_capture(self):
  for stamp,near in [(1.,True),(1.1,False)]:
   g,o=self.make(stamp,near);self.assertGreater(g.update([-.4,.7],1.1)[0],-.9);self.assertNotIn('capture_s',g.last)
 def test_external_fine_head_owner_still_wins(self):
  g,o=self.make();o.external_fine_step_head_owned=True;g.update([0,0],1.1);self.assertIn('neutral fine-step owns',g.last['source'])
