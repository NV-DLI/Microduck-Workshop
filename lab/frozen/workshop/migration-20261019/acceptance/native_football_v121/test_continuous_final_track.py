import unittest
import numpy as np
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet,mem
class ContinuousFinalTrack(unittest.TestCase):
 def test_track_heading_equilibrium_independent_of_side_error(self):
  for y in [-.09,.04]:
   f=NeutralFineStep();v=f.track_filtered([.3,y,-.08],0.,.02,packet(1.))
   self.assertEqual(abs(v[1]),.2);self.assertAlmostEqual(v[2],0.)
 def test_track_final_stop_preserves_zero_authority(self):
  f=NeutralFineStep();f.state='TRACK';f.motion_kind='TRACK';f.started=.5;f.last_policy=1.;f.capture_s=1.;f.capture_safe=True;f.anchor_s=.5;f.goal_s=1.;f.goal_heading=0.;f.step_start=.8;f.head=np.zeros(4)
  m=mem(1.02,anchor=.5);m['point_level_m']=[.114,-.035,-.08]
  r=f.policy(t=1.02,packet=packet(1.02),memory=m,actual_head=[0]*4)
  self.assertEqual(f.state,'SETTLE');self.assertEqual(r['velocity'],[0,0,0]);self.assertFalse(r['kick_authorized'])
 def test_track_old_midpoint_no_longer_stops(self):
  f=NeutralFineStep();f.state='TRACK';f.motion_kind='TRACK';f.started=.5;f.last_policy=1.;f.capture_s=1.;f.capture_safe=True;f.anchor_s=.5;f.goal_s=1.;f.goal_heading=0.;f.step_start=.8;f.head=np.zeros(4)
  m=mem(1.02,anchor=.5);m['point_level_m']=[.21,-.035,-.08]
  r=f.policy(t=1.02,packet=packet(1.02),memory=m,actual_head=[0]*4);self.assertEqual(f.state,'TRACK');self.assertEqual(r['velocity'][0],.3)
