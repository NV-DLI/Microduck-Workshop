import unittest
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet,mem
class TrackAlignment(unittest.TestCase):
 def start(self,h):
  f=NeutralFineStep();m=mem(1.,p=(.3,-.04,-.08));f.capture(t=1.,guard_safe=True,complete=True,trigger=True,actual_head=[0]*4,memory=m,goal_heading=h,goal_fix_s=1.,motion_kind='TRACK')
  for k in range(51,78):
   t=k/50;f.capture_s=t;f.goal_s=t;r=f.policy(t=t,packet=packet(t),memory=m,actual_head=[0]*4)
  return f,m,r
 def test_unaligned_track_becomes_pure_goal_turn_without_forward_motion(self):
  for h in [-.15,-.081,.081,.15]:
   f,m,r=self.start(h);self.assertEqual(f.state,'TURN');self.assertTrue(f.turn_goal_only);self.assertEqual(r['velocity'][:2],[0.,0.]);self.assertEqual(r['velocity'][2],.8*(1 if h>0 else -1))
 def test_aligned_track_keeps_existing_route(self):
  f,m,r=self.start(.07);self.assertEqual(f.state,'TRACK');self.assertFalse(f.turn_goal_only);self.assertEqual(r['velocity'][0],.3)
 def test_turn_stops_and_requires_reacquisition(self):
  f,m,r=self.start(.15)
  for k in range(78,84):
   t=k/50;f.capture_s=t;f.goal_s=t;f.goal_heading=0.;r=f.policy(t=t,packet=packet(t),memory=m,actual_head=[0]*4)
  self.assertEqual(f.state,'SETTLE');self.assertEqual(r['velocity'],[0.,0.,0.]);self.assertIsNone(f.step_start)
if __name__=='__main__':unittest.main()
