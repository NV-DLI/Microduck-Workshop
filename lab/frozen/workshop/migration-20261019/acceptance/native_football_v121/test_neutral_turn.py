import unittest
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet,mem
class NeutralTurn(unittest.TestCase):
 def make(self):
  f=NeutralFineStep();m=mem(1.,p=(.3,-.035,-.08));f.capture(t=1.,guard_safe=True,complete=True,trigger=True,actual_head=[0]*4,memory=m,goal_heading=.12,goal_fix_s=1.,motion_kind='TURN');return f,m
 def test_actual_quiet_before_turn_and_bounded_brake(self):
  f,m=self.make();self.assertEqual(f.state,'RETURN')
  for k in range(51,78):
   t=k/50;f.capture_s=t;f.goal_s=t;r=f.policy(t=t,packet=packet(t),memory=m,actual_head=[0]*4)
  self.assertEqual(f.state,'TURN');self.assertEqual(r['velocity'],[0,0,.6])
  for k in range(78,120):
   t=k/50;f.capture_s=t;f.goal_s=t;r=f.policy(t=t,packet=packet(t),memory=m,actual_head=[0]*4)
   if f.state=='SETTLE':break
  self.assertEqual(f.state,'SETTLE');self.assertLessEqual(f.step_end-f.turn_start,.801);self.assertEqual(r['velocity'],[0,0,0])
 def test_expired_anchor_never_turns(self):
  f,m=self.make();f.state='TURN';f.turn_start=5.5;f.last_policy=5.5;f.capture_s=5.52;f.goal_s=5.52
  r=f.policy(t=5.52,packet=packet(5.52),memory=m,actual_head=[0]*4);self.assertEqual(f.state,'FAILED');self.assertEqual(r['velocity'],[0,0,0])
 def test_stale_goal_revokes_turn(self):
  f,m=self.make();f.state='TURN';f.turn_start=1.;f.last_policy=1.5;f.capture_s=1.52
  r=f.policy(t=1.52,packet=packet(1.52),memory=m,actual_head=[0]*4);self.assertEqual(f.reason,'neutral_turn_fresh_goal_required');self.assertEqual(r['velocity'],[0,0,0])
