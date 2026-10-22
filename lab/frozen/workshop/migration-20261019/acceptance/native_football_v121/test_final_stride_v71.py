import unittest
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import mem,packet
class FinalStride(unittest.TestCase):
 def enter(self,p=(.16,-.035,-.08),head=(-.7,.7,0,0)):
  f=NeutralFineStep();m=mem(1.,p=p);f.capture(t=1.,guard_safe=True,complete=True,trigger=True,actual_head=head,memory=m,goal_heading=.05,goal_fix_s=1.);return f,m
 def test_short_budget_final_stride_only(self):
  f,m=self.enter();self.assertTrue(f.final_stride);self.assertEqual(f.maximum_step_s,1.2);self.assertEqual(f.status(1,m)['maximum_memory_age_s'],4.5)
  g,_=self.enter(p=(.20,-.035,-.08));self.assertFalse(g.final_stride);self.assertEqual(g.maximum_step_s,.6)
 def test_faster_return_is_bounded(self):
  f,m=self.enter();f.capture_s=1.02;f.policy(t=1.02,packet=packet(1.02),memory=m,actual_head=[-.7,.7,0,0]);self.assertAlmostEqual(f.head[0],-.69)
 def test_final_stride_stops_on_public_range(self):
  f,m=self.enter();f.state='STEP';f.step_start=1.;f.step_goal_heading=.05;f.goal_heading=.05;f.capture_s=1.02;m['point_level_m']=[.111,-.035,-.08]
  r=f.policy(t=1.02,packet=packet(1.02),memory=m,actual_head=[0]*4);self.assertEqual(r['state'],'SETTLE');self.assertEqual(r['velocity'],[0,0,0])
