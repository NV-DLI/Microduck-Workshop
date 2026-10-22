import unittest,math
from neutral_fine_step import NeutralFineStep
from final_approach import FinalApproach
from test_neutral_fine_step import packet,mem
class Direction(unittest.TestCase):
 def start(self,p=(.20876635349096115,-.035,-.08160351074880644),heading=-.15585650312274382):
  f=NeutralFineStep();m=mem(1.,p=p);f.capture(t=1.,guard_safe=True,complete=True,trigger=True,actual_head=[0]*4,memory=m,goal_heading=heading,goal_fix_s=1.);return f,m
 def test_body_forward_foot_line_legal(self):
  f,m=self.start();self.assertEqual(f.state,'RETURN');self.assertLess(abs(f.waypoint_bearing(m['point_level_m'],f.goal_heading)),.15)
  for k in range(51,80):
   t=k/50;f.capture_s=t;f.goal_s=t;d=f.policy(t=t,packet=packet(t),memory=m,actual_head=[0]*4)
  self.assertEqual(f.state,'STEP');self.assertEqual(f.step_goal_heading,-.15585650312274382);self.assertEqual(d['velocity'],[.2,0,0]);self.assertEqual(d['step_heading_drift_rad'],0.)
 def test_waypoint_outside_original15_rejected(self):
  f,_=self.start(p=(.2,.02,0),heading=0.);self.assertEqual(f.state,'FAILED');self.assertEqual(f.reason,'neutral_step_initial_public_domain_rejected')
 def test_return_must_recheck_actual_current_waypoint(self):
  f,m=self.start();m['point_level_m']=[.2,.02,0]
  for k in range(51,80):
   t=k/50;f.capture_s=t;f.goal_s=t;f.policy(t=t,packet=packet(t),memory=m,actual_head=[0]*4)
  self.assertEqual(f.state,'FAILED');self.assertEqual(f.reason,'neutral_step_waypoint_direction_required')
 def test_step_relative_heading_and_stale_revoke(self):
  for changed in ('drift','stale'):
   f,m=self.start();f.state='STEP';f.step_start=1.;f.step_goal_heading=-.3;f.goal_heading=-.3;f.capture_s=1.02
   if changed=='drift':f.goal_heading=-.3+.150001
   else:f.goal_s=-.1
   d=f.policy(t=1.02,packet=packet(1.02),memory=m,actual_head=[0]*4)
   self.assertEqual(f.state,'FAILED');self.assertEqual(d['velocity'],[0,0,0])
 def test_relative_heading_wrap(self):
  f,m=self.start();f.state='STEP';f.step_start=1.;f.step_goal_heading=math.pi-.01;f.goal_heading=-math.pi+.01;f.capture_s=1.02
  d=f.policy(t=1.02,packet=packet(1.02),memory=m,actual_head=[0]*4)
  self.assertEqual(f.state,'STEP');self.assertAlmostEqual(d['step_heading_drift_rad'],.02)
 def test_near_kick_box_still_aims_final_goal_not_forward(self):
  f=FinalApproach()
  for t in [0,.07,.135,.2,.27,.335,.4]:
   out,d=f.update(t=t,point=[.09,-.035,0],gyro_body=[0,0,0],gyro_level=0,goal_heading=.2,authorized=True,uncertainty=.02)
  self.assertEqual(out,[0,0,.4]);self.assertEqual(d['alignment_mode'],'final_goal_heading');self.assertEqual(f.advances,0)
if __name__=='__main__':unittest.main()
