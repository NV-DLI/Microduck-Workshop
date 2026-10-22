import unittest,math
from final_approach import FinalApproach
class CloseForward(unittest.TestCase):
 def step(self,f,t,p=(.2,-.035,0),**kw):
  a=dict(t=t,point=p,gyro_body=[0,0,0],gyro_level=0.,goal_heading=0.,authorized=True,uncertainty=.02);a.update(kw);return f.update(**a)
 def ready(self,p):
  f=FinalApproach()
  for t in [0,.07,.135,.2,.27,.335,.4]:u,d=self.step(f,t,p)
  return f,u,d
 def test_close_now_holds_for_zero_body_fine_handoff(self):
  f,u,d=self.ready((.2,-.035,0));self.assertEqual(u,[0,0,0]);self.assertEqual(d['reason'],'standoff_aligned_observe');self.assertEqual(f.advances,0)
 def test_threshold_equality_and_far(self):
  p=(math.sqrt(.22**2-.035**2),-.035,0);_,u,_=self.ready(p);self.assertEqual(u,[0,0,0])
  _,u,d=self.ready((p[0]+1e-6,-.035,0));self.assertEqual(u[0],.3);self.assertLessEqual(d['duration_s'],.4)
 def test_far_crossing_stops_and_does_not_start_fine_without_quiet(self):
  f,u,_=self.ready((.23,-.035,0));self.assertEqual(u[0],.3)
  u,d=self.step(f,.465);self.assertEqual(u,[0,0,0]);self.assertEqual(d['reason'],'final_close_range_transition_stop');self.assertEqual(f.advances,1)
  u,_=self.step(f,.535);self.assertEqual(u,[0,0,0]);self.assertEqual(f.advances,1)
 def test_close_hold_does_not_resume_until_quiet(self):
  f,_,_=self.ready((.2,-.035,0));u,_=self.step(f,.465,(.23,-.035,0));self.assertEqual(u,[0,0,0]);self.assertEqual(f.advances,0)
 def test_guard_and_heading_revoke_immediately(self):
  for kwargs in [dict(authorized=False),dict(goal_heading=.2)]:
   f,_,_=self.ready((.26,-.035,0));u,_=self.step(f,.465,**kwargs);self.assertEqual(u,[0,0,0]);self.assertEqual(f.advances,1)
 def test_budget_exhaustion_not_refunded(self):
  f=FinalApproach();f.advances=20
  for t in [0,.07,.135,.2,.27,.335,.4]:u,d=self.step(f,t,(.26,-.035,0))
  self.assertEqual(u,[0,0,0]);self.assertEqual(f.failed,'final_forward_budget_exhausted');self.assertEqual(f.advances,20)
if __name__=='__main__':unittest.main()
