import unittest
import torch
from bounded_actions import BoundedActions

class BoundedActionTests(unittest.TestCase):
 def setup_arrays(self):
  q=torch.zeros(2,14);p=q+.01;limits=torch.tensor([[[-.5,.5]]*14]*2);return q,p,limits
 def test_actual_output_target_and_bias_are_not_double_compensated(self):
  q,p,l=self.setup_arrays();a=BoundedActions([[-.5,.5]]*14,max_target_rate_rad_s=100)
  raw=torch.ones_like(q)*10;original=q.clone();bias=torch.ones_like(q)*.02
  out,d=a.apply(raw,scale=.9,offset=.1,encoder_bias=bias,q_servo=q,q_passive=p,servo_limits=l)
  target=out*.9+.1-bias
  self.assertTrue(torch.allclose(target,d['executed_output_target_rad']));self.assertTrue((target<=.4650001).all());self.assertTrue(torch.equal(q,original));self.assertTrue(d['boundary_limited'].all())
 def test_slew_changes_targets_not_state(self):
  q,p,l=self.setup_arrays();a=BoundedActions([[-.5,.5]]*14,max_target_rate_rad_s=3)
  out,d=a.apply(q+.3,scale=1,offset=0,encoder_bias=q,q_servo=q,q_passive=p,servo_limits=l)
  self.assertTrue(torch.allclose(out,p+.06));self.assertTrue(torch.equal(q,torch.zeros_like(q)))
 def test_effective_limits_not_widened_by_backlash(self):
  q,p,l=self.setup_arrays();p[:]=-.1;a=BoundedActions([[-.5,.5]]*14,max_target_rate_rad_s=100)
  out,d=a.apply(q+1,scale=1,offset=0,encoder_bias=q,q_servo=q,q_passive=p,servo_limits=l)
  self.assertTrue((out-p<=.4650001).all());self.assertTrue((out<=.4650001).all())
 def test_nonfinite_action_rejected(self):
  q,p,l=self.setup_arrays();raw=q.clone();raw[0,0]=float('nan')
  with self.assertRaises(ValueError):BoundedActions([[-.5,.5]]*14).apply(raw,scale=1,offset=0,encoder_bias=q,q_servo=q,q_passive=p,servo_limits=l)

if __name__=='__main__':unittest.main(verbosity=2)
