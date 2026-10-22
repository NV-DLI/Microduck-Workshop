import unittest
import numpy as np
from neutral_fine_step import NeutralFineStep
from short_primitive import select
from test_neutral_fine_step import mem
class CaptureDomain(unittest.TestCase):
 def test_selected_outer_entry_accepted(self):
  p=[.21,-.075,-.08];h=0.
  self.assertIsNotNone(select(p,h));m=mem(1.);m['point_level_m']=p
  f=NeutralFineStep();f.capture(t=1.,guard_safe=True,complete=True,trigger=True,actual_head=np.zeros(4),memory=m,goal_heading=h,goal_fix_s=1.,motion_kind='MACRO')
  self.assertEqual(f.state,'RETURN')
 def test_selector_y_domain_matches_capture(self):
  self.assertIsNone(select([.2,-.105,-.08],0.))
