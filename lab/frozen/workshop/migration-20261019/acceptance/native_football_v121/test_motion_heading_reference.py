import unittest
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet
class MotionHeading(unittest.TestCase):
 def test_camera_jitter_cannot_change_motion_reference_after_start(self):
  f=NeutralFineStep();f.goal_s=.9
  self.assertEqual(f.motion_heading(.04,packet(1.)),.04)
  p=packet(1.02);p['gyro_body_rad_s']=[0,0,.2]
  self.assertAlmostEqual(f.motion_heading(-.1,p),.036)
  self.assertEqual(f.motion_heading_anchor['public_goal_fix_s'],.9)
 def test_duplicate_read_does_not_integrate_twice(self):
  f=NeutralFineStep();f.motion_heading(.04,packet(1.));p=packet(1.02);p['gyro_body_rad_s']=[0,0,.2]
  a=f.motion_heading(.1,p);self.assertEqual(f.motion_heading(.2,p),a)
 def test_retimestamp_and_old_sequence_rejected(self):
  for p in [dict(packet(1.02),seq=50),packet(.98)]:
   f=NeutralFineStep();f.motion_heading(.04,packet(1.))
   with self.assertRaises(ValueError):f.motion_heading(.04,p)
 def test_world_horizontal_rate_from_public_gravity(self):
  f=NeutralFineStep();f.motion_heading(.1,packet(1.));p=packet(1.02);p['gravity_body']=[-.6,0,-.8];p['gyro_body_rad_s']=[.3,0,.4]
  self.assertAlmostEqual(f.motion_heading(.1,p),.09)
