import unittest
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet, mem


class ContinuousTrack(unittest.TestCase):
    def make(self):
        f = NeutralFineStep()
        m = mem(1.0, p=(0.3, -0.035, -0.08))
        f.capture(
            t=1.0,
            guard_safe=True,
            complete=True,
            trigger=True,
            actual_head=[0] * 4,
            memory=m,
            goal_heading=0.0,
            goal_fix_s=1.0,
            motion_kind="TRACK",
        )
        return f, m

    def test_neutral_quiet_then_public_position_stop(self):
        f, m = self.make()
        for k in range(51, 78):
            t = k / 50
            f.capture_s = t
            f.goal_s = t
            r = f.policy(t=t, packet=packet(t), memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "TRACK")
        self.assertEqual(r["velocity"], [0.3, 0.0, 0.0])
        m["point_level_m"] = [0.111, -0.035, -0.08]
        f.capture_s = 1.56
        f.goal_s = 1.56
        r = f.policy(t=1.56, packet=packet(1.56), memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "SETTLE")
        self.assertEqual(r["velocity"], [0, 0, 0])

    def test_public_shot_frame_control_signs(self):
        a = NeutralFineStep.track_command([0.3, -0.02, -0.08], 0.0)
        self.assertGreater(a[1], 0.0)
        self.assertEqual(a[2], 0.0)
        a = NeutralFineStep.track_command([0.3, -0.035, -0.08], 0.2)
        self.assertEqual(a[2], 0.2)
        self.assertTrue(0.25 <= a[0] <= 0.3)
        self.assertTrue(abs(a[1]) <= 0.2)

    def test_expired_memory_cannot_authorize_tracking(self):
        f, m = self.make()
        f.state = "TRACK"
        f.step_start = 5.5
        f.last_policy = 5.5
        f.capture_s = 5.52
        f.goal_s = 5.52
        r = f.policy(t=5.52, packet=packet(5.52), memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "FAILED")
        self.assertEqual(r["velocity"], [0, 0, 0])
