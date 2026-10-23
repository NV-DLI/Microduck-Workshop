import unittest
import numpy as np
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet, mem


class TurnGoalView(unittest.TestCase):
    def make(self, h=-0.67, kind="TURN"):
        f = NeutralFineStep()
        m = mem(1.0, p=(0.25, -0.10, -0.08))
        f.capture(
            t=1.0,
            guard_safe=True,
            complete=True,
            trigger=True,
            actual_head=[0] * 4,
            memory=m,
            goal_heading=h,
            goal_fix_s=1.0,
            motion_kind=kind,
        )
        return f, m

    def test_large_turn_requests_bounded_goal_yaw(self):
        f, m = self.make()
        self.assertAlmostEqual(f.return_target[2], -0.67)
        self.assertEqual(list(f.return_target[[0, 1, 3]]), [0, 0, 0])
        self.assertLessEqual(abs(f.return_target[2]), 1.1)

    def test_tracking_and_small_turn_still_return_neutral(self):
        for h, k in [(0.12, "TURN"), (0.3, "TRACK")]:
            f, m = self.make(h, k)
            self.assertTrue(np.array_equal(f.return_target, np.zeros(4)))

    def test_actual_command_must_reach_fixed_target_before_turn(self):
        f, m = self.make()
        f.head = f.return_target.copy()
        for k in range(51, 85):
            t = k / 50
            f.capture_s = t
            f.goal_s = t
            r = f.policy(t=t, packet=packet(t), memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "RETURN")
        self.assertEqual(r["velocity"], [0, 0, 0])
        for k in range(85, 112):
            t = k / 50
            f.capture_s = t
            f.goal_s = t
            r = f.policy(
                t=t, packet=packet(t), memory=m, actual_head=f.return_target.tolist()
            )
        self.assertEqual(f.state, "TURN")
        self.assertEqual(r["velocity"], [0, 0, -0.6])
        self.assertFalse(r["kick_authorized"])
