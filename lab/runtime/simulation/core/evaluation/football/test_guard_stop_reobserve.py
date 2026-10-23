import unittest
from types import SimpleNamespace
import numpy as np
from neutral_fine_step import NeutralFineStep
from fine_kick import route_at_step_end
from test_neutral_fine_step import packet, mem


class GuardStop(unittest.TestCase):
    def test_no_old_motion_after_guard_recovers(self):
        f = NeutralFineStep()
        f.state = "TRACK"
        f.motion_kind = "TRACK"
        f.started = 0.5
        f.last_policy = 1.0
        f.capture_s = 1.02
        f.capture_safe = False
        f.head = np.zeros(4)
        f.step_start = 0.8
        f.anchor_s = 1.0
        f.goal_s = 1.0
        f.goal_heading = 0.0
        f.track_velocity = [0.3, 0.1, 0.2]
        r = f.policy(t=1.02, packet=packet(1.02), memory=mem(1.0), actual_head=[0] * 4)
        self.assertEqual(r["velocity"], [0, 0, 0])
        self.assertEqual(f.state, "SETTLE")
        self.assertEqual(f.anchor_s, 1.0)
        self.assertEqual(f.started, 0.5)
        for k in range(52, 80):
            t = k / 50
            f.capture_s = t
            f.capture_safe = True
            r = f.policy(t=t, packet=packet(t), memory=mem(1.0), actual_head=[0] * 4)
            self.assertEqual(r["velocity"], [0, 0, 0])
            self.assertFalse(r["kick_authorized"])
        self.assertEqual(f.state, "REACQUIRE")
        f.capture(
            t=1.60,
            guard_safe=True,
            complete=False,
            trigger=False,
            actual_head=[0] * 4,
            memory=mem(1.0),
            goal_heading=0.0,
            goal_fix_s=1.60,
        )
        self.assertEqual(f.state, "REACQUIRE")
        f.capture(
            t=1.67,
            guard_safe=True,
            complete=True,
            trigger=False,
            actual_head=[0] * 4,
            memory=mem(1.67),
            goal_heading=0.0,
            goal_fix_s=1.67,
        )
        self.assertEqual(f.state, "IDLE")
        self.assertEqual(f.attempts, 0)
        f.capture(
            t=1.74,
            guard_safe=True,
            complete=True,
            trigger=True,
            actual_head=[0] * 4,
            memory=mem(1.74, 1.74, p=(0.2, -0.035, -0.08)),
            goal_heading=0.0,
            goal_fix_s=1.74,
            motion_kind="TRACK",
        )
        self.assertEqual(f.state, "RETURN")
        self.assertIsNone(f.reason)
        self.assertEqual(f.anchor_s, 1.74)

    def test_guard_stop_cannot_route_to_kick_even_if_later_capture_safe(self):
        f = NeutralFineStep()
        f.state = "SETTLE"
        f.reason = "public_guard_stop_requires_fresh_reacquisition"
        self.assertEqual(
            route_at_step_end(
                now_s=1.0,
                fine=f,
                handoff=SimpleNamespace(latest_capture=None),
                memory={},
                goal_heading=0.0,
                goal_fix_s=1.0,
            ),
            "keep:guard_stop_requires_fresh_reacquisition",
        )


if __name__ == "__main__":
    unittest.main()
