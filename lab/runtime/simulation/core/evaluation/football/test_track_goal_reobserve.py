import unittest
from types import SimpleNamespace
import numpy as np
from neutral_fine_step import NeutralFineStep
from fine_kick import route_at_step_end
from test_neutral_fine_step import packet, mem


class GoalStop(unittest.TestCase):
    def ready(self):
        f = NeutralFineStep()
        f.state = "TRACK"
        f.motion_kind = "TRACK"
        f.started = 0.5
        f.last_policy = 1.0
        f.capture_s = 1.02
        f.capture_safe = True
        f.head = np.zeros(4)
        f.step_start = 0.8
        f.anchor_s = 1.0
        f.goal_s = 0.50
        f.goal_heading = 0.0
        f.track_velocity = [0.3, 0.1, 0.2]
        return f

    def test_stale_goal_withdraws_motion_and_requires_new_observation(self):
        f = self.ready()
        r = f.policy(t=1.02, packet=packet(1.02), memory=mem(1.0), actual_head=[0] * 4)
        self.assertEqual(f.state, "SETTLE")
        self.assertEqual(r["velocity"], [0, 0, 0])
        self.assertEqual(f.goal_s, 0.50)
        self.assertEqual(f.anchor_s, 1.0)
        self.assertEqual(f.started, 0.5)
        self.assertEqual(
            route_at_step_end(
                now_s=1.02,
                fine=f,
                handoff=SimpleNamespace(latest_capture=None),
                memory={},
                goal_heading=0.0,
                goal_fix_s=1.02,
            ),
            "keep:guard_stop_requires_fresh_reacquisition",
        )
        for k in range(52, 80):
            t = k / 50
            f.capture_s = t
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

    def test_large_heading_is_still_terminal(self):
        f = self.ready()
        f.goal_s = 1.02
        f.goal_heading = 0.46
        r = f.policy(t=1.02, packet=packet(1.02), memory=mem(1.0), actual_head=[0] * 4)
        self.assertEqual(f.state, "FAILED")
        self.assertEqual(r["velocity"], [0, 0, 0])

    def test_total_deadline_cannot_be_reset_by_goal_stop(self):
        f = self.ready()
        f.started = -23.0
        r = f.policy(t=1.02, packet=packet(1.02), memory=mem(1.0), actual_head=[0] * 4)
        self.assertEqual(f.reason, "neutral_step_total_deadline")


if __name__ == "__main__":
    unittest.main()
