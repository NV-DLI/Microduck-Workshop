import unittest
import numpy as np
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet, mem


class Brake(unittest.TestCase):
    def setup_track(self):
        f = NeutralFineStep()
        f.state = "TRACK"
        f.motion_kind = "TRACK"
        f.started = 0.5
        f.last_policy = 1.0
        f.capture_s = 1.0
        f.capture_safe = True
        f.anchor_s = 1.0
        f.goal_s = 1.0
        f.goal_heading = 0.0
        f.step_start = 0.8
        f.head = np.zeros(4)
        return f, mem(0.5, p=(0.124, -0.035, -0.08))

    def test_continuous_brake_then_irrevocable_zero(self):
        f, m = self.setup_track()
        speeds = []
        for k in range(51, 61):
            t = k / 50
            f.capture_s = t
            f.goal_s = t
            r = f.policy(t=t, packet=packet(t), memory=m, actual_head=[0] * 4)
            speeds.append(r["velocity"][0])
        self.assertEqual(f.state, "SETTLE")
        self.assertEqual(speeds[0], 0.25)
        self.assertEqual(speeds[-1], 0.0)
        self.assertTrue(all(b <= a for a, b in zip(speeds, speeds[1:])))
        self.assertEqual(f.step_end, 1.18)

    def test_stale_capture_revokes_while_braking(self):
        f, m = self.setup_track()
        f.policy(t=1.02, packet=packet(1.02), memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "TRACK")
        self.assertIsNotNone(f.track_brake_start)
        f.capture_s = 0.8
        r = f.policy(t=1.04, packet=packet(1.04), memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "FAILED")
        self.assertEqual(r["velocity"], [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
