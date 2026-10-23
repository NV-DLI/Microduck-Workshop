import unittest
import numpy as np
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet, mem


class SettleGuardHold(unittest.TestCase):
    def ready(self, state="SETTLE"):
        f = NeutralFineStep()
        f.state = state
        f.started = 0.5
        f.last_policy = 1.0
        f.capture_s = 1.02
        f.capture_safe = False
        f.head = np.zeros(4)
        f.quiet_start = 0.6
        return f

    def test_missing_geometry_cannot_move_or_preserve_quiet_credit(self):
        f = self.ready()
        r = f.policy(t=1.02, packet=packet(1.02), memory=mem(1.0), actual_head=[0] * 4)
        self.assertEqual(f.state, "SETTLE")
        self.assertEqual(r["velocity"], [0, 0, 0])
        self.assertIsNone(f.quiet_start)
        self.assertFalse(r["kick_authorized"])
        f.capture_s = 1.04
        f.capture_safe = True
        f.policy(t=1.04, packet=packet(1.04), memory=mem(1.0), actual_head=[0] * 4)
        self.assertEqual(f.quiet_start, 1.04)
        self.assertEqual(f.state, "SETTLE")

    def test_active_movement_still_fails_on_same_guard(self):
        f = self.ready("TURN")
        r = f.policy(t=1.02, packet=packet(1.02), memory=mem(1.0), actual_head=[0] * 4)
        self.assertEqual(f.state, "SETTLE")
        self.assertEqual(r["velocity"], [0, 0, 0])
        self.assertEqual(f.reason, "public_guard_stop_requires_fresh_reacquisition")

    def test_missing_capture_clock_still_fails(self):
        f = self.ready()
        f.capture_s = 0.5
        r = f.policy(t=1.02, packet=packet(1.02), memory=mem(1.0), actual_head=[0] * 4)
        self.assertEqual(f.state, "FAILED")
