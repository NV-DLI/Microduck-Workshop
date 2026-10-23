"""Public-input ordering regression; synthetic tests are not physics evidence."""

import math
import unittest
import numpy as np
from final_approach import FinalApproach, rotation
from fine_kick import standoff_entry


class Standoff(unittest.TestCase):
    def tick(self, f, t, p, h, authorized=True):
        return f.update(
            t=t,
            point=p,
            gyro_body=[0.0, 0.0, 0.0],
            gyro_level=0.0,
            goal_heading=h,
            authorized=authorized,
            uncertainty=0.02,
        )

    def ready(self, p, h, phase="POSITION", authorized=True):
        f = FinalApproach()
        f.staging_phase = phase
        for t in [0.0, 0.07, 0.135, 0.2, 0.27, 0.335, 0.4]:
            v, d = self.tick(f, t, p, h, authorized)
        return f, v, d

    def test_aligns_at_standoff_before_forward(self):
        for h in [-0.4, 0.4, 0.65]:
            p = [*(rotation(h) @ [0.26, -0.035]), -0.08]
            f, v, d = self.ready(p, h)
            self.assertEqual(f.staging_phase, "GOAL_ALIGN")
            self.assertEqual(v, [0.0, 0.0, math.copysign(0.4, h)])
            self.assertEqual(f.advances, 0)
            self.assertEqual(d["alignment_mode"], "final_goal_heading")

    def test_turn_keeps_goal_target_when_crossing_entry_threshold(self):
        f, _, _ = self.ready([0.26, -0.035, -0.08], 0.1, "GOAL_ALIGN")
        # A target switch here would steer toward a different bearing mid-turn.
        _, d = self.tick(f, 0.47, [0.26, -0.035, -0.08], 0.02)
        self.assertAlmostEqual(d["bearing_rad"], 0.0)
        self.assertEqual(d["alignment_mode"], "final_goal_heading")

    def test_misaligned_lateral_position_does_not_commit_standoff(self):
        f, _, _ = self.ready([0.26, 0.03, -0.08], 0.0)
        self.assertEqual(f.staging_phase, "POSITION")

    def test_close_alignment_yields_zero_body_fresh_entry(self):
        f, v, d = self.ready([0.20, -0.035, -0.08], 0.0, "GOAL_ALIGN")
        self.assertEqual(v, [0.0, 0.0, 0.0])
        self.assertEqual(d["phase"], "OBSERVE")
        ok, why = standoff_entry(
            t=0.4,
            handoff_state="APPROACH",
            fine_active=False,
            fine_state="IDLE",
            final_diag=d,
            velocity_body_xyw=v,
            complete=True,
            guard_safe=True,
            memory={
                "memory_available": True,
                "ball_capture_timestamp_s": 0.4,
                "point_level_m": [0.20, -0.035, -0.08],
            },
            goal_heading=0.0,
            goal_fix_s=0.3,
            actual_head=[-0.5, 0.5, 0.0, 0.0],
        )
        self.assertTrue(ok, why)

    def test_dead_zone_holds_without_fallback_forward(self):
        f, v, d = self.ready([0.16, -0.035, -0.08], 0.0, "GOAL_ALIGN")
        self.assertEqual(v, [0.0, 0.0, 0.0])
        self.assertEqual(f.advances, 0)

    def test_forward_stops_on_goal_heading_drift(self):
        f, v, _ = self.ready([0.26, -0.035, -0.08], 0.0, "GOAL_ALIGN")
        self.assertGreater(v[0], 0.0)
        v, d = self.tick(f, 0.47, [0.25, -0.035, -0.08], 0.19)
        self.assertEqual(v, [0.0, 0.0, 0.0])
        self.assertEqual(d["reason"], "standoff_heading_drift_stop")

    def test_guard_blocks_staging_state_change(self):
        f, v, _ = self.ready([0.26, -0.035, -0.08], 0.3, authorized=False)
        self.assertEqual(f.staging_phase, "POSITION")
        self.assertEqual(v, [0.0, 0.0, 0.0])
        self.assertEqual(f.turns, 0)

    def test_clock_gap_invalidates_motion(self):
        f, _, _ = self.ready([0.26, -0.035, -0.08], 0.3, "GOAL_ALIGN")
        v, d = self.tick(f, 0.8, [0.26, -0.035, -0.08], 0.3)
        self.assertEqual(v, [0.0, 0.0, 0.0])
        self.assertEqual(d["reason"], "final_capture_clock_gap")

    def test_original_kick_box_is_not_widened(self):
        _, v, d = self.ready([0.09, -0.035, -0.08], 0.0)
        self.assertEqual(v, [0.0, 0.0, 0.0])
        self.assertTrue(d["final_alignment_parameters"])
        _, v, d = self.ready([0.11, -0.035, -0.08], 0.0, "GOAL_ALIGN")
        self.assertFalse(d["final_alignment_parameters"])

    def test_nonfinite_heading_holds(self):
        _, v, d = self.ready([0.26, -0.035, -0.08], float("nan"))
        self.assertEqual(v, [0.0, 0.0, 0.0])
        self.assertFalse(d["final_alignment_parameters"])


if __name__ == "__main__":
    unittest.main()
