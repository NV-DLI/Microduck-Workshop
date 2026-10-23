import unittest, math, sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent
M = D.parents[1]
sys.path[:0] = [
    str(D),
    str(M / "evaluation"),
    str(M / "football"),
    str(M / "sensors"),
    str(M / "gait"),
]
from final_approach import FinalApproach
from head_handoff import HeadHandoff


class Leg:
    def estimate(self, *a):
        return {
            "velocity_body_m_s": [0.0, 0.0, 0.0],
            "diagnostic_low_confidence": False,
        }


def packet(t):
    return {
        "source": "body_powerboard_control_imu",
        "axes": "trunk FLU",
        "seq": int(round(t / 0.02)),
        "timestamp_s": t,
        "read_timestamp_s": t,
        "delivered_timestamp_s": t,
        "encoder_read_timestamp_s": t,
        "orientation_valid": True,
        "encoder_q_rad": [0.0] * 14,
        "encoder_qd_rad_s": [0.0] * 14,
        "gyro": [0.0, 0.0, 0.0],
        "gravity": [0.0, 0.0, -1.0],
        "gyro_body_rad_s": [0.0, 0.0, 0.0],
        "gravity_body": [0.0, 0.0, -1.0],
        "gyro_clipped": [False] * 3,
        "accel_clipped": [False] * 3,
    }


def geometry(p=(0.09, -0.035, 0.0)):
    return {
        "ball": {
            "accepted": True,
            "geometry": {
                "point_reference_m": list(p),
                "estimated_range_uncertainty_m": 0.02,
                "observed_boundary_required": True,
                "accepted": True,
            },
        }
    }


class TestPhases(unittest.TestCase):
    def ready(self, p=(0.09, -0.035, 0.0)):
        h = HeadHandoff(Leg(), [str(i) for i in range(14)])
        h.policy(now_s=1.0, packet=packet(1.0))
        h.capture(
            now_s=1.0,
            geometry=geometry(p),
            head_offsets=[-0.6, 0.6, 0.0, -0.3],
            parameters=True,
            goal_heading=0.0,
            goal_fix_s=1.0,
        )
        return h

    def test_actual_slew_and_complete_quiet(self):
        h = self.ready()
        prev = np.array(h.head)
        maximum = 0.0
        for i in range(1, 130):
            t = 1 + i * 0.02
            if i % 3 == 0:
                h.capture(
                    now_s=t - 0.001,
                    geometry=None,
                    head_offsets=None,
                    parameters=False,
                    goal_heading=0.0,
                    goal_fix_s=t - 0.001,
                )
            s = h.policy(now_s=t, packet=packet(t))
            maximum = max(maximum, float(abs(np.array(h.head) - prev).max() / 0.02))
            prev = h.head.copy()
        self.assertLessEqual(maximum, 0.3000000001)
        self.assertEqual(h.state, "KICK")
        self.assertGreaterEqual(h.kick_start, 3.48)
        self.assertFalse(s["uncertainty_envelope_inside"])
        self.assertEqual(s["memory"]["original_range_uncertainty_m"], 0.02)

    def test_bad_clock_does_not_poison_approach(self):
        f = FinalApproach()
        before = f.__dict__.copy()
        with self.assertRaises(ValueError):
            f.update(
                t=math.nan,
                point=[0.2, 0.0, 0.0],
                gyro_body=[0.0, 0.0, 0.0],
                gyro_level=0.0,
                goal_heading=0.0,
                authorized=True,
                uncertainty=0.02,
            )
        self.assertEqual(before, f.__dict__)

    def test_invalid_nonempty_packet_revokes_active_kick(self):
        h = self.ready()
        h.state = "KICK"
        h.kick_start = 1.0
        bad = packet(1.02)
        bad["orientation_valid"] = False
        s = h.policy(now_s=1.02, packet=bad)
        self.assertEqual(s["state"], "FAILED")
        self.assertFalse(s["kick_runtime_authorized"])

    def test_no_fresh_goal_no_kick(self):
        h = self.ready()
        for i in range(1, 130):
            h.policy(now_s=1 + i * 0.02, packet=packet(1 + i * 0.02))
        self.assertEqual(h.state, "FAILED")
        self.assertEqual(h.reason, "fresh_goal_fix_required_after_headreturn")

    def test_gap_cannot_authorize(self):
        h = self.ready()
        s = h.policy(now_s=1.1, packet=None)
        self.assertEqual(s["state"], "FAILED")
        self.assertFalse(s["kick_runtime_authorized"])

    def test_outside_box_does_not_start(self):
        h = self.ready((0.11, -0.035, 0.0))
        self.assertEqual(h.state, "APPROACH")

    def test_motion_needs_real_quiet(self):
        f = FinalApproach()
        inp = dict(
            point=[0.25, -0.035, 0.0],
            gyro_body=[0.0, 0.0, 0.0],
            gyro_level=0.0,
            goal_heading=0.0,
            authorized=True,
            uncertainty=0.02,
        )
        for i in range(6):
            u, d = f.update(t=i / 15, **inp)
            self.assertEqual(u, [0.0, 0.0, 0.0])
        u, d = f.update(t=0.4, **inp)
        self.assertEqual(u, [0.3, 0.0, 0.0])
        self.assertEqual(d["duration_s"], 0.4)
        u, d = f.update(t=0.47, **{**inp, "authorized": False})
        self.assertEqual(u, [0.0, 0.0, 0.0])
        self.assertEqual(f.advances, 1)

    def test_fresh_center_stays_still(self):
        f = FinalApproach()
        for i in range(8):
            u, d = f.update(
                t=i / 15,
                point=[0.09, -0.035, 0.0],
                gyro_body=[0.0, 0.0, 0.0],
                gyro_level=0.0,
                goal_heading=0.0,
                authorized=True,
                uncertainty=0.02,
            )
        self.assertTrue(d["final_alignment_parameters"])
        self.assertEqual(u, [0.0, 0.0, 0.0])

    def test_unknown_target_holds(self):
        f = FinalApproach()
        u, d = f.update(
            t=0.0,
            point=[math.nan, 0.0, 0.0],
            gyro_body=[0.0, 0.0, 0.0],
            gyro_level=0.0,
            goal_heading=0.0,
            authorized=True,
            uncertainty=0.02,
        )
        self.assertEqual(u, [0.0, 0.0, 0.0])
        self.assertFalse(d["final_alignment_parameters"])

    def test_finite_motion_budget(self):
        f = FinalApproach()
        f.advances = 20
        for i in range(8):
            u, d = f.update(
                t=i / 15,
                point=[0.25, -0.035, 0.0],
                gyro_body=[0.0, 0.0, 0.0],
                gyro_level=0.0,
                goal_heading=0.0,
                authorized=True,
                uncertainty=0.02,
            )
        self.assertEqual(u, [0.0, 0.0, 0.0])
        self.assertEqual(f.failed, "final_forward_budget_exhausted")


if __name__ == "__main__":
    unittest.main()
