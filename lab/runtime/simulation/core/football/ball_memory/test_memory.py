import math
import unittest
from dataclasses import replace
import numpy as np
from memory import BallMemory, PublicRead, level_rotation

NAMES = tuple(str(i) for i in range(14))


class KnownKinematics:
    """Analytic fixture only; physical readback tests use actual current CAD."""

    def estimate(self, q, qd, gyro, gravity, names):
        return {"velocity_body_m_s": qd[:3], "diagnostic_low_confidence": q[0] > 0.5}


def packet(t, velocity=(0.0, 0.0, 0.0), omega=0.0):
    return PublicRead(
        t,
        t,
        True,
        (0.0,) * 14,
        (*velocity, *(0.0,) * 11),
        (0.0, 0.0, omega),
        (0.0, 0.0, -1.0),
        NAMES,
    )


def seed(m, t=0.0, point=(0.2, -0.035, -0.09)):
    return m.observe_complete_ball(
        timestamp_s=t,
        point_body_m=point,
        estimated_range_uncertainty_m=0.01,
        complete=True,
        geometry_accepted=True,
    )


class ContractTests(unittest.TestCase):
    def make(self):
        return BallMemory(KnownKinematics(), NAMES)

    def test_translation_at_both_public_rates(self):
        for hz in (50, 15):
            m = self.make()
            m.update(packet(0, (0.02, 0.0, 0.0)))
            seed(m)
            for t in np.arange(1, 3 * hz + 1) / hz:
                out = m.update(packet(t, (0.02, 0.0, 0.0)))
            np.testing.assert_allclose(
                out["point_level_m"], [0.14, -0.035, -0.09], atol=1e-12
            )
            self.assertFalse(out["kick_authorized"])

    def test_turning_transforms_stationary_ball_opposite_way(self):
        m = self.make()
        m.update(packet(0, omega=0.2))
        seed(m, point=(0.2, 0.0, 0.0))
        for t in np.arange(1, 51) / 50:
            out = m.update(packet(t, omega=0.2))
        np.testing.assert_allclose(
            out["point_body_m"],
            [0.2 * math.cos(0.2), -0.2 * math.sin(0.2), 0.0],
            atol=1e-12,
        )

    def test_expiry_never_extends_from_sensor_updates(self):
        m = self.make()
        m.update(packet(0))
        seed(m)
        for t in np.arange(1, 231) / 50:
            out = m.update(packet(t))
        self.assertFalse(out["memory_available"])
        self.assertEqual(out["reason"], "ball_memory_expired")

    def test_gap_requires_new_image(self):
        m = self.make()
        m.update(packet(0))
        seed(m)
        out = m.update(packet(0.2))
        self.assertFalse(out["memory_available"])
        self.assertFalse(m.update(packet(0.22))["memory_available"])
        self.assertTrue(seed(m, 0.22)["memory_available"])

    def test_invalid_read_revokes_old_memory(self):
        for bad in (
            replace(packet(0.02), orientation_valid=False),
            replace(packet(0.02), encoder_timestamp_s=0.01),
            replace(packet(0.02), q_rad=(float("nan"),) * 14),
            replace(packet(0.02), joint_names=NAMES[::-1]),
        ):
            m = self.make()
            m.update(packet(0))
            seed(m)
            with self.assertRaises(ValueError):
                m.update(bad)
            self.assertFalse(m.status(0.02)["memory_available"])

    def test_duplicate_and_reverse_time_revoke_memory(self):
        for t in (0.0, -0.001):
            m = self.make()
            m.update(packet(0))
            seed(m)
            with self.assertRaises(ValueError):
                m.update(packet(t))
            self.assertFalse(m.status(0.0)["memory_available"])

    def test_crop_or_stale_image_cannot_refresh_anchor(self):
        m = self.make()
        m.update(packet(0))
        seed(m)
        for change in (
            {"complete": False},
            {"geometry_accepted": False},
            {"timestamp_s": 0.01},
        ):
            args = dict(
                timestamp_s=0.0,
                point_body_m=(0.2, 0.0, 0.0),
                estimated_range_uncertainty_m=0.01,
                complete=True,
                geometry_accepted=True,
            )
            args.update(change)
            with self.assertRaises(ValueError):
                m.observe_complete_ball(**args)
        self.assertEqual(m.capture_time, 0.0)

    def test_reobservation_reports_innovation(self):
        m = self.make()
        m.update(packet(0))
        seed(m)
        out = seed(m, point=(0.21, -0.025, -0.09))
        np.testing.assert_allclose(
            out["reobservation_innovation_level_m"], [0.01, 0.01, 0.0], atol=1e-12
        )

    def test_declared_image_read_offset_uses_causal_hold(self):
        m = self.make()
        m.update(packet(0, (0.02, 0.0, 0.0)))
        out = m.observe_complete_ball(
            timestamp_s=0.015,
            point_body_m=(0.2, 0.0, 0.0),
            estimated_range_uncertainty_m=0.01,
            complete=True,
            geometry_accepted=True,
            capture_read_age_s=0.015,
        )
        np.testing.assert_allclose(out["point_body_m"], [0.2, 0.0, 0.0], atol=1e-12)
        out = m.update(packet(0.02, (0.02, 0.0, 0.0)))
        np.testing.assert_allclose(out["point_body_m"], [0.1999, 0.0, 0.0], atol=1e-12)
        self.assertEqual(out["capture_sensor_read_age_s"], 0.015)
        self.assertFalse(out["public_motion_hold_is_measurement"])
        with self.assertRaises(ValueError):
            m.observe_complete_ball(
                timestamp_s=0.05,
                point_body_m=(0.2, 0.0, 0.0),
                estimated_range_uncertainty_m=0.01,
                complete=True,
                geometry_accepted=True,
                capture_read_age_s=0.03,
            )

    def test_no_truth_keyword_or_hidden_mapping(self):
        m = self.make()
        with self.assertRaises(ValueError):
            m.update({"root_pose": (0, 0, 0)})
        with self.assertRaises(TypeError):
            PublicRead(**{**packet(0).__dict__, "root_pose": (0, 0, 0)})

    def test_body_level_frame_roundtrip(self):
        g = (0.1, 0.2, -math.sqrt(0.95))
        m = self.make()
        m.update(replace(packet(0), gravity_body=g))
        out = seed(m)
        np.testing.assert_allclose(
            out["point_body_m"], [0.2, -0.035, -0.09], atol=1e-12
        )
        np.testing.assert_allclose(
            out["point_level_m"],
            level_rotation(g) @ np.array([0.2, -0.035, -0.09]),
            atol=1e-12,
        )

    def test_low_confidence_is_visible_and_reset(self):
        m = self.make()
        m.update(packet(0))
        seed(m)
        out = m.update(replace(packet(0.02), q_rad=(1.0,) + (0.0,) * 13))
        self.assertEqual(out["low_confidence_samples_since_capture"], 1)
        self.assertFalse(out["propagated_uncertainty_calibrated"])
        m.reset()
        self.assertFalse(m.status(0.02)["memory_available"])


if __name__ == "__main__":
    unittest.main()
