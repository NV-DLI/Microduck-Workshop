import unittest, sys, json, hashlib, math
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parents[1]))
from beta_body_imu import BetaBodyImu, BetaBodyImuProfile
from adapter import FiniteRangeBodyImu, install_body_before_first_tick


class TestScalar(unittest.TestCase):
    def tick(self, s, t, omega=(0.0, 0.0, 0.0), acc=(0.0, 0.0, 0.0)):
        return s.poll(
            t,
            q_world_from_trunk_xyzw=(0.0, 0.0, 0.0, 1.0),
            angular_velocity_world_rad_s=omega,
            linear_acceleration_world_m_s2=acc,
        )

    def test_exact_no_clip_packets_and_rng(self):
        a = BetaBodyImu(seed=888)
        b = FiniteRangeBodyImu(seed=888)
        for i in range(500):
            self.assertEqual(
                self.tick(a, i * 0.005, (0.01, 0.0, 0.03)),
                self.tick(b, i * 0.005, (0.01, 0.0, 0.03)),
            )
        self.assertEqual(a.chip.rng.getstate(), b.chip.rng.getstate())

    def test_only_changed_loss_condition(self):
        c = json.loads((H / "change-contract.json").read_text())
        p = Path(c["parent"])
        self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), c["parent_sha256"])
        self.assertEqual(
            "".join((H / "finite_range_chip.py").read_text().splitlines(True)[2:]),
            p.read_text().replace(*c["replace"]),
        )

    def test_clipped_gyro_continues_with_bounded_measurement(self):
        s = FiniteRangeBodyImu(
            profile=BetaBodyImuProfile(
                gyro_noise_std_rad_s=0.0, accel_noise_std_m_s2=0.0
            )
        )
        for i in range(201):
            self.tick(s, i * 0.005)
        self.assertTrue(s.latest["orientation_valid"])
        oldq = s.latest["quat"]
        for i in range(201, 241):
            self.tick(s, i * 0.005, (20.0, 0.0, 0.0))
        self.assertTrue(s.latest["orientation_valid"])
        self.assertTrue(s.latest["gyro_clipped"][0])
        self.assertNotEqual(s.latest["quat"], oldq)
        self.assertLessEqual(abs(s.latest["gyro"][0]), math.radians(500.0175))

    def test_initial_saturation_and_real_gap_still_invalid(self):
        s = FiniteRangeBodyImu()
        for i in range(401):
            self.tick(s, i * 0.005, (20.0, 0.0, 0.0))
        self.assertFalse(s.latest["orientation_valid"])
        for i in range(401, 701):
            self.tick(s, i * 0.005)
        self.assertTrue(s.latest["orientation_valid"])
        self.tick(s, 4.0)
        self.assertFalse(s.latest["orientation_valid"])

    def test_no_midstream_switch_and_reset_retains_parameters(self):
        s = FiniteRangeBodyImu()
        self.tick(s, 0.0)
        with self.assertRaises(ValueError):
            install_body_before_first_tick(s)
        cls = type(s.chip)
        s.reset()
        self.assertIs(type(s.chip), cls)


if __name__ == "__main__":
    unittest.main()
