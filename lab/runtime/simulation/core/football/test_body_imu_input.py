import unittest
from body_imu_input import BodyImuInput


def packet(**updates):
    p = dict(
        seq=1,
        timestamp_s=0.02,
        read_timestamp_s=0.02,
        delivered_timestamp_s=0.04,
        axes="trunk FLU",
        source="body_powerboard_control_imu",
        orientation_valid=True,
        gyro=[0.1, 0.2, 0.3],
        gravity=[0.01, 0.02, -0.97],
    )
    p.update(updates)
    return p


class BodyInputTests(unittest.TestCase):
    def source(self):
        x = BodyImuInput()
        x.record_encoders(0.02, [0.1] * 14, [0.2] * 14)
        return x

    def test_median_values_not_normalized_or_delayed_twice(self):
        x = self.source()
        x.ingest([packet()], now_s=0.04)
        r = x.observe(0.04)
        self.assertEqual(r["gravity_body"], [0.01, 0.02, -0.97])
        self.assertAlmostEqual(r["measurement_age_s"], 0.02)

    def test_warmup_and_expired_hold(self):
        x = self.source()
        self.assertIsNone(x.observe(0.0))
        x.ingest([packet(orientation_valid=False)], now_s=0.04)
        self.assertIsNone(x.observe(0.04))
        x.ingest([packet(seq=2)], now_s=0.04)
        self.assertIsNone(x.observe(0.1001))

    def test_future_duplicate_and_wrong_sensor_rejected(self):
        x = self.source()
        with self.assertRaises(ValueError):
            x.ingest([packet()], now_s=0.03)
        with self.assertRaises(ValueError):
            x.ingest([packet(source="head")], now_s=0.04)
        x.ingest([packet()], now_s=0.04)
        with self.assertRaises(ValueError):
            x.ingest([packet()], now_s=0.04)

    def test_invalid_latest_never_falls_back_to_previous(self):
        x = self.source()
        x.ingest([packet()], now_s=0.04)
        x.ingest(
            [
                packet(
                    seq=2,
                    timestamp_s=0.04,
                    delivered_timestamp_s=0.06,
                    orientation_valid=False,
                )
            ],
            now_s=0.06,
        )
        self.assertIsNone(x.observe(0.06))

    def test_same_bus_encoder_not_current_state(self):
        x = self.source()
        x.record_encoders(0.04, [0.8] * 14, [0.9] * 14)
        x.ingest([packet()], now_s=0.04)
        r = x.observe(0.04)
        self.assertLess(abs(r["encoder_q_rad"][0] - 0.1), 0.001)
        self.assertEqual(r["encoder_read_timestamp_s"], 0.02)

    def test_no_substituting_missing_encoder_sample(self):
        x = BodyImuInput()
        with self.assertRaises(ValueError):
            x.ingest([packet()], now_s=0.04)


if __name__ == "__main__":
    unittest.main()
