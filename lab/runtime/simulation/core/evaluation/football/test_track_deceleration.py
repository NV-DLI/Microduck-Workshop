import unittest
import numpy as np
from neutral_fine_step import NeutralFineStep


class NearRangeSpeed(unittest.TestCase):
    def test_public_range_switch_keeps_feedback_and_model_commands(self):
        packet = dict(
            seq=1,
            read_timestamp_s=1.0,
            gyro_body_rad_s=[0.0, 0.0, 0.0],
            gravity_body=[0.0, 0.0, -1.0],
        )
        for x, expected in ((0.3, 0.3), (0.17001, 0.3), (0.17, 0.25), (0.12, 0.25)):
            f = NeutralFineStep()
            v = f.track_filtered(np.array([x, -0.035, -0.08]), 0.0, 0.02, packet)
            self.assertEqual(v, [expected, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
