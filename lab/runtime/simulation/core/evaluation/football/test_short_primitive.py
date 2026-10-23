import unittest
from short_primitive import select


class TestPrimitive(unittest.TestCase):
    def test_measured_lateral_step_matches_box(self):
        p = select([0.21, -0.075, -0.08], 0.0)
        self.assertIsNotNone(p)
        self.assertEqual(p["velocity_command"], [0.3, -0.2, 0])
        self.assertTrue(0.4 <= p["forward_duration_s"] <= 0.8)
        self.assertTrue(
            all(
                0.075 <= x <= 0.105 and -0.05 <= y <= -0.02
                for x, y, h in p["predicted_endpoints"]
            )
        )

    def test_nonfinite_no_action(self):
        self.assertIsNone(select([float("nan"), 0, 0], 0))

    def test_no_endpoint_fit_no_action(self):
        self.assertIsNone(select([0.13, 0.06, -0.08], 0.0))
