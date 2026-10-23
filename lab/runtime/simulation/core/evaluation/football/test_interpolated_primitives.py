import unittest
from short_primitive import select, CANDIDATES


class Interpolated(unittest.TestCase):
    def test_observed_between_grid_start_has_parameters(self):
        r = select([0.18, -0.05, -0.08], 0.0)
        self.assertIsNotNone(r)
        self.assertTrue(
            all(
                0.075 <= p[0] <= 0.105 and -0.05 <= p[1] <= -0.02
                for p in r["predicted_endpoints"]
            )
        )

    def test_no_extrapolation_or_larger_source_command(self):
        for r in CANDIDATES:
            self.assertTrue(0.4 <= r["forward_duration_s"] <= 0.8)
            self.assertTrue(-0.2 <= r["velocity_command"][1] <= 0.2)
            self.assertIn(r["forward_duration_s"], (0.6, 0.8))
            self.assertIn(r["velocity_command"][1], (-0.2, -0.1, 0.0))

    def test_original_public_entry_domain(self):
        for p, h in [
            ([0.3, -0.035, 0], 0),
            ([0.09, -0.035, 0], 0),
            ([0.18, -0.035, 0], 0.31),
        ]:
            self.assertIsNone(select(p, h))
