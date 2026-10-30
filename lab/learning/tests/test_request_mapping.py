import math
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from request_mapping import map_request, mapping_identity, DEMONSTRATED_FORWARD


class RequestMappingTests(unittest.TestCase):
    def test_default_matches_original_clipping(self):
        for row in (
            np.random.default_rng(54).uniform(-2, 2, size=(100, 4)).astype(np.float32)
        ):
            mapped, bounded = map_request(row)
            expected = np.clip(row, [-0.5, -0.3, -0.2, 0], [0.6, 0.3, 1, 1]).tolist()
            self.assertEqual(mapped, expected)
            self.assertEqual(bounded, expected)
        self.assertEqual(mapping_identity("continuous"), {})

    def test_strict_stop_boundary_and_unchanged_other_requests(self):
        for forward in [-0.5, 0.0, 0.049, 0.05, 0.050001, 0.3, 0.6, 1.0]:
            actual, bounded = map_request(
                [forward, 0.12, 0.45, 0.7], "demonstrated-forward"
            )
            self.assertEqual(actual[0], 0.6 if forward > 0.05 else forward)
            self.assertEqual(actual[1:], [0.12, 0.45, 0.7])
            self.assertEqual(bounded[0], min(forward, 0.6))
        self.assertEqual(
            mapping_identity("demonstrated-forward")["inference_output_mapping"],
            DEMONSTRATED_FORWARD,
        )

    def test_nonfinite_and_invalid_request_rejected(self):
        for row in [
            [0.0, 0.0, 0.0],
            [math.nan, 0.0, 0.0, 0.0],
            [0.0, 0.0, math.inf, 0.0],
        ]:
            with self.assertRaises(ValueError):
                map_request(row)
        with self.assertRaises(ValueError):
            map_request([0.0] * 4, "unknown")


if __name__ == "__main__":
    unittest.main()
