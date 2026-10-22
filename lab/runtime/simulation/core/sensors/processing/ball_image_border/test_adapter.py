import unittest
import numpy as np
from adapter import wrap_estimator
from test_guard import scene


class AdapterTests(unittest.TestCase):
    def test_old_rejection_is_same_object_without_new_evaluation(self):
        old = {"accepted": False, "reason": "stale_image_or_FK", "point_reference_m": None}
        wrapped = wrap_estimator(lambda *args, **kwargs: old)
        self.assertIs(wrapped(scene(), [34, 34, 86, 86]), old)

    def test_accepted_geometry_and_clock_unchanged_for_clear_component(self):
        old = {"accepted": True, "reason": "old", "point_reference_m": [.1, -.03, 0.],
               "capture_timestamp_s": 3., "supported_boundary_sectors": 21}
        wrapped = wrap_estimator(lambda *args, **kwargs: old)
        actual = wrapped(scene(), [34, 34, 86, 86])
        for key, value in old.items():
            self.assertEqual(actual[key], value)
        self.assertNotIn("observed_image_border_guard", old)

    def test_inset_box_cannot_authorize_missing_bottom(self):
        old = {"accepted": True, "reason": "old", "point_reference_m": [.1, -.03, 0.],
               "capture_timestamp_s": 3., "supported_boundary_sectors": 21}
        actual = wrap_estimator(lambda *args, **kwargs: old)(scene((60, 100)), [35, 74, 85, 114])
        self.assertFalse(actual["accepted"])
        self.assertIsNone(actual["point_reference_m"])
        self.assertEqual(actual["capture_timestamp_s"], 3.)
        self.assertEqual(actual["supported_boundary_sectors"], 21)
        self.assertTrue(old["accepted"])

    def test_unknown_component_rejects_and_double_install_rejected(self):
        wrapped = wrap_estimator(lambda *args, **kwargs: {"accepted": True, "point_reference_m": [1, 2, 3]})
        self.assertFalse(wrapped(np.zeros((120, 120, 3), np.uint8), [34, 34, 86, 86])["accepted"])
        with self.assertRaises(ValueError):
            wrap_estimator(wrapped)


if __name__ == "__main__":
    unittest.main()
