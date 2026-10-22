import copy
import unittest
import cv2
import numpy as np
from guard import check_observed_ball_border as check


def scene(center=(60, 60), radius=25):
    a = np.full((120, 120, 3), [100, 170, 140], dtype=np.uint8)
    cv2.circle(a, center, radius, (220, 220, 220), -1)
    return a


class TestBorder(unittest.TestCase):
    def test_complete(self):
        result = check(scene(), [34, 34, 86, 86])
        self.assertTrue(result["accepted"])
        self.assertFalse(result["body_authorized"])
        self.assertFalse(result["kick_authorized"])

    def test_four_edges_with_inset_detector_boxes(self):
        for center, box in [
            ((60, 20), [35, 5, 85, 46]), ((60, 100), [35, 74, 85, 114]),
            ((20, 60), [5, 35, 46, 85]), ((100, 60), [74, 35, 114, 85]),
        ]:
            with self.subTest(center=center):
                r = check(scene(center), box)
                self.assertEqual(r["reason"], "observed_ball_component_touches_image_edge")

    def test_corners(self):
        for center, box in [
            ((20, 20), [5, 5, 46, 46]), ((100, 20), [74, 5, 114, 46]),
            ((20, 100), [5, 74, 46, 114]), ((100, 100), [74, 74, 114, 114]),
        ]:
            with self.subTest(center=center):
                self.assertFalse(check(scene(center), box)["accepted"])

    def test_full_ball_one_pixel_inside(self):
        self.assertTrue(check(scene((60, 93)), [34, 67, 86, 118])["accepted"])

    def test_unrelated_white_border_not_assigned_to_ball(self):
        a = scene()
        a[-1, :] = 220
        self.assertTrue(check(a, [34, 34, 86, 86])["accepted"])

    def test_two_components_ambiguous(self):
        a = scene((44, 60), 10)
        cv2.circle(a, (76, 60), 10, (220, 220, 220), -1)
        self.assertEqual(check(a, [30, 44, 90, 76])["reason"], "ambiguous_ball_component")

    def test_missing_foreground_unknown(self):
        self.assertEqual(check(np.zeros((120, 120, 3), np.uint8), [34, 34, 86, 86])["reason"], "unknown_ball_component")

    def test_ROI_truncation_unknown(self):
        self.assertEqual(check(scene(radius=40), [50, 50, 70, 70])["reason"], "unknown_component_truncated_by_ROI")

    def test_invalid_finite_shape_and_no_input_mutation(self):
        a = scene()
        b = [34., 34., 86., 86.]
        aa, bb = a.copy(), copy.deepcopy(b)
        check(a, b)
        np.testing.assert_array_equal(a, aa)
        self.assertEqual(b, bb)
        for box in ([0, 0, 0, 3], [0, 0, float("nan"), 3], [0, 0, float("inf"), 3], [0, 0, 121, 3], [1]):
            self.assertFalse(check(a, box)["accepted"])
        self.assertFalse(check(a.astype(float), b)["accepted"])


if __name__ == "__main__":
    unittest.main()
