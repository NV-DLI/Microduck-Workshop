import unittest
import cv2, numpy as np
from beta_sensor_profile import BetaCameraProfile
from beta_perception_geometry import CaptureCameraPose, PlaneEstimate
from vision.beta_box_geometry import estimate_box_sphere
from vision.beta_goal_points import extract_beta_goal_bases


class ImageGeometryTests(unittest.TestCase):
    def test_supported_sphere_and_partial_occlusion(self):
        camera = BetaCameraProfile(width_px=960, height_px=540)
        radius_px = 90
        image = np.zeros((540, 960, 3), np.uint8)
        image[:] = [80, 150, 90]
        cv2.circle(image, (480, 270), radius_px, (230, 230, 230), -1)
        # Face straight down; the support plane is declared, not inferred from a label.
        pose = CaptureCameraPose(
            1.0,
            "body",
            ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, -1.0)),
            (0.0, 0.0, 0.302),
            "synthetic camera fixture",
        )
        ground = PlaneEstimate(
            "body", (0.0, 0.0, 1.0), 0.0, "synthetic flat support plane"
        )
        call = lambda rgb, box: estimate_box_sphere(
            rgb,
            box,
            radius_m=0.035,
            pose=pose,
            ground=ground,
            camera=camera,
            capture_timestamp_s=1.0,
            now_s=1.0,
        )
        self.assertTrue(call(image, [390, 180, 570, 360])["accepted"])
        hidden = image.copy()
        hidden[180:361, 480:571] = [80, 150, 90]
        self.assertEqual(
            call(hidden, [390, 180, 570, 360])["reason"],
            "partial_occlusion_or_box_boundary_unsupported",
        )
        self.assertEqual(call(image, [0, 180, 180, 360])["reason"], "cropped_ball")

    def test_goal_lower_opening_rejects_filled_rectangle(self):
        camera = BetaCameraProfile(width_px=960, height_px=540)
        image = np.full((540, 960, 3), 225, np.uint8)
        image[310:] = [80, 150, 90]
        cv2.rectangle(image, (300, 200), (314, 350), (230, 230, 230), -1)
        cv2.rectangle(image, (486, 200), (500, 350), (230, 230, 230), -1)
        cv2.rectangle(image, (300, 200), (500, 214), (230, 230, 230), -1)
        result = extract_beta_goal_bases(image, [300, 200, 500, 350], [0, 1, 0], camera)
        self.assertTrue(result["accepted"], result)
        cv2.rectangle(image, (300, 200), (500, 350), (230, 230, 230), -1)
        result = extract_beta_goal_bases(image, [300, 200, 500, 350], [0, 1, 0], camera)
        self.assertFalse(result["accepted"])


if __name__ == "__main__":
    unittest.main()
