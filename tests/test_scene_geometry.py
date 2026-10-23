"""CPU regression cases for measured scene geometry and RGB post extraction."""

from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))
from scene_goal_points import extract_scene_goal_bases


class PostEvidence(unittest.TestCase):
    def image(self, color=(205, 205, 205)):
        image = np.full((240, 320, 3), (45, 130, 40), np.uint8)
        image[40:180, 60:65] = color
        image[40:180, 255:260] = color
        image[40:45, 60:260] = color
        return image

    def extract(self, image, box=(58, 38, 262, 182), g=(0, 1, 0)):
        return extract_scene_goal_bases(
            image, box, g, SimpleNamespace(width_px=320, height_px=240)
        )

    def test_real_post_shape(self):
        r = self.extract(self.image())
        self.assertTrue(r["accepted"])
        np.testing.assert_allclose(
            r["footpoints_upright_px"], [[62, 179], [257, 179]], atol=2
        )

    def test_transverse_line_does_not_replace_base(self):
        image = self.image()
        image[187:190, 40:285] = 220
        r = self.extract(image)
        self.assertTrue(r["accepted"])
        self.assertLess(max(p[1] for p in r["footpoints_upright_px"]), 182)

    def test_colored_clutter(self):
        for color in [(0, 0, 0), (20, 20, 220)]:
            self.assertFalse(self.extract(self.image(color))["accepted"])

    def test_cropped_or_invalid(self):
        self.assertFalse(self.extract(self.image(), box=(0, 38, 262, 182))["accepted"])
        self.assertFalse(self.extract(self.image(), g=(np.nan, 1, 0))["accepted"])
        self.assertFalse(self.extract(self.image(), g=(0, 1))["accepted"])

    def test_rear_net_without_front_posts(self):
        image = np.full((240, 320, 3), (45, 130, 40), np.uint8)
        image[40:180, 105:108] = 210
        image[40:180, 212:215] = 210
        self.assertFalse(self.extract(image)["accepted"])


class SupportSurface(unittest.TestCase):
    def test_ramp_and_landings_have_large_support(self):
        from terrain import ramp_mesh

        for angle in [-15, 0, 15]:
            meshes, origin = ramp_mesh(0, SimpleNamespace(slope_deg=angle))
            vertices = meshes[0].vertices - origin
            self.assertLessEqual(vertices[:, 0].min(), -50)
            self.assertGreaterEqual(vertices[:, 0].max(), 50)
            self.assertLessEqual(vertices[:, 1].min(), -50)
            self.assertGreaterEqual(vertices[:, 1].max(), 50)
            expected = np.clip(vertices[:, 0] - 0.35, 0, 1) * np.tan(np.deg2rad(angle))
            np.testing.assert_allclose(vertices[:, 2], expected, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
