"""Geometry/split fixtures only. These synthetic arrays never become a dataset."""

import copy, sys, unittest, math
from pathlib import Path

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P.parent / "sensors"))
sys.path.insert(0, str(P.parent / "football"))
import numpy as np
from beta_sensor_profile import BetaCameraProfile
from nav2_bridge import quaternion_xyzw
from teacher_labels import labels, rays, sphere_depth, box_depth
from plan_dataset import make_plan, validate_plan


class PerceptionCpuTests(unittest.TestCase):
    def test_complete_scene_disjoint_plan(self):
        plan = make_plan()
        r = validate_plan(plan)
        self.assertEqual(r["views"], 384)
        self.assertEqual(r["scenes"], 96)
        self.assertEqual(plan["actual_images"], 0)

    def test_scene_leakage_rejected(self):
        p = make_plan()
        p["rows"][1]["split"] = "test"
        with self.assertRaises(ValueError):
            validate_plan(p)

    def fixture(self):
        profile = BetaCameraProfile(width_px=320, height_px=180)
        origin = np.array([-0.5, 0, 0.2])
        R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])
        camera = {
            "position_world_m": origin.tolist(),
            "quaternion_ros_xyzw": quaternion_xyzw(R),
        }
        ball = np.array([0.25, 0, 0.035])
        d = rays(profile)
        depth = sphere_depth(d, R.T @ (ball - origin), 0.035)
        for center, size in [
            ((0.93, 0.1325, 0.09625), (0.025, 0.025, 0.1925)),
            ((0.93, -0.1325, 0.09625), (0.025, 0.025, 0.1925)),
            ((0.93, 0, 0.1925), (0.025, 0.29, 0.025)),
        ]:
            depth = np.minimum(depth, box_depth(origin, d @ R.T, center, size))
        return profile, camera, ball, depth.reshape(180, 320)

    def test_depth_confirmed_objects_labelled(self):
        p, c, b, d = self.fixture()
        out, diag = labels(p, c, d, b)
        self.assertEqual({o["class_id"] for o in out}, {0, 1})
        self.assertTrue(all(o["teacher_only"] for o in out))

    def test_occluded_projection_not_labelled(self):
        p, c, b, d = self.fixture()
        d[:] = 0.08
        out, diag = labels(p, c, d, b)
        self.assertEqual(out, [])

    def test_detector_does_not_import_teacher(self):
        s = (P / "detector.py").read_text()
        self.assertNotIn("import teacher_labels", s)
        self.assertNotIn("ball_world", s)

    def test_export_parity_cannot_hide_missing_or_wrong_class(self):
        from validate_export import compare

        a = [{"class_id": 0, "confidence": 0.8, "box": [10.0, 20.0, 30.0, 40.0]}]
        self.assertTrue(compare(a, copy.deepcopy(a))["passed"])
        self.assertFalse(compare(a, [])["passed"])
        b = copy.deepcopy(a)
        b[0]["class_id"] = 1
        self.assertFalse(compare(a, b)["passed"])

    def test_all_planned_lenses_fit_actual_overscan(self):
        import cv2
        from types import SimpleNamespace
        from collect_isaac import configure_optics

        plan = make_plan()
        for hfov in (60, 70, 80, 90):
            profile = BetaCameraProfile(
                width_px=960,
                height_px=540,
                assumed_horizontal_fov_deg=hfov,
                distortion_k1_k2_p1_p2_k3=(-0.1, 0.01, 0.0, 0.0, 0.0),
            )
            K = np.asarray(profile.K)
            focal = (K[0, 0] + K[1, 1]) / 2
            xx, yy = np.meshgrid(np.arange(960), np.arange(540))
            p = np.stack([xx, yy], axis=-1).astype(float).reshape(-1, 1, 2)
            u = cv2.undistortPoints(
                p, K, np.asarray(profile.distortion_k1_k2_p1_p2_k3)
            ).reshape(540, 960, 2)
            rig = SimpleNamespace(
                width=960,
                height=540,
                profile=profile,
                render_focal_px=focal,
                render_width=2 * int(math.ceil(np.abs(u[:, :, 0]).max() * focal + 4)),
                render_height=2 * int(math.ceil(np.abs(u[:, :, 1]).max() * focal + 4)),
            )
            for d in {
                tuple(r["distortion"])
                for r in plan["rows"]
                if r["horizontal_fov_deg"] == hfov
            }:
                with self.subTest(hfov=hfov, distortion=d):
                    configure_optics(rig, d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
