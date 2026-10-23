import copy, json, math, unittest
from pathlib import Path
import numpy as np, cv2
from beta_sensor_profile import BetaCameraProfile
from beta_perception_geometry import raw_pixel_to_unit_ray
from perception import BetaPerception, validate_observation
from hardware_head_fk import CourseHeadFK, HEAD_JOINTS
from vision.goal_points import extract_goal_bases
from contact_scoring import PhysicalGoalScorer
from scoring import goal_crossing

NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


class CpuTests(unittest.TestCase):
    def test_beta_native_no_cw90(self):
        f = CourseHeadFK().forward(dict(zip(HEAD_JOINTS, [0.2, 0.3, 0.1, -0.1])))
        self.assertEqual(f["camera_raw_optical"], f["camera_upright_optical"])

    def test_current_usd_fk_geometry_agrees(self):
        p = BetaPerception()
        self.assertEqual(p.vertices["left_foot"]["vertex_count"], 7896)

    def test_parameterized_native_rasters(self):
        for w, h in [(640, 360), (960, 540), (1280, 720), (1920, 1080)]:
            c = BetaCameraProfile(width_px=w, height_px=h)
            p = BetaPerception(c)
            ray = raw_pixel_to_unit_ray([c.cx, c.cy], c)
            self.assertEqual(ray, (0.0, 0.0, 1.0))

    def test_hidden_state_rejected(self):
        o = {
            "timestamp_s": 0.0,
            "joint_names": NAMES,
            "q_rad": [0.0] * 14,
            "qd_rad_s": [0.0] * 14,
            "gyro_body_rad_s": [0.0, 0.0, 0.0],
            "gravity_body": [0.0, 0.0, -1.0],
        }
        validate_observation(o)
        o["root_xyz"] = [0.0, 0.0, 0.1]
        with self.assertRaises(ValueError):
            validate_observation(o)

    def test_native_goal_pixels(self):
        c = BetaCameraProfile(width_px=960, height_px=540)
        im = np.zeros((540, 960, 3), np.uint8)
        im[:] = [50, 100, 70]
        cv2.rectangle(im, (300, 100), (600, 115), (255, 255, 255), -1)
        cv2.rectangle(im, (300, 100), (315, 300), (255, 255, 255), -1)
        cv2.rectangle(im, (585, 100), (600, 300), (255, 255, 255), -1)
        result = extract_goal_bases(im, [290, 90, 610, 310], [0.0, 1.0, 0.0], c)
        self.assertTrue(result["accepted"], result["reason"])
        self.assertTrue(all(p[1] > 290 for p in result["footpoints_upright_px"]))

    def test_whole_ball_not_center(self):
        self.assertFalse(goal_crossing([0.92, 0, 0.035], [0.94, 0, 0.035]))
        self.assertTrue(goal_crossing([0.96, 0, 0.035], [0.97, 0, 0.035]))

    def test_no_contact_no_goal(self):
        s = PhysicalGoalScorer()
        for t, x in [(0.0, 0.96), (0.005, 0.97)]:
            s.update(
                timestamp_s=t,
                ball_world_xyz=[x, 0, 0.035],
                paired_foot_forces_world_n=[[0, 0, 0], [0, 0, 0]],
                foot_names=["ankle_left", "ankle_right"],
                active_policy="kick_right",
                kick_requested=True,
            )
        self.assertFalse(s.summary()["whole_ball_crossed_inside_goal"])

    def test_real_contact_then_goal(self):
        s = PhysicalGoalScorer()
        for t, x in [(0.0, 0.96), (0.005, 0.97)]:
            s.update(
                timestamp_s=t,
                ball_world_xyz=[x, 0, 0.035],
                paired_foot_forces_world_n=[[0, 0, 0], [0.1, 0, 0]],
                foot_names=["ankle_left", "ankle_right"],
                active_policy="kick_right",
                kick_requested=True,
            )
        self.assertTrue(s.summary()["whole_ball_crossed_inside_goal"])

    def test_wrong_foot_during_kick_not_qualified(self):
        s = PhysicalGoalScorer()
        for t, x in [(0.0, 0.96), (0.005, 0.97)]:
            s.update(
                timestamp_s=t,
                ball_world_xyz=[x, 0, 0.035],
                paired_foot_forces_world_n=[[0.1, 0, 0], [0, 0, 0]],
                foot_names=["ankle_left", "ankle_right"],
                active_policy="kick_right",
                kick_requested=True,
            )
        self.assertFalse(s.summary()["whole_ball_crossed_inside_goal"])

    def test_zero_gravity_rejected(self):
        o = {
            "timestamp_s": 0.0,
            "joint_names": NAMES,
            "q_rad": [0.0] * 14,
            "qd_rad_s": [0.0] * 14,
            "gyro_body_rad_s": [0.0, 0.0, 0.0],
            "gravity_body": [0.0, 0.0, 0.0],
        }
        with self.assertRaises(ValueError):
            validate_observation(o)

    def test_missing_contact_channel_rejected(self):
        with self.assertRaises(ValueError):
            PhysicalGoalScorer().update(
                timestamp_s=0.0,
                ball_world_xyz=[0, 0, 0.035],
                paired_foot_forces_world_n=[[0.1, 0, 0]],
                foot_names=["ankle_right"],
                active_policy="kick_right",
                kick_requested=True,
            )

    def test_score_is_not_controller_import(self):
        source = (Path(__file__).parent / "controller.py").read_text()
        self.assertNotIn("from scoring", source)
        self.assertNotIn("from contact_scoring", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
