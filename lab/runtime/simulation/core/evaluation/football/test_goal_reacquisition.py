import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image
from goal_reacquisition_native import (
    GoalAwareGazeSlew,
    GoalReacquisitionNavigator,
    visible_goal_attention,
)
from native_head_domain import HeadCommandDomainAdapter
from controller import ControllerConfig

O = Path("@COURSE@/configuration/core")
RUN = O / "football-v39-small-forward40s"


class Goal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = json.loads((RUN / "sensor-inputs.jsonl").read_text().splitlines()[-1])
        cls.d = json.loads((RUN / "detections.jsonl").read_text().splitlines()[-1])
        cls.c = json.loads((RUN / "control.jsonl").read_text().splitlines()[-1])

    def nav(self):
        n = GoalReacquisitionNavigator(
            None, config=ControllerConfig(ground_mode="tof-virtual-floor")
        )
        n.state = "REFRESH_GOAL"
        n.pose = np.array(self.c["estimated_pose_map_xyyaw"])
        n.first_time = 30.0
        n.last_time = self.s["timestamp_s"] - 0.065
        n.goal_time = 30.0
        n.gaze.current[:] = [-0.4, 0]
        n.gaze.timestamp = n.last_time
        return n

    def test_real_cropped_goal_full_chain_moves_head_beyond_legacy_domain_and_holds_body(
        self,
    ):
        n = self.nav()
        out = n.step(
            np.asarray(Image.open(RUN / self.s["rgb"])),
            self.s["proprio"],
            self.s["tof_frame"],
            now_s=self.s["timestamp_s"],
            detections=self.d["detections"],
        )
        self.assertEqual(out["state"], "REFRESH_GOAL")
        self.assertLess(out["gaze_body_yaw_pitch_rad"][0], -0.4)
        self.assertEqual(out["velocity_body_xyw"], [0, 0, 0])
        self.assertFalse(out["motion_authorization"]["allowed"])
        self.assertFalse(out["kick_request"])
        self.assertTrue(out["native_goal_image_servo"]["cropped"])
        self.assertFalse(
            out["native_goal_image_servo"]["metric_goal_position_estimated"]
        )
        self.assertLess(
            out["native_goal_image_servo"]["bounded_step"]["objective_after"],
            out["native_goal_image_servo"]["bounded_step"]["objective_before"],
        )
        head = HeadCommandDomainAdapter().solve(out["gaze_body_yaw_pitch_rad"])
        self.assertTrue(head["accepted"])

    def test_negative_future_stale_and_nonfinite_goal_inputs_cannot_retarget(self):
        n = self.nav()
        cam = n.vision.camera
        t = self.s["timestamp_s"]
        for capture, now in [
            (-1, t),
            (t + 0.1, t),
            (t - 1, t),
            (float("nan"), t),
            (t, float("nan")),
        ]:
            self.assertIsNone(
                visible_goal_attention(
                    self.d["detections"], camera=cam, capture_s=capture, now_s=now
                )
            )
        self.assertIsNone(
            visible_goal_attention(
                [
                    {
                        "class_name": "goal",
                        "confidence": float("nan"),
                        "bbox_xyxy_px": [1, 1, 2, 2],
                    }
                ],
                camera=cam,
                capture_s=t,
                now_s=t,
            )
        )
        n.goal_attention_input = None
        prior = n.gaze.current.copy()
        n.gaze.update([-1.1, 0.2], t)
        self.assertTrue(np.array_equal(n.gaze.current, prior))

    def test_too_close_keeps_head_and_body_authority_zero(self):
        n = self.nav()
        n.refresh_too_close = True
        t = self.s["timestamp_s"]
        att = visible_goal_attention(
            self.d["detections"], camera=n.vision.camera, capture_s=t, now_s=t
        )
        n.goal_attention_input = (att, np.eye(3))
        prior = n.gaze.current.copy()
        n.gaze.update([-1, 0], t)
        self.assertTrue(np.array_equal(n.gaze.current, prior))

    def test_invalid_time_rejected_before_any_state_change(self):
        n = self.nav()
        prior = n.gaze.current.copy()
        stamp = n.gaze.timestamp
        n.goal_attention_context_valid = True
        for bad in (float("nan"), float("inf"), -1.0, stamp - 0.01):
            with self.assertRaises(ValueError):
                n.gaze.update([-1, 0], bad)
            self.assertTrue(np.array_equal(n.gaze.current, prior))
            self.assertEqual(n.gaze.timestamp, stamp)

    def test_missing_goal_uses_public_map_to_aim_head_only(self):
        n = self.nav()
        n.gaze.current[:] = [-0.2, 1.0]
        out = n.step(
            np.asarray(Image.open(RUN / self.s["rgb"])),
            self.s["proprio"],
            self.s["tof_frame"],
            now_s=self.s["timestamp_s"],
            detections=[],
        )
        self.assertEqual(out["state"], "REFRESH_GOAL")
        self.assertLess(out["gaze_body_yaw_pitch_rad"][1], 1.0)
        self.assertLess(out["gaze_body_yaw_pitch_rad"][0], -0.2)
        self.assertEqual(out["velocity_body_xyw"], [0, 0, 0])
        self.assertFalse(
            out["native_goal_image_servo"]["metric_goal_position_estimated"]
        )
        self.assertFalse(out["motion_authorization"]["allowed"])

    def test_actual_too_close_chain_remains_stopped(self):
        n = self.nav()
        tof = json.loads(json.dumps(self.s["tof_frame"]))
        tof["cells"][0][0]["contains_too_close"] = True
        out = n.step(
            np.asarray(Image.open(RUN / self.s["rgb"])),
            self.s["proprio"],
            tof,
            now_s=self.s["timestamp_s"],
            detections=self.d["detections"],
        )
        self.assertEqual(out["velocity_body_xyw"], [0, 0, 0])
        self.assertFalse(out["kick_request"])
        self.assertFalse(out["motion_authorization"]["allowed"])

    def test_return_from_extended_goal_gaze_is_slewed_not_clamped(self):
        n = self.nav()
        n.state = "NEAR_ALIGN"
        n.gaze.current[:] = [-1.0, 0.1]
        n.gaze.timestamp = 0.0
        values = []
        for k in range(1, 21):
            values.append(n.gaze.update([-0.3, 0.8], k / 15))
        delta = np.diff(np.vstack(([-1.0, 0.1], values)), axis=0)
        self.assertLessEqual(abs(delta).max(), 0.6 / 15 + 1e-10)
        self.assertGreater(values[-1][0], -0.4)
        self.assertTrue(
            all(HeadCommandDomainAdapter().solve(x)["accepted"] for x in values)
        )


if __name__ == "__main__":
    unittest.main()
