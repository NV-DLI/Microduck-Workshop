import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from PIL import Image
import perception
from visibility_gate import install
from ball_reacquisition import BallReacquisitionNavigator
from controller import ControllerConfig
from head_handoff import HeadHandoff
from visual_localization import rotate


class VisibilityConsumers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path("@COURSE@/outputs/migration-20261019/football-v60-full-task180s")
        read = lambda n: [json.loads(s) for s in (cls.root / n).read_text().splitlines()]
        cls.controls = read("control.jsonl")
        cls.sensors = {x["timestamp_s"]: x for x in read("sensor-inputs.jsonl")}
        cls.detections = {x["timestamp_s"]: x for x in read("detections.jsonl")}

    def setUp(self):
        self.original_estimator = install()

    def tearDown(self):
        perception.estimate_box_sphere = self.original_estimator

    def test_all_four_actual_crops_do_not_refresh_metric_ball_time_or_hint(self):
        for t in (48.27, 48.335, 48.4, 48.47):
            original = next(x for x in self.controls if x["timestamp_s"] == t)
            prior = max((x for x in self.controls if x["timestamp_s"] < t), key=lambda x: x["timestamp_s"])
            n = BallReacquisitionNavigator(None, config=ControllerConfig(ground_mode="tof-virtual-floor"))
            n.pose = np.array(original["estimated_pose_map_xyyaw"])
            n.final_entry = True
            n.first_time = 2.34
            n.last_time = prior["timestamp_s"]
            n.goal_time = 31.8
            n.ball_time = 35.4
            n.ball = n.pose[:2] + rotate(n.pose[2], original["public_metric_ball"]["point_level_m"][:2])
            previous_ball = n.ball.copy()
            n.attention_owner.state = "BALL_REACQUIRE"
            n.attention_owner.started = n.attention_owner.phase_started = 35.47
            n.attention_owner.last_t = prior["timestamp_s"]
            n.gaze.current = np.array(prior["gaze_body_yaw_pitch_rad"])
            n.gaze.timestamp = prior["timestamp_s"]
            n.ball_search.begin(35.47, [-.17133256514572132, 1.3904477599867453])
            n.ball_view_hint.observe(now_s=35.4, capture_s=35.4, gaze=[-.4, 1.], complete=True)
            sr = self.sensors[t]
            out = n.step(np.asarray(Image.open(self.root / sr["rgb"])), sr["proprio"], sr["tof_frame"],
                         now_s=t, detections=self.detections[t]["detections"])
            self.assertFalse(n.last_geometry["ball"]["accepted"])
            self.assertIsNone(n.last_geometry["ball"]["geometry"]["point_reference_m"])
            self.assertIsNone(out["public_metric_ball"])
            self.assertEqual(n.ball_time, 35.4)
            np.testing.assert_array_equal(n.ball, previous_ball)
            self.assertEqual(n.ball_view_hint.capture_s, 35.4)
            np.testing.assert_array_equal(n.ball_view_hint.gaze, [-.4, 1.])
            self.assertEqual(n.attention_owner.started, 35.47)
            self.assertEqual(n.ball_search.started, 35.47)
            self.assertEqual(out["velocity_body_xyw"], [0., 0., 0.])
            self.assertFalse(out["kick_request"])
            # Instrument the actual handoff method with an explicit synthetic
            # memory spy. Rejected original RGB must never invoke anchoring.
            h = HeadHandoff(n.vision.leg, sr["proprio"]["joint_names"])
            def forbidden_anchor(**kwargs):
                self.fail("A rejected complete-ball capture attempted memory anchoring")
            old_memory = {"ball_capture_timestamp_s": 35.4, "memory_available": False}
            h.adapter.memory = SimpleNamespace(
                observe_complete_ball=forbidden_anchor, status=lambda now_s: copy.deepcopy(old_memory))
            before = h.status(t)
            after = h.capture(now_s=t, geometry=n.last_geometry, head_offsets=[0., 0., 0., 0.],
                              candidate=True, goal_heading=0., goal_fix_s=t)
            self.assertEqual(before, after)
            self.assertEqual(after["memory"]["ball_capture_timestamp_s"], 35.4)
            self.assertFalse(after["kick_runtime_authorized"])

    def test_confirmed_complete_near_bottom_frame_still_accepted(self):
        root = self.root.with_name("football-v61-full-task180s")
        t = 23.07
        sr = next(json.loads(s) for s in (root / "sensor-inputs.jsonl").read_text().splitlines() if json.loads(s)["timestamp_s"] == t)
        dr = next(json.loads(s) for s in (root / "detections.jsonl").read_text().splitlines() if json.loads(s)["timestamp_s"] == t)
        g = perception.BetaPerception().estimate(np.asarray(Image.open(root / sr["rgb"])),
            sr["proprio"], dr["detections"], now_s=t, tof_frame=sr["tof_frame"])
        self.assertTrue(g["ball"]["accepted"])
        self.assertTrue(g["ball"]["geometry"]["observed_image_border_guard"]["accepted"])

    def test_wrapper_is_same_function_installed_by_new_entry(self):
        self.assertTrue(getattr(perception.estimate_box_sphere, "_observed_border_wrapped", False))
        source = (Path(__file__).parent / "execute.py").read_text()
        self.assertIn("from visibility_gate import install as install_visibility_gate", source)
        self.assertIn("install_visibility_gate()", source)


if __name__ == "__main__":
    unittest.main()
