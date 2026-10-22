import copy
import json
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from attention_head_permission import head_only_permission
from ball_reacquisition import BallReacquisitionNavigator
from controller import ControllerConfig
from head_feedback import head_context_allowed, ObservationHeadFeedback
from visual_localization import rotate

R = Path("@COURSE@/configuration/core/football-v56-full-task180s")


class HeadAttention(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        read = lambda name: [json.loads(x) for x in (R / name).read_text().splitlines()]
        cls.controls = read("control.jsonl")
        cls.sensors = {x["timestamp_s"]: x for x in read("sensor-inputs.jsonl")}
        cls.detections = {x["timestamp_s"]: x for x in read("detections.jsonl")}
        cls.last = [x for x in cls.controls if x.get("public_metric_ball")][-1]

    def navigator(self, t, owner="BALL_REACQUIRE"):
        row = min(self.controls, key=lambda x: abs(x["timestamp_s"] - t))
        t = row["timestamp_s"]
        n = BallReacquisitionNavigator(None, config=ControllerConfig(ground_mode="tof-virtual-floor"))
        n.pose = np.array(row["estimated_pose_map_xyyaw"])
        n.ball = np.array(self.last["estimated_pose_map_xyyaw"][:2]) + rotate(
            self.last["estimated_pose_map_xyyaw"][2], self.last["public_metric_ball"]["point_level_m"][:2])
        n.ball_time = self.last["timestamp_s"]
        n.first_time = 2.34
        n.last_time = t - .065
        n.goal_time = 65.135
        n.final_entry = True
        n.last_observed_ball_gaze = [-.4, 1.3359329201709067]
        n.gaze.current[:] = row["gaze_body_yaw_pitch_rad"]
        n.gaze.timestamp = t - .065
        n.attention_owner.started = 72.535
        n.attention_owner.phase_started = t - .1 if owner == "GOAL_REFRESH" else 72.535
        n.attention_owner.state = owner
        n.ball_search.begin(72.535, n.gaze.current)
        return n, t

    def actual(self, n, t):
        s = self.sensors[t]
        return n.step(np.asarray(Image.open(R / s["rgb"])), s["proprio"], s["tof_frame"],
                      now_s=t, detections=self.detections[t]["detections"])

    def test_three_actual_ToF_hazard_frames_move_only_head(self):
        for t in (72.67, 75., 90.):
            n, t = self.navigator(t)
            before = n.gaze.current.copy()
            out = self.actual(n, t)
            self.assertTrue(n.refresh_too_close)
            self.assertTrue(n.last_geometry["floor_observation"]["accepted"])
            self.assertFalse(n.last_geometry["ball"]["accepted"])
            self.assertTrue(out["public_attention_head_permission"]["accepted"])
            self.assertGreater(np.linalg.norm(n.gaze.current - before), 1e-8)
            self.assertEqual(out["velocity_body_xyw"], [0., 0., 0.])
            self.assertFalse(out["kick_request"])
            self.assertFalse(out["motion_authorization"]["allowed"])
            self.assertEqual(n.ball_time, 72.47)
            self.assertEqual(n.ball_search.started, 72.535)
            self.assertEqual(out["ball_reacquisition"]["waypoint_index"], 0)
            domain = out["ball_reacquisition"]["parameters_native_domain"]
            self.assertTrue(domain["accepted"])
            self.assertLessEqual(max(abs(np.array(domain["head_pose_native_offsets_rad"]))
                                     - np.array([1.1, 1.1, 1.4, .31])), 1e-9)
            self.assertTrue(head_context_allowed(
                now_s=t, capture_s=t, control=out, geometry=n.last_geometry,
                too_close=True, ball_context=True, goal_context=False))

    def test_goal_attention_TooClose_still_zero_body(self):
        n, t = self.navigator(75., "GOAL_REFRESH")
        before = n.gaze.current.copy()
        out = self.actual(n, t)
        self.assertTrue(n.refresh_too_close)
        self.assertEqual(out["public_attention_owner"]["owner"], "GOAL_REFRESH")
        self.assertGreater(np.linalg.norm(n.gaze.current - before), 1e-8)
        self.assertEqual(out["velocity_body_xyw"], [0., 0., 0.])
        self.assertFalse(out["kick_request"])
        self.assertEqual(n.ball_time, 72.47)

    def test_floor_is_not_a_head_pose_certificate(self):
        s = self.sensors[75.]["proprio"]
        for floor in ({}, {"accepted": False}, {"accepted": True}):
            z = head_only_permission(t=75., owner="BALL_REACQUIRE", proprio=s,
                                     geometry={"timestamp_s": 75., "floor_observation": floor})
            self.assertTrue(z["accepted"])
            self.assertFalse(z["body_authorized"])
            self.assertFalse(z["kick_authorized"])

    def test_invalid_capture_pose_and_inactive_owner_rejected(self):
        original = self.sensors[75.]["proprio"]
        mutations = [("timestamp_s", 74.7), ("timestamp_s", 75.1), ("timestamp_s", float("nan")),
                     ("q_rad", [0.] * 13), ("q_rad", [float("nan")] * 14),
                     ("qd_rad_s", [float("inf")] * 14), ("gyro_body_rad_s", [0.]),
                     ("gravity_body", [0., 0., 0.]), ("joint_names", ["duplicate"] * 14)]
        for key, value in mutations:
            s = copy.deepcopy(original)
            s[key] = value
            before = repr(s)
            self.assertFalse(head_only_permission(
                t=75., owner="BALL_REACQUIRE", proprio=s, geometry={"timestamp_s": 75.})["accepted"])
            self.assertEqual(repr(s), before)
        for owner in ("NONE", "FAILED", "KICK", "RETURN"):
            self.assertFalse(head_only_permission(
                t=75., owner=owner, proprio=original, geometry={"timestamp_s": 75.})["accepted"])
        for t in (-1., float("nan"), float("inf")):
            self.assertFalse(head_only_permission(
                t=t, owner="BALL_REACQUIRE", proprio=original, geometry={"timestamp_s": t})["accepted"])

    def test_scope_cannot_authorize_body_or_kick_delivery(self):
        n, t = self.navigator(75.)
        out = self.actual(n, t)
        for changes in ({"velocity_body_xyw": [.2, 0., 0.]}, {"kick_request": True},
                        {"public_attention_owner": {"owner": "NONE"}},
                        {"public_attention_head_permission": {"accepted": True, "owner": "BALL_REACQUIRE", "capture_s": 74.}}):
            changed = {**out, **changes}
            self.assertFalse(head_context_allowed(
                now_s=t, capture_s=t, control=changed, geometry=n.last_geometry,
                too_close=True, ball_context=True, goal_context=False))

    def test_same_read_invalid_still_holds_actual_head(self):
        f = ObservationHeadFeedback()
        nominal = [0., .1, 0., 0.]
        sent, d = f.update(now_s=75., nominal=nominal, packet=None, velocity=[0., 0., 0.],
                           state="REACQUIRE_BALL", handoff_state="APPROACH",
                           navigation_enabled=True, head_motion_allowed=True)
        self.assertFalse(d["packet_valid"])
        self.assertFalse(d["head_motion_permitted"])
        self.assertFalse(d["integration_enabled"])

    def test_original_deadline_fails_with_valid_head_pose(self):
        n, t = self.navigator(96.535)
        before = n.gaze.current.copy()
        out = self.actual(n, t)
        self.assertEqual(n.attention_owner.state, "FAILED")
        self.assertIsNotNone(n.ball_search.failed)
        self.assertFalse(out["public_attention_head_permission"]["accepted"])
        np.testing.assert_array_equal(n.gaze.current, before)
        self.assertEqual(out["velocity_body_xyw"], [0., 0., 0.])
        self.assertFalse(out["kick_request"])


if __name__ == "__main__":
    unittest.main()
