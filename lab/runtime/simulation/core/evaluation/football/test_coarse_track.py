import unittest, copy, math
from types import SimpleNamespace
import numpy as np
from coarse_track import CoarseObserver, coarse_entry, path_authority
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet, mem
from fine_kick import route_at_step_end


class CoarseTrack(unittest.TestCase):
    def inputs(self):
        obs = CoarseObserver()
        for t in [1.0, 1.07, 1.14, 1.21, 1.28, 1.35]:
            d = obs.update(t, [0.7, -0.028, -0.08], [0, 0, 0], 0.0, True)
        return dict(
            path={"allowed": True, "layout": "empty", "capture_s": 1.35},
            t=1.35,
            handoff_state="APPROACH",
            fine_active=False,
            fine_state="IDLE",
            final_diag=d,
            velocity_body_xyw=[0.0, 0.0, 0.0],
            complete=True,
            guard_safe=True,
            memory=mem(1.35, anchor=1.35, p=(0.7, -0.028, -0.08)),
            goal_heading=0.0,
            goal_fix_s=1.35,
            actual_head=[0] * 4,
        )

    def make(self):
        f = NeutralFineStep()
        m = mem(1.0, p=(0.7, -0.028, -0.08))
        f.capture(
            t=1.0,
            guard_safe=True,
            complete=True,
            trigger=True,
            actual_head=[0] * 4,
            memory=m,
            goal_heading=0.0,
            goal_fix_s=1.0,
            motion_kind="COARSE_TRACK",
        )
        for k in range(51, 78):
            t = k / 50
            f.capture_s = t
            f.goal_s = t
            r = f.policy(t=t, packet=packet(t), memory=m, actual_head=[0] * 4)
        return f, m, r

    def test_measured_coarse_entry(self):
        self.assertTrue(coarse_entry(**self.inputs())[0])

    def test_missing_stale_wrong_layout_path_rejected(self):
        for path in (
            None,
            {},
            {"allowed": True, "capture_s": 1.2, "layout": "empty"},
            {"allowed": True, "capture_s": 1.35, "layout": "one-obstacle"},
        ):
            a = self.inputs()
            a["path"] = path
            self.assertFalse(coarse_entry(**a)[0])

    def test_sensor_guard_and_incomplete_capture_rejected(self):
        for key in ("complete", "guard_safe"):
            a = self.inputs()
            a[key] = False
            self.assertFalse(coarse_entry(**a)[0])

    def test_unobserved_or_moving_quiet_window_rejected(self):
        for key, value in [
            ("gyro_3D_RMS_rad_s", 0.13),
            ("ball_XY_diameter_m", 0.021),
            ("quiet_sample_times", [1.35]),
        ]:
            a = self.inputs()
            a["final_diag"][key] = value
            self.assertFalse(coarse_entry(**a)[0])

    def test_coarse_stop_never_enters_kick_handoff(self):
        f, m, r = self.make()
        self.assertEqual(f.state, "TRACK")
        self.assertTrue(f.coarse_track)
        self.assertEqual(r["velocity"], [0.3, 0.0, 0.0])
        m["point_level_m"] = [0.35, -0.028, -0.08]
        f.capture_s = f.goal_s = 1.56
        r = f.policy(t=1.56, packet=packet(1.56), memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "SETTLE")
        self.assertEqual(r["velocity"], [0.0, 0.0, 0.0])
        self.assertIsNone(f.track_brake_start)
        # Even an accidental much closer stop may never use coarse routing to kick.
        m["point_level_m"] = [0.09, -0.035, -0.08]
        h = SimpleNamespace(
            state="APPROACH", latest_capture={"capture_s": 1.56, "guard_safe": True}
        )
        self.assertEqual(
            route_at_step_end(
                now_s=1.56,
                fine=f,
                handoff=h,
                memory=m,
                goal_heading=0.0,
                goal_fix_s=1.56,
            ),
            "keep:coarse_requires_fresh_reacquisition",
        )

    def test_far_heading_is_corrected_during_track_without_tiny_turn_loop(self):
        for heading in (-0.15, 0.1, 0.3):
            f = NeutralFineStep()
            m = mem(1.0, p=(0.7, 0.21, -0.08))
            f.capture(
                t=1.0,
                guard_safe=True,
                complete=True,
                trigger=True,
                actual_head=[0] * 4,
                memory=m,
                goal_heading=heading,
                goal_fix_s=1.0,
                motion_kind="COARSE_TRACK",
            )
            for k in range(51, 78):
                t = k / 50
                f.capture_s = f.goal_s = t
                r = f.policy(t=t, packet=packet(t), memory=m, actual_head=[0] * 4)
            self.assertEqual(f.state, "TRACK")
            self.assertEqual(r["velocity"][0], 0.3)
            self.assertFalse(f.turn_goal_only)

    def test_public_y_point_within_existing_coarse_memory_bound_is_admitted(self):
        a = self.inputs()
        a["memory"]["point_level_m"] = [0.72, 0.205, -0.08]
        a["goal_heading"] = 0.1
        self.assertTrue(coarse_entry(**a)[0])

    def test_coarse_time_cap(self):
        f, m, r = self.make()
        for k in range(78, 178):
            t = k / 50
            f.capture_s = f.goal_s = t
            r = f.policy(t=t, packet=packet(t), memory=m, actual_head=[0] * 4)
            if f.state == "SETTLE":
                break
        self.assertEqual(f.state, "SETTLE")
        self.assertLessEqual(f.step_end - f.step_start, 2.000001)

    def test_memory_expiry_and_hazard_still_stop_coarse(self):
        f, m, r = self.make()
        f.capture_safe = False
        r = f.policy(t=1.56, packet=packet(1.56), memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "SETTLE")
        self.assertEqual(r["velocity"], [0.0, 0.0, 0.0])
        f, m, r = self.make()
        f.last_policy = 5.5
        f.capture_s = f.goal_s = 5.52
        r = f.policy(t=5.52, packet=packet(5.52), memory=m, actual_head=[0] * 4)
        self.assertEqual(f.state, "FAILED")

    def test_quiet_gap_and_guard_clear_history(self):
        a = CoarseObserver()
        for t in [1.0, 1.07, 1.14, 1.21, 1.28, 1.35]:
            d = a.update(t, [0.7, 0, 0], [0, 0, 0], 0, True)
        self.assertTrue(d["measured_quiet"])
        self.assertFalse(
            a.update(2.0, [0.7, 0, 0], [0, 0, 0], 0, True)["measured_quiet"]
        )
        a.update(2.07, [0.7, 0, 0], [0, 0, 0], 0, False)
        self.assertFalse(a.samples)

    def test_wrong_scene_and_stale_pose_do_not_request_map(self):
        o = SimpleNamespace(
            config=SimpleNamespace(layout="one-obstacle"),
            pose=np.array([-0.66, 0, 0]),
            goal_time=1.0,
        )
        r = path_authority(o, {}, dict(timestamp_s=1.0), np.eye(3), 0.12, [0.7, 0, 0])
        self.assertFalse(r["allowed"])
        o.config.layout = "empty"
        o.goal_time = -10.0
        self.assertFalse(
            path_authority(o, {}, dict(timestamp_s=1.0), np.eye(3), 0.12, [0.7, 0, 0])[
                "allowed"
            ]
        )


class CoarseRealMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_zone_floor_projection import FloorProjection

        FloorProjection.setUpClass()
        cls.frame = FloorProjection.frame
        cls.request = FloorProjection.request

    def owner(self):
        from zone_floor_projection import project_floor_zones

        sample, _ = project_floor_zones(
            self.request["case"]["tof_samples"][0], self.frame, floor_accepted=True
        )
        o = SimpleNamespace(
            config=SimpleNamespace(layout="empty"),
            pose=np.array(self.request["case"]["odom"]["pose_xyyaw"]),
            goal_time=self.frame["timestamp_s"],
            goal_map=np.array([0.93, 0.0]),
            image_sha="0" * 64,
        )
        o._tof_sample = lambda *args: (sample, False)
        return o

    def test_actual_map_and_original_envelope(self):
        o = self.owner()
        t = self.frame["timestamp_s"]
        r = path_authority(
            o, self.frame, dict(timestamp_s=t), np.eye(3), 0.12, [0.712, 0.238, -0.081]
        )
        self.assertTrue(r["allowed"])
        self.assertEqual(r["envelope"]["test_radius_m"], 0.1)
        self.assertIn("map_request", r)

    def test_current_hazard_and_field_boundary_rejected(self):
        o = self.owner()
        t = self.frame["timestamp_s"]
        sample, _ = o._tof_sample()
        o._tof_sample = lambda *args: (sample, True)
        self.assertFalse(
            path_authority(
                o, self.frame, dict(timestamp_s=t), np.eye(3), 0.12, [0.7, 0, -0.08]
            )["allowed"]
        )
        o = self.owner()
        self.assertFalse(
            path_authority(
                o, self.frame, dict(timestamp_s=t), np.eye(3), 0.12, [0.7, 1.0, -0.08]
            )["allowed"]
        )


if __name__ == "__main__":
    unittest.main()
