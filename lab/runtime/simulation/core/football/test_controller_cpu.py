"""Synthetic state-transition tests, explicitly not robot qualification."""

import copy, unittest
from types import SimpleNamespace
import numpy as np
from controller import BetaNavigator
from test_cpu import NAMES


class SensorFixture:
    def __init__(self):
        self.goal_visible = True
        self.ball_visible = True
        self.leg = SimpleNamespace(
            estimate=lambda *args: {"velocity_body_m_s": [0.0, 0.0, 0.0]}
        )
        self.head = SimpleNamespace(
            forward_policy14=lambda *args, **kwargs: {
                "tof_flu": SimpleNamespace(
                    translation_m=[0.0, 0.0, 0.0], rotation=np.eye(3)
                )
            }
        )

    def estimate(self, rgb, o, detections, *, now_s, tof_frame=None):
        return {
            "timestamp_s": o["timestamp_s"],
            "image_sha256": "0" * 64,
            "height": {"conditional_root_height_m": 0.2},
            "pose": {"translation_m": [0.074, 0.0, 0.12]},
            "goal": (
                {
                    "accepted": True,
                    "geometry": {
                        "post_points_reference_m": [
                            [0.93, 0.1325, -0.2],
                            [0.93, -0.1325, -0.2],
                        ]
                    },
                }
                if self.goal_visible
                else None
            ),
            "ball": (
                {
                    "accepted": True,
                    "geometry": {"point_reference_m": [0.105, -0.035, -0.165]},
                }
                if self.ball_visible
                else None
            ),
        }


def observation(t):
    return {
        "timestamp_s": t,
        "joint_names": NAMES,
        "q_rad": [0.0] * 14,
        "qd_rad_s": [0.0] * 14,
        "gyro_body_rad_s": [0.0, 0.0, 0.0],
        "gravity_body": [0.0, 0.0, -1.0],
    }


def tof(t):
    return {
        "timestamp_s": t,
        "cells": [
            [
                {
                    "status": "no_return",
                    "range_m": None,
                    "ray_sensor_flu": [1.0, 0.0, 0.0],
                }
                for j in range(8)
            ]
            for i in range(8)
        ],
    }


class ControllerTests(unittest.TestCase):
    def make(self):
        nav = SimpleNamespace(
            query=lambda *args: (_ for _ in ()).throw(
                AssertionError("No global route needed for this fixture")
            )
        )
        n = BetaNavigator(nav)
        n.vision = SensorFixture()
        return n

    def run_step(self, n, t):
        return n.step(np.zeros((1, 1, 3), np.uint8), observation(t), tof(t), now_s=t)

    def test_three_observed_frames_before_localization(self):
        n = self.make()
        self.assertEqual(self.run_step(n, 0)["state"], "SEARCH_GOAL")
        self.assertEqual(self.run_step(n, 1 / 15)["state"], "SEARCH_GOAL")
        self.assertEqual(self.run_step(n, 2 / 15)["state"], "SETTLE")

    def test_half_second_settle_then_one_shot_request(self):
        n = self.make()
        rows = [self.run_step(n, i / 15) for i in range(20)]
        kick = [r for r in rows if r["kick_request"]]
        self.assertEqual(len(kick), 1)
        self.assertGreaterEqual(kick[0]["timestamp_s"], 2 / 15 + 0.5)
        self.assertTrue(all(r["football_success"] is None for r in rows))

    def test_lost_ball_stops_without_kick(self):
        n = self.make()
        for i in range(3):
            self.run_step(n, i / 15)
        n.vision.ball_visible = False
        out = None
        for i in range(3, 8):
            out = self.run_step(n, i / 15)
        self.assertEqual(out["state"], "SEARCH_BALL")
        self.assertEqual(out["velocity_body_xyw"], [0.0, 0.0, 0.0])
        self.assertFalse(out["kick_request"])

    def test_control_gap_forces_relocalization(self):
        n = self.make()
        for i in range(3):
            self.run_step(n, i / 15)
        out = self.run_step(n, 1.0)
        self.assertEqual(out["state"], "SEARCH_GOAL")
        self.assertIn("control_gap", out["reason"])
        self.assertIsNone(n.pose)

    def test_tof_capture_mismatch_stops(self):
        n = self.make()
        for i in range(3):
            self.run_step(n, i / 15)
        out = n.step(
            np.zeros((1, 1, 3), np.uint8), observation(0.2), tof(0.18), now_s=0.2
        )
        self.assertIn("acquisition-time", out["reason"])
        self.assertEqual(out["velocity_body_xyw"], [0.0, 0.0, 0.0])
        self.assertFalse(out["kick_request"])

    def test_tooclose_never_suppressed(self):
        n = self.make()
        n.pose = np.zeros(3)
        frame = tof(0.2)
        frame["cells"][0][0]["contains_too_close"] = True
        _, hazard = n._tof_sample(
            frame, observation(0.2), np.eye(3), 0.2, np.array([0.105, 0.0, -0.165])
        )
        self.assertTrue(hazard)


if __name__ == "__main__":
    unittest.main(verbosity=2)
