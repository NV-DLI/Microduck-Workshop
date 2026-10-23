#!/usr/bin/env python3
"""Angular navigation native: Microduck simulation and control."""
from pathlib import Path
import numpy as np
from image_attention_native import ImageHeadNavigator
from tof_ball_association import associate


class AngularBallNavigator(ImageHeadNavigator):
    def reset(self):
        super().reset()
        self.last_tof_association = None

    def _tof_sample(self, frame, o, R, height, observed_ball_level_xyz=None):
        sample, _ = super()._tof_sample(frame, o, R, height, None)
        t = frame["timestamp_s"]
        geo = self.last_geometry
        ball = (
            geo.get("ball")
            if geo and geo["timestamp_s"] == t and observed_ball_level_xyz is not None
            else None
        )
        fresh = bool(ball and ball["accepted"])
        fk = self.vision.head.forward_policy14(
            o["q_rad"], joint_names=o["joint_names"]
        )["tof_flu"]
        rotation = np.asarray(fk.rotation)
        origin = np.asarray(fk.translation_m)
        observed = np.asarray(ball["geometry"]["point_reference_m"]) if fresh else None
        direction = (
            None
            if observed is None
            else observed - np.asarray(geo["pose"]["translation_m"])
        )
        if direction is not None:
            direction /= np.linalg.norm(direction)
        decisions = []
        hazard = False
        for cell in sum(frame["cells"], []):
            if cell.get("contains_too_close", False):
                hazard = True
                decisions.append(
                    {
                        "zone": [cell["row"], cell["col"]],
                        "associated": False,
                        "reason": "too_close",
                    }
                )
                continue
            if cell["status"] != "valid" or cell["range_m"] is None:
                continue
            body = origin + rotation @ (
                np.asarray(cell["ray_sensor_flu"]) * cell["range_m"]
            )
            point = R @ body
            if not (
                0.04 < point[2] + height < 0.45
                and 0 < point[0] < 0.105
                and abs(point[1]) < 0.09
            ):
                continue
            result = {
                "associated": False,
                "reason": "fresh_complete_metric_ball_unavailable",
            }
            if fresh:
                result = associate(
                    cell,
                    center_tof=rotation.T @ (observed - origin),
                    camera_to_ball_unit_tof=rotation.T @ direction,
                    range_uncertainty_m=ball["geometry"][
                        "estimated_range_uncertainty_m"
                    ],
                    timestamp_s=t,
                    ball_timestamp_s=ball["geometry"]["capture_timestamp_s"],
                    hfov=frame["profile"]["horizontal_fov_deg"],
                    vfov=frame["profile"]["vertical_fov_deg"],
                )
            result["zone"] = [cell["row"], cell["col"]]
            decisions.append(result)
            if not result["associated"]:
                hazard = True
        self.last_tof_association = {
            "timestamp_s": t,
            "hazard": hazard,
            "fresh_metric_ball": fresh,
            "cells": decisions,
            "semantic_exception": "Only the nearest range associated to fresh RGB sphere; no free-space claim for the angular cell",
            "truth_used": False,
            "surrounding_obstacle_guard_unchanged": True,
        }
        return sample, hazard

    def _output(self, t, command=(0.0, 0.0, 0.0), reason="", gaze=None):
        out = super()._output(t, command, reason, gaze)
        if (
            self.last_tof_association is not None
            and self.last_tof_association["timestamp_s"] == t
        ):
            out["tof_ball_association"] = self.last_tof_association
        return out
