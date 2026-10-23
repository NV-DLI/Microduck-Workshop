"""Visible-goal angular centering, with no metric or base-motion authority."""

import math
import numpy as np
from gaze_tracking import GazeSlew
from image_head_servo import attention_correction as image_geometry
from bounded_image_head_servo import bounded_step
from command_navigation_native import NativeCommandNavigator


def visible_goal_attention(detections, *, camera, capture_s, now_s):
    if (
        not all(math.isfinite(x) and x >= 0 for x in (capture_s, now_s))
        or not 0 <= now_s - capture_s <= 0.2
    ):
        return None
    parameterss = []
    for row in detections or []:
        confidence = row.get("confidence", float("nan"))
        if (
            row.get("class_name") != "goal"
            or not math.isfinite(confidence)
            or not 0.45 <= confidence <= 1
        ):
            continue
        box = row.get("bbox_xyxy_px", [])
        if len(box) != 4 or not all(math.isfinite(x) for x in box):
            continue
        x0, y0, x1, y1 = box
        if not 0 <= x0 < x1 < camera.width_px or not 0 <= y0 < y1 < camera.height_px:
            continue
        parameterss.append(row)
    if not parameterss:
        return None
    row = max(parameterss, key=lambda x: x["confidence"])
    x0, y0, x1, y1 = row["bbox_xyxy_px"]
    return {
        "observed_visible_box_center_px": [(x0 + x1) / 2, (y0 + y1) / 2],
        "observed_bbox_xyxy_px": [x0, y0, x1, y1],
        "capture_timestamp_s": capture_s,
        "cropped": min(x0, y0, camera.width_px - 1 - x1, camera.height_px - 1 - y1) < 2,
        "metric_goal_position_estimated": False,
        "base_motion_authorized": False,
        "kick_authorized": False,
    }


class GoalAwareGazeSlew(GazeSlew):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def update(self, target, t):
        if (
            not math.isfinite(t)
            or t < 0
            or self.timestamp is not None
            and t < self.timestamp
        ):
            raise ValueError("Nonnegative monotone finite gaze time required")
        owner = self.owner
        diagnostic = None
        goal_mode = owner.state == "REFRESH_GOAL"
        if goal_mode:
            dt = 0.0 if self.timestamp is None else t - self.timestamp
            target = self.current.copy()
            data = owner.goal_attention_input
            if (
                owner.goal_attention_context_valid
                and owner.pose is not None
                and not owner.refresh_too_close
                and np.isfinite(owner.pose).all()
                and np.isfinite(owner.goal_map).all()
            ):
                # The existing public map/pose may aim attention before the
                # goal enters the camera. Its age does not grant metric,
                # body-motion or kick authority.
                relative = owner.goal_map - owner.pose[:2]
                angle = math.atan2(relative[1], relative[0]) - owner.pose[2]
                angle = (angle + math.pi) % (2 * math.pi) - math.pi
                target = np.array([np.clip(angle, -1.2, 1.2), 0.0])
                diagnostic = {
                    "mode": "public map/pose attention only, awaiting visible goal pixels",
                    "public_goal_bearing_body_rad": angle,
                    "metric_goal_position_estimated": False,
                    "base_motion_authorized": False,
                    "kick_authorized": False,
                    "truth_used": False,
                }
            if data is not None and not owner.refresh_too_close:
                attention, rotation = data
                measured = image_geometry(
                    attention,
                    command=self.current,
                    rotation_body_from_raw=rotation,
                    camera=owner.vision.camera,
                    now_s=t,
                    dt_s=dt,
                )
                if measured is not None:
                    lower = np.maximum(-0.45 * dt, np.array([-1.2, 0.0]) - self.current)
                    upper = np.minimum(0.45 * dt, np.array([1.2, 1.4]) - self.current)
                    delta, box = bounded_step(
                        measured["local_image_jacobian"],
                        np.array(measured["image_error_normalized"]) * 2 * dt,
                        lower,
                        upper,
                    )
                    target = self.current + delta
                    diagnostic = {
                        **attention,
                        "image_error_normalized": measured["image_error_normalized"],
                        "local_image_jacobian": measured["local_image_jacobian"],
                        "bounded_step": box,
                        "increment_rad": delta.tolist(),
                        "old_command_rad": self.current.tolist(),
                        "parameters_command_rad": target.tolist(),
                        "mode": "fresh visible goal center; same-capture encoder FK",
                        "maximum_increment_rate_rad_s": 0.45,
                        "truth_used": False,
                    }
            raw = np.asarray(target, float)
            limited = np.clip(raw, [-1.2, 0.0], [1.2, 1.4])
            self.current += np.clip(
                limited - self.current, -self.rate * dt, self.rate * dt
            )
            self.timestamp = t
            self.last = {
                "requested_body_yaw_pitch_rad": raw.tolist(),
                "bounded_body_yaw_pitch_rad": limited.tolist(),
                "executed_body_yaw_pitch_rad": self.current.tolist(),
                "slew_limit_rad_s": self.rate,
                "limit_clipped": bool(np.any(raw != limited)),
                "world_stabilized": False,
                "gait_head_command_generalization_validated": False,
                "scope": "Goal-only parameters gaze envelope; final native-head FK and source-command gate remains required",
            }
            owner.goal_servo_diagnostic = diagnostic or {
                "mode": "hold current head; no fresh valid goal angular observation",
                "truth_used": False,
                "base_motion_authorized": False,
                "kick_authorized": False,
            }
            return self.current.tolist()
        owner.goal_servo_diagnostic = None
        return super().update(target, t)


class GoalReacquisitionNavigator(NativeCommandNavigator):
    def reset(self):
        super().reset()
        self.goal_attention_input = None
        self.goal_servo_diagnostic = None
        self.goal_attention_context_valid = False
        self.gaze = GoalAwareGazeSlew(self)

    def step(self, rgb, proprio, tof_frame, *, now_s, detections=None):
        self.goal_attention_input = None
        self.goal_attention_context_valid = bool(
            proprio is not None
            and math.isfinite(now_s)
            and math.isfinite(proprio["timestamp_s"])
            and 0 <= now_s - proprio["timestamp_s"] <= 0.2
        )
        if proprio is not None:
            attention = visible_goal_attention(
                detections,
                camera=self.vision.camera,
                capture_s=proprio["timestamp_s"],
                now_s=now_s,
            )
            if attention is not None:
                fk = self.vision.head.forward_policy14(
                    proprio["q_rad"], joint_names=proprio["joint_names"]
                )["camera_raw_optical"]
                self.goal_attention_input = (attention, fk.rotation)
        out = super().step(rgb, proprio, tof_frame, now_s=now_s, detections=detections)
        if out["state"] == "REFRESH_GOAL":
            self.native_commands.reset()
            out["velocity_body_xyw"] = [0.0, 0.0, 0.0]
            out["kick_request"] = False
            out["native_command_adapter"] = {
                "phase": "OBSERVE",
                "reason": "goal_image_reacquisition_head_only",
                "guard_zero_preserved": True,
                "truth_used": False,
            }
            out["native_goal_image_servo"] = self.goal_servo_diagnostic
            out["motion_authorization"] = {
                **out["motion_authorization"],
                "allowed": False,
                "output_reason": "goal_image_reacquisition_head_only",
            }
            out["legacy_gaze_domain_diagnostic"] = out.pop("trained_gaze_domain", None)
            out["native_goal_gaze_domain"] = {
                "parameters_yaw_rad": [-1.2, 1.2],
                "source_head_offsets_checked_by": "HeadCommandDomainAdapter",
                "learned_tracking_validated": False,
            }
        return out
