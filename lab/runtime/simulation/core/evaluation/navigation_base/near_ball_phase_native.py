"""Native-reference near-ball goal memory and declared K8 strike-center geometry."""

import math
import numpy as np
from guarded_navigation_feedback import (
    GuardedFeedbackNavigator,
    MOTION_REASONS,
    ORIGINAL_MOTION_REASONS,
    UPSTREAM_HOLDS,
)
from perception import level_rotation
from visual_localization import rotate, wrap


class NearBallNavigator(GuardedFeedbackNavigator):
    motion_reasons = {
        **MOTION_REASONS,
        "near_goal_heading_alignment": "NEAR_ALIGN",
        "near_preparation_heading_alignment": "NEAR_ALIGN",
        "near_ball_position_alignment": "NEAR_ALIGN",
    }
    original_motion_reasons = ORIGINAL_MOTION_REASONS | {
        "near_goal_heading_alignment",
        "near_preparation_heading_alignment",
        "near_ball_position_alignment",
    }

    def reset(self):
        super().reset()
        self.current_sensor_pair = None

    def public_near_context(self, t):
        """One metric predicate shared by near geometry, attention and adapter.

        A cropped frame may retain the old public map predicate for head
        attention only. It still cannot supply fresh metric motion authority.
        """
        if self.pose is None or self.ball is None or self.current_sensor_pair is None:
            return False
        geo = self.last_geometry
        ball = geo.get("ball") if geo else None
        if (
            geo
            and geo.get("timestamp_s") == t
            and ball
            and ball.get("accepted")
            and geo.get("floor_observation", {}).get("accepted")
        ):
            R = level_rotation((-np.asarray(geo["ground"]["normal"])).tolist())
            point = R @ np.asarray(ball["geometry"]["point_reference_m"], float)
            uncertainty = float(ball["geometry"]["estimated_range_uncertainty_m"])
            return bool(
                np.isfinite(point).all()
                and math.isfinite(uncertainty)
                and uncertainty >= 0
                and np.linalg.norm(point[:2]) - uncertainty <= 0.34
            )
        return bool(np.linalg.norm(self.ball - self.pose[:2]) <= 0.34)

    def step(self, rgb, proprio, tof_frame, **kwargs):
        self.current_sensor_pair = (proprio, tof_frame)
        return super().step(rgb, proprio, tof_frame, **kwargs)

    def _output(self, t, command=(0.0, 0.0, 0.0), reason="", gaze=None):
        near = self.public_near_context(t)
        goal_age = t - self.goal_time
        use = (
            near
            and 0 <= goal_age <= 8.0
            and reason
            in {
                "visual_localization_refresh",
                "fresh_ball_required",
                "fresh_RGB_position_error",
                "actual_Navfn_path",
            }
            and (reason != "actual_Navfn_path" or np.any(command))
        )
        diagnostic = None
        attention = self.ball_attention
        if use:
            diagnostic = {
                "mode": "near_ball_observe_align",
                "goal_fix_age_s": goal_age,
                "maximum_goal_fix_age_s": 8.0,
                "ball_metric_maximum_age_s": 0.2,
                "goal_map_from": "actual stable goal-base pixel observations plus public encoder/IMU odometry",
                "goal_lifetime_is_validated": False,
                "kick_authorized": False,
                "engineering_assumption": "Prospective bounded short-term goal memory; audit drift before qualification",
            }
            if (
                attention is not None
                and 0 <= t - attention["capture_timestamp_s"] <= 0.2
            ):
                gaze = attention["gaze_body_yaw_pitch_rad"]
            else:
                gaze = self.last_observed_ball_gaze or self.gaze.current.tolist()
            geo = self.last_geometry
            ball = geo.get("ball") if geo is not None else None
            fresh = bool(
                geo
                and geo["timestamp_s"] == t
                and ball
                and ball["accepted"]
                and t - self.ball_time <= 0.2
            )
            if not fresh:
                self.state = "NEAR_OBSERVE"
                command = (0.0, 0.0, 0.0)
                reason = "near_ball_fresh_metric_required"
            else:
                R = level_rotation((-np.asarray(geo["ground"]["normal"])).tolist())
                point = R @ np.asarray(ball["geometry"]["point_reference_m"])
                proprio, tof = self.current_sensor_pair
                try:
                    _, hazard = self._tof_sample(
                        tof,
                        proprio,
                        R,
                        geo["height"]["conditional_root_height_m"],
                        point,
                    )
                except (ValueError, KeyError):
                    hazard = True
                if hazard:
                    self.state = "NEAR_OBSERVE"
                    command = (0.0, 0.0, 0.0)
                    reason = "actual_ToF_near_obstacle_or_too_close"
                else:
                    shot = self.goal_map - self.ball
                    unit = shot / max(float(np.linalg.norm(shot)), 1e-9)
                    heading_error = wrap(math.atan2(unit[1], unit[0]) - self.pose[2])
                    side = np.array([-unit[1], unit[0]])
                    target = self.ball - 0.09 * unit + 0.035 * side
                    error = np.asarray(rotate(-self.pose[2], target - self.pose[:2]))
                    translation_heading = math.atan2(float(error[1]), float(error[0]))
                    diagnostic.update(
                        heading_error_rad=heading_error,
                        relative_position_error_m=error.tolist(),
                        translation_heading_error_rad=translation_heading,
                        body_heading_owner="public_preparation_waypoint",
                        final_goal_heading_alignment_armed=False,
                    )
                    self.state = "NEAR_ALIGN"
                    if abs(translation_heading) > 0.10:
                        command = (
                            0.0,
                            0.0,
                            float(np.clip(translation_heading, -0.18, 0.18)),
                        )
                        reason = "near_preparation_heading_alignment"
                    elif abs(error[0]) >= 0.012 or abs(error[1]) >= 0.008:
                        command = (
                            float(np.clip(0.7 * error[0], -0.03, 0.03)),
                            float(np.clip(0.7 * error[1], -0.035, 0.035)),
                            float(np.clip(heading_error, -0.12, 0.12)),
                        )
                        reason = "near_ball_position_alignment"
                    else:
                        self.state = "SETTLE"
                        command = (0.0, 0.0, 0.0)
                        reason = "zero_command_observe_real_settling"
            # During position approach, the public preparation point owns body
            # heading. Pixel attention owns the head; final aiming is unarmed.
            self.ball_attention = None
        try:
            out = super()._output(t, command, reason, gaze)
        finally:
            self.ball_attention = attention
        if diagnostic is not None:
            out["near_ball_phase"] = diagnostic
            out["kick_request"] = False
        return out
