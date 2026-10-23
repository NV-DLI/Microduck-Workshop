"""Explicit motion authority for the complete visual-navigation output chain.

The yaw servo can correct an authorized motion request. It cannot turn a
sensor guard, preparation state or observation-only hold into a motion request.
"""

import numpy as np
from execute_observed_ball_turn import ObservedTurnNavigator
from measured_yaw_feedback import MeasuredYawFeedback
from perception import level_rotation


MOTION_REASONS = {
    "actual_Navfn_path": "NAV2_APPROACH",
    "fresh_RGB_position_error": "VISUAL_ALIGN",
    "observed_ball_direction_body_turn": "SEARCH_BALL",
    "sensor_pose_goal_reacquisition": "REFRESH_GOAL",
}
ORIGINAL_MOTION_REASONS = {
    "actual_Navfn_path",
    "fresh_RGB_position_error",
    "fresh_ball_required",
    "visual_localization_refresh",
}
UPSTREAM_HOLDS = {
    "stale_proprioception",
    "control_gap_requires_visual_relocalization",
    "actual_ToF_floor_unavailable",
    "actual_ToF_near_obstacle_or_too_close",
    "180s_teaching_limit",
    "local_segment_hits_surveyed_obstacle",
    "ball_goal_direction_degenerate",
    "zero_command_observe_real_settling",
    "visual_and_IMU_settled_parameters",
}


class GuardedFeedbackNavigator(ObservedTurnNavigator):
    motion_reasons = MOTION_REASONS
    original_motion_reasons = ORIGINAL_MOTION_REASONS

    def reset(self):
        super().reset()
        self.yaw_feedback = MeasuredYawFeedback()

    def _output(self, t, command=(0.0, 0.0, 0.0), reason="", gaze=None):
        original_reason = reason
        # Observing a nearby hazard must not silently request a new head pose.
        # Keep the actual command continuity; stale observations cannot retarget.
        if gaze is None and (
            reason in UPSTREAM_HOLDS or reason.startswith("Nav2_failed:")
        ):
            gaze = self.gaze.current.tolist()
        out = super()._output(t, command, reason, gaze)
        allowed = (
            original_reason in self.original_motion_reasons
            and self.motion_reasons.get(out["reason"]) == out["state"]
        )
        if self.refresh_too_close:
            allowed = False
        out["motion_authorization"] = {
            "allowed": allowed,
            "upstream_reason": original_reason,
            "output_reason": out["reason"],
            "state": out["state"],
            "authority": "explicit sensor-navigation mode; unknown modes hold",
        }
        if not allowed:
            out["velocity_body_xyw"] = [0.0, 0.0, 0.0]
            out["kick_request"] = False
        return out

    def step(self, rgb, proprio, tof_frame, **kwargs):
        out = super().step(rgb, proprio, tof_frame, **kwargs)
        geo = self.last_geometry
        valid = bool(
            out.get("motion_authorization", {}).get("allowed")
            and geo
            and geo["timestamp_s"] == proprio["timestamp_s"]
            and geo["floor_observation"]["accepted"]
            and not self.refresh_too_close
        )
        if not valid:
            self.yaw_feedback.reset()
            out["velocity_body_xyw"] = [0.0, 0.0, 0.0]
            out["kick_request"] = False
            out["measured_yaw_feedback"] = {
                "active": False,
                "reason": "motion_authority_denied",
                "integral_rad_s": 0.0,
            }
            return out
        omega = float(
            (
                level_rotation((-np.asarray(geo["ground"]["normal"])).tolist())
                @ np.asarray(proprio["gyro_body_rad_s"])
            )[2]
        )
        command, diagnostic = self.yaw_feedback.update(
            float(out["velocity_body_xyw"][2]),
            omega,
            timestamp=proprio["timestamp_s"],
            valid=True,
        )
        out["velocity_body_xyw"][2] = command
        out["measured_yaw_feedback"] = diagnostic
        return out
