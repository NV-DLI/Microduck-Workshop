"""Head-only image feedback using an uncalibrated ideal-axes approximation.

The measured encoder rotation fixes the current optical frame. The yaw/pitch
command derivative still uses ideal body axes; actual joint tracking and camera
translation are not reproduced by this local Jacobian. Partial visible box
centres authorize attention only, never metric position, base motion or kick.
"""

import math
import numpy as np
from beta_perception_geometry import raw_pixel_to_unit_ray


def attention_correction(
    attention,
    *,
    command,
    rotation_body_from_raw,
    camera,
    now_s,
    dt_s,
    gain=2.0,
    maximum_rate=0.45
):
    if attention is None or not 0 <= now_s - attention["capture_timestamp_s"] <= 0.2:
        return None
    if not 0 < dt_s <= 0.2:
        return None
    command = np.asarray(command, float)
    R = np.asarray(rotation_body_from_raw, float)
    if (
        command.shape != (2,)
        or R.shape != (3, 3)
        or not np.isfinite(command).all()
        or not np.isfinite(R).all()
    ):
        raise ValueError(
            "Finite current command and same-capture encoder rotation required"
        )
    pixel = attention["observed_visible_box_center_px"]
    local = np.asarray(raw_pixel_to_unit_ray(pixel, camera))
    ray = R @ local
    if local[2] <= 0.1:
        raise ValueError("Forward observed image ray required")
    # Camera angular sensitivity: yaw about body Z; positive down-pitch
    # about the horizontal left axis at the current commanded yaw.
    yaw = float(command[0])
    axes = np.array([[0.0, 0.0, 1.0], [-math.sin(yaw), math.cos(yaw), 0.0]])
    derivatives = np.stack([-R.T @ np.cross(axis, ray) for axis in axes], axis=1)
    J = (derivatives[:2] * local[2] - local[:2, None] * derivatives[2]) / local[2] ** 2
    error = local[:2] / local[2]
    delta = -np.linalg.solve(J.T @ J + np.eye(2) * 0.025, J.T @ error) * gain * dt_s
    delta = np.clip(delta, -maximum_rate * dt_s, maximum_rate * dt_s)
    desired = np.clip(command + delta, [-0.4, 0.0], [0.4, 1.4])
    return {
        "gaze_body_yaw_pitch_rad": desired.tolist(),
        "capture_timestamp_s": attention["capture_timestamp_s"],
        "observed_visible_box_center_px": list(pixel),
        "image_error_normalized": error.tolist(),
        "local_image_jacobian": J.tolist(),
        "unclipped_increment_rad": delta.tolist(),
        "old_command_rad": command.tolist(),
        "cropped": attention["cropped"],
        "maximum_command_rate_rad_s": maximum_rate,
        "control": "incremental visible-pixel error with same-time encoder rotation",
        "distance_estimated": False,
        "map_ball_update_allowed": False,
        "base_motion_authorized": False,
        "kick_authorized": False,
        "uncalibrated_local_jacobian": True,
    }
