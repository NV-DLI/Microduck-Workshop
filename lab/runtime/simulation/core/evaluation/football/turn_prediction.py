"""Public yaw-rate hint for stopping; never a motion authorization."""

import math


def near_turn_stop_hint(bearing_rad, gyro_level_z_rad_s, direction):
    values = (bearing_rad, gyro_level_z_rad_s, direction)
    if not all(math.isfinite(float(v)) for v in values) or direction not in (-1.0, 1.0):
        raise ValueError(
            "Finite public bearing/rate and an active signed turn required"
        )
    horizon = 0.15
    predicted = float(bearing_rad) - horizon * float(gyro_level_z_rad_s)
    toward = float(gyro_level_z_rad_s) * direction > 0.12
    stop = (
        toward and float(bearing_rad) * direction > 0 and predicted * direction <= 0.06
    )
    return {
        "current_bearing_rad": float(bearing_rad),
        "measured_level_yaw_rate_rad_s": float(gyro_level_z_rad_s),
        "turn_direction": float(direction),
        "horizon_s": horizon,
        "predicted_bearing_rad": predicted,
        "signed_remaining_target_rad": 0.06,
        "minimum_toward_rate_rad_s": 0.12,
        "stop_hint": bool(stop),
        "translation_coupling_modelled": False,
        "future_bearing_is_guaranteed": False,
        "motion_authorized": False,
        "truth_used": False,
    }
