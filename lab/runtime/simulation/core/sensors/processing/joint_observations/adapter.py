"""Causal capture-pose estimate from one actual quantized BODY50 read.

The pose timestamp is the time being estimated, never a replacement for the
hardware read timestamp. All original times remain in returned provenance.
No actual sensor, robot or image state is overwritten by this pure function.
"""

import hashlib
import json
import math
import numpy as np

JOINT_NAMES = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)


def _time(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean timestamp")
    t = float(value)
    if not math.isfinite(t) or t < 0:
        raise ValueError("Finite nonnegative time required")
    return t


def _vector(value, n):
    v = np.asarray(value, dtype=np.float64)
    if v.shape != (n,) or not np.isfinite(v).all():
        raise ValueError("Malformed finite sensor vector")
    return v


def estimate_capture_proprio(
    packet, *, available_at_s, capture_timestamp_s, now_s, joint_names
):
    """Use only a read already available at capture; extrapolate by at most20ms.

    Caller selects the last delivered public read available before capture and
    retains its immutable source packet. A later read is rejected, even if its
    acquisition time precedes the camera timestamp.
    """
    rejected = {
        "accepted": False,
        "reason": "invalid_public_capture_input",
        "observation": None,
        "kick_authorized": False,
        "truth_used": False,
        "uncertainty_calibrated": False,
    }
    try:
        if tuple(joint_names) != JOINT_NAMES:
            raise ValueError("Declared native14 encoder order required")
        if (
            packet["source"] != "body_powerboard_control_imu"
            or packet["axes"] != "trunk FLU"
            or packet["gyro_unit"] != "rad/s"
        ):
            raise ValueError("Actual FLU body sensor source required")
        if type(packet["seq"]) is not int or packet["seq"] < 0:
            raise ValueError("Integral sequence required")
        if packet["orientation_valid"] is not True:
            return {**rejected, "reason": "orientation_unready"}
        sample, read, encoder, delivery, available, capture, now = map(
            _time,
            (
                packet["timestamp_s"],
                packet["read_timestamp_s"],
                packet["encoder_read_timestamp_s"],
                packet["delivered_timestamp_s"],
                available_at_s,
                capture_timestamp_s,
                now_s,
            ),
        )
        if (
            not sample <= read + 1e-9
            or abs(read - encoder) > 1e-7
            or not read <= delivery + 1e-9
            or not delivery <= available + 1e-9
            or not available <= capture + 1e-9
            or not capture <= now + 1e-9
        ):
            return {**rejected, "reason": "not_same_read_causal_available_capture"}
        age = capture - read
        if not 0 <= age <= 0.02000001 or not 0 <= now - capture <= 0.2:
            return {**rejected, "reason": "stale_read_or_capture"}
        q = _vector(packet["encoder_q_rad"], 14)
        qd = _vector(packet["encoder_qd_rad_s"], 14)
        gyro = _vector(packet["gyro"], 3)
        gravity = _vector(packet["gravity"], 3)
        if not np.array_equal(
            gyro, _vector(packet["gyro_body_rad_s"], 3)
        ) or not np.array_equal(gravity, _vector(packet["gravity_body"], 3)):
            return {**rejected, "reason": "ambiguous_body_sensor_aliases"}
        if not 0.5 < float(np.linalg.norm(gravity)) < 1.5:
            return {**rejected, "reason": "invalid_gravity_norm"}
        estimate = q + qd * age
        provenance = {
            "seq": packet["seq"],
            "sample_timestamp_s": sample,
            "read_timestamp_s": read,
            "encoder_read_timestamp_s": encoder,
            "delivered_timestamp_s": delivery,
            "available_at_s": available,
            "capture_estimate_timestamp_s": capture,
            "evaluated_at_s": now,
            "prediction_seconds": age,
            "raw_encoder_q_rad": q.tolist(),
            "raw_encoder_qd_rad_s": qd.tolist(),
            "actual_gyro_body_rad_s": gyro.tolist(),
            "actual_gravity_body": gravity.tolist(),
            "method": "q_read + qd_read * (capture-read); hold measured gyro/gravity; no future read",
            "capture_timestamp_is_estimate_not_new_hardware_read": True,
        }
        provenance["public_input_sha256"] = hashlib.sha256(
            json.dumps(provenance, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        observation = {
            "timestamp_s": capture,
            "joint_names": list(JOINT_NAMES),
            "q_rad": estimate.tolist(),
            "qd_rad_s": qd.tolist(),
            "gyro_body_rad_s": gyro.tolist(),
            "gravity_body": gravity.tolist(),
        }
        return {
            **rejected,
            "accepted": True,
            "reason": "causal_quantized_read_pose_prediction",
            "observation": observation,
            "provenance": provenance,
        }
    except (KeyError, TypeError, ValueError, OverflowError):
        return rejected
