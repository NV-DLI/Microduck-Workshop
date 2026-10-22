"""Head attention needs fresh public pose, not a valid floor or ball metric.

This permission is valid only inside the explicit zero-body attention owner.
It never changes TooClose, a measured range, body permission, or kick permission.
Actual50Hz native-head delivery retains its independent same-read packet gate.
"""
import math
import numpy as np


def head_only_permission(*, t, owner, proprio, geometry):
    diagnostic = {
        "accepted": False, "owner": owner, "capture_s": t,
        "body_authorized": False, "kick_authorized": False,
        "floor_or_ToF_body_permission_inferred": False, "truth_used": False,
    }
    if owner not in ("GOAL_REFRESH", "BALL_REACQUIRE"):
        return {**diagnostic, "reason": "no_active_zero_body_attention_owner"}
    try:
        timestamp = proprio["timestamp_s"]
        geometry_time = geometry["timestamp_s"]
        q = np.asarray(proprio["q_rad"], float)
        qd = np.asarray(proprio["qd_rad_s"], float)
        gyro = np.asarray(proprio["gyro_body_rad_s"], float)
        gravity = np.asarray(proprio["gravity_body"], float)
        names = proprio["joint_names"]
        clocks = (t, timestamp, geometry_time)
        valid = (all(not isinstance(x, bool) and math.isfinite(x) and x >= 0 for x in clocks)
                 and 0 <= t - timestamp <= .2 and geometry_time == t
                 and q.shape == qd.shape == (14,) and gyro.shape == gravity.shape == (3,)
                 and all(np.isfinite(x).all() for x in (q, qd, gyro, gravity))
                 and .9 <= np.linalg.norm(gravity) <= 1.1
                 and len(names) == len(set(names)) == 14)
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        return {**diagnostic, "reason": "invalid_or_stale_public_head_pose"}
    return {**diagnostic, "accepted": True,
            "reason": "fresh_public_head_pose_for_zero_body_attention_only",
            "proprio_capture_s": timestamp,
            "TooClose_releases_only_head_attention_not_body": True}
