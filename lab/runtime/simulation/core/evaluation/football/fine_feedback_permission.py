"""A narrow public permission for feedback after the original fine-step quiet."""

import math
import numpy as np


def fine_reacquire_feedback_allowed(
    *,
    now_s,
    capture_s,
    control,
    fine_status,
    handoff_state,
    head_motion_allowed,
    velocity,
    kick_requested
):
    """No body/kick authority; packet validity and original5s timer remain separate."""
    if (
        not control
        or not fine_status
        or not head_motion_allowed
        or handoff_state != "APPROACH"
    ):
        return False
    if fine_status.get("state") != "REACQUIRE" or kick_requested is not False:
        return False
    try:
        clocks = [
            now_s,
            capture_s,
            fine_status["started_s"],
            fine_status["step_end_s"],
            fine_status["quiet_started_s"],
        ]
        if not all(
            not isinstance(x, bool) and math.isfinite(x) and x >= 0 for x in clocks
        ):
            return False
        if not (
            0 <= now_s - capture_s <= 0.2
            and 0 <= now_s - fine_status["started_s"] < 24
            and fine_status["started_s"]
            <= fine_status["step_end_s"]
            <= fine_status["quiet_started_s"]
            and now_s - fine_status["quiet_started_s"] >= 0.5 - 1e-9
        ):
            return False
        v = np.asarray(velocity, float)
        if v.shape != (3,) or not np.isfinite(v).all() or np.any(v):
            return False
        permission = control.get("public_attention_head_permission") or {}
        owner = (control.get("public_attention_owner") or {}).get("owner")
        return bool(
            control.get("state") == "NEUTRAL_FINE_REACQUIRE"
            and owner in ("GOAL_REFRESH", "BALL_REACQUIRE")
            and permission.get("accepted") is True
            and permission.get("owner") == owner
            and permission.get("capture_s") == capture_s
            and permission.get("body_authorized") is False
            and permission.get("kick_authorized") is False
            and control.get("velocity_body_xyw") == [0.0, 0.0, 0.0]
            and control.get("kick_request") is False
            and fine_status.get("velocity") == [0.0, 0.0, 0.0]
            and fine_status.get("kick_authorized") is False
        )
    except (KeyError, ValueError, TypeError):
        return False
