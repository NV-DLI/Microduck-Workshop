"""Fine kick: Microduck simulation and control."""

import math
import sys
from pathlib import Path

import numpy as np

V64 = Path(__file__).resolve().parent
for p in (
    str(V64),
    str(V64.parent),
    str(V64.parents[1] / "football"),
    str(V64.parents[1] / "sensors"),
):
    if p not in sys.path:
        sys.path.append(p)
from neutral_fine_step import NeutralFineStep  # noqa: E402  (frozen v64)
from head_handoff import HeadHandoff  # noqa: E402  (frozen v64)

ENTRY_GOAL_HEADING = (
    0.065  # measured quiet required separately; final K10 gate remains .08
)
DEAD_ZONE_X = (
    0.142,
    0.177,
)  # advisory 3-sample model (mean STEP+settle dx -.037): .105+.037 / .14+.037; landing in blind band outside box
OBSERVABLE_X = 0.14  # 0/161 hazards over [.14,.22) in v63+v64 (limited evidence)
CAPTURE_MAX_AGE_S = 0.2
NATIVE_HEAD_BOUNDS = np.array([1.1, 1.1, 1.4, 0.31])


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def standoff_entry(
    *,
    t,
    handoff_state,
    fine_active,
    fine_state,
    final_diag,
    velocity_body_xyw,
    complete,
    guard_safe,
    memory,
    goal_heading,
    goal_fix_s,
    actual_head
):
    """All args are the executor locals at the fine-trigger line. Returns (trigger, reason)."""
    fd = final_diag or {}
    mem = memory if isinstance(memory, dict) else {}
    checks = [
        ("final_phase_observe", fd.get("phase") == "OBSERVE"),
        (
            "exact_zero_body",
            isinstance(velocity_body_xyw, (list, tuple))
            and len(velocity_body_xyw) == 3
            and all(_num(v) and v == 0 for v in velocity_body_xyw),
        ),
        ("handoff_approach", handoff_state == "APPROACH"),
        ("fine_idle", fine_active is False and fine_state == "IDLE"),
        ("complete_fresh_ball", complete is True),
        ("body_guard_safe", guard_safe is True),
        (
            "memory_anchored_now",
            mem.get("memory_available") is True
            and _num(t)
            and mem.get("ball_capture_timestamp_s") == t,
        ),
    ]
    for name, ok in checks:
        if not ok:
            return False, "rejected:" + name
    try:
        p = np.asarray(mem.get("point_level_m"), float)
        h = np.asarray(actual_head, float)
    except (TypeError, ValueError):
        return False, "rejected:nonfinite_input"
    if (
        p.shape != (3,)
        or not np.isfinite(p).all()
        or h.shape != (4,)
        or not np.isfinite(h).all()
        or not _num(goal_heading)
        or not _num(goal_fix_s)
    ):
        return False, "rejected:nonfinite_input"
    rest = [
        (
            "original_domain",
            0.13 <= p[0] and np.linalg.norm(p[:2]) <= 0.22 and -0.10 <= p[1] <= 0.08,
        ),
        ("goal_fix_age_0_to_8s", 0 <= t - goal_fix_s <= 8),
        (
            "waypoint_bearing_le_0p15",
            abs(NeutralFineStep.waypoint_bearing(p, goal_heading)) <= 0.15,
        ),
        ("native_head_in_bounds", bool(np.all(np.abs(h) <= NATIVE_HEAD_BOUNDS + 1e-9))),
        ("goal_heading_le_public_margin", abs(goal_heading) <= ENTRY_GOAL_HEADING),
        (
            "not_in_model_dead_zone",
            not DEAD_ZONE_X[0] < p[0] < DEAD_ZONE_X[1]
            or (abs(p[1] + 0.035) <= 0.008 and max(abs(h)) <= 0.75),
        ),
    ]
    for name, ok in rest:
        if not ok:
            return False, "rejected:" + name
    return True, "standoff_entry"


class FineStepV65(NeutralFineStep):
    def release_to_kick_handoff(self, t):
        if self.state != "SETTLE":
            raise ValueError("ownership release only at the STEP->SETTLE boundary")
        self.state = "IDLE"
        self.reason = "released_to_kick_handoff"
        self.last_finished = t
        self.quiet_start = None


class HeadHandoffV65(HeadHandoff):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.latest_capture = None
        self.routed_from_fine = False

    def note_capture(self, *, capture_s, guard_safe, fresh_floor):
        """Executor records every capture's public body guard (fine_guard) here."""
        self.latest_capture = {
            "capture_s": capture_s,
            "guard_safe": bool(guard_safe is True and fresh_floor is True),
        }

    def enter_from_fine(self, *, now_s, goal_heading, goal_fix_s):
        if self.state != "APPROACH":
            raise ValueError("kick handoff already owned")
        self.head = np.zeros(4)
        self.started = now_s
        self.last_t = now_s
        self.initial_capture_s = None
        self.quiet_start = None
        self.routed_from_fine = True
        self.attempts += 1
        self.state = "HEAD_RETURN"
        if _num(goal_heading) and _num(goal_fix_s) and 0 <= now_s - goal_fix_s <= 0.5:
            self.last_goal_heading = float(goal_heading)
            self.last_goal_s = float(goal_fix_s)

    def policy(self, *, now_s, packet):
        before = self.state
        out = super().policy(now_s=now_s, packet=packet)
        if before != "KICK" and self.state == "KICK":
            c = self.latest_capture
            if (
                c is None
                or not _num(c.get("capture_s"))
                or not 0 <= now_s - c["capture_s"] <= CAPTURE_MAX_AGE_S
            ):
                self.state = "FAILED"
                self.reason = "kick_requires_fresh_public_capture"
                self.kick_start = None
            elif c["guard_safe"] is not True:
                self.state = "FAILED"
                self.reason = "kick_requires_body_guard_safe"
                self.kick_start = None
            if self.state == "FAILED":
                out = self.status(now_s)
        return out


def route_at_step_end(*, now_s, fine, handoff, memory, goal_heading, goal_fix_s):
    """Call on the policy tick where fine.state changed STEP->SETTLE. Returns the decision.
    Fails closed: anything not clearly routable keeps the existing v64 fine path (body stays zero).
    """
    mem = memory if isinstance(memory, dict) else {}
    p = mem.get("point_level_m")
    c = handoff.latest_capture
    if getattr(fine, "coarse_track", False):
        return "keep:coarse_requires_fresh_reacquisition"
    if fine.reason in (
        "public_guard_stop_requires_fresh_reacquisition",
        "public_goal_stop_requires_fresh_reacquisition",
    ):
        return "keep:guard_stop_requires_fresh_reacquisition"
    if fine.state != "SETTLE" or handoff.state != "APPROACH":
        return "keep:not_at_step_end_or_handoff_owned"
    if (
        mem.get("memory_available") is not True
        or p is None
        or not np.isfinite(np.asarray(p, float)).all()
    ):
        return "keep:memory_invalid"
    if (
        c is None
        or not _num(c.get("capture_s"))
        or not 0 <= now_s - c["capture_s"] <= CAPTURE_MAX_AGE_S
    ):
        return "keep:stale_capture"
    if c["guard_safe"] is not True:
        return "keep:body_guard_unsafe"
    if not _num(goal_heading):
        return "keep:goal_heading_nonfinite"
    if p[0] >= OBSERVABLE_X:
        return "keep:observable_reacquire"
    fine.release_to_kick_handoff(now_s)
    handoff.enter_from_fine(
        now_s=now_s, goal_heading=goal_heading, goal_fix_s=goal_fix_s
    )
    return "route_to_kick_handoff"
