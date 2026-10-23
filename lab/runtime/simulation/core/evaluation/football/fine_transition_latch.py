"""Fine transition latch: Microduck simulation and control."""

import math

import numpy as np

from neutral_fine_step import NeutralFineStep

ORIGINAL_REASON = "final_public_forward"
TRANSITION_REASON = "final_close_range_transition_stop"
NEXT_CAPTURE_MAX_GAP_S = 0.08
NATIVE_HEAD_BOUNDS = np.array([1.1, 1.1, 1.4, 0.31])
SCHEMA = "microduck.fine-transition-latch.v2"


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def original_trigger(*, handoff_state, fine_active, final_diag):
    """Exact v63 executor.py:514 expression."""
    final_diag = final_diag or {}
    return bool(
        handoff_state == "APPROACH"
        and not fine_active
        and final_diag.get("reason") == ORIGINAL_REASON
        and final_diag.get("forward_command") == 0.2
    )


def _ordered(checks, name, ok):
    checks[name] = bool(ok)
    return bool(ok)


def _common_state(
    checks,
    *,
    handoff_state,
    fine_active,
    fine_state,
    velocity_body_xyw,
    complete,
    guard_safe,
    memory,
    t,
):
    ok = _ordered(
        checks,
        "exact_zero_body_command",
        isinstance(velocity_body_xyw, (list, tuple))
        and len(velocity_body_xyw) == 3
        and all(_num(v) and v == 0 for v in velocity_body_xyw),
    )
    ok = ok and _ordered(checks, "handoff_approach", handoff_state == "APPROACH")
    ok = ok and _ordered(
        checks, "fine_idle_not_active", fine_active is False and fine_state == "IDLE"
    )
    ok = ok and _ordered(checks, "complete_same_capture_ball", complete is True)
    ok = ok and _ordered(checks, "body_guard_safe", guard_safe is True)
    ok = ok and _ordered(
        checks,
        "memory_anchored_this_capture",
        isinstance(memory, dict)
        and memory.get("memory_available") is True
        and memory.get("ball_capture_timestamp_s") == t,
    )
    return ok


def _entry(checks, *, memory, goal_heading, goal_fix_s, actual_head, t):
    """NeutralFineStep.capture IDLE-entry validity on the CURRENT capture."""
    try:
        p = np.asarray(memory.get("point_level_m"), float)
    except (TypeError, ValueError):
        p = np.full(1, np.nan)
    ok = _ordered(
        checks, "memory_point_finite", p.shape == (3,) and np.isfinite(p).all()
    )
    ok = ok and _ordered(
        checks,
        "original_entry_domain",
        0.13 <= p[0] and np.linalg.norm(p[:2]) <= 0.22 and -0.10 <= p[1] <= 0.08,
    )
    ok = ok and _ordered(
        checks, "goal_fix_age_0_to_8s", _num(goal_fix_s) and 0 <= t - goal_fix_s <= 8
    )
    ok = ok and _ordered(checks, "goal_heading_finite", _num(goal_heading))
    bearing = NeutralFineStep.waypoint_bearing(p, goal_heading) if ok else None
    ok = ok and _ordered(
        checks,
        "waypoint_bearing_le_0p15",
        bearing is not None and math.isfinite(bearing) and abs(bearing) <= 0.15,
    )
    if ok:
        h = (
            np.asarray(actual_head, float)
            if isinstance(actual_head, (list, tuple, np.ndarray))
            else np.full(1, np.nan)
        )
    ok = ok and _ordered(
        checks,
        "native_head_finite_in_bounds",
        h.shape == (4,)
        and np.isfinite(h).all()
        and np.all(np.abs(h) <= NATIVE_HEAD_BOUNDS + 1e-9),
    )
    return ok, (p.tolist() if p.shape == (3,) else None), bearing


def _arm_inputs_finite(checks, *, memory, goal_heading, goal_fix_s, actual_head):
    try:
        p = np.asarray(memory.get("point_level_m"), float)
        h = np.asarray(actual_head, float)
    except (TypeError, ValueError):
        return _ordered(checks, "finite_public_inputs", False)
    return _ordered(
        checks,
        "finite_public_inputs",
        p.shape == (3,)
        and np.isfinite(p).all()
        and h.shape == (4,)
        and np.isfinite(h).all()
        and _num(goal_heading)
        and _num(goal_fix_s),
    )


class FineTransitionLatch:
    """Call evaluate() exactly once on every controller capture; cancel() on any capture
    that cannot evaluate (exception path). Holds at most one pending opportunity."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pending = None
        self.last_t = None
        self.last_index = None

    def cancel(self, *, capture_index=None, t=None, reason):
        dropped = self.pending
        self.pending = None
        return {
            "schema": SCHEMA,
            "cancelled_pending": dropped,
            "reason": "cancelled:" + str(reason),
            "capture_index": capture_index,
            "timestamp_s": t,
        }

    def evaluate(
        self,
        *,
        capture_index,
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
        actual_head,
    ):
        final_diag = final_diag or {}
        orig = original_trigger(
            handoff_state=handoff_state, fine_active=fine_active, final_diag=final_diag
        )
        pending, self.pending = (
            self.pending,
            None,
        )  # consumed on this capture, whatever happens
        out = {
            "schema": SCHEMA,
            "capture_index": capture_index,
            "timestamp_s": t,
            "original_trigger": orig,
            "consumed_pending": pending,
            "armed": None,
            "body_authorized": False,
            "kick_authorized": False,
            "feedback_bias_used": False,
            "truth_used": False,
        }
        checks = {}
        monotone = (
            isinstance(capture_index, int)
            and not isinstance(capture_index, bool)
            and capture_index >= 0
            and _num(t)
            and t >= 0
            and (self.last_index is None or capture_index > self.last_index)
            and (self.last_t is None or t > self.last_t)
        )
        _ordered(checks, "capture_index_and_time_strictly_increasing", monotone)
        if monotone:
            self.last_index, self.last_t = capture_index, t
        if orig:
            return {
                **out,
                "trigger": True,
                "kind": "original_0p2_forward_pulse",
                "reason": "original_v63_trigger_unchanged",
                "checks": checks,
            }
        if not monotone:
            return {
                **out,
                "trigger": False,
                "kind": None,
                "reason": "rejected:capture_index_and_time_strictly_increasing",
                "checks": checks,
            }
        reason = final_diag.get("reason")
        if reason == TRANSITION_REASON:
            ok = _ordered(
                checks,
                "stop_came_from_0p3_pulse_at_0p22",
                final_diag.get("forward_command") == 0.3
                and final_diag.get("close_range_threshold_m") == 0.22,
            )
            ok = ok and _common_state(
                checks,
                handoff_state=handoff_state,
                fine_active=fine_active,
                fine_state=fine_state,
                velocity_body_xyw=velocity_body_xyw,
                complete=complete,
                guard_safe=guard_safe,
                memory=memory,
                t=t,
            )
            ok = ok and _arm_inputs_finite(
                checks,
                memory=memory,
                goal_heading=goal_heading,
                goal_fix_s=goal_fix_s,
                actual_head=actual_head,
            )
            if not ok:
                failed = next(k for k, v in checks.items() if not v)
                return {
                    **out,
                    "trigger": False,
                    "kind": None,
                    "reason": f"transition_not_armed:{failed}",
                    "checks": checks,
                }
            entry_checks = {}
            entry, p, bearing = _entry(
                entry_checks,
                memory=memory,
                goal_heading=goal_heading,
                goal_fix_s=goal_fix_s,
                actual_head=actual_head,
                t=t,
            )
            checks.update({"entry:" + k: v for k, v in entry_checks.items()})
            if entry:
                return {
                    **out,
                    "trigger": True,
                    "kind": "close_range_transition_immediate",
                    "checks": checks,
                    "reason": "transition_capture_meets_all_entry_predicates",
                    "memory_point_level_m": p,
                    "waypoint_bearing_rad": bearing,
                }
            self.pending = {"capture_index": capture_index, "timestamp_s": t}
            failed = next(k for k, v in entry_checks.items() if not v)
            return {
                **out,
                "trigger": False,
                "kind": None,
                "armed": dict(self.pending),
                "checks": checks,
                "reason": f"armed_for_next_capture_only:entry_failed_now:{failed}",
                "memory_point_level_m": p,
            }
        if pending is None:
            return {
                **out,
                "trigger": False,
                "kind": None,
                "reason": "no_pending_transition",
                "checks": checks,
            }
        ok = _ordered(
            checks,
            "very_next_capture_index",
            capture_index == pending["capture_index"] + 1,
        )
        ok = ok and _ordered(
            checks,
            "gap_gt0_le_0p08s",
            0 < t - pending["timestamp_s"] <= NEXT_CAPTURE_MAX_GAP_S,
        )
        # FinalApproach.update's ordinary quiet-sampling OBSERVE output carries no 'reason'
        # key; every invalidate/turn/advance/prepared/acceptable output carries one.
        # 'measured_quiet' is written only by that sampling branch (final_approach.py:81).
        ok = ok and _ordered(
            checks,
            "ordinary_observe",
            final_diag.get("phase") == "OBSERVE"
            and final_diag.get("reason") is None
            and "measured_quiet" in final_diag
            and final_diag.get("final_alignment_parameters") is False,
        )
        ok = ok and _common_state(
            checks,
            handoff_state=handoff_state,
            fine_active=fine_active,
            fine_state=fine_state,
            velocity_body_xyw=velocity_body_xyw,
            complete=complete,
            guard_safe=guard_safe,
            memory=memory,
            t=t,
        )
        p = bearing = None
        if ok:
            ok, p, bearing = _entry(
                checks,
                memory=memory,
                goal_heading=goal_heading,
                goal_fix_s=goal_fix_s,
                actual_head=actual_head,
                t=t,
            )
        if ok:
            return {
                **out,
                "trigger": True,
                "kind": "close_range_transition_latched_next_capture",
                "checks": checks,
                "reason": "next_capture_meets_all_entry_predicates",
                "memory_point_level_m": p,
                "waypoint_bearing_rad": bearing,
            }
        failed = next(k for k, v in checks.items() if not v)
        return {
            **out,
            "trigger": False,
            "kind": None,
            "reason": f"pending_expired:{failed}",
            "checks": checks,
        }
