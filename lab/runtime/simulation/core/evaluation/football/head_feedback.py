"""Public outer head correction with no base/skill/observation authority."""

import math
import numpy as np
from head_feedback_core import HeadSetpointFeedback, HOME

OBSERVING = frozenset(
    (
        "SEARCH_BALL",
        "SEARCH_GOAL",
        "REFRESH_GOAL",
        "REACQUIRE_BALL",
        "NEAR_OBSERVE",
        "OBSERVE",
        "FINAL_ALIGN",
    )
)


def head_context_allowed(
    *, now_s, capture_s, control, geometry, too_close, ball_context, goal_context
):
    if (
        not control
        or not geometry
        or not math.isfinite(capture_s)
        or not 0 <= now_s - capture_s <= 0.2
    ):
        return False
    attention = control.get("public_attention_head_permission") or {}
    owner = control.get("public_attention_owner", {}).get("owner")
    if (
        owner in ("GOAL_REFRESH", "BALL_REACQUIRE")
        and attention.get("accepted") is True
        and attention.get("owner") == owner
        and attention.get("capture_s") == capture_s
        and geometry.get("timestamp_s") == capture_s
        and control.get("velocity_body_xyw") == [0.0, 0.0, 0.0]
        and control.get("kick_request") is False
    ):
        # The explicit attention owner cannot issue body/kick commands.
        # Native50Hz output still validates its fresh actual packet separately.
        return True
    if (
        too_close
        or geometry.get("timestamp_s") != capture_s
        or not geometry.get("floor_observation", {}).get("accepted")
    ):
        return False
    reason = control.get("reason", "")
    from guarded_navigation_feedback import UPSTREAM_HOLDS

    if (
        reason in UPSTREAM_HOLDS
        or reason.startswith("Nav2_failed:")
        or control.get("state") == "CONTROL_ERROR"
    ):
        return False
    if control.get("public_attention_owner", {}).get("owner") in (
        "GOAL_REFRESH",
        "BALL_REACQUIRE",
        "FAILED",
    ):
        return bool(ball_context)
    if control.get("state") == "REFRESH_GOAL":
        return bool(goal_context)
    return True


class ObservationHeadFeedback:
    def __init__(self):
        self.core = HeadSetpointFeedback(1)
        self.observe_started = None
        self.withdraw_started = None
        self.compensated = False
        self.failure = None

    def update(
        self,
        *,
        now_s,
        nominal,
        packet,
        velocity,
        state,
        handoff_state,
        navigation_enabled,
        head_motion_allowed,
        fine_reacquire_head_only=False,
        macro_neutral_return=False
    ):
        # Invalid clocks must never mutate state, including any deadline.
        if (
            not math.isfinite(now_s)
            or now_s < 0
            or self.core.last_now is not None
            and now_s <= self.core.last_now
        ):
            raise ValueError("Finite increasing actual policy time required")
        nominal = np.asarray(nominal, float)
        velocity = np.asarray(velocity, float)
        if (
            nominal.shape != (4,)
            or velocity.shape != (3,)
            or not np.isfinite(nominal).all()
            or not np.isfinite(velocity).all()
        ):
            raise ValueError("Finite native head4 and velocity3 required")
        valid = False
        why = "missing public packet"
        q = HOME.copy()
        seq = -1
        read = now_s
        if packet is not None:
            try:
                seq = packet["seq"]
                read = float(packet["read_timestamp_s"])
                sample = float(packet["timestamp_s"])
                encoder_time = float(packet["encoder_read_timestamp_s"])
                qdummy = np.asarray(packet["encoder_q_rad"], float)
                valid = bool(
                    packet.get("orientation_valid") is True
                    and isinstance(seq, int)
                    and not isinstance(seq, bool)
                    and seq >= 0
                    and qdummy.shape == (14,)
                    and np.isfinite(qdummy).all()
                    and all(
                        math.isfinite(x) and x >= 0 and 0 <= now_s - x <= 0.08
                        for x in (read, sample, encoder_time)
                    )
                    and encoder_time == read
                )
                if valid:
                    q = qdummy[5:9].copy()
                    why = "same-read public packet valid"
                else:
                    why = "public packet invalid or stale"
            except (KeyError, TypeError, ValueError):
                valid = False
                why = "malformed public packet"
        moving = bool(np.any(velocity))
        if moving and self.compensated and self.withdraw_started is None:
            self.withdraw_started = now_s
        ordinary = handoff_state == "APPROACH"
        reacquire_only = bool(
            fine_reacquire_head_only is True
            and handoff_state == "NEUTRAL_FINE_STEP"
            and state == "NEUTRAL_FINE_REACQUIRE"
        )
        observing = bool(
            navigation_enabled
            and (ordinary and state in OBSERVING or reacquire_only)
            and not moving
            and head_motion_allowed
            and valid
            and self.withdraw_started is None
            and self.failure is None
        )
        if observing:
            if self.observe_started is None:
                self.observe_started = now_s
        else:
            self.observe_started = None
        integrate = bool(observing and now_s - self.observe_started >= 5.0 - 1e-9)
        permitted = (
            valid
            and (head_motion_allowed or not navigation_enabled or not ordinary)
            and self.failure is None
        )
        returning = bool(
            macro_neutral_return is True
            and handoff_state == "NEUTRAL_FINE_STEP"
            and not moving
            and valid
            and permitted
            and not integrate
            and np.all(abs(nominal) <= abs(self.core.sent[0]) + 1e-9)
            and np.all(nominal * self.core.sent[0] >= -1e-12)
        )
        sent, diagnostic = self.core.update(
            now_s=now_s,
            nominal=nominal[None],
            measured_q=q[None],
            seq=[seq if valid else -1],
            read_s=[read if valid else now_s],
            valid=[permitted],
            enabled=[integrate],
            slew_rad_s=0.65 if returning else 0.3,
        )
        actual = sent[0]
        if np.any(abs(self.core.bias[0]) > 1e-12):
            self.compensated = True
        settled = bool(
            np.max(abs(actual - nominal)) <= 1e-9 and not self.core.bias.any()
        )
        if (
            self.withdraw_started is not None
            and not settled
            and now_s - self.withdraw_started >= 2.0 - 1e-9
        ):
            self.failure = "head_feedback_withdrawal_timeout"
        if settled:
            self.withdraw_started = None
            self.compensated = False
        hold_body = bool(self.withdraw_started is not None or self.failure is not None)
        if handoff_state == "KICK" and np.max(abs(actual)) > 1e-9:
            self.failure = "head_feedback_actual_command_not_neutral_at_kick"
            hold_body = True
        return actual.tolist(), dict(
            {
                k: (v[0].tolist() if isinstance(v, np.ndarray) else v)
                for k, v in diagnostic.items()
            },
            packet_valid=valid,
            packet_reason=why,
            head_motion_permitted=bool(permitted),
            integration_enabled=integrate,
            macro_neutral_return_active=returning,
            fine_reacquire_head_only=reacquire_only,
            observation_started_s=self.observe_started,
            observation_phase=state,
            handoff_state=handoff_state,
            nominal_head_offsets_rad=nominal.tolist(),
            requested_body_command=velocity.tolist(),
            body_zero_for_withdrawal=hold_body,
            withdrawal_started_s=self.withdraw_started,
            withdrawal_deadline_s=2.0,
            failure=self.failure,
            qualification_claimed=False,
            truth_used=False,
        )
