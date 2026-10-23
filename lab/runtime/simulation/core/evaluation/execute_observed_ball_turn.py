#!/usr/bin/env python3
"""Execute observed ball turn: Microduck simulation and control."""
import math
import execute_observed_attention as attention


class ObservedTurnNavigator(attention.AttentionNavigator):
    def _output(self, t, command=(0.0, 0.0, 0.0), reason="", gaze=None):
        view = self.ball_attention
        valid = (
            self.state == "SEARCH_BALL"
            and view is not None
            and 0 <= t - view["capture_timestamp_s"] <= 0.2
        )
        evidence = None
        if valid:
            yaw = float(view["gaze_body_yaw_pitch_rad"][0])
            turn = max(-0.2, min(0.2, 0.5 * yaw)) if abs(yaw) > 0.35 else 0.0
            if self.refresh_too_close:
                turn = 0.0
            command = (0.0, 0.0, turn)
            evidence = {
                "measured_visible_ray_yaw_rad": yaw,
                "body_turn_command_rad_s": turn,
                "head_domain_rad": [-0.4, 0.4],
                "body_turn_deadband_rad": 0.35,
                "translation_authorized": False,
                "kick_authorized": False,
                "TooClose_guard": self.refresh_too_close,
                "source": "fresh RGB visible pixels plus same-time encoder FK",
            }
            reason = "observed_ball_direction_body_turn" if turn else reason
        out = super()._output(t, command, reason, gaze)
        if evidence:
            out["observed_ball_body_turn"] = evidence
        return out


def build(a):
    cfg = attention.build(a)
    entry = attention.active.base.entry
    entry.contract["source_sha256"][str(entry.Path(__file__).resolve())] = entry.sha(
        __file__
    )
    return cfg


if __name__ == "__main__":
    entry = attention.active.base.entry
    attention.active.base.controller.BetaNavigator = ObservedTurnNavigator
    entry.isaac_execute.build_cfg = build
    entry.isaac_execute.run = entry.run
    entry.nav2_bridge.Nav2Session = entry.BoundedLifetimeSession
    entry.isaac_execute.main()
