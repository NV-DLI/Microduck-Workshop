"""Public visual-bearing phases with actual delivered-gyro stopping evidence.

Far approach and a separate bounded near-observation alignment are armed. Every original sensor/navigation hold remains
authoritative. No state, achieved velocity or policy action is overwritten.
"""

import math
from collections import deque
import numpy as np
from angular_navigation_native import AngularBallNavigator
from perception import level_rotation


class NativePhaseCommands:
    forward_s = 0.6
    turn_min_s = 0.4
    turn_max_s = 1.2
    near_turn_max_s = 2.4
    near_turn_magnitude = 0.8
    observe_min_s = 0.4
    observe_warning_s = 2.0
    near_hold_m = 0.18
    near_entry_m = 0.34

    def __init__(self):
        self.reset()

    def reset(self):
        self.phase = "OBSERVE"
        self.started = None
        self.last_t = None
        self.turn_sign = 0.0
        self.turn_integral = 0.0
        self.active_turn_max_s = self.turn_max_s
        self.active_turn_magnitude = 0.6
        self.samples = deque()
        self.active_duration = self.forward_s
        self.strafe_sign = 0.0

    def _hold(self, t, why, clear=True):
        if clear:
            self.phase = "OBSERVE"
            self.started = t
            self.samples.clear()
        return [0.0, 0.0, 0.0], {
            "phase": self.phase,
            "reason": why,
            "guard_zero_preserved": True,
            "truth_used": False,
        }

    def update(
        self,
        *,
        timestamp_s,
        authorized,
        requested,
        point_level,
        range_uncertainty_m,
        gyro_level_z,
        near_goal_heading=None,
        near_position_error=None
    ):
        t = float(timestamp_s)
        u = np.asarray(requested, float)
        p = (
            np.asarray(point_level, float)
            if point_level is not None
            else np.full(3, np.nan)
        )
        good = (
            math.isfinite(t)
            and t >= 0
            and u.shape == (3,)
            and p.shape == (3,)
            and np.isfinite(u).all()
            and np.isfinite(p).all()
            and math.isfinite(range_uncertainty_m)
            and range_uncertainty_m >= 0
            and math.isfinite(gyro_level_z)
        )
        if not good:
            self.reset()
            return [0.0, 0.0, 0.0], {
                "phase": "OBSERVE",
                "reason": "invalid_public_input",
                "guard_zero_preserved": True,
                "truth_used": False,
            }
        dt = 0.0 if self.last_t is None else t - self.last_t
        if self.last_t is not None and not 0 < dt <= 0.2:
            self.reset()
            self.last_t = t
            return self._hold(t, "nonmonotone_or_gap")
        self.last_t = t
        if not authorized or not np.any(u):
            return self._hold(t, "upstream_hold")
        distance = float(np.linalg.norm(p[:2]))
        visual_bearing = float(math.atan2(p[1], p[0]))
        near = distance - range_uncertainty_m <= self.near_entry_m
        error = None
        if near:
            error = (
                np.asarray(near_position_error, float)
                if near_position_error is not None
                else np.full(2, np.nan)
            )
            if (
                near_goal_heading is None
                or not math.isfinite(near_goal_heading)
                or error.shape != (2,)
                or not np.isfinite(error).all()
            ):
                return self._hold(t, "fresh_near_goal_heading_and_position_required")
        bearing = float(near_goal_heading) if near else visual_bearing
        if p[0] <= 0 or distance - range_uncertainty_m <= self.near_hold_m:
            return self._hold(t, "near_or_behind_ball_requires_separate_validation")
        if self.started is None:
            self.started = t
        self.samples.append((t, gyro_level_z, p[:2].copy()))
        while self.samples and t - self.samples[0][0] > 0.45:
            self.samples.popleft()
        diagnostic = {
            "phase": self.phase,
            "bearing_rad": bearing,
            "range_lower_m": distance - range_uncertainty_m,
            "near_mode": near,
            "visual_ball_bearing_rad": visual_bearing,
            "body_heading_owner": (
                "public_preparation_waypoint" if near else "public_ball_bearing"
            ),
            "public_gyro_integral_rad": self.turn_integral,
            "truth_used": False,
            "achieved_velocity_calibrated": False,
        }
        if self.phase == "TURN":
            self.turn_integral += gyro_level_z * dt
            crossed = bearing * self.turn_sign <= 0
            done = t - self.started >= self.turn_min_s and abs(bearing) <= 0.06
            if crossed or done or t - self.started >= self.active_turn_max_s:
                return self._hold(
                    t,
                    "turn_bearing_complete" if crossed or done else "turn_duration_cap",
                )
            diagnostic.update(
                phase="TURN",
                elapsed_s=t - self.started,
                public_gyro_integral_rad=self.turn_integral,
                maximum_duration_s=self.active_turn_max_s,
            )
            return [0.0, 0.0, self.active_turn_magnitude * self.turn_sign], diagnostic
        if self.phase == "ADVANCE":
            if (
                abs(bearing) > (0.15 if near else 0.35)
                or t - self.started >= self.active_duration
            ):
                return self._hold(t, "advance_direction_or_duration_cap")
            diagnostic.update(phase="ADVANCE", elapsed_s=t - self.started)
            return [0.3, 0.0, 0.0], diagnostic
        if self.phase == "STRAFE":
            if (
                not near
                or error[1] * self.strafe_sign <= 0
                or abs(error[1]) <= 0.02
                or abs(bearing) > 0.15
                or t - self.started >= 2.0
            ):
                return self._hold(t, "lateral_error_or_duration_cap")
            diagnostic.update(
                phase="STRAFE",
                elapsed_s=t - self.started,
                lateral_error_m=float(error[1]),
            )
            return [0.0, 0.3 * self.strafe_sign, 0.0], diagnostic
        elapsed = t - self.started
        covered = len(self.samples) >= 6 and t - self.samples[0][0] >= 0.33
        gyro_mean = float(np.mean([x[1] for x in self.samples]))
        visual_motion = float(np.linalg.norm(self.samples[-1][2] - self.samples[0][2]))
        quiet = covered and abs(gyro_mean) <= 0.12 and visual_motion <= 0.02
        diagnostic.update(
            phase="OBSERVE",
            elapsed_s=elapsed,
            gyro_window_mean_rad_s=gyro_mean,
            ball_relative_change_m=visual_motion,
            measured_quiet=quiet,
            settling_warning=elapsed > self.observe_warning_s,
        )
        if elapsed < self.observe_min_s or not quiet:
            return [0.0, 0.0, 0.0], diagnostic
        self.started = t
        self.samples.clear()
        if abs(bearing) > (0.10 if near else 0.18):
            self.phase = "TURN"
            self.turn_sign = math.copysign(1.0, bearing)
            self.active_turn_max_s = self.near_turn_max_s if near else self.turn_max_s
            self.active_turn_magnitude = self.near_turn_magnitude if near else 0.6
            self.turn_integral = 0.0
            return [0.0, 0.0, self.active_turn_magnitude * self.turn_sign], {
                **diagnostic,
                "phase": "TURN",
                "reason": "fresh_visual_bearing",
            }
        if u[0] <= 0:
            return self._hold(t, "upstream_path_does_not_authorize_forward")
        self.phase = "ADVANCE"
        self.active_duration = 0.4 if near else self.forward_s
        return [0.3, 0.0, 0.0], {
            **diagnostic,
            "phase": "ADVANCE",
            "reason": "bearing_aligned_fresh_path",
        }


class NativeCommandNavigator(AngularBallNavigator):
    def reset(self):
        super().reset()
        self.native_commands = NativePhaseCommands()

    def step(self, rgb, proprio, tof_frame, **kwargs):
        out = super().step(rgb, proprio, tof_frame, **kwargs)
        # Keep upstream authorization but remove the obsolete trained-gait PI
        # correction from this reference's command profile. No accumulated PI
        # state can choose a native turn or prevent a visually aligned advance.
        previous = out.get("measured_yaw_feedback", {})
        requested = list(out["velocity_body_xyw"])
        requested[2] = float(previous.get("desired_yaw_rate_rad_s", 0.0))
        self.yaw_feedback.reset()
        out["measured_yaw_feedback"] = {
            "active": False,
            "reason": "replaced_by_public_visual_phase",
            "old_desired_rate_diagnostic": requested[2],
        }
        geo = self.last_geometry
        ball = geo.get("ball") if geo else None
        fresh = bool(
            geo
            and geo["timestamp_s"] == proprio["timestamp_s"]
            and ball
            and ball["accepted"]
            and geo["floor_observation"]["accepted"]
        )
        point = None
        uncertainty = 0.0
        omega = 0.0
        if fresh:
            R = level_rotation((-np.asarray(geo["ground"]["normal"])).tolist())
            point = R @ np.asarray(ball["geometry"]["point_reference_m"])
            uncertainty = float(ball["geometry"]["estimated_range_uncertainty_m"])
            omega = float((R @ np.asarray(proprio["gyro_body_rad_s"]))[2])
        allowed = bool(
            out.get("motion_authorization", {}).get("allowed")
            and fresh
            and not self.refresh_too_close
        )
        near_diag = out.get("near_ball_phase", {})
        near_heading = (
            near_diag.get("translation_heading_error_rad")
            if 0 <= near_diag.get("goal_fix_age_s", -1) <= 8.0
            else None
        )
        near_error = near_diag.get("relative_position_error_m")
        actual, diagnostic = self.native_commands.update(
            timestamp_s=proprio["timestamp_s"],
            authorized=allowed,
            requested=requested,
            point_level=point,
            range_uncertainty_m=uncertainty,
            gyro_level_z=omega,
            near_goal_heading=near_heading,
            near_position_error=near_error,
        )
        out["velocity_body_xyw"] = actual
        out["native_command_adapter"] = diagnostic
        out["velocity_semantics"] = (
            "Published community command primitives, not achieved physical velocity"
        )
        out["kick_request"] = False
        return out
