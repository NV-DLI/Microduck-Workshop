"""Range progress: Microduck simulation and control."""

import math


class RangeProgress:
    minimum_progress_m = 0.005
    unproven_pulses_before_probe = 2
    probe_duration_s = 0.4
    probe_command_magnitude = 0.3
    total_probe_pulse_budget = 3
    start_lower_m = 0.28
    active_stop_lower_m = 0.24
    comparison_maximum_age_s = 3.0

    def __init__(self):
        self.probe_pulses = 0
        self.last_comparison = None
        self.invalidate()

    def invalidate(self):
        self.anchor = None
        self.completed = False
        self.unproven = 0

    def begin(self, t, radius, uncertainty, duration, probe=False):
        if (
            not all(math.isfinite(x) for x in (t, radius, uncertainty, duration))
            or t < 0
            or radius <= 0
            or uncertainty < 0
            or duration <= 0
        ):
            raise ValueError(
                "Finite nonnegative timestamp/uncertainty and positive range/duration required"
            )
        if probe and self.probe_pulses >= self.total_probe_pulse_budget:
            raise ValueError("Stronger pulse budget exhausted")
        self.anchor = (t, radius, uncertainty, duration)
        self.completed = False
        if probe:
            self.probe_pulses += 1
            assert self.probe_pulses <= self.total_probe_pulse_budget

    def end(self, t, complete):
        if not math.isfinite(t) or t < 0:
            raise ValueError("Invalid completion timestamp")
        if not self.anchor or not complete:
            self.invalidate()
        else:
            self.completed = True

    def settled(self, t, radius, uncertainty):
        if (
            not all(math.isfinite(x) for x in (t, radius, uncertainty))
            or t < 0
            or radius <= 0
            or uncertainty < 0
        ):
            raise ValueError("Invalid settled public geometry")
        if not self.anchor or not self.completed:
            return
        start, r0, u0, duration = self.anchor
        self.anchor = None
        self.completed = False
        if not 0 < t - start <= self.comparison_maximum_age_s:
            self.unproven = 0
            self.last_comparison = {"accepted": False, "reason": "comparison_age"}
            return
        lower = (r0 - u0) - (radius + uncertainty)
        upper = (r0 + u0) - (radius - uncertainty)
        confirmed = lower >= self.minimum_progress_m
        receding = upper < -self.minimum_progress_m
        self.unproven = 0 if confirmed or receding else self.unproven + 1
        self.last_comparison = {
            "accepted": True,
            "start_s": start,
            "settled_s": t,
            "start_range_interval_m": [r0 - u0, r0 + u0],
            "settled_range_interval_m": [radius - uncertainty, radius + uncertainty],
            "radial_approach_interval_m": [lower, upper],
            "nominal_radial_approach_m": r0 - radius,
            "confirmed_minimum_progress": confirmed,
            "confirmed_receding": receding,
            "classification": (
                "confirmed_progress"
                if confirmed
                else (
                    "receding"
                    if receding
                    else "insufficient_evidence_of_minimum_progress"
                )
            ),
            "source": "complete fresh RGB/FK/floor horizontal norms at two measured-quiet decisions",
            "yaw_rotation_cannot_change_horizontal_norm": True,
            "interval_is_not_a_calibrated_statistical_confidence_bound": True,
        }

    def choose(self, current_lower):
        if not math.isfinite(current_lower) or current_lower < 0:
            raise ValueError("Invalid current range lower bound")
        if (
            self.unproven < self.unproven_pulses_before_probe
            or current_lower < self.start_lower_m
        ):
            return 0.4, "original_short_pulse"
        if self.probe_pulses >= self.total_probe_pulse_budget:
            return None, "bounded_progress_probe_budget_exhausted"
        return self.probe_duration_s, "bounded_stronger_probe_after_unproven_progress"

    def status(self):
        return {
            "unproven_completed_pulses": self.unproven,
            "probe_pulses_started": self.probe_pulses,
            "probe_pulse_budget": self.total_probe_pulse_budget,
            "last_comparison": self.last_comparison,
            "pulse_plus_settling_allocation_m": 0.16,
            "remaining_after_probe_stop_allocation_m": 0.12,
            "allocations_are_engineering_assumptions_not_proven_bounds": True,
            "kick_authorized": False,
            "truth_used": False,
        }
