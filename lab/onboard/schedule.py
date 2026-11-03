"""Private physics/control/capture/display schedule for the student walking executor (phase 13).

Pure Python (no Kit/torch). `Schedule` decides, per physics tick k (state at t_k = k*dt before the
step, t_{k+1} after it), which events fire, using exact integer arithmetic; timestamps are
k * (1.0 / physics_hz), which reproduces the original `tick * 0.005` floats bit-for-bit at 200 Hz.

Rules (identical to the original executor at 200/50/15 Hz):
  control  : before stepping tick k iff some j has ceil(j * P / C) == k  (original: tick % 4 == 0)
  period end: control_at(k + 1)                                         (original: tick % 4 == 3)
  capture  : after tick k iff some n >= 1 has ceil(n * P / S) == k + 1  (original: tick + 1 >=
             ceil(n*200/15 - 1e-10), next_capture counter)
  display  : same rule as capture at display_hz, separate callback, does not touch sensors
P physics Hz, C control Hz, S capture Hz. When P is not a multiple of C (60 physics, 50 control)
control falls on the first tick at/after each nominal time: intervals are uneven and the policy is
NOT run at its trained fixed 20 ms hold; `record()` reports nominal vs actual times and lags.
"""
from fractions import Fraction
import math

TRAINED_CONTROL_HZ = 50
TRAINED_PHYSICS_HZ = 200
TRAINED_DELAY_S = Fraction(1, 50)  # course: 4 steps at 200 Hz (actuator_bam_cfg: delay in physics steps)


class Schedule:
    def __init__(self, physics_hz=200, control_hz=50, capture_hz=15, display_hz=None, delay_steps=None):
        for name, v in (("physics_hz", physics_hz), ("control_hz", control_hz), ("capture_hz", capture_hz)):
            if type(v) is not int or v <= 0:
                raise ValueError(name + " must be a positive integer")
        if control_hz > physics_hz or capture_hz > physics_hz or (display_hz and display_hz > physics_hz):
            raise ValueError("event rate above physics rate")
        self.P, self.C, self.S, self.D = physics_hz, control_hz, capture_hz, display_hz
        self.physics_dt = 1.0 / physics_hz
        exact_delay = TRAINED_DELAY_S * physics_hz
        if delay_steps is None:
            if exact_delay.denominator != 1:
                raise ValueError(f"20 ms actuator delay is {float(exact_delay)} steps at {physics_hz} Hz; "
                                 "pass delay_steps explicitly")
            delay_steps = int(exact_delay)
        self.delay_steps = int(delay_steps)
        self.control_exact = physics_hz % control_hz == 0
        self.decimation = physics_hz // control_hz if self.control_exact else None
        self._control_ticks = {}
        self._capture_ticks = {}
        self._display_ticks = {}

    # -- exact event tables (computed lazily per index) -------------------------------------
    @staticmethod
    def _tick_of(index, P, rate):
        return -(-index * P // rate)  # ceil(index * P / rate) with integers

    def time(self, tick):
        return tick * self.physics_dt

    def ticks_for(self, seconds):
        """Number of physics steps covering `seconds` (original: math.ceil(seconds / 0.005))."""
        return math.ceil(seconds / self.physics_dt)

    def control_at(self, tick):
        if self.control_exact:
            return tick % self.decimation == 0
        j = tick * self.C // self.P  # last nominal index with nominal time <= t_k
        return self._tick_of(j, self.P, self.C) == tick

    def control_period_end(self, tick):
        return self.control_at(tick + 1)

    def control_period_s(self, tick):
        """Duration of the control period that ends after tick (for interval events)."""
        k = tick
        while k > 0 and not self.control_at(k):
            k -= 1
        return (tick + 1 - k) * self.physics_dt

    def _after(self, tick, rate):
        n = -(-(tick + 1) * rate // self.P)  # smallest n with n/rate >= t_{k+1}? use exact match below
        hits = [m for m in (n - 1, n) if m >= 1 and self._tick_of(m, self.P, rate) == tick + 1]
        if len(hits) > 1:
            raise AssertionError("two events in one tick")
        return bool(hits)

    def capture_after(self, tick):
        return self._after(tick, self.S)

    def display_after(self, tick):
        return bool(self.D) and self._after(tick, self.D)

    # -- reporting ----------------------------------------------------------------------------
    def simulate(self, ticks):
        control = [k for k in range(ticks) if self.control_at(k)]
        capture = [k + 1 for k in range(ticks) if self.capture_after(k)]
        display = [k + 1 for k in range(ticks) if self.display_after(k)]
        ctl_t = [self.time(k) for k in control]
        nominal = [j / self.C for j in range(len(control))]
        holds = [b - a for a, b in zip(ctl_t, ctl_t[1:])]
        return dict(control_ticks=control, control_times=ctl_t,
                    control_lag_s=[a - b for a, b in zip(ctl_t, nominal)],
                    control_hold_s=sorted(set(round(h, 9) for h in holds)),
                    capture_times=[self.time(k) for k in capture], display_times=[self.time(k) for k in display])

    def record(self, seconds=1.0):
        sim = self.simulate(self.ticks_for(seconds))
        return dict(
            physics_hz=self.P, physics_dt_s=self.physics_dt, control_hz=self.C, capture_hz=self.S,
            display_hz=self.D, decimation=self.decimation, control_exact=self.control_exact,
            control_events_per_s=len(sim["control_ticks"]) / seconds,
            control_hold_values_s=sim["control_hold_s"], max_control_lag_s=max(sim["control_lag_s"]),
            policy_rate_matches_training=self.control_exact and self.C == TRAINED_CONTROL_HZ,
            physics_matches_training=self.P == TRAINED_PHYSICS_HZ,
            actuator_delay_steps=self.delay_steps, actuator_delay_s=self.delay_steps / self.P,
            actuator_delay_matches_course_20ms=abs(self.delay_steps / self.P - 0.02) < 1e-12,
            capture_events_per_s=len(sim["capture_times"]) / seconds,
            display_events_per_s=(len(sim["display_times"]) / seconds) if self.D else None,
        )
