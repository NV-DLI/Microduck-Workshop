"""Simulation-time intervals for manually sampled RTX camera products."""
from __future__ import annotations

import math


class CaptureClock:
    """Advance only after a successful capture, in the caller's time coordinates."""

    def __init__(self, timestamp_s: float = 0.0):
        self.reset(timestamp_s)

    def reset(self, timestamp_s: float = 0.0) -> None:
        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("Capture reset time must be finite")
        self.last_timestamp_s = timestamp_s

    def elapsed(self, timestamp_s: float) -> float:
        timestamp_s = float(timestamp_s)
        if not math.isfinite(timestamp_s):
            raise ValueError("Capture time must be finite")
        elapsed = timestamp_s - self.last_timestamp_s
        if elapsed < 0:
            raise ValueError("Capture time moved backwards; reset the sensor clock explicitly")
        return elapsed

    def commit(self, timestamp_s: float) -> None:
        self.elapsed(timestamp_s)
        self.last_timestamp_s = float(timestamp_s)
