"""Live display profile for the course's windowed checks (display only).

Used by the windowed course entries (walk check, keyboard football) so the Isaac Sim window
runs close to real time on DGX Spark:
- the viewport render texture is set to 960x540 (the image Kit renders for the window
  only; camera sensors, recorded videos and policy inputs do not use it);
- the viewport is redrawn every 2nd control step (25 Hz);
- each control step is paced against a cumulative real-time deadline, so one slow step is
  followed by steps without waiting until the run is back on schedule (lag beyond 0.1 s
  is dropped rather than caught up).
Physics (200 Hz), policy (50 Hz) and the action cadence are unchanged; results use
simulated time only. Offline course videos are rendered by a separate camera with its own
resolution. MICRODUCK_VIEWPORT_RESOLUTION="WxH" and MICRODUCK_RENDER_EVERY=N override.
"""
import os
import time

VIEWPORT_RESOLUTION = "960x540"
RENDER_EVERY = 2
CONTROL_DT = 0.02
MAX_LAG_S = 0.1


def parse_resolution(text):
    width, height = (int(v) for v in text.lower().split("x"))
    if not (320 <= width <= 3840 and 240 <= height <= 2160):
        raise ValueError(f"viewport resolution out of range: {text}")
    return width, height


def viewport_resolution(requested=None):
    """Set (when requested) and read the active viewport's render-texture resolution."""
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
    except Exception as error:  # no Kit viewport (headless)
        return dict(status=f"unavailable: {error!r}")
    if viewport is None:
        return dict(status="no active viewport")
    if requested is not None:
        viewport.resolution = requested
    return dict(status="ok", requested=list(requested) if requested else None,
                render_texture=list(viewport.resolution), resolution_scale=viewport.resolution_scale)


def apply_viewport_profile():
    """Apply the configured viewport render size; returns the record for the run report."""
    size = parse_resolution(os.environ.get("MICRODUCK_VIEWPORT_RESOLUTION", VIEWPORT_RESOLUTION))
    record = viewport_resolution(size)
    record["render_every_control_steps"] = render_every()
    return record


def render_every():
    value = int(os.environ.get("MICRODUCK_RENDER_EVERY", str(RENDER_EVERY)))
    if value < 1:
        raise ValueError("MICRODUCK_RENDER_EVERY must be >= 1")
    return value


class Pacer:
    """Real-time pacing against a cumulative deadline (one control step = CONTROL_DT)."""

    def __init__(self, clock=time.monotonic, sleep=time.sleep):
        self.clock, self.sleep = clock, sleep
        self.start = self.deadline = clock()
        self.steps = 0

    def wait(self):
        self.steps += 1
        self.deadline = max(self.deadline + CONTROL_DT, self.clock() - MAX_LAG_S)
        self.sleep(max(0.0, self.deadline - self.clock()))

    def real_time_factor(self):
        elapsed = self.clock() - self.start
        return self.steps * CONTROL_DT / elapsed if elapsed > 0 else None
