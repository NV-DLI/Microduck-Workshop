"""Live head-sensor panel docked next to the Isaac Sim viewport (display only).

`attach(env)` wraps `env.beta_rig.capture`, which both the onboard football executor and the
GR00T walking executor call every 1/15 s of simulation time. After each capture the panel
shows the frames that call returned: the head-camera RGB (downscaled) and the 64 ToF zone
distances (not the 32x32 render the ToF model samples). Nothing else is read: no extra camera,
render, GPU readback or physics state, and the returned dictionary is passed on unchanged.

Kit redraws this window during the next capture's renders, so the frame on screen is one
capture period older than the scene in the viewport; the panel prints that age in simulation
time. Without a Kit window (headless runs) `attach` does nothing and imports no UI module.
Display errors are recorded and switch the panel off; they never stop an episode.
"""
import time

from sensor_payload import (
    CAPTURE_PERIOD_S, INVALID_FILL, NO_RETURN_FILL, freshness, legend_image, rgb_preview, tof_view,
)

TITLE = "Microduck head sensors"
RGB_FACTOR = 3  # 960x540 -> 320x180
CELL_PX = 28
PANEL_PX = 360  # window width; content is at most 320 px wide
MAX_LINE_CHARS = 44  # one line of the default 14 px Kit font fits about 320 px
# Upper bound on panel updates per wall-clock second (sensor frames arrive at 15 Hz of sim time).
MAX_UPDATES_PER_S = 15.0
# Which consumer each sensor feeds, from the executors: onboard, the detector reads rgb and
# controller.step(rgb, proprio, tof) reads the ToF frame; GR00T, WalkingPolicy.capture(rgb)
# only (the rig still produces ToF, but GR00T does not receive it).
ROLES = {
    "onboard": ("Detector input", "Controller input"),
    "groot": ("GR00T image input", "Not a GR00T input"),
}
LEGEND_LINES = ("NR: no return, unknown (not free space)", "<0.1: too close   X: invalid")
ORIENTATION_LINE = "Robot's view: row 0 top, column 0 left"


def dock_ratio(viewport_width_px, panel_px=PANEL_PX):
    """Share of the Viewport's dock area given to the panel (e.g. 0.19 of a 1920 px viewport)."""
    return min(max(panel_px / max(float(viewport_width_px), 1.0), 0.12), 0.35)


def _abgr(rgb, alpha=255):
    r, g, b = rgb
    return (alpha << 24) | (b << 16) | (g << 8) | r


class PanelState:
    """What the panel shows; pure Python so it can be tested without Kit."""

    def __init__(self, role="onboard", period_s=CAPTURE_PERIOD_S, min_interval_s=1 / MAX_UPDATES_PER_S,
                 clock=time.monotonic):
        self.roles = ROLES[role]
        self.period_s = period_s
        self.min_interval_s = min_interval_s
        self.clock = clock
        self.sim_now_s = None
        self.frame_t_s = None
        self.rgb = None
        self.tof = None
        self.rgb_shape = None
        self.last_update_wall = None
        self.stats = dict(captures=0, shown=0, skipped_for_cadence=0, update_seconds=0.0)

    def set_clock(self, sim_now_s):
        if sim_now_s is not None:
            self.sim_now_s = float(sim_now_s)

    def accept(self, capture):
        """Convert one returned capture; False when skipped to bound the refresh rate."""
        self.stats["captures"] += 1
        now = self.clock()
        if self.last_update_wall is not None and now - self.last_update_wall < self.min_interval_s:
            self.stats["skipped_for_cadence"] += 1
            return False
        start = time.perf_counter()
        rgb = capture["rgb_raw"]
        self.rgb = rgb_preview(rgb, RGB_FACTOR)
        self.rgb_shape = tuple(rgb.shape[:2])
        self.tof = tof_view(capture["tof"])
        self.frame_t_s = float(capture["timestamp_s"])
        self.set_clock(self.frame_t_s if self.sim_now_s is None else max(self.sim_now_s, self.frame_t_s))
        self.last_update_wall = now
        self.stats["shown"] += 1
        self.stats["update_seconds"] += time.perf_counter() - start
        return True

    def status(self):
        return freshness(self.sim_now_s, self.frame_t_s, self.period_s)

    def lines(self):
        """Every text line of the panel, each short enough for one 320 px line."""
        f = self.status()
        status = ("Waiting for the first frame" if f.state == "waiting"
                  else f"{f.state.upper()}, frame age {f.age_s * 1000:.0f} ms")
        clock = "" if self.sim_now_s is None else f"Simulation time {self.sim_now_s:.1f} s"
        if self.rgb_shape is None:
            rgb = "No frame yet"
        else:
            h, w = self.rgb_shape
            rgb = f"Delivered {w}x{h} frame, shown at 1/{RGB_FACTOR}"
        if self.tof is None:
            zones = "64 zones"
        elif self.tof.frame_dropped:
            zones = "Frame dropped: all 64 zones invalid"
        else:
            c = self.tof.counts
            zones = f"64 zones: {c['valid']} valid, {c['no_return']} NR, {c['too_close'] + c['invalid']} invalid"
        return dict(status=status, clock=clock, rgb_title="Head camera RGB", rgb_detail=rgb,
                    rgb_role=self.roles[0], tof_title="ToF 8x8 distance (m)", tof_zones=zones, tof_role=self.roles[1])

    # Short accessors kept for logs and tests.
    def header(self):
        lines = self.lines()
        return lines["status"] + ("  " + lines["clock"] if lines["clock"] else "")

    def rgb_caption(self):
        lines = self.lines()
        return f"{lines['rgb_title']}: {lines['rgb_detail']}. {lines['rgb_role']}"

    def tof_caption(self):
        lines = self.lines()
        return f"{lines['tof_title']}: {lines['tof_zones']}. {lines['tof_role']}"


class SensorPanel:
    """omni.ui window; created only when a Kit window exists."""

    def __init__(self, state, title=TITLE):
        import omni.ui as ui

        self.ui = ui
        self.state = state
        self.window = ui.Window(title, width=PANEL_PX, height=820)
        self.rgb_provider = ui.ByteImageProvider()
        self.legend_provider = ui.ByteImageProvider()
        self.cells = []
        self.labels = {}
        bold = {"font_size": 15, "color": 0xFFFFFFFF}
        small = {"font_size": 13, "color": 0xFFC8C8C8}

        def line(key, style, height=18, wrap=False):
            # A wrapped line keeps room for two lines so nothing is clipped at this width.
            self.labels[key] = ui.Label("", height=36 if wrap else height, word_wrap=wrap, style=style)

        with self.window.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(spacing=1, height=0):
                    line("status", bold, 20)
                    ui.Spacer(height=4)
                    line("rgb_title", bold, 20)
                    line("rgb_role", small)
                    ui.ImageWithProvider(self.rgb_provider, height=ui.Pixel(128),
                                         fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                    ui.Spacer(height=4)
                    line("tof_title", bold, 20)
                    line("tof_zones", small)
                    line("tof_role", small)
                    with ui.HStack(height=8 * CELL_PX + 7):
                        ui.Spacer()
                        with ui.VStack(width=8 * CELL_PX + 7, spacing=1):
                            for _ in range(8):
                                row = []
                                with ui.HStack(height=CELL_PX, spacing=1):
                                    for _ in range(8):
                                        with ui.ZStack(width=CELL_PX, height=CELL_PX):
                                            rect = ui.Rectangle(style={"background_color": _abgr(NO_RETURN_FILL, 90)})
                                            label = ui.Label("", alignment=ui.Alignment.CENTER,
                                                             style={"font_size": 13})
                                        row.append((rect, label))
                                self.cells.append(row)
                        ui.Spacer()
                    ui.ImageWithProvider(self.legend_provider, height=ui.Pixel(10),
                                         fill_policy=ui.IwpFillPolicy.IWP_STRETCH)
                    with ui.HStack(height=16):
                        self.legend_min = ui.Label("0.1 m", style=small)
                        ui.Spacer()
                        self.legend_max = ui.Label("4.0 m", alignment=ui.Alignment.RIGHT, style=small)
                    for text, fill in zip(LEGEND_LINES, (NO_RETURN_FILL, INVALID_FILL)):
                        with ui.HStack(height=18, spacing=6):
                            ui.Rectangle(width=14, height=14, style={"background_color": _abgr(fill),
                                                                     "border_color": 0xFF808080, "border_width": 1})
                            ui.Label(text, style=small)
                    ui.Label(ORIENTATION_LINE, height=18, style=small)
        self._set_image(self.legend_provider, legend_image())
        self.dock_attempts = 0
        self.docked = self._dock()
        self.refresh()

    def _dock(self):
        """Split the Viewport's dock area; retried on the next refreshes if Kit has not laid it out yet."""
        self.dock_attempts += 1
        ui = self.ui
        viewport = ui.Workspace.get_window("Viewport")
        panel = ui.Workspace.get_window(self.window.title)
        if viewport is None or panel is None:
            return False
        if not panel.docked:
            panel.dock_in(viewport, ui.DockPosition.RIGHT, dock_ratio(viewport.width))
        # Kit applies docking on an update; retry until the handle confirms it.
        return bool(panel.docked)

    @staticmethod
    def _set_image(provider, rgba):
        h, w = rgba.shape[:2]
        # strict=True raises on a malformed upload instead of logging and skipping it.
        provider.set_data_array(rgba, [w, h], strict=True)

    def refresh(self, images=False):
        handle = self.ui.Workspace.get_window(self.window.title)
        self.docked = bool(handle is not None and handle.docked)
        if not self.docked and self.dock_attempts < 5:
            self.docked = self._dock()
        s = self.state
        for key, text in s.lines().items():
            if key in self.labels:
                self.labels[key].text = text
        self.labels["status"].tooltip = s.lines()["clock"]
        self.labels["rgb_title"].tooltip = s.lines()["rgb_detail"]
        stale = s.status().state != "live"
        self.labels["status"].set_style({"font_size": 15, "color": 0xFF3C8CF0 if stale else 0xFF00B976})
        if not images:
            return
        self._set_image(self.rgb_provider, s.rgb)
        self.legend_min.text = f"{s.tof.min_m:.1f} m"
        self.legend_max.text = f"{s.tof.max_m:.1f} m"
        for row, widgets in zip(s.tof.cells, self.cells):
            for cell, (rect, label) in zip(row, widgets):
                rect.set_style({"background_color": _abgr(cell.fill)})
                label.text = cell.text
                label.set_style({"color": _abgr(cell.text_fill), "font_size": 13})

    def destroy(self):
        if self.window is not None:
            self.window.visible = False
            self.window.destroy()
            self.window = None


def kit_window_requested(argv):
    """Same rule as the executors: a window exists only with `--visualizer kit`."""
    for i, arg in enumerate(argv):
        if arg == "--visualizer" and "kit" in argv[i + 1:i + 2]:
            return True
        if arg.startswith("--visualizer=") and "kit" in arg.split("=", 1)[1].split(","):
            return True
    return False


class Attachment:
    """Wraps rig.capture; keeps a record for the run's scene contract."""

    def __init__(self, rig, panel_factory=SensorPanel, state=None, role="onboard"):
        self.rig = rig
        self.state = state or PanelState(role)
        self.panel = None
        self.record = dict(kind="live head-sensor panel (display only)", role=role, enabled=True, error=None,
                           source="IsaacBetaSensors.capture() return value: rgb_raw, tof, timestamp_s",
                           extra_render_or_readback=False)
        try:
            self.panel = panel_factory(self.state)
            self.record["docked_next_to_viewport"] = getattr(self.panel, "docked", None)
        except Exception as error:  # display only
            self._fail(error)
        self.original = rig.capture
        rig.capture = self.capture

    def _fail(self, error):
        self.record.update(enabled=False, error=repr(error))
        if self.panel is not None:
            try:
                self.panel.destroy()
            except Exception:
                pass
            self.panel = None

    def capture(self, *args, **kwargs):
        if self.panel is not None:
            try:
                self.state.set_clock(kwargs.get("timestamp_s"))
                self.panel.refresh()
            except Exception as error:
                self._fail(error)
        result = self.original(*args, **kwargs)
        if self.panel is not None:
            try:
                if self.state.accept(result):
                    self.panel.refresh(images=True)
            except Exception as error:
                self._fail(error)
        return result

    def close(self):
        if self.rig.capture == self.capture:
            self.rig.capture = self.original
        if self.panel is not None:
            self.record["docked_next_to_viewport"] = getattr(self.panel, "docked", None)
            try:
                self.panel.destroy()
            except Exception as error:
                self.record["close_error"] = repr(error)
            self.panel = None
        self.record["stats"] = dict(self.state.stats)
        return self.record


def attach(env, argv):
    """Attach the panel when a Kit window was requested; returns the Attachment or None."""
    if not kit_window_requested(argv):
        return None
    return Attachment(env.beta_rig, role="groot" if "--walking-policy-server" in argv else "onboard")
