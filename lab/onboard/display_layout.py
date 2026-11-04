"""Private, display-only dock layout for the homepage hero capture (not course code).

Replaces only the *construction and sizing* of the live head-sensor panel and asks Kit's own
docking (omni.ui.Workspace.set_dock_id_width / set_dock_id_height) to narrow the Stage/Property
column and the Content/Console row. It reads nothing new: the panel still shows exactly what
`IsaacBetaSensors.capture()` returned (sensor_panel.Attachment and PanelState are unchanged),
with the same text, the same 1/3 RGB preview and the same 15-per-second refresh bound.
No physics, policy, model, sensor, render-resolution or capture-rate setting is touched.

Every phase is written to a sidecar JSON next to the run output and into the panel record that
football_scene.py stores in scene-contract.json, so the run states that its layout was changed.
"""
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = _load("hero_layout_plan", HERE / "layout_plan.py")
SETTLE_REFRESHES = 4     # Kit applies dock changes on later updates; refresh runs once per capture call
MAX_ATTEMPTS = 5         # outer + inner sizing passes per main-window size
MAX_REPLANS = 4          # main window size changes handled after the first plan


class Recorder:
    """Writes the layout record (sidecar) atomically; never raises into the episode."""

    def __init__(self, path, header):
        self.path = Path(path)
        self.record = dict(header, phase="created", events=[], final=None)
        self.write()

    def event(self, kind, sim_s=None, **data):
        self.record["events"].append(dict(event=kind, wall_epoch=round(time.time(), 3), sim_s=sim_s, **data))
        self.write()

    def write(self):
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.record, indent=1))
            tmp.replace(self.path)
        except Exception as error:  # display only
            self.record["write_error"] = repr(error)


def make_panel_class(sp):
    """LayoutPanel built on the course module `sp` (lab/course_labs/sensor_panel.py, unmodified)."""

    class LayoutPanel(sp.SensorPanel):
        def __init__(self, state, title=sp.TITLE, recorder=None):
            import omni.ui as ui

            self.ui, self.state, self.recorder = ui, state, recorder
            self.fonts = dict(bold=plan.BASE["bold"], small=plan.BASE["small"], cell_font=plan.BASE["cell_font"])
            self.phase, self.refreshes, self.last_step, self.attempts, self.replans = "docking", 0, -99, 0, 0
            self.main_size, self.metrics = None, None
            self.window = ui.Window(title, width=sp.PANEL_PX, height=820)
            self.rgb_provider = ui.ByteImageProvider()
            self.legend_provider = ui.ByteImageProvider()
            self.cells, self.labels, self.rows, self.swatches, self.legend_rows = [], {}, [], [], []
            bold = {"font_size": self.fonts["bold"], "color": 0xFFFFFFFF}
            small = {"font_size": self.fonts["small"], "color": 0xFFC8C8C8}
            self.kind = {}

            def line(key, style, height=18):
                self.labels[key] = ui.Label("", height=height, style=style)
                self.kind[key] = "bold" if style is bold else "small"

            # Same widgets, order and text as SensorPanel.__init__; sizes become adjustable.
            with self.window.frame:
                with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                    with ui.VStack(spacing=1, height=0):
                        line("status", bold, 20)
                        ui.Spacer(height=4)
                        line("rgb_title", bold, 20)
                        line("rgb_role", small)
                        self.rgb_image = ui.ImageWithProvider(self.rgb_provider, height=ui.Pixel(128),
                                                              fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                        ui.Spacer(height=4)
                        line("tof_title", bold, 20)
                        line("tof_zones", small)
                        line("tof_role", small)
                        with ui.HStack(height=8 * sp.CELL_PX + 7) as self.grid_row:
                            ui.Spacer()
                            with ui.VStack(width=8 * sp.CELL_PX + 7, spacing=1) as self.grid_col:
                                for _ in range(8):
                                    row = []
                                    with ui.HStack(height=sp.CELL_PX, spacing=1) as hrow:
                                        self.rows.append(hrow)
                                        for _ in range(8):
                                            with ui.ZStack(width=sp.CELL_PX, height=sp.CELL_PX) as z:
                                                rect = ui.Rectangle(
                                                    style={"background_color": sp._abgr(sp.NO_RETURN_FILL, 90)})
                                                label = ui.Label("", alignment=ui.Alignment.CENTER,
                                                                 style={"font_size": self.fonts["cell_font"]})
                                            row.append((rect, label, z))
                                    self.cells.append(row)
                            ui.Spacer()
                        self.legend_image = ui.ImageWithProvider(self.legend_provider, height=ui.Pixel(10),
                                                                 fill_policy=ui.IwpFillPolicy.IWP_STRETCH)
                        with ui.HStack(height=16) as self.legend_minmax:
                            self.legend_min = ui.Label("0.1 m", style=small)
                            ui.Spacer()
                            self.legend_max = ui.Label("4.0 m", alignment=ui.Alignment.RIGHT, style=small)
                        for text, fill in zip(sp.LEGEND_LINES, (sp.NO_RETURN_FILL, sp.INVALID_FILL)):
                            with ui.HStack(height=18, spacing=6) as legend_row:
                                swatch = ui.Rectangle(width=14, height=14, style={
                                    "background_color": sp._abgr(fill), "border_color": 0xFF808080, "border_width": 1})
                                legend_label = ui.Label(text, style=small)
                            self.legend_rows.append(legend_row)
                            self.swatches.append((swatch, legend_label))
                        self.orientation = ui.Label(sp.ORIENTATION_LINE, height=18, style=small)
            self._set_image(self.legend_provider, sp.legend_image())
            self.dock_attempts = 0
            self.docked = self._dock()
            self.refresh()

        def _event(self, kind, **data):
            if self.recorder is not None:
                self.recorder.event(kind, sim_s=self.state.sim_now_s, **data)

        def _dock(self):
            """As SensorPanel._dock, with the target share instead of the 360 pt default."""
            self.dock_attempts += 1
            ui = self.ui
            viewport = ui.Workspace.get_window("Viewport")
            panel = ui.Workspace.get_window(self.window.title)
            if viewport is None or panel is None:
                return False
            if not panel.docked:
                panel.dock_in(viewport, ui.DockPosition.RIGHT, plan.PANEL_SHARE["target"])
            return bool(panel.docked)

        # ------------------------------------------------------------ Kit dock layout
        def _handles(self):
            W = self.ui.Workspace
            h = {k: W.get_window(t) for k, t in
                 (("stage", "Stage"), ("content", "Content"), ("viewport", "Viewport"), ("panel", self.window.title))}
            missing = [k for k, v in h.items() if v is None or not v.docked]
            return h, missing

        def measure(self):
            W = self.ui.Workspace
            h, missing = self._handles()
            if missing:
                return None, missing
            main = (W.get_main_window_width(), W.get_main_window_height())
            m = dict(main_pt=list(main), dpi=W.get_dpi_scale(),
                     right_w=W.get_dock_id_width(h["stage"].dock_id), right_h=W.get_dock_id_height(h["stage"].dock_id),
                     bottom_w=W.get_dock_id_width(h["content"].dock_id),
                     bottom_h=W.get_dock_id_height(h["content"].dock_id),
                     viewport_w=W.get_dock_id_width(h["viewport"].dock_id),
                     viewport_h=W.get_dock_id_height(h["viewport"].dock_id),
                     panel_w=W.get_dock_id_width(h["panel"].dock_id), panel_h=W.get_dock_id_height(h["panel"].dock_id),
                     panel_window=[self.window.width, self.window.height],
                     dock_ids={k: int(v.dock_id) for k, v in h.items()})
            m["fractions"] = plan.fractions(main[0], main[1], m["right_w"], m["bottom_h"], m["viewport_w"], m["panel_w"])
            m["within"] = plan.within(m["fractions"])
            m["scene_image_fraction"] = plan.scene_image_fraction(m["viewport_w"], m["viewport_h"], *main)
            return m, []

        def _layout_step(self):
            self.refreshes += 1
            if not self.docked or self.phase in ("failed",):
                return
            W = self.ui.Workspace
            main = (W.get_main_window_width(), W.get_main_window_height())
            if main != self.main_size:
                if self.main_size is not None:
                    self.replans += 1
                    if self.replans > MAX_REPLANS:
                        self.phase = "failed"
                        self._event("failed", reason="main window kept changing", main_pt=list(main))
                        return
                self.main_size, self.phase, self.attempts = main, "outer", 0
                self._event("main_window", main_pt=list(main), dpi=W.get_dpi_scale())
            if self.refreshes - self.last_step < SETTLE_REFRESHES or self.phase == "applied":
                return
            self.last_step = self.refreshes
            h, missing = self._handles()
            if missing:
                self.attempts += 1
                if self.attempts > MAX_ATTEMPTS:
                    self.phase = "failed"
                self._event("missing_windows", missing=missing, attempts=self.attempts)
                return
            if self.phase == "outer":
                self.attempts += 1
                if self.attempts > MAX_ATTEMPTS:
                    self.phase = "failed"
                    self._event("failed", reason="outer sizes not reached", measured=self.measure()[0])
                    return
                t = plan.targets(*main)
                W.set_dock_id_width(h["stage"].dock_id, t["right_w"])
                W.set_dock_id_height(h["content"].dock_id, t["bottom_h"])
                self.phase = "inner"
                self._event("set_outer", attempt=self.attempts, request=t)
            elif self.phase == "inner":
                m, _ = self.measure()
                if not (m["within"]["right"] and m["within"]["bottom"]):
                    self.phase = "outer"
                    self._event("outer_not_reached", measured=m)
                    return
                width = plan.panel_target(m["viewport_w"] + m["panel_w"], m["viewport_h"])
                W.set_dock_id_width(h["panel"].dock_id, width)
                self.phase = "verify"
                self._event("set_panel", request=dict(panel_w=width), measured=m)
            elif self.phase == "verify":
                m, _ = self.measure()
                if not all(m["within"].values()):
                    self.phase = "outer"
                    self._event("verify_failed", measured=m)
                    return
                self.apply_metrics(plan.panel_metrics(*m["panel_window"]))
                self.phase = "applied"
                self._event("applied", measured=m, metrics=self.metrics)
                if self.recorder is not None:
                    self.recorder.record.update(phase="applied", final=dict(measured=m, metrics=self.metrics))
                    self.recorder.write()

        def apply_metrics(self, m):
            ui, k = self.ui, m["text_scale"]
            self.metrics = m
            self.fonts = dict(bold=m["bold"], small=m["small"], cell_font=m["cell_font"])
            for key, label in self.labels.items():
                size = self.fonts[self.kind[key]]
                style = {"font_size": size, "color": 0xFFFFFFFF if self.kind[key] == "bold" else 0xFFC8C8C8}
                label.set_style(style)
                label.height = ui.Pixel(size + 6)
            small = {"font_size": m["small"], "color": 0xFFC8C8C8}
            self.rgb_image.height = ui.Pixel(m["rgb_h"])
            self.grid_row.height = ui.Pixel(m["grid"])
            self.grid_col.width = ui.Pixel(m["grid"])
            for hrow, row in zip(self.rows, self.cells):
                hrow.height = ui.Pixel(m["cell"])
                for _, label, z in row:
                    z.width = z.height = ui.Pixel(m["cell"])
            self.legend_image.height = ui.Pixel(m["legend_h"])
            self.legend_minmax.height = ui.Pixel(m["small"] + 6)
            for widget in (self.legend_min, self.legend_max, self.orientation):
                widget.set_style(small)
            self.orientation.height = ui.Pixel(m["small"] + 6)
            for legend_row, (swatch, legend_label) in zip(self.legend_rows, self.swatches):
                legend_row.height = ui.Pixel(m["small"] + 8)
                swatch.width = swatch.height = ui.Pixel(round(14 * k))
                legend_label.set_style(small)

        def refresh(self, images=False):
            handle = self.ui.Workspace.get_window(self.window.title)
            self.docked = bool(handle is not None and handle.docked)
            if not self.docked and self.dock_attempts < 5:
                self.docked = self._dock()
            try:
                self._layout_step()
            except Exception as error:  # layout is cosmetic; keep showing the sensors
                self.phase = "failed"
                self._event("error", error=repr(error))
            s = self.state
            lines = s.lines()
            for key, text in lines.items():
                if key in self.labels:
                    self.labels[key].text = text
            self.labels["status"].tooltip = lines["clock"]
            self.labels["rgb_title"].tooltip = lines["rgb_detail"]
            stale = s.status().state != "live"
            self.labels["status"].set_style({"font_size": self.fonts["bold"],
                                             "color": 0xFF3C8CF0 if stale else 0xFF00B976})
            if not images:
                return
            self._set_image(self.rgb_provider, s.rgb)
            self.legend_min.text = f"{s.tof.min_m:.1f} m"
            self.legend_max.text = f"{s.tof.max_m:.1f} m"
            for row, widgets in zip(s.tof.cells, self.cells):
                for cell, (rect, label, _) in zip(row, widgets):
                    rect.set_style({"background_color": sp._abgr(cell.fill)})
                    label.text = cell.text
                    label.set_style({"color": sp._abgr(cell.text_fill), "font_size": self.fonts["cell_font"]})

    return LayoutPanel


def install(sp, sidecar, header):
    """Route sensor_panel.attach (looked up by football_scene at build time) through LayoutPanel."""
    LayoutPanel = make_panel_class(sp)
    state = dict(recorder=None)

    def attach(env, argv):
        if not sp.kit_window_requested(argv):
            return None
        recorder = Recorder(sidecar, header)
        state["recorder"] = recorder
        attachment = sp.Attachment(env.beta_rig, panel_factory=lambda s: LayoutPanel(s, recorder=recorder))
        # Stored in scene-contract.json by football_scene.py through Attachment.close().
        attachment.record["display_layout"] = dict(kind="course native dock layout (display only)",
                                                   sidecar=str(sidecar), wrapper_sha256=header["wrapper_sha256"],
                                                   record=recorder.record)
        return attachment

    sp.attach = attach
    return state
