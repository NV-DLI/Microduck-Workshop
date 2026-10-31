"""CPU tests of the live head-sensor panel: conversion, validity, staleness, no input mutation.

Run with any Python that has numpy (no Kit, no GPU):
  python -m unittest tests/test_sensor_panel.py
Frames come from the course's own ToF sampler (beta_tof_geometry.sample_tof_from_depth).
Set MICRODUCK_SENSOR_RECORDING=<episode folder with sensor-inputs.jsonl> to also check one
recorded run.
"""
from pathlib import Path
import copy
import json
import os
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "lab/course_labs"), str(ROOT / "lab/runtime/simulation/core/sensors")]
import sensor_payload as sp  # noqa: E402
import sensor_panel  # noqa: E402
from beta_tof_geometry import TOF_PROFILE, sample_tof_from_depth, tof_zone_rays  # noqa: E402


def frame(depth, t=1.0, **kwargs):
    """A ToF frame dict exactly as IsaacBetaSensors.capture() returns it."""
    return sample_tof_from_depth(
        np.asarray(depth, np.float32), profile=TOF_PROFILE, depth_kind="ray_range",
        source_horizontal_fov_deg=50.0, source_vertical_fov_deg=50.0, samples_per_zone=4,
        timestamp_s=t, **kwargs,
    ).to_dict()


def capture(t, depth=None, rgb=None, **kwargs):
    rng = np.random.default_rng(int(t * 1000))
    return {
        "rgb_raw": rgb if rgb is not None else rng.integers(0, 256, (540, 960, 3), dtype=np.uint8),
        "tof": frame(np.full((32, 32), 1.2) if depth is None else depth, t, **kwargs),
        "tof_source_ray_range": np.zeros((32, 32), np.float32),
        "timestamp_s": t,
    }


class TofConversion(unittest.TestCase):
    def test_valid_ranges_keep_metres_and_scale(self):
        view = sp.tof_view(frame(np.full((32, 32), 1.2)))
        self.assertEqual(view.counts["valid"], 64)
        self.assertEqual((view.min_m, view.max_m), (0.1, 4.0))
        for row in view.cells:
            for cell in row:
                self.assertEqual(cell.text, "1.20")
                self.assertAlmostEqual(cell.range_m, 1.2, places=3)

    def test_no_return_and_too_close_are_distinct_from_ranges(self):
        depth = np.full((32, 32), np.inf)
        depth[16:, :] = 0.05  # lower half closer than the reliable minimum
        view = sp.tof_view(frame(depth))
        kinds = {cell.kind for row in view.cells for cell in row}
        self.assertEqual(kinds, {"no_return", "too_close"})
        for row in view.cells:
            for cell in row:
                self.assertIsNone(cell.range_m)
                self.assertIn(cell.text, ("NR", "<0.1"))
                self.assertIn(cell.fill, (sp.NO_RETURN_FILL, sp.INVALID_FILL))
        ranges = {sp.range_color(m) for m in np.linspace(0.1, 4.0, 40)}
        self.assertFalse(ranges & {sp.NO_RETURN_FILL, sp.INVALID_FILL})

    def test_modeled_dropouts_are_invalid(self):
        view = sp.tof_view(frame(np.full((32, 32), 1.0), dropout_probability=1.0))
        self.assertEqual(view.counts["invalid"], 64)
        self.assertTrue(all(c.text == "X" for r in view.cells for c in r))
        dropped = sp.tof_view(frame(np.full((32, 32), 1.0), frame_dropout_probability=1.0))
        self.assertTrue(dropped.frame_dropped)
        self.assertEqual(dropped.counts["valid"], 0)

    def test_near_is_warm_far_is_dark_and_clamped(self):
        near, far = sp.range_color(0.1), sp.range_color(4.0)
        self.assertGreater(sum(near), sum(far))
        self.assertEqual(sp.range_color(-1.0), near)
        self.assertEqual(sp.range_color(9.0), far)
        legend = sp.legend_image(64, 4)
        self.assertEqual(tuple(legend[0, 0, :3]), near)
        self.assertEqual(tuple(legend[0, -1, :3]), far)

    def test_grid_orientation_matches_sensor_rays(self):
        # Label "row 0 top, column 0 left (as seen from the robot)" must match the FLU rays.
        rays = tof_zone_rays()
        self.assertGreater(rays[0][1], 0)  # +y is left
        self.assertGreater(rays[0][2], 0)  # +z is up
        self.assertLess(rays[63][1], 0)
        self.assertLess(rays[63][2], 0)
        depth = np.full((32, 32), 3.0)
        depth[:, :16] = 0.5  # image left (optical +x right) is sensor left
        view = sp.tof_view(frame(depth))
        self.assertLess(view.cells[4][0].range_m, view.cells[4][7].range_m)

    def test_rejects_malformed_frames(self):
        good = frame(np.full((32, 32), 1.0))
        bad = copy.deepcopy(good)
        bad["cells"] = bad["cells"][:7]
        with self.assertRaises(ValueError):
            sp.tof_view(bad)
        bad = copy.deepcopy(good)
        bad["cells"][0][0]["status"] = "maybe"
        with self.assertRaises(ValueError):
            sp.tof_view(bad)
        bad = copy.deepcopy(good)
        bad["cells"][0][0]["range_m"] = None
        with self.assertRaises(ValueError):
            sp.tof_view(bad)


class RgbPreview(unittest.TestCase):
    def test_area_average_and_shape(self):
        rgb = np.zeros((540, 960, 3), np.uint8)
        rgb[:3, :3] = [[[9, 0, 255]] * 3, [[0, 0, 255]] * 3, [[0, 0, 255]] * 3]
        out = sp.rgb_preview(rgb, 3)
        self.assertEqual(out.shape, (180, 320, 4))
        self.assertEqual(tuple(out[0, 0]), (3, 0, 255, 255))
        self.assertFalse(np.shares_memory(out, rgb))

    def test_cv2_and_numpy_paths_agree(self):
        rgb = np.random.default_rng(3).integers(0, 256, (540, 960, 3), dtype=np.uint8)
        fast = sp.rgb_preview(rgb)
        saved = sys.modules.get("cv2")
        sys.modules["cv2"] = None  # import cv2 now raises ImportError
        try:
            plain = sp.rgb_preview(rgb)
        finally:
            if saved is None:
                del sys.modules["cv2"]
            else:
                sys.modules["cv2"] = saved
        np.testing.assert_array_equal(fast, plain)

    def test_rejects_non_rgb(self):
        for bad in (np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32)):
            with self.assertRaises(ValueError):
                sp.rgb_preview(bad)


class Freshness(unittest.TestCase):
    def test_states(self):
        self.assertEqual(sp.freshness(1.0, None).state, "waiting")
        live = sp.freshness(1.0 + 1 / 15, 1.0)
        self.assertEqual(live.state, "live")
        self.assertIn("age 67 ms", live.text)
        self.assertEqual(sp.freshness(1.0 + 2 / 15, 1.0).state, "live")
        self.assertEqual(sp.freshness(1.0 + 3 / 15, 1.0).state, "stale")
        with self.assertRaises(ValueError):
            sp.freshness(0.9, 1.0)


class FakeRig:
    def __init__(self):
        self.calls = []

    def capture(self, *, timestamp_s, render_subframes=3):
        self.calls.append(timestamp_s)
        self.last = capture(timestamp_s)
        return self.last


class FakePanel:
    def __init__(self, state, fail_on=None):
        self.state, self.fail_on = state, fail_on
        self.refreshes, self.destroyed, self.docked = [], False, True

    def refresh(self, images=False):
        if self.fail_on == ("images" if images else "clock"):
            raise RuntimeError("display failure")
        self.refreshes.append((images, self.state.sim_now_s, self.state.frame_t_s))

    def destroy(self):
        self.destroyed = True


class Hook(unittest.TestCase):
    def test_passes_capture_through_unchanged_and_shows_age(self):
        rig = FakeRig()
        original = rig.capture
        state = sensor_panel.PanelState(min_interval_s=0)
        hook = sensor_panel.Attachment(rig, FakePanel, state)
        first = rig.capture(timestamp_s=1.0, render_subframes=4)
        self.assertIs(first, rig.last)
        rgb_copy, tof_copy = first["rgb_raw"].copy(), copy.deepcopy(first["tof"])
        rig.capture(timestamp_s=1.0 + 1 / 15)
        # The frame shown during the second capture's renders is the first one, 67 ms old.
        images, now, shown = hook.panel.refreshes[2]
        self.assertEqual((images, now, shown), (False, 1.0 + 1 / 15, 1.0))
        np.testing.assert_array_equal(first["rgb_raw"], rgb_copy)
        self.assertEqual(first["tof"], tof_copy)
        self.assertEqual(rig.calls, [1.0, 1.0 + 1 / 15])
        panel = hook.panel
        record = hook.close()
        self.assertTrue(panel.destroyed)
        self.assertEqual(rig.capture, original)
        self.assertEqual(record["stats"]["shown"], 2)
        self.assertFalse(record["extra_render_or_readback"])

    def test_refresh_cadence_is_bounded(self):
        ticks = iter([0.0, 0.01, 0.02, 0.2])
        state = sensor_panel.PanelState(min_interval_s=0.1, clock=lambda: next(ticks))
        hook = sensor_panel.Attachment(FakeRig(), FakePanel, state)
        for k in range(4):
            hook.rig.capture(timestamp_s=1 + k / 15)
        self.assertEqual(state.stats, dict(captures=4, shown=2, skipped_for_cadence=2,
                                           update_seconds=state.stats["update_seconds"]))
        self.assertEqual(state.frame_t_s, 1 + 3 / 15)

    def test_display_failure_never_stops_capture(self):
        for where in ("images", "clock"):
            rig = FakeRig()
            hook = sensor_panel.Attachment(rig, lambda s: FakePanel(s, fail_on=where),
                                           sensor_panel.PanelState(min_interval_s=0))
            for k in range(3):
                self.assertIs(rig.capture(timestamp_s=1 + k / 15), rig.last)
            self.assertFalse(hook.record["enabled"])
            self.assertIn("display failure", hook.record["error"])
            self.assertEqual(len(rig.calls), 3)

    def test_failed_window_creation_is_recorded(self):
        def broken(state):
            raise ImportError("no omni.ui")

        rig = FakeRig()
        hook = sensor_panel.Attachment(rig, broken)
        self.assertIs(rig.capture(timestamp_s=0.5), rig.last)
        self.assertFalse(hook.record["enabled"])

    def test_headless_needs_no_ui(self):
        before = set(sys.modules)
        env = type("Env", (), {"beta_rig": FakeRig()})()
        for argv in ([], ["--visualizer", "none"], ["--headless"]):
            self.assertIsNone(sensor_panel.attach(env, argv))
        self.assertFalse({m for m in set(sys.modules) - before if m.startswith("omni")})
        self.assertTrue(sensor_panel.kit_window_requested(["--seed", "1", "--visualizer", "kit"]))
        self.assertTrue(sensor_panel.kit_window_requested(["--visualizer=kit"]))
        self.assertFalse(sensor_panel.kit_window_requested(["--visualizer"]))

    def test_panel_text(self):
        state = sensor_panel.PanelState(min_interval_s=0)
        self.assertIn("Waiting", state.header())
        depth = np.full((32, 32), 2.0)
        depth[24:] = 0.05
        state.accept(capture(2.0, depth))
        state.set_clock(2.0 + 1 / 15)
        self.assertEqual(state.lines()["status"], "LIVE, frame age 67 ms")
        state.set_clock(2.0 + 4 / 15)
        self.assertIn("STALE", state.header())
        self.assertIn("960x540", state.rgb_caption())
        self.assertRegex(state.lines()["tof_zones"], r"^64 zones: \d+ valid, 0 NR, \d+ invalid$")
        self.assertEqual(state.lines()["tof_role"], "Controller input")
        self.assertIn("Detector input", state.rgb_caption())
        groot = sensor_panel.PanelState("groot")
        self.assertIn("Not a GR00T input", groot.tof_caption())
        self.assertIn("GR00T image input", groot.rgb_caption())

    def test_every_line_fits_the_panel(self):
        dropped = capture(3.0, np.full((32, 32), 1.0), frame_dropout_probability=1.0)
        for role in sensor_panel.ROLES:
            for frame in (None, capture(2.0), dropped):
                state = sensor_panel.PanelState(role, min_interval_s=0)
                if frame is not None:
                    state.accept(frame)
                    state.set_clock(frame["timestamp_s"] + 123.4)
                texts = [*state.lines().values(), *sensor_panel.LEGEND_LINES, sensor_panel.ORIENTATION_LINE]
                for text in texts:
                    self.assertLessEqual(len(text), sensor_panel.MAX_LINE_CHARS, text)
        self.assertIn("not free space", sensor_panel.LEGEND_LINES[0])

    def test_layout_for_4k_desktop(self):
        # Kit window about 1920 px wide on a 3840x2160 desktop: the panel takes 360 px.
        self.assertAlmostEqual(sensor_panel.dock_ratio(1920), 360 / 1920)
        self.assertEqual(sensor_panel.dock_ratio(3840), 0.12)
        self.assertEqual(sensor_panel.dock_ratio(600), 0.35)
        self.assertGreaterEqual(sensor_panel.PANEL_PX, 8 * sensor_panel.CELL_PX + 7 + 20)
        self.assertGreaterEqual(sensor_panel.PANEL_PX, 960 // sensor_panel.RGB_FACTOR + 20)

    def test_role_follows_command(self):
        env = type("Env", (), {"beta_rig": FakeRig()})()
        hook = sensor_panel.attach(env, ["--visualizer", "kit", "--walking-policy-server", "http://x"])
        self.assertEqual(hook.record["role"], "groot")
        hook.close()
        hook = sensor_panel.attach(env, ["--visualizer", "kit"])
        self.assertEqual(hook.record["role"], "onboard")
        hook.close()


class FakeUi:
    """Minimal stand-in for omni.ui 3.3 that records what the panel builds and uploads."""

    def __init__(self, viewport_width=1920, dock_ok=True):
        import types

        ui = types.ModuleType("omni.ui")
        self.ui, self.uploads, self.windows, self.labels = ui, [], [], []
        test = self

        class Widget:
            def __init__(self, *args, **kwargs):
                self.args, self.kwargs, self.style = args, kwargs, kwargs.get("style")
                self.text = args[0] if args and isinstance(args[0], str) else ""

            def set_style(self, style):
                self.style = style

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        class Label(Widget):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                test.labels.append(self)

        class Provider:
            def set_data_array(self, array, sizes, strict=False):
                if not strict:
                    raise AssertionError("uploads must be strict")
                if array.dtype != np.uint8 or list(array.shape[1::-1]) != list(sizes) or array.shape[2] != 4:
                    raise ValueError("malformed upload")
                test.uploads.append(tuple(sizes))

        class Window(Widget):
            def __init__(self, title, **kwargs):
                super().__init__(title, **kwargs)
                self.title, self.docked = title, False
                self.frame, self.visible, self.destroyed = Widget(), True, False
                test.windows.append(self)

            def dock_in(self, target, position, ratio):
                self.dock = (target.title, position, ratio)
                self.docked = dock_ok

            def destroy(self):
                self.destroyed = True

        viewport = type("Viewport", (), {"title": "Viewport", "width": viewport_width})()
        enum = lambda *names: type("Enum", (), {n: n for n in names})  # noqa: E731
        ui.Window, ui.Label, ui.ByteImageProvider = Window, Label, Provider
        for name in ("VStack", "HStack", "ZStack", "Rectangle", "Spacer", "ImageWithProvider", "ScrollingFrame"):
            setattr(ui, name, Widget)
        ui.Pixel = ui.Fraction = lambda v: v
        ui.Alignment, ui.DockPosition = enum("CENTER", "RIGHT"), enum("RIGHT")
        ui.IwpFillPolicy = enum("IWP_PRESERVE_ASPECT_FIT", "IWP_STRETCH")
        ui.ScrollBarPolicy = enum("SCROLLBAR_ALWAYS_OFF")
        def get_window(title):
            return viewport if title == "Viewport" else next(
                (w for w in test.windows if w.title == title and not w.destroyed), None)
        ui.Workspace = type("Workspace", (), {"get_window": staticmethod(get_window)})

    def __enter__(self):
        import types

        self.saved = {k: sys.modules.get(k) for k in ("omni", "omni.ui")}
        omni = types.ModuleType("omni")
        omni.ui = self.ui
        sys.modules.update({"omni": omni, "omni.ui": self.ui})
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class PanelWidgets(unittest.TestCase):
    """Builds the real SensorPanel against a recording omni.ui stand-in (no Kit, no GPU)."""

    def test_build_refresh_and_close(self):
        with FakeUi() as fake:
            rig = FakeRig()
            hook = sensor_panel.Attachment(rig, state=sensor_panel.PanelState("groot", min_interval_s=0))
            self.assertTrue(hook.record["enabled"], hook.record["error"])
            window = fake.windows[0]
            self.assertEqual(window.dock[0], "Viewport")
            self.assertAlmostEqual(window.dock[2], 360 / 1920)
            rig.capture(timestamp_s=1.0)
            rig.capture(timestamp_s=1.0 + 1 / 15)
            panel = hook.panel
            self.assertEqual(fake.uploads[0], (256, 10))  # legend strip
            self.assertEqual(fake.uploads[1:], [(320, 180), (320, 180)])
            texts = [[label.text for _, label in row] for row in panel.cells]
            want = [[c.text for c in row] for row in sp.tof_view(rig.last["tof"]).cells]
            self.assertEqual(texts, want)
            shown = {key: label.text for key, label in panel.labels.items()}
            self.assertEqual(shown["status"], "LIVE, frame age 0 ms")
            self.assertEqual(shown["tof_role"], "Not a GR00T input")
            record = hook.close()
            self.assertTrue(window.destroyed)
            self.assertTrue(record["docked_next_to_viewport"])
            self.assertIs(rig.capture.__func__, FakeRig.capture)

    def test_dock_retry_then_float(self):
        with FakeUi(dock_ok=False):
            rig = FakeRig()
            hook = sensor_panel.Attachment(rig, state=sensor_panel.PanelState(min_interval_s=0))
            for k in range(8):
                rig.capture(timestamp_s=1 + k / 15)
            self.assertEqual(hook.panel.dock_attempts, 5)
            self.assertTrue(hook.record["enabled"])
            self.assertFalse(hook.close()["docked_next_to_viewport"])

    def test_late_docking_is_observed_after_retry_limit(self):
        with FakeUi(dock_ok=False) as fake:
            rig = FakeRig()
            hook = sensor_panel.Attachment(rig, state=sensor_panel.PanelState(min_interval_s=0))
            for k in range(8):
                rig.capture(timestamp_s=1 + k / 15)
            self.assertEqual(hook.panel.dock_attempts, 5)
            fake.windows[0].docked = True  # Kit applies the pending layout on an update.
            rig.capture(timestamp_s=2.0)
            self.assertTrue(hook.close()["docked_next_to_viewport"])

    def test_malformed_upload_disables_panel_not_run(self):
        with FakeUi():
            rig = FakeRig()
            hook = sensor_panel.Attachment(rig, state=sensor_panel.PanelState(min_interval_s=0))
            bad = capture(1.0)
            bad["rgb_raw"] = bad["rgb_raw"][:, :, :2]
            rig.last = None
            original = hook.original
            hook.original = lambda **kw: bad
            self.assertIs(rig.capture(timestamp_s=1.0), bad)
            self.assertFalse(hook.record["enabled"])
            hook.original = original


class SceneWiring(unittest.TestCase):
    """football_scene.py is the shared entry of onboard football and GR00T validation."""

    def test_hook_attaches_closes_and_hashes(self):
        text = (ROOT / "lab/football_scene.py").read_text()
        self.assertIn("attach_sensor_panel(env, remaining)", text)
        self.assertIn('scene_record["sensor_panel"] = panel["attachment"].close()', text)
        finally_block = text[text.index("def run_and_finalize"):]
        self.assertLess(finally_block.index("finally:"), finally_block.index('panel["attachment"].close()'))
        for name in ("sensor_panel.py", "sensor_payload.py"):
            self.assertIn(f'HERE / "course_labs/{name}"', text)


@unittest.skipUnless(os.environ.get("MICRODUCK_SENSOR_RECORDING"), "no recording given")
class Recorded(unittest.TestCase):
    def test_recorded_frames(self):
        folder = Path(os.environ["MICRODUCK_SENSOR_RECORDING"])
        with (folder / "sensor-inputs.jsonl").open() as stream:
            rows = [json.loads(line) for _, line in zip(range(30), stream)]
        for row in rows:
            view = sp.tof_view(row["tof_frame"])
            statuses = [c["status"] for r in row["tof_frame"]["cells"] for c in r]
            self.assertEqual(view.counts["valid"], statuses.count("valid"))
            self.assertEqual(view.counts["no_return"], statuses.count("no_return"))
            self.assertEqual(view.timestamp_s, row["timestamp_s"])


if __name__ == "__main__":
    unittest.main()
