"""Isaac Sim window check of the head-sensor panel, without physics (needs a GPU and a display).

Opens Kit with the course window settings, attaches the real panel to a stand-in rig that
returns frames from a finished episode (frames/*-rgb.png and the 8x8 tof_frame in
sensor-inputs.jsonl, at their recorded simulation times), and redraws the window
`--render-subframes` times per capture as the executors' rig does. It checks docking,
the cell texts against the recorded ToF frame, the LIVE/STALE header, the bounded refresh
and the cleanup, then writes probe-result.json and one swapchain screenshot.

This is a replay of recorded sensor frames to test the display; it is not a simulation
result and its screenshot is not course media. The live check is a short course run
(see the runtime report).

  python lab/course_labs/sensor_panel_probe.py --recording <episode> --output <new folder>
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Match the course launcher's ARM library ordering before importing Isaac Lab.
openmp = Path("/lib/aarch64-linux-gnu/libgomp.so.1")
preload = os.environ.get("LD_PRELOAD", "")
if openmp.is_file() and str(openmp) not in preload.replace(":", " ").split():
    env = dict(os.environ, LD_PRELOAD=":".join(filter(None, (str(openmp), preload))))
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--recording", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--frames", type=int, default=45)
parser.add_argument("--start", type=int, default=0, help="first capture ordinal to replay")
parser.add_argument("--render-subframes", type=int, default=4)
parser.add_argument("--role", choices=["onboard", "groot"], default="onboard",
                    help="which consumer captions to show (groot for a GR00T episode)")
a = parser.parse_args()
if a.output.exists():
    raise SystemExit("Choose a new output folder: " + str(a.output))
rows = []
with (a.recording / "sensor-inputs.jsonl").open() as stream:
    for i, line in enumerate(stream):
        if i >= a.start + a.frames:
            break
        if i >= a.start:
            rows.append(json.loads(line))
if len(rows) < 3:
    raise SystemExit("Need at least three recorded captures")
a.output.mkdir(parents=True)

from isaaclab.app import launch_simulation  # noqa: E402
from kit_settings import kit_args  # noqa: E402

checks = {}
result = dict(kind="sensor panel display probe (replayed recorded frames; not a simulation)",
                          recording=str(a.recording.resolve()), captures=len(rows),
                          render_subframes=a.render_subframes, checks=checks)
with launch_simulation(None, {"device": "cuda:0", "require_kit": True, "visualizer": ["kit"],
                              "kit_args": kit_args()}):
    import cv2
    import omni.kit.app
    import omni.ui as ui
    import sensor_panel

    app = omni.kit.app.get_app()
    for _ in range(10):
        app.update()

    class ReplayRig:
        def __init__(self):
            self.index = 0

        def capture(self, *, timestamp_s, render_subframes=3):
            for _ in range(render_subframes):
                app.update()  # the real rig calls sim.render() here
            row = rows[self.index]
            if abs(row["timestamp_s"] - timestamp_s) > 1e-9:
                raise RuntimeError("Replay out of order")
            self.index += 1
            bgr = cv2.imread(str(a.recording / row["rgb"]), cv2.IMREAD_COLOR)
            return dict(rgb_raw=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), tof=row["tof_frame"],
                        timestamp_s=row["timestamp_s"])

    rig = ReplayRig()
    hook = sensor_panel.Attachment(rig, role=a.role)
    checks["window_created"] = hook.record["enabled"] and hook.panel is not None
    wall = time.perf_counter()
    for row in rows:
        rig.capture(timestamp_s=row["timestamp_s"], render_subframes=a.render_subframes)
    result["replay_wall_seconds"] = time.perf_counter() - wall
    for _ in range(a.render_subframes):
        app.update()
    panel, last = hook.panel, rows[-1]
    window = ui.Workspace.get_window(sensor_panel.TITLE)
    checks["window_visible"] = bool(window is not None and window.visible)
    checks["docked_next_to_viewport"] = bool(getattr(panel, "docked", False)) and bool(getattr(window, "docked", False))
    if panel is not None and panel.state.tof is not None:
        shown_t = panel.state.frame_t_s
        expected = next(r for r in rows if r["timestamp_s"] == shown_t)["tof_frame"]
        texts = [[label.text for _, label in widgets] for widgets in panel.cells]
        want = [[c.text for c in row] for row in sensor_panel.tof_view(expected).cells]
        checks["tof_cell_texts_match_recorded_frame"] = texts == want
        status = panel.labels["status"]
        checks["header_live"] = status.text.startswith("LIVE")
        result["panel_text"] = {key: label.text for key, label in panel.labels.items()}
        checks["lines_fit_panel"] = all(len(t) <= sensor_panel.MAX_LINE_CHARS for t in result["panel_text"].values())
        result["window_size_px"] = [window.width, window.height] if window is not None else None
        renderer = None
        try:
            import omni.kit.renderer_capture as capture

            renderer = capture.acquire_renderer_capture_interface()
            renderer.capture_next_frame_swapchain(str(a.output / "probe-ui.png"))
            for _ in range(3):
                app.update()
            renderer.wait_async_capture()
            result["screenshot"] = "probe-ui.png (replayed recorded frames; layout check only)"
        except Exception as error:
            result["screenshot_error"] = repr(error)
        panel.state.set_clock(last["timestamp_s"] + 0.5)
        panel.refresh()
        app.update()
        checks["header_stale_after_0_5_s"] = status.text.startswith("STALE")
        result["header_stale_text"] = status.text
    record = hook.close()
    for _ in range(3):
        app.update()
    window = ui.Workspace.get_window(sensor_panel.TITLE)
    checks["closed"] = window is None or not window.visible
    checks["rig_capture_restored"] = rig.capture.__func__ is ReplayRig.capture
    stats = record["stats"]
    checks["refresh_bounded"] = stats["shown"] + stats["skipped_for_cadence"] == stats["captures"] == len(rows)
    result["panel_record"] = record
    result["mean_update_ms"] = 1000 * stats["update_seconds"] / max(stats["shown"], 1)
    result["passed"] = all(checks.values())
    (a.output / "probe-result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("SENSOR_PANEL_PROBE " + json.dumps(dict(passed=result["passed"], checks=checks)), flush=True)
