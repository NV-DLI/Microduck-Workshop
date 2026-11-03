"""One Isaac window, independently initialized and recorded football episodes."""
from pathlib import Path
import json
import random
import sys
import time
import gc
import weakref
import os
import stat
import math

JITTER_X_M = .10
JITTER_Y_M = .10
JITTER_YAW_RAD = math.pi / 6


def initial_position(seed, index, center):
    return initial_pose(seed, index, center)[:2]


def initial_pose(seed, index, center):
    """Nominal first round; later independent reproducible XY and yaw offsets."""
    if index == 1:
        return (*center, 0.0)
    rng = random.Random(seed + index - 1)
    return (center[0] + rng.uniform(-JITTER_X_M, JITTER_X_M),
            center[1] + rng.uniform(-JITTER_Y_M, JITTER_Y_M),
            rng.uniform(-JITTER_YAW_RAD, JITTER_YAW_RAD))


def set_argument(argv, flag, value):
    """Replace all spellings, including --flag=value, without duplicate values."""
    result = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag:
            i += 2
        elif token.startswith(flag + "="):
            i += 1
        else:
            result.append(token)
            i += 1
    argv[:] = [*result, flag, str(value)]


def close_environment_window(env):
    """Break native widget callback roots before IsaacLab drops its window handle."""
    window = getattr(env.unwrapped, "_window", None)
    if window is None:
        return
    native = getattr(window, "ui_window", None)
    if native is not None:
        native.frame.clear()
        native.visible = False
        native.destroy()
        window.ui_window = None
    window.ui_window_elements.clear()
    window._ui_listeners.clear()
    window.env = None
    window.stage = None
    env.unwrapped._window = None


def release_archive_cache(folder):
    """Best-effort cache advice for this closed episode's own immutable evidence.

    Spark shares RAM with CUDA. Keep archived camera frames on disk without
    accumulating a page-cache copy over an unlimited interactive session.
    This never removes files, changes their contents, or drops system caches.
    """
    result = dict(files=0, bytes=0, errors=0)
    if not hasattr(os, "posix_fadvise"):
        return dict(result, supported=False)
    for path in folder.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                info = os.fstat(fd)
                if stat.S_ISREG(info.st_mode):
                    os.fsync(fd)
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    result["files"] += 1
                    result["bytes"] += info.st_size
            finally:
                os.close(fd)
        except OSError:
            result["errors"] += 1
    return result


class ResetSession:
    def __init__(self, args):
        import carb.input
        import omni.appwindow
        import omni.kit.app
        self.args = args
        self.root = args.output
        self.root.mkdir(parents=True, exist_ok=False)
        if args.ball_xy is None:
            args.ball_xy = [.02, 0.0]
        self.seed = args.seed
        self.center = (args.root_x, args.root_y if args.root_y is not None else .12)
        self.index = 0
        self.requested = False
        self.down = False
        self.rows = []
        self.resources = []
        self.previous_env = None
        self.previous_unwrapped = None
        self.app = omni.kit.app.get_app()
        self.input = carb.input.acquire_input_interface()
        self.keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        def key(event, *unused):
            if event.input != carb.input.KeyboardInput.R:
                return True
            if event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                self.down = False
            elif event.type == carb.input.KeyboardEventType.KEY_PRESS and not self.down:
                self.down = True
                self.requested = True
            return True
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, key)
        self.window = None
        self.write_status("starting")

    def write_status(self, state):
        self.state = state
        value = dict(state=state, episode=self.index, output=str(self.args.output),
                     seed=self.args.seed, root_xy=[self.args.root_x, self.args.root_y],
                     root_yaw_offset_rad=getattr(self.args, "root_yaw_rad", 0.0),
                     jitter_xy_m=[JITTER_X_M, JITTER_Y_M], jitter_yaw_rad=JITTER_YAW_RAD,
                     yaw_convention="world-Z offset applied to the measured standing61 XYZW quaternion",
                     episodes=self.rows, resources=self.resources)
        tmp = self.root / "session.json.tmp"
        tmp.write_text(json.dumps(value, indent=2) + "\n")
        tmp.replace(self.root / "session.json")

    def next_episode(self):
        if not self.app.is_running():
            return False
        if self.index and not self.requested:
            return False
        if self.args.session_episodes and self.index >= self.args.session_episodes:
            return False
        self.index += 1
        self.requested = False
        self.args.output = self.root / f"episode-{self.index:03d}"
        self.args.seed = self.seed + self.index - 1
        self.args.root_x, self.args.root_y, self.args.root_yaw_rad = initial_pose(self.seed, self.index, self.center)
        # Native contract and scene hooks derive provenance from the current argv.
        for flag, value in (("--output", self.args.output), ("--seed", self.args.seed),
                            ("--root-x", self.args.root_x), ("--root-y", self.args.root_y),
                            ("--root-yaw-rad", self.args.root_yaw_rad)):
            set_argument(sys.argv, flag, value)
        self.resources.append(self.resource_snapshot("before_build"))
        self.write_status("running")
        return True

    def should_interrupt(self):
        return self.requested or not self.app.is_running()

    def hold(self, env, report):
        import carb.settings
        import omni.timeline
        import omni.ui as ui
        self.previous_env = weakref.ref(env)
        self.previous_unwrapped = weakref.ref(env.unwrapped)
        self.resources.append(self.resource_snapshot("episode_finished"))
        reason = report.get("terminal_reason")
        interrupted = reason in ("user_reset", "window_closed")
        self.rows.append(dict(index=self.index, seed=self.args.seed,
            root_xy=[self.args.root_x, self.args.root_y], output=str(self.args.output),
            root_yaw_offset_rad=self.args.root_yaw_rad,
            applied_root_pose_xyzw=report.get("fixture", {}).get("root_pose_xyzw"),
            applied_yaw_rad=report.get("initialization_reference", {}).get("applied_yaw_rad"),
            interrupted=interrupted, terminal_reason=reason,
            physical_success=bool(report.get("scorer", {}).get("physical_episode_success")),
            simulated_seconds=report.get("simulated_seconds"),
            loop_wall_seconds=report.get("loop_wall_seconds")))
        if interrupted:
            self.write_status("interrupted")
            return
        if self.args.session_episodes and self.index >= self.args.session_episodes:
            self.write_status("limit_reached")
            return
        settings = carb.settings.get_settings()
        prior = settings.get("/app/player/playSimulations")
        omni.timeline.get_timeline_interface().pause()
        settings.set_bool("/app/player/playSimulations", False)
        success = self.rows[-1]["physical_success"]
        self.window = ui.Window("Football | R to restart", width=370, height=90)
        with self.window.frame:
            with ui.VStack(spacing=8):
                ui.Label("Goal!" if success else "Episode finished", alignment=ui.Alignment.CENTER)
                ui.Label("Press R to restart / 按 R 再试一次", alignment=ui.Alignment.CENTER)
        self.write_status("waiting_for_r")
        print("FOOTBALL_WAITING_FOR_R", self.index, flush=True)
        try:
            while self.app.is_running() and not self.requested:
                self.app.update()
                time.sleep(.05)
        finally:
            self.window.destroy()
            self.window = None
            if prior is not None:
                settings.set("/app/player/playSimulations", prior)
            self.write_status("rebuilding" if self.requested else "closed")

    def resource_snapshot(self, phase):
        import torch
        free, total = torch.cuda.mem_get_info()
        mem = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value = line.split(":", 1)
            if name in ("MemFree", "MemAvailable", "Cached"):
                mem[name + "_kb"] = int(value.split()[0])
        return dict(episode=self.index, phase=phase, cuda_free_bytes=free,
            cuda_total_bytes=total, torch_allocated_bytes=torch.cuda.memory_allocated(),
            torch_reserved_bytes=torch.cuda.memory_reserved(), **mem)

    def finalized(self):
        # Scene/native finalizers have dropped the closed rig and panel references.
        # Drain Kit destruction callbacks, then return unused allocator blocks.
        import torch
        for _ in range(3):
            if self.app.is_running():
                self.app.update()
        gc.collect()
        torch.cuda.empty_cache()
        cache = release_archive_cache(self.args.output)
        row = self.resource_snapshot("after_cleanup")
        row["closed_archive_cache"] = cache
        row.update(old_env_released=self.previous_env is None or self.previous_env() is None,
                   old_unwrapped_released=self.previous_unwrapped is None or self.previous_unwrapped() is None)
        self.resources.append(row)
        self.write_status("episode_finalized")

    def record_technical_error(self, error, phase):
        """Keep a completed physical result while reporting a lifecycle failure."""
        row = next((r for r in self.rows if r["index"] == self.index), None)
        if row is None:
            row = dict(index=self.index, seed=self.args.seed,
                root_xy=[self.args.root_x, self.args.root_y], output=str(self.args.output),
                root_yaw_offset_rad=getattr(self.args, "root_yaw_rad", 0.0),
                status="technical_error")
            self.rows.append(row)
        row.update(technical_error=True, technical_error_detail=repr(error),
                   technical_error_phase=phase)
        self.write_status("technical_error")

    def close(self):
        if self.index and not any(row["index"] == self.index for row in self.rows):
            path = self.args.output / "manifest.json"
            report = json.loads(path.read_text()) if path.exists() else {}
            self.rows.append(dict(index=self.index, seed=self.args.seed,
                root_xy=[self.args.root_x, self.args.root_y], output=str(self.args.output),
                root_yaw_offset_rad=getattr(self.args, "root_yaw_rad", 0.0),
                status="technical_error", error=report.get("error", "episode did not finalize")))
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)
        if self.window is not None:
            self.window.destroy()
        self.write_status("closed")
