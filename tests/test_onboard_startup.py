"""CPU tests of the onboard football start-up (standing61 joint brake) and its entry wiring.

Run with the course runtime Python (numpy and torch; no Kit, no GPU):
  CUDA_VISIBLE_DEVICES= <runtime python> -m unittest tests/test_onboard_startup.py

Covered without physics:
- the start-up gate never enables policy, navigation, scoring or the task clock before the
  release, and never before settled + delivered RGB/ToF + finite BODY50 data;
- the external wrench is zero on every row and any nonzero actual wrench fails the run;
- the joint brake restores every original limit and physical field, and refuses to report a
  release when a field changed;
- in the executor source, HOME is held while braked, and the release happens after the first
  actor output is computed and before it is applied (static order check);
- the onboard entry selects the student executor, refuses GR00T options and preloaded GR00T
  skill modules; the executor requires exactly one route;
- the grounded reference pose is the recorded one for this football pitch.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import sys
import types
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "lab/groot_walking/student"
EXECUTOR = STUDENT / "walking_student_executor.py"
sys.path[:0] = [
    str(ROOT / "lab/course_labs"),
    str(ROOT / "lab/runtime/simulation/core/sensors"),
    str(STUDENT),
    str(ROOT / "lab"),
]
import onboard_startup as gate  # noqa: E402
from beta_tof_geometry import TOF_PROFILE, sample_tof_from_depth  # noqa: E402
from joint_brake import JointBrake, NoSupportStartup  # noqa: E402

DT = 0.005


def tof(depth_m=1.0, t=1.0):
    """A ToF frame dict exactly as IsaacBetaSensors.capture() returns it."""
    return sample_tof_from_depth(
        np.full((32, 32), depth_m, np.float32), profile=TOF_PROFILE, depth_kind="ray_range",
        source_horizontal_fov_deg=50.0, source_vertical_fov_deg=50.0, samples_per_zone=4,
        timestamp_s=t,
    ).to_dict()


def rgb(seed=0):
    return np.random.default_rng(seed).integers(0, 256, gate.RGB_SHAPE, dtype=np.uint8)


def body_packet():
    return dict(
        gyro_body_rad_s=[0.001, -0.002, 0.0], gravity_body=[0.0, 0.0, -1.0],
        encoder_q_rad=[0.1] * 14, encoder_qd_rad_s=[0.0] * 14,
    )


def still(t, gate_or_inner, moving=False):
    return gate_or_inner.update(
        t, position_world=[-0.55, 0.04, 0.1168], quaternion_xyzw=[0, 0, 0, 1],
        velocity_world=[0.01 if moving else 0.0, 0, 0], angular_velocity_world=[0, 0, 0],
        imu_orientation_valid=t >= 0.5,
    )


def new_gate():
    return gate.OnboardBrakeStartup(NoSupportStartup([-0.55, 0.04, 0.12, 0, 0, 0, 1], 0.6))


def run_until(g, t_end, start=0.0, **kwargs):
    rows, t = [], start
    while t < t_end - 1e-9:
        t = round(t + DT, 6)
        rows.append(still(t, g, **kwargs))
    return rows, t


class CaptureAndImuReadiness(unittest.TestCase):
    def test_complete_capture_is_ready(self):
        status = gate.capture_status(1.0, rgb(), tof(t=1.0))
        self.assertTrue(status["ok"], status)
        self.assertEqual(status["tof_valid_zones"], 64)

    def test_incomplete_captures_are_refused(self):
        cases = {
            "rgb_not_960x540_uint8": (rgb()[:, :, :2], tof()),
            "rgb_constant_image": (np.zeros(gate.RGB_SHAPE, np.uint8), tof()),
            "tof_no_valid_zone": (rgb(), tof(depth_m=np.inf)),
            "tof_timestamp_mismatch": (rgb(), tof(t=0.9)),
        }
        dropped = tof()
        dropped["frame_dropped"] = True
        cases["tof_frame_missing_or_dropped"] = (rgb(), dropped)
        short = tof()
        short["cells"] = short["cells"][:7]
        cases["tof_not_8x8"] = (rgb(), short)
        nan_range = copy.deepcopy(tof())
        nan_range["cells"][3][4]["range_m"] = float("nan")
        cases["tof_valid_zone_without_finite_range"] = (rgb(), nan_range)
        for reason, (image, frame) in cases.items():
            with self.subTest(reason):
                status = gate.capture_status(1.0, image, frame)
                self.assertFalse(status["ok"])
                self.assertIn(reason, status["reasons"])
        self.assertFalse(gate.capture_status(1.0, rgb(), None)["ok"])

    def test_capture_check_does_not_modify_inputs(self):
        image, frame = rgb(), tof()
        before_image, before_frame = image.copy(), copy.deepcopy(frame)
        gate.capture_status(1.0, image, frame)
        self.assertTrue(np.array_equal(image, before_image))
        self.assertEqual(frame, before_frame)

    def test_body_imu_must_be_finite(self):
        self.assertTrue(gate.imu_status(body_packet())["ok"])
        self.assertEqual(gate.imu_status(None)["reasons"], ["body_imu_not_delivered"])
        for key, value, reason in [
            ("gyro_body_rad_s", [math.nan, 0, 0], "gyro_not_finite"),
            ("gravity_body", [0, 0, -9.81], "gravity_not_unit"),
            ("gravity_body", [0, math.inf, -1], "gravity_not_finite"),
            ("encoder_q_rad", [0.1] * 13, "encoder_q_not_finite"),
            ("encoder_qd_rad_s", [0.0] * 13 + [math.nan], "encoder_qd_not_finite"),
        ]:
            with self.subTest(reason):
                packet = dict(body_packet(), **{key: value})
                self.assertIn(reason, gate.imu_status(packet)["reasons"])


class BrakeGate(unittest.TestCase):
    def test_nothing_enabled_and_no_wrench_before_release(self):
        g = new_gate()
        rows, t = run_until(g, 5.0)
        self.assertIsNotNone(g.settled_at, "the locked robot settles on the stationary check")
        # NoSupportStartup: at least 3 s locked and 0.5 s stationary (here since the IMU at 0.5 s).
        self.assertGreaterEqual(g.settled_at, 3.0 - 1e-9)
        for row in rows:
            self.assertFalse(row["policy_enabled"] or row["navigation_enabled"] or row["score_enabled"])
            self.assertIsNone(row["task_time_s"])
            self.assertIsNone(row["failure"])
            self.assertEqual(row["force_world_n"], [0.0] * 3)
            self.assertEqual(row["torque_world_nm"], [0.0] * 3)
            self.assertEqual(row["support_gain"], 0.0)
        self.assertEqual(rows[-1]["phase"], "JOINT_BRAKE_AWAITING_FIRST_WALK_OUTPUT")
        self.assertIsNone(g.task_started_at, "the inner gate reached TASK but the task clock is off")

    def test_no_release_before_all_inputs_are_ready(self):
        g = new_gate()
        _, t = run_until(g, 2.0)
        g.note_capture(t, rgb(), tof(t=t))
        self.assertFalse(g.ready(t, body_packet()))
        self.assertIn("locked_robot_not_settled", g.last_check["reasons"])
        g2 = new_gate()
        _, t = run_until(g2, 4.0)
        self.assertFalse(g2.ready(t, body_packet()))
        self.assertIn("no_capture_delivered", g2.last_check["reasons"])
        g2.note_capture(t, rgb(), tof(t=t))
        self.assertFalse(g2.ready(t, None))
        g2.note_capture(t, rgb(), tof(depth_m=np.inf, t=t))
        self.assertFalse(g2.ready(t, body_packet()))
        g2.note_capture(t - 0.2, rgb(), tof(t=t - 0.2))
        self.assertFalse(g2.ready(t, body_packet()))
        self.assertIn("capture_not_fresh", g2.last_check["reasons"])
        with self.assertRaises(RuntimeError):
            g2.mark_released(t)
        g2.note_capture(t - 0.06, rgb(), tof(t=t - 0.06))
        self.assertTrue(g2.ready(t, body_packet()), g2.last_check)

    def test_release_starts_the_task_clock(self):
        g = new_gate()
        _, t = run_until(g, 4.0)
        g.note_capture(t, rgb(), tof(t=t))
        self.assertTrue(g.ready(t, body_packet()))
        with self.assertRaises(RuntimeError):
            g.mark_released(t + DT)  # the ready check must be from the same tick
        g.mark_released(t)
        with self.assertRaises(RuntimeError):
            g.mark_released(t)
        self.assertEqual(g.task_started_at, t)
        row = still(t + 0.02, g)
        self.assertTrue(row["navigation_enabled"] and row["score_enabled"])
        self.assertAlmostEqual(row["task_time_s"], 0.02)
        self.assertEqual(row["phase"], "TASK")
        self.assertEqual(row["force_world_n"], [0.0] * 3)
        meta = g.metadata()
        self.assertTrue(meta["external_wrench_all_times_zero"] and meta["task_clock_starts_at_release"])
        self.assertEqual(meta["task_started_at_s"], t)

    def test_fail_closed_when_sensors_never_ready(self):
        g = new_gate()
        rows, _ = run_until(g, 10.0)
        self.assertEqual(rows[-1]["failure"], "onboard_sensors_not_ready_before_timeout")
        self.assertFalse(rows[-1]["navigation_enabled"])
        self.assertIsNone(g.task_started_at)
        self.assertFalse(g.ready(10.0, body_packet()))

    def test_fail_closed_when_locked_robot_never_settles(self):
        g = new_gate()
        rows, _ = run_until(g, 10.0, moving=True)
        self.assertIsNone(g.settled_at)
        self.assertEqual(rows[-1]["failure"], "locked_grounded_initialization_not_stable")

    def test_any_actual_external_wrench_fails(self):
        g = new_gate()
        g.check_wrench(0.0)
        for peak in (1e-6, math.nan, math.inf):
            with self.assertRaises(RuntimeError):
                g.check_wrench(peak)


def tilted(t, g, roll=0.3, speed=0.03, spin=0.2):
    return g.update(
        t, position_world=[-0.55, 0.04, 0.1168],
        quaternion_xyzw=[math.sin(roll / 2), 0, 0, math.cos(roll / 2)],
        velocity_world=[speed, 0, 0], angular_velocity_world=[spin, 0, 0], imu_orientation_valid=True,
    )


class StationaryUntilRelease(unittest.TestCase):
    """The inner check latches once passed; the onboard gate must still be stationary at release."""

    def test_motion_after_settling_blocks_release_until_a_new_full_window(self):
        g = new_gate()
        run_until(g, 3.1)
        self.assertEqual(g.settled_at, 3.0)
        # Camera arrives late; the free root tilts and moves at 3.2 s.
        row = tilted(3.2, g)
        self.assertFalse(row["navigation_enabled"] or row["score_enabled"])
        packet = dict(body_packet(), gyro_body_rad_s=[0.2, 0, 0],
                      gravity_body=[0, math.sin(0.3), -math.cos(0.3)])
        g.note_capture(3.2, rgb(), tof(t=3.2))
        self.assertFalse(g.ready(3.2, packet))
        self.assertIn("locked_robot_not_stationary", g.last_check["reasons"])
        with self.assertRaises(RuntimeError):
            g.mark_released(3.2)
        self.assertEqual(g.settled_at, 3.0, "the inner latch is unchanged; only the gate refuses")
        # Still again from 3.205 s: a fresh capture alone is not enough before 0.5 s.
        _, t = run_until(g, 3.6, start=3.2)
        g.note_capture(t, rgb(), tof(t=t))
        self.assertFalse(g.ready(t, body_packet()))
        self.assertIn("locked_robot_not_stationary", g.last_check["reasons"])
        _, t = run_until(g, 3.705, start=t)
        g.note_capture(t, rgb(), tof(t=t))
        self.assertTrue(g.ready(t, body_packet()), g.last_check)
        g.mark_released(t)
        meta = g.metadata()["stationary_until_release"]
        self.assertEqual(meta["resets_before_release"], 1)
        self.assertAlmostEqual(meta["stable_since_s"], 3.205)

    def test_any_single_moving_tick_resets_the_window(self):
        g = new_gate()
        _, t = run_until(g, 4.0)
        for bad in (dict(roll=0.2, speed=0, spin=0), dict(roll=0, speed=0.003, spin=0),
                    dict(roll=0, speed=0, spin=0.031)):
            with self.subTest(bad):
                t = round(t + DT, 6)
                tilted(t, g, **bad)
                _, t = run_until(g, t + 0.495, start=t)
                g.note_capture(t, rgb(), tof(t=t))
                self.assertFalse(g.ready(t, body_packet()))
                self.assertEqual(g.last_check["reasons"], ["locked_robot_not_stationary"])
                _, t = run_until(g, t + 0.01, start=t)
                g.note_capture(t, rgb(), tof(t=t))
                self.assertTrue(g.ready(t, body_packet()), g.last_check)

    def test_nonfinite_state_is_not_stationary(self):
        g = new_gate()
        _, t = run_until(g, 4.0)
        t = round(t + DT, 6)
        g.update(t, position_world=[-0.55, 0.04, 0.1168], quaternion_xyzw=[0, 0, 0, 1],
                 velocity_world=[math.nan, 0, 0], angular_velocity_world=[0, 0, 0],
                 imu_orientation_valid=True)
        g.note_capture(t, rgb(), tof(t=t))
        self.assertFalse(g.ready(t, body_packet()))
        self.assertIn("locked_robot_not_stationary", g.last_check["reasons"])

    def test_stability_must_be_from_the_same_tick(self):
        g = new_gate()
        _, t = run_until(g, 4.0)
        g.note_capture(t, rgb(), tof(t=t))
        self.assertFalse(g.ready(t + DT, body_packet()), "no update at this tick yet")
        self.assertIn("locked_robot_not_stationary", g.last_check["reasons"])

    def test_timeout_names_lost_stationarity(self):
        g = new_gate()
        _, t = run_until(g, 9.9)
        while t < 10.0 - 1e-9:
            t = round(t + DT, 6)
            row = tilted(t, g)
        self.assertEqual(row["failure"], "locked_robot_not_stationary_before_timeout")
        self.assertIsNone(g.task_started_at)

    def test_thresholds_match_no_support_startup(self):
        cases = []
        for roll in (0.0, 0.09, 0.1, 0.11, math.nan):
            for speed in (0.0, 0.0019, 0.0021, math.inf):
                for spin in (0.0, 0.029, 0.031):
                    for valid in (True, False):
                        cases.append(dict(
                            position_world=[-0.55, 0.04, 0.1168],
                            quaternion_xyzw=[math.sin(roll / 2), 0, 0, math.cos(roll / 2)],
                            velocity_world=[speed, 0, 0], angular_velocity_world=[0, 0, spin],
                            imu_orientation_valid=valid))
        for state in cases:
            inner = NoSupportStartup([-0.55, 0.04, 0.12, 0, 0, 0, 1], 0.6)
            with np.errstate(invalid="ignore"):
                inner.update(1.0, **state)
            with self.subTest(state):
                self.assertEqual(gate.stationary(**state), inner.stable_since is not None)


class FakeArray:
    def __init__(self, value):
        self.value = np.array(value)

    def numpy(self):
        return self.value


class FakeRobot:
    def __init__(self, floating=True):
        lo = torch.linspace(-1.0, -0.5, 28)
        self.data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.linspace(-0.2, 0.2, 28)),
            joint_pos_limits=SimpleNamespace(torch=torch.stack([lo, -lo], dim=-1)),
            default_joint_pos=SimpleNamespace(torch=torch.zeros(28)),
        )
        self.model = SimpleNamespace(**{
            name: FakeArray(np.arange(28.0) + i)
            for i, name in enumerate(
                ["joint_limit_lower", "joint_limit_upper", "joint_armature", "body_mass", "body_inertia"])
        })
        self._backend_manager = SimpleNamespace(get_model=lambda: self.model)
        self.root_view = SimpleNamespace(is_floating_base=floating)
        self.calls = []

    def write_joint_position_limit_to_sim_index(self, *, limits, warn_limit_violation):
        self.calls.append(("limits", limits.clone()))
        self.data.joint_pos_limits.torch = limits.clone()


class JointBrakeRestore(unittest.TestCase):
    def test_lock_then_release_restores_every_original_field(self):
        robot = FakeRobot()
        original = robot.data.joint_pos_limits.torch.clone()
        brake = JointBrake(robot)
        brake.lock()
        q = robot.data.joint_pos.torch
        self.assertTrue(brake.active)
        self.assertTrue(torch.allclose(robot.data.joint_pos_limits.torch[:, 0], q - 1e-6))
        self.assertTrue(torch.allclose(robot.data.joint_pos_limits.torch[:, 1], q + 1e-6))
        self.assertEqual(robot.data.joint_pos_limits.torch.shape, (28, 2), "all 28 joint axes locked")
        brake.release(4.2)
        self.assertTrue(torch.equal(robot.data.joint_pos_limits.torch, original))
        self.assertEqual([c[0] for c in robot.calls], ["limits", "limits"], "lock, then restore")
        meta = brake.metadata()
        self.assertTrue(meta["root_free"] and meta["restored_before_first_policy_action"])
        self.assertFalse(meta["external_body_support"] or meta["motor_model_retuned"])
        self.assertEqual(meta["release_timestamp_s"], 4.2)
        brake.release(5.0)  # idempotent once released
        self.assertEqual(brake.release_time, 4.2)

    def test_release_refuses_changed_physics(self):
        robot = FakeRobot()
        brake = JointBrake(robot)
        brake.lock()
        robot.model.joint_armature.value[3] += 0.01
        with self.assertRaisesRegex(RuntimeError, "joint_armature"):
            brake.release(4.0)
        self.assertIsNone(brake.release_time)

    def test_fixed_base_is_refused(self):
        with self.assertRaises(ValueError):
            JointBrake(FakeRobot(floating=False))


def _run_function():
    tree = ast.parse(EXECUTOR.read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run")


def _calls(node, dotted):
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call) and ast.unparse(n.func) == dotted:
            found.append(n)
    return sorted(found, key=lambda n: n.lineno)


class ExecutorProtocolOrder(unittest.TestCase):
    """Static checks of the executor loop; physics behaviour is verified in the GUI run."""

    @classmethod
    def setUpClass(cls):
        cls.run_fn = _run_function()

    def test_release_after_first_output_and_before_apply(self):
        release_ifs = [
            n for n in ast.walk(self.run_fn)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "release_now"
        ]
        self.assertEqual(len(release_ifs), 1)
        block = release_ifs[0]
        brake = _calls(block, "initialization_brake.release")
        marked = _calls(block, "onboard.mark_released")
        self.assertEqual(len(brake), 1)
        self.assertEqual(len(marked), 1)
        self.assertLess(brake[0].lineno, marked[0].lineno, "restore check before the task clock starts")
        actor = [c for c in _calls(self.run_fn, "model") if ast.unparse(c.args[0]) == "policy_obs"]
        apply = _calls(self.run_fn, "env.action_manager.process_action")
        self.assertEqual(len(actor), 1)
        self.assertEqual(len(apply), 1)
        self.assertLess(actor[0].lineno, block.lineno)
        self.assertLess(block.end_lineno, apply[0].lineno)

    def test_home_is_held_while_braked(self):
        hold = [
            n for n in ast.walk(self.run_fn)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "last_body_imu is None or hold_home"
        ]
        self.assertEqual(len(hold), 1)
        self.assertIn("torch.zeros", ast.unparse(hold[0].body[0]))
        self.assertNotIn("model(", ast.unparse(hold[0].body[0]))
        assign = next(
            n for n in ast.walk(self.run_fn)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "release_now"
        )
        text = ast.unparse(assign.value)
        for term in ("onboard is not None", "initialization_brake.active", "not kick_requested",
                     "onboard.ready(t, last_body_imu)"):
            self.assertIn(term, text)

    def test_gr00t_release_stays_inside_the_gr00t_branch(self):
        brakes = _calls(self.run_fn, "initialization_brake.release")
        self.assertEqual(len(brakes), 2)
        gr00t = [
            n for n in ast.walk(self.run_fn)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "walking_policy is not None"
            and _calls(n, "walking_policy.step")
        ]
        self.assertEqual(len(gr00t), 1)
        self.assertEqual(len(_calls(gr00t[0], "initialization_brake.release")), 1)

    def test_route_wiring(self):
        source = ast.unparse(self.run_fn)
        self.assertIn(
            "nav = Nav2Unused() if a.walking_policy_server else Nav2Session(output / 'nav2', "
            "max_wall_s=max(1800.0, a.seconds * 60))", source)
        mapping = _calls(self.run_fn, "k10.verify_action_mapping")
        self.assertEqual(len(mapping), 1)
        guarded = [
            n for n in ast.walk(self.run_fn)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "a.walking_policy_server"
            and _calls(n, "k10.verify_action_mapping")
        ]
        self.assertEqual(len(guarded), 1)
        self.assertEqual(len(_calls(self.run_fn, "OnboardBrakeStartup")), 1)
        capture = _calls(self.run_fn, "onboard.note_capture")
        self.assertEqual([ast.unparse(a) for a in capture[0].args], ["t", "rgb", "capture['tof']"])
        wrench = _calls(self.run_fn, "onboard.check_wrench")
        written = _calls(self.run_fn, "support_adapter.check_after_write")
        self.assertLess(written[0].lineno, wrench[0].lineno)


@contextlib.contextmanager
def executor_module():
    """Import the executor with a stub launcher; only arguments() is used."""
    saved_path, saved_modules = list(sys.path), dict(sys.modules)
    stub = types.ModuleType("isaaclab.app")
    stub.add_launcher_args = lambda p: p.add_argument("--device", default=None)
    sys.modules.update({"isaaclab": types.ModuleType("isaaclab"), "isaaclab.app": stub})
    sys.path.insert(0, str(ROOT / "lab/runtime/simulation/core/evaluation/football"))
    try:
        spec = importlib.util.spec_from_file_location("executor_under_test", EXECUTOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = saved_path
        for name in set(sys.modules) - set(saved_modules):
            del sys.modules[name]


class ExecutorRoute(unittest.TestCase):
    BASE = ["x", "--model", "m.onnx", "--output", "/nonexistent/out", "--development",
            "--recovery-approval", "a.json"]

    def parse_error(self, extra):
        with executor_module() as module:
            argv, sys.argv = sys.argv, [*self.BASE, *extra]
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
                    module.arguments()
            finally:
                sys.argv = argv
        return err.getvalue()

    def test_exactly_one_route(self):
        self.assertIn("exactly one route", self.parse_error(["--detector-deployment", "d.json"]))
        self.assertIn("exactly one route", self.parse_error(
            ["--detector-deployment", "d.json", "--course-onboard", "--walking-policy-server", "h:1"]))

    def test_onboard_requires_detector(self):
        self.assertIn("trained detector", self.parse_error(["--development-contour", "--course-onboard"]))


class OnboardEntry(unittest.TestCase):
    ARGV = ["--model", "velstand.onnx", "--output", "/tmp/x", "--detector-deployment", "d.json"]

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("onboard_scene", ROOT / "lab/onboard_scene.py")
        cls.scene = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.scene)

    def test_arguments(self):
        out = self.scene.onboard_argv(list(self.ARGV))
        self.assertEqual(out.count("--course-onboard"), 1)
        self.assertEqual(self.scene.onboard_argv(out).count("--course-onboard"), 1)
        for bad in (["--walking-policy-server", "h:1"], ["--walking-policy-server=h:1"],
                    ["--development-contour"], ["--calibration-command", "0", "0", "0"]):
            with self.subTest(bad), self.assertRaises(ValueError):
                self.scene.onboard_argv([*self.ARGV, *bad])
        with self.assertRaises(ValueError):
            self.scene.onboard_argv(["--model", "m", "--output", "/tmp/x"])

    def test_executor_redirect_only(self):
        football_scene = self.scene.football_scene
        original = football_scene.load
        seen = []
        football_scene.load = lambda name, path: seen.append((name, Path(path)))
        try:
            self.scene.install_executor()
            football_scene.load("isaac_execute", Path("/runtime/evaluation/football/executor.py"))
            football_scene.load("scene_preserved_controller", Path("/runtime/evaluation/football/execute.py"))
        finally:
            football_scene.load = original
        self.assertEqual(seen[0], ("isaac_execute", EXECUTOR))
        self.assertEqual(seen[1][1], Path("/runtime/evaluation/football/execute.py"))

    def test_refuses_preloaded_gr00t_modules(self):
        argv = sys.argv
        sys.argv = ["onboard_scene.py", *self.ARGV]
        sys.modules["kick_runtime"] = types.ModuleType("kick_runtime")
        try:
            with self.assertRaisesRegex(RuntimeError, "preloaded"):
                self.scene.main()
        finally:
            del sys.modules["kick_runtime"]
            sys.argv = argv

    def test_hashed_sources_exist(self):
        for p in self.scene.SOURCES:
            self.assertTrue(p.is_file(), p)


class AdapterRecord(unittest.TestCase):
    def test_recorded_hashes_match_current_files(self):
        adapter = json.loads((STUDENT / "ADAPTER.json").read_text())
        for entry in adapter["files"]:
            with self.subTest(entry["file"]):
                path = STUDENT.parent / entry["file"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["sha256"])
                if "repository_path" in entry:
                    self.assertEqual(path.resolve(), (ROOT / entry["repository_path"]).resolve())
                base = STUDENT.parent / entry.get("based_on", "")
                if entry.get("based_on", "").startswith("rollout/"):
                    self.assertEqual(hashlib.sha256(base.read_bytes()).hexdigest(), entry["based_on_sha256"])


class InitializationIdentity(unittest.TestCase):
    def test_grounded_pose_is_the_recorded_football_reference(self):
        path = STUDENT / "grounded-pose.json"
        adapter = json.loads((STUDENT / "ADAPTER.json").read_text())
        recorded = next(f["sha256"] for f in adapter["files"] if f["file"] == "student/grounded-pose.json")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), recorded)
        pose = json.loads(path.read_text())
        self.assertEqual(pose["case"], "ankle_+0.00")
        self.assertFalse(pose["fell"])
        self.assertEqual(pose["external_wrench_peak"], 0.0)
        self.assertGreater(pose["upright"], 0.9999)
        self.assertLess(np.linalg.norm(pose["v"]), 1e-3)
        self.assertLess(np.linalg.norm(pose["w"]), 1e-2)
        self.assertEqual(len(pose["q"]), 28)
        self.assertAlmostEqual(np.linalg.norm(pose["quat"]), 1.0, places=5)
        # Course football start (scripts/run.py indoor: --root-x -0.55, --root-y 0.04) on the flat pitch.
        self.assertLess(np.hypot(pose["root"][0] + 0.55, pose["root"][1] - 0.04), 0.01)
        self.assertTrue(0.10 < pose["root"][2] < 0.125)


if __name__ == "__main__":
    unittest.main()
