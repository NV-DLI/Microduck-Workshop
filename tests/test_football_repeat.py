"""CPU tests of `run.py football --repeat` with fake child processes (no Isaac Sim, no GPU).

Run with any Python 3.11+:
  python3 -m unittest tests/test_football_repeat.py

Covered:
- N episodes run in fresh OUTPUT/episode-NNN folders with seeds seed, seed+1, ...; an existing
  output root or episode folder is refused;
- an evaluated task failure (every command exits 0) is counted and printed and the loop continues;
- any nonzero exit is a technical error: a crash, a missing result, or an evaluator whose
  independent checks failed even though it wrote a result. The loop stops, no further episode
  starts, and that episode is not counted as a task outcome;
- descendants left behind by a command that exited normally are stopped before the next command;
- Ctrl+C stops the running child and its grandchild, starts no next command or episode, records
  the interrupted episode and exits 130; a child that ignores SIGINT is escalated to SIGTERM, and
  a second Ctrl+C or SIGTERM during cleanup escalates to SIGKILL, also during the cleanup after a
  normal exit;
- --repeat 0 continues until Ctrl+C;
- argument checks (negative count, rollers, legacy scene), and the single-run walking command
  line is unchanged.
"""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import repeat_episodes  # noqa: E402

FAKE = textwrap.dedent(
    """
    import json, os, signal, subprocess, sys, time
    from pathlib import Path
    kind, out, mode = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    if kind == "physics":
        if mode == "crash":
            sys.exit(3)
        out.mkdir()
        if mode.startswith("hang"):
            if mode == "hang-ignore-int":
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
            (out / "pids.json").write_text(json.dumps([os.getpid(), child.pid]))
            time.sleep(600)
        if mode == "slow":
            time.sleep(0.3)
        if mode.startswith("orphan"):
            code = "import signal, time; " + ("signal.signal(signal.SIGINT, signal.SIG_IGN); " if mode == "orphan-ignore-int" else "") + "time.sleep(600)"
            child = subprocess.Popen([sys.executable, "-c", code])
            time.sleep(0.2)  # let the descendant install its signal handler
            (out / "pids.json").write_text(json.dumps([os.getpid(), child.pid]))
    else:
        passed = mode in ("pass", "pass-badcheck")
        if mode != "noresult":
            (out / "manifest.json").write_text(json.dumps({"terminal_reason": "development_kick_window_complete" if passed else "task_duration_limit"}))
            (out / "result.json").write_text(json.dumps({
                "status": "completed", "task_seconds": 19.4,
                "scorer": {"goal_time_s": 16.1 if passed else None, "real_foot_ball_contact": passed,
                           "whole_ball_crossed_inside_goal": passed, "fell": False},
                "course_evaluation": {"passed": passed and mode != "pass-badcheck", "recovered": passed,
                                      "audit_exit_codes": [0]}}))
        sys.exit(1 if mode in ("badcheck", "pass-badcheck", "noresult") else 0)
    """
)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers signal 0; it has exited.
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


class RepeatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fake = self.tmp / "fake.py"
        self.fake.write_text(FAKE)

    def plan(self, physics_modes, evaluate_modes):
        def prepare(index, out):
            return dict(
                seed=100 + index - 1,
                commands=[
                    [sys.executable, self.fake, "physics", out, physics_modes[index - 1]],
                    [sys.executable, self.fake, "evaluate", out, evaluate_modes[index - 1]],
                ],
            )

        return prepare

    def outcome(self, out):
        sys.path.insert(0, str(ROOT / "scripts"))
        import run

        return run.football_outcome(out)

    def repeat(self, count, physics, evaluate):
        return repeat_episodes.repeat(
            self.tmp / "root", count, self.plan(physics, evaluate), self.outcome, grace_s=2, term_s=2
        )

    def summary(self):
        return json.loads((self.tmp / "root/summary.json").read_text())

    def test_count_fresh_folders_seeds_and_failed_outcome_continues(self):
        code = self.repeat(3, ["ok"] * 3, ["pass", "fail", "pass"])
        self.assertEqual(code, 0)
        s = self.summary()
        self.assertEqual(s["state"], "completed")
        self.assertEqual([r["output"] for r in s["episodes"]], ["episode-001", "episode-002", "episode-003"])
        self.assertEqual([r["seed"] for r in s["episodes"]], [100, 101, 102])
        self.assertEqual([r["state"] for r in s["episodes"]], ["success", "failed", "success"])
        self.assertEqual(s["counts"], {"success": 2, "failed": 1, "interrupted": 0, "technical_error": 0})
        self.assertEqual(s["episodes"][1]["terminal_reason"], "task_duration_limit")
        self.assertTrue(s["simulator_relaunched_each_episode"])
        self.assertFalse((self.tmp / "root/episode-004").exists())

    def test_existing_root_is_refused(self):
        (self.tmp / "root").mkdir()
        with self.assertRaises(FileExistsError):
            self.repeat(1, ["ok"], ["pass"])

    def test_existing_episode_folder_is_refused(self):
        root = self.tmp / "root"
        calls = []

        def make(index, out):
            calls.append(index)
            if index == 1:
                (root / "episode-002").mkdir()
            return dict(seed=index, commands=[[sys.executable, self.fake, "physics", out, "ok"],
                                              [sys.executable, self.fake, "evaluate", out, "pass"]])

        with self.assertRaises(FileExistsError):
            repeat_episodes.repeat(root, 3, make, self.outcome)
        self.assertEqual(calls, [1])

    def test_evaluator_check_failure_stops_and_is_not_counted(self):
        code = self.repeat(2, ["ok", "ok"], ["pass-badcheck", "pass"])
        self.assertEqual(code, 1)
        s = self.summary()
        self.assertEqual(s["state"], "stopped_after_technical_error")
        self.assertEqual([r["state"] for r in s["episodes"]], ["technical_error"])
        self.assertIn("result_not_counted", s["episodes"][0])
        self.assertEqual(s["counts"], {"success": 0, "failed": 0, "interrupted": 0, "technical_error": 1})
        self.assertTrue((self.tmp / "root/episode-001/result.json").exists())  # kept for inspection
        self.assertFalse((self.tmp / "root/episode-002").exists())

    def test_descendant_left_after_normal_exit_is_stopped(self):
        code = self.repeat(1, ["orphan"], ["pass"])
        self.assertEqual(code, 0)
        leader, child = json.loads((self.tmp / "root/episode-001/pids.json").read_text())
        self.assertFalse(alive(child), "descendant survived its command")
        self.assertEqual(self.summary()["episodes"][0]["state"], "success")

    def test_physics_crash_stops_before_next_episode(self):
        code = self.repeat(3, ["ok", "crash", "ok"], ["pass", "pass", "pass"])
        self.assertEqual(code, 3)
        s = self.summary()
        self.assertEqual(s["state"], "stopped_after_technical_error")
        self.assertEqual([r["state"] for r in s["episodes"]], ["success", "technical_error"])
        self.assertEqual(s["episodes"][1]["exit_codes"], [3])
        self.assertFalse((self.tmp / "root/episode-003").exists())

    def test_evaluator_error_without_result_stops(self):
        code = self.repeat(2, ["ok", "ok"], ["noresult", "pass"])
        self.assertEqual(code, 1)
        self.assertEqual([r["state"] for r in self.summary()["episodes"]], ["technical_error"])
        self.assertFalse((self.tmp / "root/episode-002").exists())

    def harness(self, count, physics, evaluate="pass", grace=2.0):
        script = textwrap.dedent(
            f"""
            import sys
            sys.path[:0] = [{str(ROOT / 'scripts')!r}]
            import repeat_episodes, run
            from pathlib import Path
            def prepare(index, out):
                return dict(seed=index, commands=[
                    [sys.executable, {str(self.fake)!r}, "physics", out, {physics!r}],
                    [sys.executable, {str(self.fake)!r}, "evaluate", out, {evaluate!r}]])
            sys.exit(repeat_episodes.repeat(Path({str(self.tmp / 'root')!r}), {count}, prepare,
                     run.football_outcome, grace_s={grace}, term_s=2))
            """
        )
        return subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

    def wait_for(self, path, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if path.exists() and path.stat().st_size:
                return json.loads(path.read_text())
            time.sleep(0.05)
        self.fail("timed out waiting for " + str(path))

    def interrupt_and_check(self, physics, expected_signals):
        proc = self.harness(3, physics)
        pids = self.wait_for(self.tmp / "root/episode-001/pids.json")
        self.assertTrue(all(alive(p) for p in pids))
        proc.send_signal(signal.SIGINT)
        output, _ = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 130, output)
        deadline = time.monotonic() + 5
        while any(alive(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(any(alive(p) for p in pids), "child or grandchild survived Ctrl+C")
        s = self.summary()
        self.assertEqual(s["state"], "interrupted")
        self.assertEqual([r["state"] for r in s["episodes"]], ["interrupted"])
        self.assertEqual(s["episodes"][0]["stop_signals"], expected_signals)
        self.assertFalse((self.tmp / "root/episode-002").exists())
        self.assertIn("interrupted 1", output)
        return output

    def test_ctrl_c_stops_child_and_grandchild_and_next_episode(self):
        self.interrupt_and_check("hang", ["SIGINT"])

    def test_ctrl_c_escalates_when_child_ignores_sigint(self):
        start = time.monotonic()
        self.interrupt_and_check("hang-ignore-int", ["SIGINT", "SIGTERM"])
        self.assertGreater(time.monotonic() - start, 1.5)  # waited the SIGINT grace first

    def test_ctrl_c_during_cleanup_after_normal_exit_stops_evaluation(self):
        proc = self.harness(3, "orphan-ignore-int", grace=20)
        leader, child = self.wait_for(self.tmp / "root/episode-001/pids.json")
        deadline = time.monotonic() + 10
        while alive(leader) and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.3)  # the leader has exited; cleanup of its descendant is waiting on SIGINT
        self.assertTrue(alive(child))
        start = time.monotonic()
        proc.send_signal(signal.SIGINT)
        output, _ = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 130, output)
        self.assertLess(time.monotonic() - start, 1.5)  # escalated, did not wait out the grace
        self.assertFalse(alive(child))
        s = self.summary()
        self.assertEqual([r["state"] for r in s["episodes"]], ["interrupted"])
        self.assertEqual(s["episodes"][0]["stop_signals"], ["SIGINT", "SIGKILL"])
        self.assertNotIn("fake.py evaluate", output)
        self.assertFalse((self.tmp / "root/episode-001/result.json").exists())
        self.assertFalse((self.tmp / "root/episode-002").exists())

    def test_second_sigterm_escalates_to_sigkill(self):
        proc = self.harness(3, "hang-ignore-int", grace=20)
        pids = self.wait_for(self.tmp / "root/episode-001/pids.json")
        start = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.5)
        proc.send_signal(signal.SIGTERM)
        output, _ = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 130, output)
        self.assertLess(time.monotonic() - start, 10)
        time.sleep(0.2)
        self.assertFalse(any(alive(p) for p in pids))
        s = self.summary()
        self.assertEqual(s["episodes"][0]["stop_signals"], ["SIGINT", "SIGKILL"])
        self.assertFalse((self.tmp / "root/episode-002").exists())

    def test_sigterm_is_handled_like_ctrl_c(self):
        proc = self.harness(3, "hang")
        pids = self.wait_for(self.tmp / "root/episode-001/pids.json")
        proc.send_signal(signal.SIGTERM)
        proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 130)
        time.sleep(0.3)
        self.assertFalse(any(alive(p) for p in pids))
        self.assertEqual(self.summary()["state"], "interrupted")

    def test_zero_repeats_until_ctrl_c(self):
        proc = self.harness(0, "slow")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                done = [r for r in self.summary()["episodes"] if r["state"] == "success"]
            except (OSError, ValueError):
                done = []
            if len(done) >= 3:
                break
            time.sleep(0.05)
        proc.send_signal(signal.SIGINT)
        output, _ = proc.communicate(timeout=30)
        self.assertEqual(proc.returncode, 130, output)
        s = self.summary()
        self.assertEqual(s["requested"], "until Ctrl+C")
        states = [r["state"] for r in s["episodes"]]
        self.assertGreaterEqual(states.count("success"), 3)
        self.assertEqual(states[-1] in ("interrupted", "success"), True)
        self.assertNotIn("technical_error", states)
        names = [r["output"] for r in s["episodes"]]
        self.assertEqual(names, [f"episode-{i:03d}" for i in range(1, len(names) + 1)])


FAKE_RUNTIME = textwrap.dedent(
    """
    import json, sys
    from pathlib import Path
    args = sys.argv[1:]
    if args[0].endswith("onboard_scene.py"):
        out = Path(args[args.index("--output") + 1])
        out.mkdir()
        (out / "argv.json").write_text(json.dumps(args))
        (out / "manifest.json").write_text(json.dumps({"terminal_reason": "fake"}))
    elif args[0].endswith("evaluate_task.py"):
        out = Path(args[args.index("--run") + 1])
        argv = json.loads((out / "argv.json").read_text())
        passed = int(argv[argv.index("--seed") + 1]) % 2 == 0
        (out / "result.json").write_text(json.dumps({
            "status": "completed", "task_seconds": 20.0,
            "scorer": {"goal_time_s": 15.0 if passed else None, "real_foot_ball_contact": passed,
                       "whole_ball_crossed_inside_goal": passed, "fell": False},
            "course_evaluation": {"passed": passed, "recovered": passed, "audit_exit_codes": [0]}}))
    else:
        sys.exit(9)
    """
)


class RunPyTest(unittest.TestCase):
    """run.py football --repeat end to end, with a fake runtime Python instead of Isaac Sim."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        state = self.tmp / "state"
        state.mkdir()
        fake = self.tmp / "fake-python"
        fake.write_text("#!" + sys.executable + "\n" + FAKE_RUNTIME)
        fake.chmod(0o755)
        (state / "prepare.ok").write_text("")
        (state / "runtime.json").write_text(json.dumps({
            "python": str(fake), "ffmpeg": "/usr/bin/ffmpeg",
            "materialized": str(self.tmp / "runtime/simulation/core"),
        }))
        self.env = dict(os.environ, MICRODUCK_STATE=str(state))

    def call(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/run.py"), "football", *args],
            env=self.env, capture_output=True, text=True, timeout=120,
        )

    def test_repeat_three_fresh_episodes(self):
        out = self.tmp / "repeat"
        r = self.call("--repeat", "3", "--seconds", "45", "--no-record-hd", "--output", str(out))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        s = json.loads((out / "summary.json").read_text())
        self.assertEqual([e["seed"] for e in s["episodes"]], [983102, 983103, 983104])
        self.assertEqual([e["state"] for e in s["episodes"]], ["success", "failed", "success"])
        self.assertEqual(s["counts"]["failed"], 1)
        for i in (1, 2, 3):
            argv = json.loads((out / f"episode-{i:03d}/argv.json").read_text())
            self.assertNotIn("--record-hd", argv)
            self.assertEqual(argv[-2:], ["--visualizer", "kit"])
            self.assertEqual(argv[argv.index("--output") + 1], str(out / f"episode-{i:03d}"))
            self.assertEqual(argv[argv.index("--seconds") + 1], "45.0")
        self.assertIn("Episodes evaluated: 3 · success 2 · failed 1", r.stdout)
        self.assertIn("Episode 2: failed - no foot-ball contact (ended: fake)", r.stdout)

    def test_existing_output_is_refused(self):
        out = self.tmp / "repeat"
        out.mkdir()
        r = self.call("--repeat", "2", "--output", str(out))
        self.assertEqual(r.returncode, 1)
        self.assertIn("Choose a new output directory", r.stderr)
        self.assertEqual(list(out.iterdir()), [])


class ArgumentTest(unittest.TestCase):
    def call(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts/run.py"), "football", *args, "--output", "/tmp/unused-repeat"],
            capture_output=True,
            text=True,
            env={**os.environ, "MICRODUCK_STATE": tempfile.mkdtemp()},
        )

    def test_negative_repeat_is_rejected(self):
        r = self.call("--repeat", "-1")
        self.assertEqual(r.returncode, 2)
        self.assertIn("--repeat needs a positive count", r.stderr)

    def test_rollers_repeat_is_rejected(self):
        r = self.call("--repeat", "2", "--locomotion", "rollers")
        self.assertEqual(r.returncode, 2)
        self.assertIn("walking onboard route", r.stderr)

    def test_legacy_scene_repeat_is_rejected(self):
        r = self.call("--repeat", "2", "--scene", "legacy")
        self.assertEqual(r.returncode, 2)
        self.assertIn("indoor scene", r.stderr)

    def test_single_run_command_line_is_unchanged(self):
        state = Path(tempfile.mkdtemp())
        env = dict(os.environ, MICRODUCK_STATE=str(state))
        code = textwrap.dedent(
            f"""
            import json, sys
            from pathlib import Path
            from types import SimpleNamespace
            sys.path.insert(0, {str(ROOT / 'scripts')!r})
            import run
            a = SimpleNamespace(seconds=45.0, scene="indoor", style="A", root_y=None, ball_xy=None,
                                record_hd=True, headless=False)
            m = Path("/rt/simulation/core")
            physics, evaluation = run.walking_football(a, Path("/o/run"), 983102, "py", m)
            print(json.dumps([[str(x) for x in physics], [str(x) for x in evaluation]]))
            """
        )
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
        physics, evaluation = json.loads(out.stdout)
        m = "/rt/simulation/core"
        self.assertEqual(
            physics,
            ["py", str(ROOT / "lab/onboard_scene.py"), "--model", m + "/models/reference/velstand.onnx",
             "--output", "/o/run", "--development", "--seconds", "45.0", "--seed", "983102",
             "--body-fusion-tau-s", "2", "--body-added-delay-s", "0", "--policy-imu", "body50",
             "--detector-deployment", str(state / "scene-detector-deployment.json"),
             "--detector-backend", "torch", "--detector-device", "cuda:0", "--shared-imu-readback",
             "--packed-physics-readback", "--root-x", "-0.55", "--recovery-approval",
             m + "/evaluation/football/recovery-approval.json", "--runtime", m, "--style", "A",
             "--render-subframes", "4", "--root-y", "0.04", "--ball-xy", "0.02", "0.0",
             "--record-hd", "--visualizer", "kit"],
        )
        self.assertEqual(
            evaluation,
            ["py", str(ROOT / "lab/evaluate_task.py"), "--runtime", m, "--run", "/o/run", "--scene-contract"],
        )
        written = json.loads((state / "scene-detector-deployment.json").read_text())
        self.assertEqual(written["weights"], str(ROOT / "models/scene-detector/detector.pt"))


if __name__ == "__main__":
    unittest.main()
