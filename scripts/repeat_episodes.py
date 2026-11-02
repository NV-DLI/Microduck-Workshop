"""Repeat one course episode command in fresh output folders (used by run.py football --repeat).

Each episode launches its commands again (for football: Isaac Sim, then the evaluator), so
the simulator reloads between episodes; nothing is reset inside a running window. Every
command runs in its own process group: Ctrl+C (or SIGTERM) stops that group, no further
command or episode starts, and the summary records the interrupted episode. Any nonzero
exit (a crash, or an evaluator whose independent checks failed) is a technical error: the
loop stops and that episode is kept for inspection but not counted as a task outcome. An
evaluated task failure (all commands exit 0, passed is false) is counted and printed, and
the loop continues.
"""
from pathlib import Path
import json
import os
import signal
import subprocess
import time


class Interrupted(Exception):
    """Ctrl+C or SIGTERM arrived; the running command's process group has been stopped."""


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def stop_group(proc, grace_s=30.0, term_s=10.0):
    """Stop a command's whole process group: SIGINT, then SIGTERM, then SIGKILL.

    The group outlives its leader, so cleanup continues after the leader has exited. A
    further Ctrl+C or SIGTERM while waiting skips straight to SIGKILL. Returns the signals
    sent and whether such an interrupt arrived during cleanup.
    """
    pgid = proc.pid
    hurry = []
    previous = {
        s: signal.signal(s, lambda *_: hurry.append(True))
        for s in (signal.SIGINT, signal.SIGTERM)
    }
    sent = []
    try:
        for sig, timeout in (
            (signal.SIGINT, grace_s),
            (signal.SIGTERM, term_s),
            (signal.SIGKILL, 5.0),
        ):
            proc.poll()
            if not _group_alive(pgid):
                break
            if hurry and sig != signal.SIGKILL:
                continue
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                break
            sent.append(signal.Signals(sig).name)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline and not (hurry and sig != signal.SIGKILL):
                proc.poll()
                if not _group_alive(pgid):
                    break
                time.sleep(0.05)
        proc.poll()
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    return sent, bool(hurry)


def supervise(argv, *, cwd=None, env=None, grace_s=30.0, term_s=10.0):
    """Run one command in its own process group and return its exit code.

    On Ctrl+C the group receives SIGINT (escalating if it does not exit) and Interrupted
    is raised. Processes the command left behind after a normal exit are stopped too, so
    the next command starts from a clean process table; an interrupt during that cleanup
    still raises Interrupted.
    """
    print("Launching:", " ".join(str(x) for x in argv), flush=True)
    proc = None
    try:
        proc = subprocess.Popen(
            [str(x) for x in argv], cwd=cwd, env=env, start_new_session=True
        )
        code = proc.wait()
        if _group_alive(proc.pid):
            sent, hurried = stop_group(proc, 2.0, 2.0)
            if hurried:
                raise Interrupted(sent)
        return code
    except KeyboardInterrupt:
        sent = stop_group(proc, grace_s, term_s)[0] if proc is not None else []
        raise Interrupted(sent) from None


def _write(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def repeat(root, count, prepare, outcome, *, cwd=None, env=None, grace_s=30.0, term_s=10.0):
    """Run episodes in root/episode-001, episode-002, ...; count 0 means until Ctrl+C.

    prepare(index, output) -> {"seed": ..., "commands": [argv, ...]}; the commands run in
    order and stop at the first nonzero exit. outcome(output) -> dict with a boolean
    "passed", or None when no evaluated result exists. Returns the process exit code:
    0 when every requested episode was evaluated, 130 after an interrupt, otherwise the
    failing command's code (1 if it was 0).
    """
    root = Path(root)
    root.mkdir(parents=False, exist_ok=False)
    summary = dict(
        requested=count if count else "until Ctrl+C",
        simulator_relaunched_each_episode=True,
        state="running",
        episodes=[],
    )
    path = root / "summary.json"
    _write(path, summary)

    def stop_on_sigterm(*_):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, stop_on_sigterm)
    code = 0
    index = 0
    row = None
    try:
        while not count or index < count:
            index += 1
            out = root / f"episode-{index:03d}"
            if out.exists():
                raise FileExistsError("Episode output already exists: " + str(out))
            row = dict(episode=index, output=out.name, state="running")
            summary["episodes"].append(row)
            print(f"\n=== Episode {index}" + (f" of {count}" if count else "") + " ===", flush=True)
            try:
                plan = prepare(index, out)
                row["seed"] = plan.get("seed")
                _write(path, summary)
                exits = []
                for argv in plan["commands"]:
                    exits.append(supervise(argv, cwd=cwd, env=env, grace_s=grace_s, term_s=term_s))
                    if exits[-1]:
                        break
                row["exit_codes"] = exits
                result = outcome(out)
            except Interrupted as stop:
                row["state"] = "interrupted"
                row["stop_signals"] = stop.args[0]
                raise KeyboardInterrupt from None
            except KeyboardInterrupt:
                raise
            except Exception:
                row["state"] = "technical_error"
                summary["state"] = "stopped_after_technical_error"
                raise
            if any(exits) or result is None:
                # Crash, missing result or failed independent checks: not a task outcome.
                row["state"] = "technical_error"
                if result is not None:
                    row["result_not_counted"] = result
                summary["state"] = "stopped_after_technical_error"
                code = next((x for x in exits if x), 1)
                print(describe(row), flush=True)
                break
            row.update(result)
            row["state"] = "success" if result.get("passed") else "failed"
            _write(path, summary)
            print(describe(row), flush=True)
        else:
            summary["state"] = "completed"
    except KeyboardInterrupt:
        if row is not None and row["state"] == "running":
            row["state"] = "interrupted"
        summary["state"] = "interrupted"
        code = 130
    finally:
        signal.signal(signal.SIGTERM, previous)
        rows = summary["episodes"]
        summary["counts"] = {
            k: sum(r["state"] == k for r in rows)
            for k in ("success", "failed", "interrupted", "technical_error")
        }
        _write(path, summary)
    print_summary(summary, path)
    return code


def describe(row):
    if row["state"] == "technical_error":
        codes = [x for x in row.get("exit_codes", []) if x]
        return (
            f"Episode {row['episode']}: stopped - "
            + (f"a command exited with code {codes[0]}" if codes else "no evaluated result")
            + "; its files are kept and it is not counted as a task outcome"
        )
    if row["state"] == "success":
        goal = row.get("goal_time_s")
        return f"Episode {row['episode']}: success" + (f" (goal at {goal:.1f} s, recovered)" if goal is not None else "")
    detail = ", ".join(
        text
        for flag, text in (
            (row.get("fell"), "fell"),
            (row.get("real_foot_ball_contact") is False, "no foot-ball contact"),
            (row.get("real_foot_ball_contact") and not row.get("whole_ball_crossed_inside_goal"), "contact but no goal"),
            (row.get("whole_ball_crossed_inside_goal") and row.get("recovered") is False, "goal but no recovery"),
        )
        if flag
    )
    end = row.get("terminal_reason")
    return f"Episode {row['episode']}: failed" + (f" - {detail}" if detail else "") + (f" (ended: {end})" if end else "")


def print_summary(summary, path):
    c = summary["counts"]
    evaluated = c["success"] + c["failed"]
    print(
        f"\nEpisodes evaluated: {evaluated} · success {c['success']} · failed {c['failed']}"
        + (" · interrupted 1" if c["interrupted"] else "")
        + (" · stopped after a technical error" if c["technical_error"] else ""),
        flush=True,
    )
    print("Summary:", path, flush=True)
