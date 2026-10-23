"""Independently check zero motion until bounded fresh reobservation after a guard stop."""

from pathlib import Path
import argparse, json, math, hashlib


def audit(run):
    rows = [json.loads(x) for x in (run / "policy-trace.jsonl").open()]
    controls = [json.loads(x) for x in (run / "control.jsonl").open()]
    seen = set()
    events = []
    active = None
    previous = None
    for r in rows:
        t = r["timestamp_s"]
        f = r["public_neutral_fine_step"]
        h = r["public_handoff"]
        state = f["state"]
        if (
            f.get("reason")
            in (
                "public_guard_stop_requires_fresh_reacquisition",
                "public_goal_stop_requires_fresh_reacquisition",
            )
            and f["attempts"] not in seen
        ):
            assert previous is not None and previous["public_neutral_fine_step"][
                "state"
            ] in ("RETURN", "STEP", "TURN", "TRACK", "MACRO")
            assert state == "SETTLE" and 0 <= t - f["capture_guard_s"] <= 0.2
            if f["reason"] == "public_guard_stop_requires_fresh_reacquisition":
                assert f["capture_safe"] is False
            else:
                assert (
                    previous["public_neutral_fine_step"]["state"] == "TRACK"
                    and f["capture_safe"]
                )
                assert f["goal_fix_s"] is None or not 0 <= t - f["goal_fix_s"] <= 0.5
            assert f["started_s"] == previous["public_neutral_fine_step"]["started_s"]
            assert (
                f["anchor_capture_s"]
                == previous["public_neutral_fine_step"]["anchor_capture_s"]
            )
            active = dict(
                attempt=f["attempts"],
                stop_s=t,
                started_s=f["started_s"],
                anchor_s=f["anchor_capture_s"],
                reacquire_s=None,
                idle_s=None,
                zero_ticks=0,
            )
            events.append(active)
            seen.add(f["attempts"])
        if active is not None:
            assert r["actual_velocity_command"] == [0.0, 0.0, 0.0]
            assert not r["kick_requested"] and h["state"] == "APPROACH"
            assert (
                f["started_s"] == active["started_s"]
                and f["anchor_capture_s"] == active["anchor_s"]
            )
            assert t - f["started_s"] < 24.0 + 1e-8
            assert state in ("SETTLE", "REACQUIRE", "IDLE", "FAILED")
            active["zero_ticks"] += 1
            if state == "SETTLE" and not f["capture_safe"]:
                assert f["quiet_started_s"] is None
            if state == "REACQUIRE" and active["reacquire_s"] is None:
                assert (
                    f["capture_safe"]
                    and f["quiet_started_s"] is not None
                    and t - f["quiet_started_s"] >= 0.5 - 1e-8
                )
                active["reacquire_s"] = t
            if state == "IDLE":
                assert active["reacquire_s"] is not None
                capture = f["last_finished_s"]
                assert active["reacquire_s"] < capture <= t
                c = next(c for c in controls if abs(c["timestamp_s"] - capture) < 1e-8)
                assert (
                    c["public_neutral_fine_step"]["state"] == "IDLE"
                    and c["public_neutral_fine_step"]["capture_safe"]
                )
                assert c.get("public_metric_ball", {}).get("point_level_m") is not None
                active["idle_s"] = t
                active = None
            elif state == "FAILED":
                active = None
        previous = r
    return dict(
        schema="microduck.actual-guard-goal-stop-audit.v2",
        passed=True,
        events=events,
        events_count=len(events),
        scope="Public trace only. Immediatezero, original24s budget, noanchorrefresh duringstop, measuredquiet andactualnewcompleteball beforeIDLE. No physics truth or training.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-stop", action="store_true")
    a = p.parse_args()
    assert not a.output.exists()
    r = audit(a.run)
    if a.require_stop:
        assert r["events_count"] > 0
    r["source_sha256"] = {
        str(x): hashlib.sha256(x.read_bytes()).hexdigest()
        for x in [Path(__file__), a.run / "policy-trace.jsonl", a.run / "control.jsonl"]
    }
    a.output.write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "source_sha256"}))
