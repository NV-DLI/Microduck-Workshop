"""Audit goal turn progress actual: Microduck simulation and control."""

from pathlib import Path
import argparse, json, math, hashlib


def audit(run):
    reads = {
        round(r["timestamp_s"], 6): r["readback"].get("actual_packet")
        for r in map(json.loads, (run / "public-memory-readback.jsonl").open())
    }
    count = 0
    stops = []
    anchors = {}
    prev = None
    for r in map(json.loads, (run / "policy-trace.jsonl").open()):
        f = r["public_neutral_fine_step"]
        t = r["timestamp_s"]
        if not f.get("turn_goal_only") or f.get("turn_start_s") is None:
            prev = r
            continue
        key = (f["attempts"], f["turn_start_s"])
        if key not in anchors:
            if f["state"] != "TURN":
                prev = r
                continue
            h = f.get("turn_initial_goal_heading_rad")
            assert h is not None and abs(h - f["goal_heading_rad"]) < 1e-10
            anchors[key] = h
        initial = anchors[key]
        assert f["turn_initial_goal_heading_rad"] == initial
        sign = math.copysign(1, initial)
        elapsed = t - f["turn_start_s"]
        h = f["goal_heading_rad"]
        if f["state"] == "TURN":
            assert r["actual_velocity_command"] == [0.0, 0.0, 0.8 * sign]
            count += 1
        if (
            prev
            and prev["public_neutral_fine_step"]["state"] == "TURN"
            and f["state"] in ("TURN", "SETTLE")
        ):
            if f["reason"] == "public_guard_stop_requires_fresh_reacquisition":
                prev = r
                continue
            p = reads[round(t, 6)]
            g = p["gravity_body"]
            w = p["gyro_body_rad_s"]
            rate = -sum(x * y for x, y in zip(w, g)) / math.sqrt(sum(x * x for x in g))
            predicted = rate * sign > 0.1 and (h - 0.12 * rate) * sign <= 0.015
            progress = (initial - h) * sign
            predicted = (
                predicted
                and elapsed >= 0.16 - 1e-9
                and f["goal_fix_s"] > f["turn_start_s"]
                and progress >= 0.03
            )
            stop = elapsed >= 0.08 - 1e-9 and (h * sign <= 0.015 or predicted)
            expected_stop = stop or elapsed >= 0.8 - 1e-9
            assert (f["state"] == "SETTLE") == expected_stop
            if expected_stop:
                assert r["actual_velocity_command"] == [0.0, 0.0, 0.0]
                stops.append(
                    dict(
                        t=t,
                        elapsed_s=elapsed,
                        initial_heading=initial,
                        heading=h,
                        net_progress_rad=progress,
                        predictive=bool(predicted),
                        direct_goal_reached=h * sign <= 0.015,
                        cap=elapsed >= 0.8 - 1e-9,
                    )
                )
        prev = r
    return dict(
        schema="microduck.goal-turn-progress-audit.v2",
        passed=True,
        goal_turn_ticks=count,
        attempts=len(anchors),
        stops=stops,
        truth_consumed=False,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--require-turn", action="store_true")
    a = p.parse_args()
    assert not a.output.exists()
    d = audit(a.run)
    if a.require_turn:
        assert d["goal_turn_ticks"] > 0
    d["source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            a.run / "policy-trace.jsonl",
            a.run / "public-memory-readback.jsonl",
        ]
    }
    a.output.write_text(json.dumps(d, indent=2) + "\n")
    print(json.dumps({k: v for k, v in d.items() if k != "source_sha256"}))
