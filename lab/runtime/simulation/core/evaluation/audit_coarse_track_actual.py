"""Reconstruct every coarse route and prove finite, fresh, no-kick execution."""

import argparse, json, sys, math, hashlib
from pathlib import Path

M = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(M / "football")]
from nav2_service.map_contract import snapshot, check_path


def rot(a, p):
    return [
        math.cos(a) * p[0] - math.sin(a) * p[1],
        math.sin(a) * p[0] + math.cos(a) * p[1],
    ]


def audit(run):
    controls = [json.loads(x) for x in (run / "control.jsonl").open()]
    maps = 0
    entries = 0
    moving = 0
    attempts = set()
    stops = set()
    validated = {}
    violations = []
    for c in controls:
        t = c["timestamp_s"]
        for key in ("coarse_track_path", "coarse_active_path"):
            route = c.get(key) or {}
            req = route.get("map_request")
            if not req:
                continue
            assert route["capture_s"] == req["case"]["query_s"] == t
            assert route["layout"] == "empty" and req["provenance"]["layout"] == "empty"
            assert (
                req["provenance"]["source_file_sha256"]["rgb_pixels"]
                == c["source_rgb_sha256"]
            )
            pose = req["case"]["odom"]["pose_xyyaw"]
            assert pose == c["estimated_pose_map_xyyaw"]
            p = route["point_level_m"]
            source = (
                c.get("public_metric_ball", {}).get("point_level_m")
                if key == "coarse_track_path"
                else c["public_neutral_fine_step"]["memory"]["point_level_m"]
            )
            assert (
                source is not None and max(abs(x - y) for x, y in zip(p, source)) < 1e-8
            ), "Route point not current public observation/memory"
            local = rot(pose[2], p[:2])
            ball = [pose[0] + local[0], pose[1] + local[1]]
            dx, dy = 0.93 - ball[0], -ball[1]
            length = math.hypot(dx, dy)
            ux, uy = dx / length, dy / length
            target = [
                ball[0] - 0.36 * ux - 0.028 * uy,
                ball[1] - 0.36 * uy + 0.028 * ux,
            ]
            snap = snapshot(req["case"], req["provenance"])
            assert max(abs(x - y) for x, y in zip(target, snap["goal"])) < 1e-8
            gate = check_path([pose[:2], target], snap)
            assert gate["accepted"] == route["allowed"]
            if gate["accepted"]:
                assert gate["test_radius_m"] == 0.1
            validated[(t, key)] = bool(gate["accepted"])
            maps += 1
        if (
            c.get("public_fine_entry", {}).get("reason")
            == "settled_coarse_track_current_path"
        ):
            assert validated.get((t, "coarse_track_path"))
            entries += 1
    bytime = {round(c["timestamp_s"], 8): c for c in controls}
    for row in map(json.loads, (run / "policy-trace.jsonl").open()):
        t = row["timestamp_s"]
        fine = row["public_neutral_fine_step"]
        state = fine["state"]
        if not fine.get("coarse_track"):
            continue
        if state == "SETTLE" and fine.get("step_end_s") is not None:
            stops.add((fine["attempts"], fine["step_end_s"]))
        if state not in ("RETURN", "TRACK", "TURN"):
            continue
        c = bytime[round(fine["capture_guard_s"], 8)]
        ct = c["timestamp_s"]
        assert 0 <= t - ct <= 0.2 + 1e-8
        key = (
            "coarse_active_path" if c.get("coarse_active_path") else "coarse_track_path"
        )
        assert validated.get((ct, key)), "No current admitted coarse path"
        assert (
            row["public_handoff"]["state"] == "APPROACH"
        ), "Coarse phase must not hand off to kick"
        assert fine["track_brake_started_s"] is None
        if state == "RETURN":
            assert row["actual_velocity_command"] == [0.0, 0.0, 0.0]
        if state == "TRACK":
            p = row["public_handoff"]["memory"]["point_level_m"]
            assert p[0] > 0.36 and t - fine["step_start_s"] < 2.0 + 1e-8
            assert t - fine["anchor_capture_s"] <= 4.5 + 1e-8
            moving += 1
            attempts.add(fine["attempts"])
    assert maps > 0 and entries > 0 and moving > 0
    return dict(
        schema="microduck.coarse-track-actual-audit.v1",
        passed=True,
        actual_maps=maps,
        actual_entries=entries,
        actual_track_attempts=len(attempts),
        track_ticks=moving,
        stop_events=len(stops),
        route_radius_m=0.1,
        truth_consumed=False,
        original_kick_authority_preserved=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    r = audit(a.run)
    with a.output.open("x") as f:
        json.dump(r, f, indent=2)
    print(json.dumps(r))
