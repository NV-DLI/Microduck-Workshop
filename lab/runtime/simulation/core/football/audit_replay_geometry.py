#!/usr/bin/env python3
"""Privileged OFFLINE error scoring only; never imported by the controller.

Only this evaluator joins RGB estimates to the retained physics truth. Missing
estimates stay in the denominator. No parameter/model selection is performed.
"""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rotation(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Retain all geometry audits")
    m = json.loads((a.episode / "manifest.json").read_text())
    r = json.loads(a.replay.read_text())
    trace = a.episode / "physics-trace.jsonl"
    if sha(trace) != m["trace_sha256"]["physics-trace.jsonl"] or r[
        "episode_manifest_sha256"
    ] != sha(a.episode / "manifest.json"):
        raise ValueError("Evidence hash mismatch")
    physical = {
        round(
            (
                (x["task_time_s"] + m["startup_support"]["task_started_at_s"])
                if x["score_enabled"]
                else x["timestamp_s"]
            ),
            6,
        ): x
        for x in map(json.loads, trace.read_text().splitlines())
    }
    out = {
        "schema": "microduck.geometry-error-audit.v1",
        "development_only": True,
        "qualification_claimed": False,
        "truth_used_by": "offline scoring only",
        "truth_sent_to_controller": False,
        "time_join": "task_time_s plus recorded fixture release for scored rows; initial v6 timestamp_s was overwritten by scorer task clock, frozen trace retained",
        "episode_manifest_sha256": sha(a.episode / "manifest.json"),
        "replay_sha256": sha(a.replay),
        "physics_trace_sha256": sha(trace),
        "rows": [],
        "post_release_rows": 0,
    }
    for sample in r["rows"]:
        actual = physical[round(sample["timestamp_s"], 6)]
        if not actual["score_enabled"]:
            continue
        out["post_release_rows"] += 1
        q = rotation(actual["root_quaternion_xyzw"])
        root = np.asarray(actual["root_position_world_m"])
        row = {
            "timestamp_s": sample["timestamp_s"],
            "ball_error_m": None,
            "goal_center_error_m": None,
            "localized_pose_error_m": None,
        }
        geo = sample["geometry"]
        for key, target, error_key in [
            ("ball", actual["ball_position_world_m"], "ball_error_m"),
            ("goal", [0.93, 0, 0], "goal_center_error_m"),
        ]:
            value = geo.get(key)
            if value and value["accepted"]:
                point = value["geometry"][
                    "point_reference_m" if key == "ball" else "goal_center_reference_m"
                ]
                estimate = root + q @ np.asarray(point)
                row[error_key] = float(np.linalg.norm(estimate - np.asarray(target)))
        pose = sample["control"].get("estimated_pose_map_xyyaw")
        if pose is not None:
            row["localized_pose_error_m"] = float(
                np.linalg.norm(np.asarray(pose[:2]) - root[:2])
            )
        out["rows"].append(row)
    out["metrics"] = {}
    for key in ("ball_error_m", "goal_center_error_m", "localized_pose_error_m"):
        values = [r[key] for r in out["rows"] if r[key] is not None]
        out["metrics"][key] = {
            "accepted": len(values),
            "total": len(out["rows"]),
            "median": float(np.median(values)) if values else None,
            "p95": float(np.quantile(values, 0.95)) if values else None,
            "max": max(values) if values else None,
        }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["metrics"], indent=2))


if __name__ == "__main__":
    main()
