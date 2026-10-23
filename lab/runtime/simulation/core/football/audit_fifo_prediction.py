#!/usr/bin/env python3
"""Offline privileged error scorer for delivered-FIFO training only."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def stats(a):
    a = np.asarray(a)
    return (
        {
            "count": len(a),
            "mean": float(a.mean()),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }
        if len(a)
        else None
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    m = json.loads((a.run / "manifest.json").read_text())
    for name, h in m["trace_sha256"].items():
        if hashlib.sha256((a.run / name).read_bytes()).hexdigest() != h:
            raise ValueError("Input evidence changed")
    load = lambda name: [
        json.loads(l) for l in (a.run / (name + ".jsonl")).read_text().splitlines()
    ]
    physics = load("physics-trace")
    policies = load("policy-trace")
    fifo = load("imu-fifo")
    lookup = {round(r["timestamp_s"], 8): r for r in physics}
    release = m["startup_support"]["task_started_at_s"]
    angle = lambda x, y: float(
        np.rad2deg(
            np.arccos(
                np.clip(np.dot(x, y) / np.linalg.norm(x) / np.linalg.norm(y), -1, 1)
            )
        )
    )
    rows = []
    samples = []
    for batch in fifo:
        for s in batch["body_samples"]:
            if not s["valid"]:
                continue
            truth = lookup[round(s["timestamp_s"], 8)]
            samples.append(
                {
                    "t": s["timestamp_s"],
                    "task_s": s["timestamp_s"] - release,
                    "gravity_error_deg": angle(
                        s["gravity_body"], truth["projected_gravity_body"]
                    ),
                    "gyro_error_rad_s": float(
                        np.linalg.norm(
                            np.asarray(s["gyro_body_rad_s"])
                            - truth["root_angular_velocity_body_rad_s"]
                        )
                    ),
                    "fifo_age_s": s["delivered_timestamp_s"] - s["timestamp_s"],
                }
            )
    for r in policies:
        prediction = r.get("IMU_prediction")
        t = r["timestamp_s"]
        if not prediction or round(t, 8) not in lookup:
            continue
        truth = lookup[round(t, 8)]
        g = np.asarray(r["observation"][3:6])
        w = np.asarray(r["observation"][:3])
        rows.append(
            {
                "t": t,
                "task_s": t - release,
                "prediction_age_s": prediction["measurement_age_s"],
                "gravity_error_deg": angle(g, truth["projected_gravity_body"]),
                "gyro_error_rad_s": float(
                    np.linalg.norm(w - truth["root_angular_velocity_body_rad_s"])
                ),
                "raw_target_error_max_rad": float(
                    np.max(abs(np.asarray(r["raw_action"]) - r["action"]))
                ),
                "gyro_norm_rad_s": float(np.linalg.norm(w)),
            }
        )
    result = {
        "schema": "microduck.offline-fifo-error.v1",
        "run": str(a.run),
        "scope": "independent privileged error audit; no truth supplied to controller",
        "source_manifest_sha256": hashlib.sha256(
            (a.run / "manifest.json").read_bytes()
        ).hexdigest(),
        "source_trace_hashes": m["trace_sha256"],
        "measurement": {},
        "control": {},
        "samples": samples,
        "control_rows": rows,
        "fall_task_s": m["task_seconds"] if m["terminal_reason"] == "fall" else None,
    }
    for label, source in [("measurement", samples), ("control", rows)]:
        for name, predicate in [
            ("supported", lambda t: t < 0),
            ("early_move_500ms", lambda t: 0 <= t < 0.5),
            ("moving", lambda t: t >= 0),
            ("last500ms", lambda t: m["task_seconds"] - 0.5 <= t <= m["task_seconds"]),
        ]:
            subset = [r for r in source if predicate(r["task_s"])]
            result[label][name] = {
                k: stats([r[k] for r in subset])
                for k in source[0]
                if k not in ["t", "task_s"]
            }
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ["measurement", "control", "fall_task_s"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
