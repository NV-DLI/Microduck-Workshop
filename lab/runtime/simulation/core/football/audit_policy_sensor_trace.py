#!/usr/bin/env python3
"""Offline privileged diagnosis. Never import this in a controller or executor."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(x):
    a = np.asarray(x)
    return (
        {
            "count": len(a),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }
        if len(a)
        else None
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Keep all offline diagnoses")
    manifest = json.loads((a.run / "manifest.json").read_text())
    for n, h in manifest["trace_sha256"].items():
        if sha(a.run / n) != h:
            raise ValueError("Trace hash mismatch: " + n)
    load = lambda n: [
        json.loads(l) for l in (a.run / (n + ".jsonl")).read_text().splitlines()
    ]
    physics = load("physics-trace")
    sensors = load("sensor-inputs")
    policy = load("policy-trace")
    release = manifest["startup_support"]["task_started_at_s"]
    times = np.array(
        [
            r["task_time_s"] + release if r.get("score_enabled") else r["timestamp_s"]
            for r in physics
        ]
    )
    rotation = Rotation.from_quat([r["root_quaternion_xyzw"] for r in physics])
    g = rotation.inv().apply(np.tile([0.0, 0.0, -1.0], (len(times), 1)))
    # Central body-frame quaternion derivative uses truth ONLY in this independent
    # scorer. It is not a deployable velocity estimate or controller observation.
    omega = np.empty((len(times), 3))
    omega[1:-1] = (rotation[:-2].inv() * rotation[2:]).as_rotvec() / (
        times[2:] - times[:-2]
    )[:, None]
    omega[0] = omega[1]
    omega[-1] = omega[-2]
    joint_names = manifest["joint_names"]
    ids = [joint_names.index(n) for n in manifest["policy_joint_names"]]
    pids = [joint_names.index(n) for n in manifest["passive_joint_names"]]
    home = np.array(
        [
            0.0,
            -0.0873,
            -0.4579,
            -0.0049,
            0.453,
            0.3491,
            0.3491,
            0.0,
            0.0,
            0.0,
            0.0873,
            0.4579,
            0.0049,
            -0.453,
        ]
    )
    records = []
    prev = None
    for r in policy:
        t = r["timestamp_s"]
        ob = np.array(r["observation"])
        valid = [
            s
            for s in sensors
            if s["timestamp_s"] <= t + 1e-9 and s["proprio"] is not None
        ]
        if not valid:
            prev = r
            continue
        s = valid[-1]
        i = int(np.argmin(abs(times - t)))
        si = int(np.argmin(abs(times - s["timestamp_s"])))
        age = t - s["timestamp_s"]
        q = np.array(physics[i]["joint_positions_absolute_rad"])
        qd = np.array(physics[i]["joint_velocities_rad_s"])
        gravity = ob[3:6]
        angle = lambda a, b: float(
            np.rad2deg(
                np.arccos(
                    np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)
                )
            )
        )
        # Causal constant body-gyro prediction, parameters diagnostic only; uses only
        # previously delivered sensor packet. No truth in prediction calculation.
        predicted = Rotation.from_rotvec(-ob[:3] * age).apply(gravity)
        rec = {
            "t": t,
            "task_t": t - release,
            "imu_age_s": age,
            "gravity_error_deg": angle(gravity, g[i]),
            "acquisition_gravity_error_deg": angle(s["proprio"]["gravity_body"], g[si]),
            "constant_body_gyro_predicted_gravity_error_deg": angle(predicted, g[i]),
            "gyro_error_norm_rad_s": float(np.linalg.norm(ob[:3] - omega[i])),
            "acquisition_gyro_error_norm_rad_s": float(
                np.linalg.norm(np.array(s["proprio"]["gyro_body_rad_s"]) - omega[si])
            ),
            "encoder_position_error_max_rad": float(
                np.max(abs(ob[6:20] - (q[ids] + q[pids] - home)))
            ),
            "raw_executed_action_error_max": float(
                np.max(abs(np.array(r["raw_action"]) - r["action"]))
            ),
            "rate_limited_joints": sum(r["bounded_action_trace"]["rate_limited"]),
            "boundary_limited_joints": sum(
                r["bounded_action_trace"]["boundary_limited"]
            ),
            "command": ob[48:51].tolist(),
            "true_root_height_m": physics[i]["root_position_world_m"][2],
        }
        if prev is not None:
            rec["action_history_matches_previous_executed_max_error"] = float(
                np.max(abs(ob[34:48] - prev["action"]))
            )
        records.append(rec)
        prev = r
    metrics = {}
    for group, predicate in [
        ("supported", lambda r: r["task_t"] < 0),
        ("moving", lambda r: r["task_t"] >= 0),
        ("early_moving_300ms", lambda r: 0 <= r["task_t"] <= 0.3),
    ]:
        subset = [r for r in records if predicate(r)]
        metrics[group] = {
            k: stats([r[k] for r in subset if k in r])
            for k in records[-1]
            if k not in ["t", "task_t", "command"]
        }
    report = {
        "schema": "microduck.offline-policy-sensor-diagnosis.v1",
        "truth_scope": "Independent offline error scoring only; executor never imports this module",
        "run": str(a.run.resolve()),
        "source_manifest_sha256": sha(a.run / "manifest.json"),
        "trace_hashes": manifest["trace_sha256"],
        "metrics": metrics,
        "per_control_step": records,
        "missing_evidence": [
            "Actual BAM delayed target and supply/torque were not logged in v7; configured20ms is not equivalent to perstep readback",
            "Exact v7 IMU raw FIFO batches were not saved; only delivered camera-aligned final samples remain",
            "v7 gyro truth is quaternion finite difference, not saved root angular velocity",
        ],
        "initialization_included": True,
        "closed_loop_parameters_prediction_tested": False,
        "diagnosis_is_not_success_rate": True,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
