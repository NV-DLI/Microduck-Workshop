#!/usr/bin/env python3
"""Audit navigation imu: Microduck simulation and control."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import audit_navigation_motion as legacy


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_body_arrays(arrays, *, nominal):
    obs = arrays["actor_observation"]
    control = arrays["control"]
    count, worlds = control.shape[:2]
    if obs.shape != (count, worlds, 61) or not np.isfinite(obs).all():
        raise ValueError("All finite delivered actor inputs are required")
    ready = arrays["body50_orientation_valid"]
    if ready.shape != (count, worlds) or not ready.all():
        raise ValueError("Unready actor observation in task")
    wrench = arrays["external_wrench_peak"]
    if (
        wrench.shape != (count * 4, worlds)
        or not np.isfinite(wrench).all()
        or np.any(wrench != 0)
    ):
        raise ValueError(
            "Actual task external wrench must be exactly zero at every physical tick"
        )
    expected_history = np.concatenate(
        (arrays["initial_previous_raw"][None], control[:-1, :, :14])
    )
    if not np.allclose(obs[:, :, 34:48], expected_history, atol=1e-6, rtol=0):
        raise ValueError("Actor history is not the preceding actual raw action")
    imu_t = arrays["body50_read_timestamp_s"]
    enc_t = arrays["body50_encoder_read_timestamp_s"]
    if imu_t.shape != (count, worlds) or not np.array_equal(imu_t, enc_t):
        raise ValueError("IMU and encoders must come from the same bus read")
    if np.any(np.diff(imu_t, axis=0) < 0) or not np.isfinite(imu_t).all():
        raise ValueError("Sensor read timestamps must be finite and monotonic")
    if nominal:
        public = np.concatenate(
            (
                arrays["body50_gyro_body_rad_s"],
                arrays["body50_gravity_body"],
                arrays["body50_encoder_q_rad"] - arrays["native_home14"][None],
                arrays["body50_encoder_qd_rad_s"],
            ),
            axis=-1,
        ).astype(np.float32)
        # The producer casts the encoder value before subtracting the FP32 home;
        # the independent calculation above has a different rounding order.
        if not np.allclose(obs[:, :, :34], public, atol=2e-6, rtol=0):
            raise ValueError(
                "Nominal actor input differs from delivered quantized sensor values"
            )
    return {
        "actor_rows": int(count * worlds),
        "task_physics_rows": int(count * 4 * worlds),
        "zero_task_wrench": True,
        "raw_history_verified": True,
        "same_read_encoders_verified": True,
        "nominal_public_sensor_values_verified": nominal,
    }


def audit_part(path, *, allow_development=False):
    result = legacy.audit_part(path, allow_development=allow_development)
    contract = result.get("body50_observation_contract", {})
    if contract != {
        "actor_dimensions": 61,
        "body_tau_s": 2.0,
        "same_read_encoders": True,
        "readiness_required": True,
        "extra_observation_dr": result["condition"] == "full_official_DR",
        "initialization_max_seconds": 8.0,
        "actor_world_state": False,
    }:
        raise ValueError("Missing exact BODY50 observation and initialization contract")
    for path, digest in result["body50_sources"].items():
        if sha(path) != digest:
            raise ValueError("BODY50 source changed after freezing")
    init = result["physical_initialization"]
    if (
        init["automatic_resets"] != 0
        or not init["task_measurement_excludes_initialization"]
    ):
        raise ValueError(
            "Initialization cannot select new worlds or count as task time"
        )
    if (
        init["physical_ticks"] > 1600
        or init["task_environments"] != result["batch_size"]
    ):
        raise ValueError("Entire original cohort must complete startup within8s")
    if result["body50_task_readback"]["actual_task_external_wrench_peak"] != 0:
        raise ValueError("Task support readback is nonzero")
    with np.load(
        Path(result["report_path"]).parent / "raw-trace.npz", allow_pickle=False
    ) as data:
        result["body50_input_audit"] = validate_body_arrays(
            data, nominal=result["condition"] == "nominal"
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--development", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Keep every audit result")
    body_parts = [
        audit_part(path, allow_development=args.development) for path in args.reports
    ]
    result = legacy.audit(args.reports, allow_development=args.development)
    result["schema"] = "microduck.navigation-v2-body50-audit.v1"
    result["body50_input_audits"] = [p["body50_input_audit"] for p in body_parts]
    result["auditor_sources"] = {
        str(Path(p).resolve()): sha(p) for p in (__file__, legacy.__file__)
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("passed", "trials", "successes", "development_only")
            }
        )
    )


if __name__ == "__main__":
    main()
