#!/usr/bin/env python3
"""Generate replayable Beta FoV/K/D assumptions and CPU optical sensitivity.

This validates ray geometry only. GPU render, detector training, task success
and Beta hardware accuracy remain separate evaluation gates.
"""
from dataclasses import asdict
from pathlib import Path
import argparse, hashlib, json, math
from beta_sensor_profile import BetaCameraProfile, HEAD_IMU_PROFILE, PROFILE_ID
from beta_tof_geometry import TOF_PROFILE
from beta_perception_geometry import raw_pixel_to_unit_ray


def distorted_pixel(p, x, y):
    k1, k2, p1, p2, k3 = p.distortion_k1_k2_p1_p2_k3
    r2 = x * x + y * y
    a = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    return (
        p.fx * (x * a + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)) + p.cx,
        p.fy * (y * a + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y) + p.cy,
    )


def build(width=960, height=540):
    scenarios = []
    for fov in (60.0, 70.0, 80.0, 90.0):
        for label, d in [
            ("pinhole", (0.0, 0.0, 0.0, 0.0, 0.0)),
            ("barrel", (-0.08, 0.01, 0.002, -0.002, 0.0)),
            ("pincushion", (0.08, -0.01, -0.002, 0.002, 0.0)),
        ]:
            scenarios.append(
                (
                    f"hfov{int(fov)}-{label}",
                    BetaCameraProfile(
                        width_px=width,
                        height_px=height,
                        assumed_horizontal_fov_deg=fov,
                        distortion_k1_k2_p1_p2_k3=d,
                    ),
                )
            )
    for offset in (-0.03, 0.03):
        scenarios.append(
            (
                f"hfov70-offset{offset:+.2f}",
                BetaCameraProfile(
                    width_px=width,
                    height_px=height,
                    principal_offset_fraction=(offset, -offset),
                ),
            )
        )
    nominal = BetaCameraProfile(width_px=width, height_px=height)
    entries = []
    for name, p in scenarios:
        points = []
        for yaw in (-25.0, -15.0, 0.0, 15.0, 25.0):
            for pitch in (-10.0, 0.0, 10.0):
                x, y = math.tan(math.radians(yaw)), math.tan(math.radians(pitch))
                uv = distorted_pixel(p, x, y)
                if not 0 <= uv[0] < width - 1 or not 0 <= uv[1] < height - 1:
                    continue
                ray = raw_pixel_to_unit_ray(uv, p)
                wrong = raw_pixel_to_unit_ray(uv, nominal)
                residual = max(abs(ray[0] / ray[2] - x), abs(ray[1] / ray[2] - y))
                err = math.degrees(math.atan2(wrong[0], wrong[2])) - yaw
                points.append(
                    {
                        "true_bearing_deg": yaw,
                        "true_elevation_tangent_deg": pitch,
                        "pixel": list(uv),
                        "roundtrip_error_normalized": residual,
                        "bearing_error_if_estimator_keeps_nominal70_deg": err,
                    }
                )
        entries.append(
            {
                "scenario_id": name,
                "profile_id": PROFILE_ID,
                "camera_kwargs": asdict(p),
                "camera": p.to_dict(),
                "detector_contract": {
                    "image_rotation_deg": 0,
                    "input_image_size_wh": [width, height],
                    "reverse_letterbox_before_geometry": True,
                    "use_scenario_K_and_D_for_geometry": True,
                    "training_scope": "vary scene camera optics across the listed scenarios; split validation by scenario and seed",
                },
                "ray_checks": {
                    "points": points,
                    "max_roundtrip_error_normalized": max(
                        t["roundtrip_error_normalized"] for t in points
                    ),
                    "worst_bearing_error_if_nominal70_kept_deg": max(
                        abs(t["bearing_error_if_estimator_keeps_nominal70_deg"])
                        for t in points
                    ),
                },
            }
        )
    return {
        "schema_version": 1,
        "profile_id": PROFILE_ID,
        "default_scenario_id": "hfov70-pinhole",
        "hardware": "Beta GC2093 + SPI VL53L8CX + LSM6DSV16X",
        "unknowns": [
            "physical Beta lens FoV",
            "physical Beta K/D",
            "unit-specific camera/IMU extrinsics",
            "measured sensor noise",
        ],
        "scenario_selection_is_hardware_calibration": False,
        "tof": TOF_PROFILE.to_dict(),
        "head_imu": HEAD_IMU_PROFILE.to_dict(),
        "passed": all(
            x["ray_checks"]["max_roundtrip_error_normalized"] < 1e-8 for x in entries
        ),
        "gate": "CPU optical geometry only",
        "gpu_render_validated": False,
        "detector_trained": False,
        "football_success_validated": False,
        "scenarios": entries,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args()
    if args.output.exists():
        ap.error("Choose a new output path; preserve previous evidence")
    result = build(args.width, args.height)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "scenarios": len(result["scenarios"]),
                "output": str(args.output),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )
    raise SystemExit(0 if result["passed"] else 1)
