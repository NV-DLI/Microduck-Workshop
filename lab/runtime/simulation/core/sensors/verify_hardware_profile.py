#!/usr/bin/env python3
"""CPU source/interface audit; never a physical calibration or task qualification."""
import argparse, hashlib, json, math, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from beta_sensor_profile import (
    CAMERA_PROFILE,
    HEAD_IMU_PROFILE,
    OFFICIAL_COMMIT,
    PROFILE_ID,
)
from beta_tof_geometry import TOF_PROFILE, tof_zone_rays

S = Path(__file__).resolve().parent
M = S.parent
C = M.parents[1]
O = C / "configuration/core"
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, default=S / "reports/hardware-profile.json")
a = p.parse_args()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


research_path = C / "outputs/research/20261019/hardware-sensors-refresh.json"
research = json.loads(research_path.read_text())
sources = []
texts = {}
checks = []


def check(name, passed, **evidence):
    checks.append({"name": name, "passed": bool(passed), **evidence})


for item in research["sources"]:
    path = C / item["path"]
    actual = sha(path)
    sources.append(
        {
            "path": str(path),
            "url": item["url"],
            "upstream_path": item["upstream_path"],
            "expected_sha256": item["sha256"],
            "actual_sha256": actual,
            "hash_matches": actual == item["sha256"],
        }
    )
    texts[item["upstream_path"]] = path.read_text()
for item in json.loads((S / "sources/manifest.json").read_text()):
    path = S / "sources" / item["path"]
    actual = sha(path)
    sources.append(
        {
            "path": str(path),
            "url": item["url"],
            "expected_sha256": item["sha256"],
            "actual_sha256": actual,
            "hash_matches": actual == item["sha256"],
        }
    )
    texts[item["path"]] = path.read_text()
check(
    "all_pinned_official_downloads_match_recorded_sha256",
    all(s["hash_matches"] for s in sources),
    source_count=len(sources),
)
check(
    "profile_and_review_use_identical_pinned_runtime_commit",
    research["upstream_commit"] == OFFICIAL_COMMIT,
    commit=OFFICIAL_COMMIT,
)
camera = CAMERA_PROFILE.to_dict()
tof = TOF_PROFILE.to_dict()
imu = HEAD_IMU_PROFILE.to_dict()
board = texts["robotd-params/src/board.rs"]
sensor = texts["mediad/src/sensor.rs"]
gc = sensor.split("pub const GC2093: Sensor = Sensor {", 1)[1].split(
    "// The GC2093", 1
)[0]
link = texts["tof/src/link.rs"]
tof_main = texts["tof/src/main.rs"]
tof_geometry = texts["kinematics/src/tof.rs"]
imu_src = texts["head_imu.rs"]
params = texts["robotd-params/src/lib.rs"]


def source_lines(key, patterns):
    lines = texts[key].splitlines()
    return {
        "file": key,
        "matched_lines": sorted(
            {
                i + 1
                for i, line in enumerate(lines)
                for pattern in patterns
                if re.search(pattern, line)
            }
        ),
    }


check(
    "beta_is_gc2093_and_native_zero_degree_mount",
    "Board::Beta => crate::CameraSensor::Gc2093" in board
    and "Board::Beta => 0" in board
    and camera["sensor"] == "GalaxyCore GC2093"
    and camera["mount_rotation_cw_deg"] == 0,
    source=source_lines(
        "robotd-params/src/board.rs",
        ["Beta => crate::CameraSensor::Gc2093", "Beta => 0"],
    ),
)
check(
    "gc2093_1080p30_raw10_mode",
    all(x in gc for x in ("width: 1920", "height: 1080", "SRGGB10_1X10"))
    and "1920×1080 raw 10-bit at 30 fps" in sensor
    and camera["hardware_driver_raster"] == [1920, 1080]
    and camera["hardware_driver_rate_hz"] == 30,
    source=source_lines("mediad/src/sensor.rs", ["1920×1080 raw", "pub const GC2093"]),
)
check(
    "beta_physical_optics_remain_unknown_not_alpha",
    all(x in gc for x in ("hfov_deg: None", "family: None"))
    and all(
        camera[k] is None
        for k in (
            "physical_K",
            "physical_D",
            "physical_lens_hfov_deg",
            "physical_lens_model",
        )
    )
    and camera["calibrated"] is False
    and camera["raw_stream_rotated"] is False,
    source=source_lines(
        "mediad/src/sensor.rs",
        ["nobody has measured", "hfov_deg: None", "family: None"],
    ),
)
check(
    "beta_l8cx_spi3_bus",
    all(
        x in link
        for x in ("beta board wires a VL53L8CX to SPI3", "/dev/spidev3.0", "8 MHz")
    )
    and tof["sensor_generations"] == ("VL53L8CX",)
    and tof["bus"] == "SPI3"
    and tof["spi_clock_hz"] == 8000000,
    source=source_lines("tof/src/link.rs", ["VL53L8CX to SPI3", "8 MHz", "BETA_SPI:"]),
)
check(
    "tof_8_by_8_15_hz_45_degree_axes",
    all(
        x in tof_geometry
        for x in ("ROWS: usize = 8", "COLS: usize = 8", "FOV_DEG: f64 = 45.0")
    )
    and "#[arg(long, default_value_t = 15)]" in tof_main
    and (
        tof["rows"],
        tof["cols"],
        tof["fps"],
        tof["horizontal_fov_deg"],
        tof["vertical_fov_deg"],
    )
    == (8, 8, 15.0, 45.0, 45.0),
    source=[
        source_lines("kinematics/src/tof.rs", ["ROWS:", "COLS:", "FOV_DEG:"]),
        source_lines("tof/src/main.rs", ["default_value_t = 15"]),
    ],
)
check(
    "lsm6dsv16x_default_enabled_robotd",
    all(
        x in params
        for x in (
            "an LSM6DSV16X",
            "board::Board::Beta => true",
            'board::Board::Beta => "robotd"',
        )
    )
    and (imu["model"], imu["enabled"], imu["reader"]) == ("LSM6DSV16X", True, "robotd"),
    source=source_lines(
        "robotd-params/src/lib.rs",
        ["an LSM6DSV16X", "Board::Beta => true", 'Board::Beta => "robotd"'],
    ),
)
check(
    "official_imu_rate_and_range_semantics",
    all(
        x in imu_src
        for x in (
            "SAMPLE_HZ: u8 = 60",
            "WAKE_HZ: u32 = 15",
            "gyro ±500 dps",
            "accel 120 Hz",
            "gyro  120 Hz",
        )
    )
    and (
        imu["simulated_sample_hz"],
        imu["fifo_read_hz"],
        imu["sensor_internal_odr_hz"],
        imu["gyro_range_deg_s"],
        imu["accel_range_g"],
    )
    == (60.0, 15.0, 120.0, 500.0, 4.0),
    source=source_lines(
        "head_imu.rs", ["SAMPLE_HZ:", "WAKE_HZ:", "gyro ±500", "accel 120", "gyro  120"]
    ),
)
check(
    "simulation_limits_explicit",
    not imu["sflp_firmware_reproduced"]
    and not imu["noise_is_measured"]
    and not tof["spi_bus_is_simulated"]
    and not tof["maximum_range_is_guaranteed"],
)
ray_error = max(abs(sum(x * x for x in ray) - 1) for ray in tof_zone_rays())
check(
    "64_tof_unit_rays",
    len(tof_zone_rays()) == 64 and ray_error < 1e-12,
    maximum_norm_squared_error=ray_error,
)
matrix_path = S / "reports/beta-optics-matrix.json"
matrix = json.loads(matrix_path.read_text())
fovs = sorted(
    {
        float(m)
        for m in re.findall(
            r'"assumed_horizontal_fov_deg": ([0-9.]+)', matrix_path.read_text()
        )
    }
)
check(
    "optics_sensitivity_keeps_60_70_80_90_fov_cases",
    all(x in fovs for x in (60.0, 70.0, 80.0, 90.0))
    and matrix["scenario_selection_is_hardware_calibration"] is False,
    values_deg=fovs,
    manifest_sha256=sha(matrix_path),
)
test = subprocess.run(
    [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(S / "tests"),
        "-p",
        "test_*.py",
        "-v",
    ],
    text=True,
    capture_output=True,
)
test_log = a.output.with_suffix(".cpu-tests.log")
test_log.parent.mkdir(parents=True, exist_ok=True)
test_log.write_text(test.stdout + test.stderr)
count = re.search(r"Ran (\d+) tests", test.stderr)
check(
    "cpu_camera_tof_mount_and_imu_initialization_regressions",
    test.returncode == 0,
    test_count=int(count.group(1)) if count else None,
    log=str(test_log),
    log_sha256=sha(test_log),
)
# Read the real earlier CUDA probe and the real sensor-only policy trace. This
# script itself stays CPU-only; it does not recreate or certify a task run.
runtime_path = O / "runtime-smoke-backlash-1024.json"
runtime = json.loads(runtime_path.read_text())
joint_names = runtime["joint_names"]
passives = [x for x in joint_names if x.startswith("passive_")]
check(
    "actual_newton_28_dof_and_61_observation_interface",
    runtime["native_bam_verified"]
    and runtime["finite"]
    and len(joint_names) == 28
    and len(passives) == 14
    and runtime["observations"]["policy"] == [1024, 61],
    report=str(runtime_path),
    sha256=sha(runtime_path),
    physical_joints=len(joint_names),
    servo_channels=14,
    passive_channels=len(passives),
    policy_observation_dim=61,
)
episode = O / "football-beta-smoke-v6"
manifest = json.loads((episode / "manifest.json").read_text())
policy_rows = [
    json.loads(x) for x in (episode / "policy-trace.jsonl").read_text().splitlines()
]
active = [r for r in policy_rows if r["IMU_source"] == "Beta head IMU + encoder FK"]
check(
    "actual_head_imu_to_61d_policy_and_14d_action_trace",
    len(active) > 0
    and all(len(r["observation"]) == 61 and len(r["action"]) == 14 for r in active)
    and manifest["onnx_parity"]["passed"],
    active_policy_ticks=len(active),
    policy_trace_sha256=sha(episode / "policy-trace.jsonl"),
    manifest=str(episode / "manifest.json"),
    manifest_sha256=sha(episode / "manifest.json"),
)
implementation = {
    str(path.relative_to(M)): sha(path) for path in sorted(S.glob("*.py"))
}
for name in (
    "skills/backlash_contract.py",
    "skills/contract.py",
    "football/head_adapter.py",
    "football/isaac_execute.py",
):
    implementation[name] = sha(M / name)
passed = all(c["passed"] for c in checks)
report = {
    "schema": "microduck.hardware-profile-source-audit.v1",
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "hardware_profile_id": PROFILE_ID,
    "state": "passed" if passed else "failed",
    "recommended_gate_scope": "hardware_source_and_simulation_interface_only",
    "source_scope_passed": passed,
    "physical_hardware_calibrated": False,
    "sensor_task_success_claimed": False,
    "football_success_claimed": False,
    "release_qualification_claimed": False,
    "gpu_jobs_started_by_this_audit": False,
    "official_runtime_commit": OFFICIAL_COMMIT,
    "research_manifest": str(research_path),
    "research_manifest_sha256": sha(research_path),
    "checks": checks,
    "official_sources": sources,
    "implementation_sha256": implementation,
    "implemented_profiles": {"camera": camera, "tof": tof, "head_imu": imu},
    "interface": {
        "physical_joints": 28,
        "servo_actions": 14,
        "passive_backlash_joints": 14,
        "native_walk_sensor_observation_dim": 61,
        "skill_training_observation_dim": 75,
        "skill_observation_note": "Independent jump/basketball/dance actor has extra privileged task features; not the 61D native walking interface.",
        "encoders": "q_servo + q_passive, likewise velocity; named 14-servo order",
        "imu": "Head FLU gyro/gravity transported through acquisition-time encoder FK; invalid gravity initialization gates inference",
    },
    "unmeasured_assumptions": [
        "GC2093 lens model, horizontal FoV, K/D and principal-point offset",
        "Unit-specific camera/ToF/IMU extrinsics; selected CAD sites are explicit proxies",
        "IMU noise/bias and complementary fusion are an approximation, not ST SFLP firmware",
        "ToF geometric range/no-return approximation, no photon/SPAD or electrical SPI simulation",
        "Renderer RGB output/resolution/acquisition rate is an explicitly recorded teaching stream, not a raw GC2093 driver capture",
    ],
    "limits": [
        "Source evidence describes a pinned beta revision, not every shipped unit or final production BOM.",
        "Existing CUDA/football traces establish the interface and stream only. The v6 episode is a startup/zero-command development run, not a football evaluation.",
        "No RK3566 deployment, thermal test or real robot sensor measurement is certified here.",
    ],
}
a.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(
    json.dumps(
        {
            "state": report["state"],
            "checks": len(checks),
            "failed": [c["name"] for c in checks if not c["passed"]],
            "report": str(a.output),
        },
        indent=2,
    )
)
if not passed:
    sys.exit(1)
