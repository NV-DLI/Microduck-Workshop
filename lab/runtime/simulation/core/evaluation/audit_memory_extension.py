"""Offline shadow only: replay six-second memories; truth never enters control."""

from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

M = Path(__file__).resolve().parents[1]
O = M.parents[1] / "configuration/core"
sys.path[:0] = [
    str(M / "evaluation"),
    str(M / "football"),
    str(M / "sensors"),
    str(M / "gait"),
]
from ball_status_adapter import parent
from leg_odometry import LegFK
from hardware_head_fk import MJCF_PATH

RUN = O / "football-v66-standoff-dev90-env2"
OUT = O / "football-v66-preparation-v1/memory6-shadow-audit-v2.json"
mm = parent.memory_module


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
    assert not OUT.exists()
    control = [json.loads(l) for l in (RUN / "control.jsonl").open()]
    ticks = [json.loads(l) for l in (RUN / "public-memory-readback.jsonl").open()]
    entries = [r for r in control if r.get("public_fine_entry", {}).get("trigger")]
    predictions = []
    for entry in entries:
        t0 = entry["timestamp_s"]
        anchor = entry["public_neutral_fine_step"]["memory"]
        start = next(
            r
            for r in reversed(ticks)
            if r["readback"].get("actual_packet", {}).get("read_timestamp_s")
            == anchor["sensor_read_timestamp_s"]
        )
        initial = start["readback"]["actual_packet"]
        memory = mm.BallMemory(
            LegFK(MJCF_PATH, require_pinned=False),
            initial["joint_names"],
            maximum_age_s=6.0,
        )

        def update(p):
            memory.update(
                mm.PublicRead(
                    p["read_timestamp_s"],
                    p["encoder_timestamp_s"],
                    p["orientation_valid"],
                    tuple(p["q_rad"]),
                    tuple(p["qd_rad_s"]),
                    tuple(p["gyro_body_rad_s"]),
                    tuple(p["gravity_body"]),
                    tuple(p["joint_names"]),
                )
            )

        update(initial)
        memory.observe_complete_ball(
            timestamp_s=t0,
            point_body_m=anchor["point_body_m"],
            estimated_range_uncertainty_m=anchor["original_range_uncertainty_m"],
            complete=True,
            geometry_accepted=True,
            capture_read_age_s=t0 - initial["read_timestamp_s"],
        )
        last = initial
        samples = []
        parity = []
        gaps = []
        for row in ticks:
            t = row["timestamp_s"]
            if t < t0 or t > t0 + 6.0:
                continue
            p = row["readback"]["actual_packet"]
            if p["seq"] != last["seq"]:
                assert p["seq"] == last["seq"] + 1
                gaps.append(p["read_timestamp_s"] - last["read_timestamp_s"])
                update(p)
                last = p
            state = memory.status(t)
            assert state["memory_available"]
            old = row["handoff"]["memory"]
            if old["memory_available"] and old["ball_capture_timestamp_s"] == t0:
                parity.append(
                    float(
                        np.max(
                            np.abs(
                                np.array(state["point_level_m"]) - old["point_level_m"]
                            )
                        )
                    )
                )
            samples.append(
                {
                    "t": t,
                    "age": t - t0,
                    "point": state["point_level_m"],
                    "gravity": p["gravity_body"],
                }
            )
        predictions.append(
            {
                "entry_s": t0,
                "max_original_4p5s_parity_error_m": max(parity, default=None),
                "max_read_gap_s": max(gaps, default=None),
                "samples": samples,
            }
        )
    want = {round(s["t"], 6): [] for a in predictions for s in a["samples"]}
    for a in predictions:
        for s in a["samples"]:
            want[round(s["t"], 6)].append(s)
    for line in (RUN / "physics-trace.jsonl").open():
        r = json.loads(line)
        key = round(r["timestamp_s"], 6)
        if key not in want:
            continue
        body = rotation(r["root_quaternion_xyzw"]).T @ (
            np.array(r["ball_position_world_m"]) - r["root_position_world_m"]
        )
        for s in want[key]:
            truth = mm.level_rotation(s["gravity"]) @ body
            s["truth_xy"] = truth[:2].tolist()
            s["error_xy_m"] = float(
                np.linalg.norm(np.array(s["point"][:2]) - truth[:2])
            )
    report = []
    for a in predictions:
        assert all("error_xy_m" in s for s in a["samples"])
        bins = {}
        for s in a["samples"]:
            b = "0_to_4p5" if s["age"] <= 4.5 else "over_4p5_to_6"
            bins[b] = max(bins.get(b, 0.0), s["error_xy_m"])
        report.append(
            {k: v for k, v in a.items() if k != "samples"}
            | {
                "samples": len(a["samples"]),
                "maximum_xy_error_m": bins,
                "last_sample": a["samples"][-1],
                "within_10mm_development_limit": all(v <= 0.01 for v in bins.values()),
            }
        )
    result = {
        "schema": "microduck.memory6-shadow-audit.v1",
        "audit_only": True,
        "controller_ttl_unchanged_s": 4.5,
        "shadow_ttl_s": 6.0,
        "attempts": report,
        "limited_evidence": True,
        "new_longer_step_validated": False,
        "truth_frame": "actual root relative ball rotated to body and then public gravity-level, audit only",
        "source_sha256": {
            str(RUN / n): hashlib.sha256((RUN / n).read_bytes()).hexdigest()
            for n in [
                "control.jsonl",
                "public-memory-readback.jsonl",
                "physics-trace.jsonl",
            ]
        },
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
