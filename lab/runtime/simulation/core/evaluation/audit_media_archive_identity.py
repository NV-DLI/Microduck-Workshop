"""Independent completed-run identity audit; no image decoding or simulator."""

import argparse
import datetime
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    media = json.loads((root / "media-archive-report.json").read_text())
    sensor = rows(root / "sensor-inputs.jsonl")
    commits = rows(root / "media-commits.jsonl")
    video = rows(root / "media-frame-ledger.jsonl")
    control = rows(root / "control.jsonl")
    detector = rows(root / "detections.jsonl")
    n = len(sensor)
    by_t = {r["timestamp_s"]: r for r in sensor}
    control_by_t = {r["timestamp_s"]: r for r in control}
    checks = {}
    checks["all_capture_counts_equal"] = (
        n > 0
        and n
        == len(commits)
        == len(video)
        == len(control)
        == manifest["camera_frames"]
        == media["submittedCaptures"]
        == media["committedCaptures"]
        == media["expectedCaptures"]
        == media["frameCount"]
    )
    checks["ordinals_complete_unique"] = (
        [r["captureOrdinal"] for r in commits]
        == [r["captureOrdinal"] for r in video]
        == list(range(n))
    )
    checks["timestamp_unique_strictly_increasing"] = len(by_t) == len(
        control_by_t
    ) == n and all(
        a["timestamp_s"] < b["timestamp_s"] for a, b in zip(sensor, sensor[1:])
    )
    checks["exact_capture_commit_video_control_time"] = all(
        s["timestamp_s"] == c["timestamp_s"] == v["timestamp_s"]
        and s["timestamp_s"] in control_by_t
        for s, c, v in zip(sensor, commits, video)
    )
    checks["physics_tick_binding"] = all(
        s["physics_step"] == c["physics_step"] == v["physics_step"]
        and abs(s["timestamp_s"] - s["physics_step"] / manifest["physics_hz"]) < 1e-10
        for s, c, v in zip(sensor, commits, video)
    )
    checks["render_generation_identity"] = all(
        s["render_generation"] == v["render_generation"] for s, v in zip(sensor, video)
    )
    checks["RGB_commit_video_identity"] = all(
        s["rgb_pixel_sha256"] == c["rgb_pixel_sha256"] == v["sourceRGBPixelSHA256"]
        and s["rgb_file_sha256"] == c["rgb_file_sha256"]
        for s, c, v in zip(sensor, commits, video)
    )
    checks["overhead_commit_video_identity"] = all(
        s["overhead_pixel_sha256"] == v["sourceOverheadPixelSHA256"]
        and s["overhead_file_sha256"] == c["overhead_file_sha256"]
        for s, c, v in zip(sensor, commits, video)
    )
    checks["video_labels_use_same_recorded_control"] = all(
        v["controlRecorded"] and v["state"] == control_by_t[v["timestamp_s"]]["state"]
        for v in video
    )
    checks["detector_uses_same_capture_RGB"] = all(
        r["timestamp_s"] in by_t
        and r["image_sha256"] == by_t[r["timestamp_s"]]["rgb_pixel_sha256"]
        for r in detector
    )
    checks["capture_payload_file_SHA"] = all(
        sha(root / s[k]) == s[h]
        for s in sensor
        for k, h in [
            ("rgb", "rgb_file_sha256"),
            ("overhead", "overhead_file_sha256"),
            ("tof_range", "tof_file_sha256"),
        ]
    )
    checks["ToF_raw_identity_and_same_timestamp_present"] = all(
        len(s["tof_source_float32_sha256"]) == 64
        and s["tof_frame"]["timestamp_s"] == s["timestamp_s"]
        for s in sensor
    )
    checks["commit_after_submit"] = all(
        s["capture_submit_unix_s"] <= c["archiveCommittedUnixSeconds"]
        for s, c in zip(sensor, commits)
    )
    checks["worker_encoders_exit_success"] = (
        media["workerExited"]
        and media["workerExitCode"] == 0
        and all(
            v["child_exited"] and v["exit_code"] == 0 for v in media["videos"].values()
        )
    )
    checks["no_drop_missing_or_incomplete"] = (
        media["status"] == "complete"
        and media["evidenceComplete"]
        and not media["missingControlTimestamps"]
    )
    checks["encoded_files_SHA_counts"] = all(
        v["frames"] == n and sha(root / v["path"]) == v["sha256"]
        for v in media["videos"].values()
    )
    checks["manifest_trace_SHA"] = all(
        sha(root / p) == digest for p, digest in manifest["trace_sha256"].items()
    )
    checks["source_snapshot_SHA"] = all(
        sha(root / v["snapshot"]) == v["sha256"]
        for v in manifest["source_snapshot"].values()
    )
    checks["sources_unchanged_during_run"] = (
        not manifest["source_changed_during_run"]
        and not manifest["native_reference_contract"]["source_changed_during_run"]
    )
    counts = {
        "camera": n,
        "commits": len(commits),
        "video": len(video),
        "control": len(control),
        "detector": len(detector),
    }
    for name, filename in [
        ("physics", "physics-trace.jsonl"),
        ("policy", "policy-trace.jsonl"),
    ]:
        with (root / filename).open() as f:
            counts[name] = sum(bool(line.strip()) for line in f)
    checks["cadence_counts_match"] = (
        counts["physics"] == manifest["physics_samples"] == counts["policy"] * 4
        and counts["policy"] == manifest["policy_updates"]
        and abs(counts["physics"] / 200 - manifest["simulated_seconds"]) < 1e-8
    )
    checks["not_task_or_hardware_qualification"] = (
        not manifest["qualification_claimed"]
        and not manifest["formal_qualification_claimed"]
        and not media["qualificationClaimed"]
    )
    paths = [
        root / n
        for n in [
            "manifest.json",
            "media-archive-report.json",
            "sensor-inputs.jsonl",
            "media-commits.jsonl",
            "media-frame-ledger.jsonl",
            "control.jsonl",
            "detections.jsonl",
        ]
    ] + [Path(__file__)]
    return {
        "schema": "microduck.media-independent-identity-review.v2",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "passed": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "simulated_seconds": manifest["simulated_seconds"],
        "task_seconds": manifest["task_seconds"],
        "loop_wall_seconds": manifest["loop_wall_seconds"],
        "wall_including_drain_seconds": manifest["wall_seconds_including_media_drain"],
        "scope": "Read-only all-record ACK/time/identity and actual archived file hashes. Does not re-decode every image or infer controller/task success. No simulator, GPU, hardware-fit or speedup claim. Raw ToF byte identity is declared at capture/worker and file rehashed here, not independently decoded here.",
        "source_sha256": {str(p): sha(p) for p in paths},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit(args.run)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "checks": len(report["checks"]),
                "failed": [k for k, v in report["checks"].items() if not v],
                "counts": report["counts"],
            }
        )
    )
    raise SystemExit(0 if report["passed"] else 1)
