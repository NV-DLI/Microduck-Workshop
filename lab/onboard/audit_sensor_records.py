"""Read-only capture identity/cadence audit for experiments without video encoding."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit(root):
    def rows(name):
        return [json.loads(line) for line in (root / name).open()]
    manifest = json.loads((root / "manifest.json").read_text())
    archive = json.loads((root / "media-archive-report.json").read_text())
    sensor, commits = rows("sensor-inputs.jsonl"), rows("media-commits.jsonl")
    controls, detections = rows("control.jsonl"), rows("detections.jsonl")
    physics, policy = rows("physics-trace.jsonl"), rows("policy-trace.jsonl")
    n = len(sensor)
    by_time = {r["timestamp_s"]: r for r in sensor}
    checks = dict(
        nonempty_captures=n > 0,
        sensor_cadence=[r["physics_step"] for r in sensor] ==
            [math.ceil(i * manifest["physics_hz"] / manifest["sensor_hz"] - 1e-9)
             for i in range(1, math.floor(manifest["simulated_seconds"] * manifest["sensor_hz"] + 1e-9) + 1)],
        no_encoded_video=not archive["videos"] and not list(root.glob("*.mp4")),
        worker_complete=archive["status"] == "complete" and archive["evidenceComplete"]
            and archive["workerExited"] and archive["workerExitCode"] == 0,
        all_captures_committed=n == len(commits) == len(controls) == manifest["camera_frames"]
            == archive["submittedCaptures"] == archive["committedCaptures"] == archive["expectedCaptures"],
        no_missing_control=not archive["missingControlTimestamps"],
        ordinals=[r["captureOrdinal"] for r in commits] == list(range(n)),
        ordered_unique_captures=len(by_time) == n and all(a["timestamp_s"] < b["timestamp_s"] for a, b in zip(sensor, sensor[1:])),
        same_capture_control_times=[r["timestamp_s"] for r in sensor] == [r["timestamp_s"] for r in controls],
        all_physics_ticks=len(physics) == manifest["physics_samples"] and all(abs(r["timestamp_s"] - (i+1)/manifest["physics_hz"]) < 1e-9 for i, r in enumerate(physics)),
        all_policy_ticks=len(policy) == manifest["policy_updates"] and all(abs(r["timestamp_s"] - i/manifest["policy_hz"]) < 1e-9 for i, r in enumerate(policy)),
        terminal_tick=abs(len(physics)/manifest["physics_hz"] - manifest["simulated_seconds"]) < 1e-9
            and len(policy) == (len(physics)+1)//2,
        detector_same_rgb=all(r["timestamp_s"] in by_time and r["image_sha256"] == by_time[r["timestamp_s"]]["rgb_pixel_sha256"] for r in detections),
        capture_commit_identity=all(s["timestamp_s"] == c["timestamp_s"] and s["physics_step"] == c["physics_step"]
            and abs(s["timestamp_s"] - s["physics_step"]/manifest["physics_hz"]) < 1e-9
            and s["rgb_pixel_sha256"] == c["rgb_pixel_sha256"]
            and s["rgb_file_sha256"] == c["rgb_file_sha256"]
            and s["overhead_file_sha256"] == c["overhead_file_sha256"]
            and s["capture_submit_unix_s"] <= c["archiveCommittedUnixSeconds"] for s, c in zip(sensor, commits)),
        capture_files=all(sha(root/s[key]) == s[digest] for s in sensor for key, digest in
            [("rgb", "rgb_file_sha256"), ("overhead", "overhead_file_sha256"), ("tof_range", "tof_file_sha256")]),
        tof_identity=all(len(s["tof_source_float32_sha256"]) == 64 and s["tof_frame"]["timestamp_s"] == s["timestamp_s"] for s in sensor),
        trace_hashes=all(sha(root/p) == digest for p, digest in manifest["trace_sha256"].items()),
        source_snapshots=all(sha(root/s["snapshot"]) == s["sha256"] for s in manifest["source_snapshot"].values()),
        sources_unchanged=not manifest["source_changed_during_run"] and not manifest["native_reference_contract"]["source_changed_during_run"],
    )
    return dict(passed=all(checks.values()), checks=checks, captures=n,
                scope="Sensor inputs and evaluation records only; no encoded-video or rendered-playback claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit(args.run)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
