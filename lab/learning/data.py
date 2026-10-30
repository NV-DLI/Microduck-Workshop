"""Validate sensor/action episodes and export disjoint LeRobot v3 datasets."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def score_episode(directory):
    """Independent full-ball, contact, no-fall and continuous-stop score."""
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    profile = report["scene_profile"]
    rows = read_rows(directory / "physical-evaluation.jsonl")
    if not rows:
        raise ValueError("Empty physics record")
    if not math.isclose(rows[0]["timestamp_s"], 0.005, abs_tol=1e-6):
        raise ValueError("Physics record is missing its initial tick")
    radius = profile["ball_diameter_m"] / 2
    goal_x = profile["goal_plane_x_m"]
    target_x = goal_x + radius
    touched = scored = fallen = overrun = False
    goal_at = None
    quiet_at = None
    longest_quiet = 0.0
    previous = None
    for r in rows:
        values = (
            [
                r[k]
                for k in (
                    "timestamp_s",
                    "ball_contact_force_n",
                    "linear_speed",
                    "angular_speed",
                    "gravity_z",
                )
            ]
            + r["ball_position"]
            + r["robot_position"]
        )
        if (
            len(r["ball_position"]) != 3
            or len(r["robot_position"]) != 3
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)
        ):
            raise ValueError("Non-finite or malformed physics sample")
        stamp = float(r["timestamp_s"])
        if previous and not math.isclose(
            stamp - previous["timestamp_s"], 0.005, abs_tol=1e-6
        ):
            raise ValueError("Non-contiguous 200 Hz physics log")
        touched |= r["ball_contact_force_n"] > 0.05
        p = r["ball_position"]
        if previous:
            old = previous["ball_position"]
            if touched and old[0] < target_x <= p[0]:
                f = (target_x - old[0]) / (p[0] - old[0])
                y, z = [old[i] + f * (p[i] - old[i]) for i in (1, 2)]
                crossing = (
                    abs(y) <= profile["goal_opening_width_m"] / 2 - radius
                    and radius - 0.001 <= z <= profile["goal_opening_height_m"] - radius
                )
                if crossing:
                    scored = True
                    goal_at = stamp if goal_at is None else goal_at
        overrun |= r["robot_position"][0] >= goal_x - 0.1
        fallen |= r["gravity_z"] > -0.6 or r["robot_position"][2] < 0.065
        quiet = (
            scored
            and r["linear_speed"] < 0.02
            and r["angular_speed"] < 0.3
            and r["gravity_z"] < -0.94
            and r["robot_position"][0] < goal_x - 0.1
        )
        quiet_at = (stamp if quiet_at is None else quiet_at) if quiet else None
        duration = 0.0 if quiet_at is None else stamp - quiet_at
        longest_quiet = max(longest_quiet, duration)
        previous = r
    final_quiet = 0.0 if quiet_at is None else rows[-1]["timestamp_s"] - quiet_at
    samples_path = directory / "learning/samples.jsonl"
    samples = read_rows(samples_path)
    if not samples:
        raise ValueError("Missing control provenance")
    for i, sample in enumerate(samples):
        if sample.get("frame") != i or not math.isclose(
            sample.get("timestamp_s", -1), i * 0.08, abs_tol=1e-6
        ):
            raise ValueError("Incomplete control frame/timestamp provenance")
        if sample.get("source") not in ("learner", "human", "automatic_expert"):
            raise ValueError("Unknown controller provenance")
    end = rows[-1]["timestamp_s"]
    if not samples[-1]["timestamp_s"] < end <= samples[-1]["timestamp_s"] + 0.080001:
        raise ValueError("Control provenance does not cover the rollout")
    sources = sorted({r["source"] for r in samples})
    physical = bool(
        touched and scored and not fallen and not overrun and final_quiet >= 1.0
    )
    return dict(
        passed=physical,
        physical_success=physical,
        policy_unassisted_success=physical and sources == ["learner"],
        control_sources=sources,
        crossed_robot_stop_boundary=overrun,
        contacted=touched,
        whole_ball_in_goal=scored,
        fallen=fallen,
        final_stop_seconds=final_quiet,
        longest_stop_seconds=longest_quiet,
        goal_at_s=goal_at,
        duration_s=rows[-1]["timestamp_s"],
        scoring="whole_ball_1mm_floor_tolerance_stop_1s_before_goal",
        seed=report["seed"],
        source=str(directory.resolve()),
    )


def segments(directory, source, corrections_only=False):
    directory = Path(directory)
    schema = json.loads((directory / "learning/schema.json").read_text())
    rows = read_rows(directory / "learning/samples.jsonl")
    groups, current = [], []
    for row in rows:
        good = row["eligible_for_training"] is True and row["source"] == source
        good &= row["correction"] is True if corrections_only else True
        if good:
            if current and (
                row["segment"] != current[-1]["segment"]
                or not math.isclose(
                    row["timestamp_s"] - current[-1]["timestamp_s"], 0.08, abs_tol=1e-6
                )
            ):
                groups.append(current)
                current = []
            if (
                len(row["state"]) != len(schema["state_names"])
                or len(row["action"]) != 4
            ):
                raise ValueError("Feature dimensions differ from declared schema")
            if not all(math.isfinite(v) for v in row["state"] + row["action"]):
                raise ValueError("Non-finite training data")
            if not (directory / "learning" / row["image"]).is_file():
                raise FileNotFoundError(row["image"])
            current.append(row)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return schema, groups


def export(manifest, output, source, corrections_only=False):
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new dataset directory")
    entries = json.loads(Path(manifest).read_text())
    if not entries:
        raise ValueError("Empty manifest")
    splits = {e.get("split") for e in entries}
    if len(splits) != 1 or not splits <= {"train", "validation", "test", "correction"}:
        raise ValueError(
            "Manifest must declare a single train/validation/test/correction split"
        )
    identities = [(e["seed"], e.get("augmentation_clip")) for e in entries]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate episode/augmentation entries")
    prepared = []
    for item in entries:
        path = Path(item["path"])
        score = score_episode(path)
        if score["seed"] != item["seed"]:
            raise ValueError("Manifest seed mismatch")
        if not corrections_only and not score["passed"]:
            raise ValueError("Demonstration failed complete-task scoring: " + str(path))
        schema, groups = segments(path, source, corrections_only)
        if (
            source == "automatic_expert"
            and not corrections_only
            and any(r["correction"] is True for rows in groups for r in rows)
        ):
            raise ValueError(
                "Automatic intervention segments require correction export and recovery qualification"
            )
        if corrections_only:
            accepted = item.get("accepted_segments", [])
            if not accepted or any(
                not r.get("reviewer") or not r.get("reason") for r in accepted
            ):
                raise ValueError(
                    "Correction export requires reviewed accepted_segments with reviewer and reason"
                )
            if source == "automatic_expert":
                try:
                    from .qualify_recovery import validate_qualification
                except ImportError:
                    from qualify_recovery import validate_qualification
                for review in accepted:
                    if not review.get("qualification_path") or not review.get(
                        "qualification_sha256"
                    ):
                        raise ValueError(
                            "Automatic correction export requires bound recovery qualification"
                        )
                    validate_qualification(
                        review["qualification_path"],
                        path,
                        review["qualification_sha256"],
                    )
            digest = hashlib.sha256(
                (path / "learning/samples.jsonl").read_bytes()
            ).hexdigest()
            if any(r.get("source_samples_sha256") != digest for r in accepted):
                raise ValueError("Correction review is not bound to this recording")
            reviewed = []
            seen_frames = set()
            for rows in groups:
                for review in accepted:
                    selected = [
                        r
                        for r in rows
                        if r["segment"] == review["segment"]
                        and review["start_frame"] <= r["frame"] <= review["end_frame"]
                    ]
                    if selected:
                        frames = {r["frame"] for r in selected}
                        if frames & seen_frames:
                            raise ValueError("Overlapping correction review ranges")
                        seen_frames.update(frames)
                        reviewed.append(selected)
            groups = reviewed
        augmentation = item.get("augmentation_clip")
        if augmentation:
            clip = Path(augmentation)
            try:
                from .cosmos_data import validate_review
            except ImportError:
                from cosmos_data import validate_review
            meta = validate_review(clip)
            if meta["episode"] != str(path.resolve()):
                raise ValueError("Generated clip does not belong to source episode")
            start = meta["start_frame"]
            end = start + meta["frames"]
            groups = [
                [
                    dict(
                        r,
                        augmentation_image=str(
                            (
                                clip / "generated" / f"{r['frame']-start:04d}.png"
                            ).resolve()
                        ),
                    )
                    for r in rows
                    if start <= r["frame"] < end
                ]
                for rows in groups
            ]
        # Long enough for at least one complete 16-step target chunk.
        for rows in groups:
            if len(rows) >= 16:
                prepared.append(
                    (path, schema, rows, score, item.get("accepted_segments", []))
                )
    if not prepared:
        raise ValueError("No eligible segments of at least 16 frames")
    schema = prepared[0][1]
    contract = schema.get("control_contract")
    if contract is None or any(
        entry[1].get("control_contract") != contract for entry in prepared
    ):
        raise ValueError(
            "Missing or mixed actuator bridge contracts; collect new compatible data"
        )
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(schema["state_names"]),),
            "names": schema["state_names"],
        },
        "action": {"dtype": "float32", "shape": (4,), "names": schema["action_names"]},
        "observation.images.front": {
            "dtype": "image",
            "shape": (270, 480, 3),
            "names": ["height", "width", "channels"],
        },
    }
    # Image storage avoids rounding 12.5 Hz to a different video time base.
    dataset = LeRobotDataset.create(
        repo_id="local/" + output.name,
        root=output,
        fps=12.5,
        robot_type="microduck_rollers",
        features=features,
        use_videos=False,
    )
    provenance = []
    for path, other_schema, rows, score, review in prepared:
        if other_schema["state_names"] != schema["state_names"]:
            raise ValueError("Mixed state schemas")
        for row in rows:
            rgb = np.asarray(
                Image.open(
                    row.get("augmentation_image", path / "learning" / row["image"])
                )
                .convert("RGB")
                .resize((480, 270))
            )
            dataset.add_frame(
                {
                    "observation.state": np.asarray(row["state"], dtype=np.float32),
                    "action": np.asarray(row["action"], dtype=np.float32),
                    "observation.images.front": rgb,
                    "task": row["task"],
                }
            )
        dataset.save_episode()
        provenance.append(
            dict(
                seed=score["seed"],
                source=str(path.resolve()),
                frames=len(rows),
                start_frame=rows[0]["frame"],
                end_frame=rows[-1]["frame"],
                label_source=source,
                augmentation=rows[0].get("augmentation_image"),
                split=next(iter(splits)),
                complete_success=score["passed"],
                corrections_only=corrections_only,
                correction_review=review,
                samples_sha256=hashlib.sha256(
                    (path / "learning/samples.jsonl").read_bytes()
                ).hexdigest(),
            )
        )
    dataset.finalize()
    (output / "split.json").write_text(json.dumps(dict(split=next(iter(splits)))))
    (output / "control-contract.json").write_text(json.dumps(contract, indent=2))
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def validate_segment_union(records):
    intervals = {}
    for record in records:
        key = (record["samples_sha256"], record.get("augmentation"))
        start, end = record["start_frame"], record["end_frame"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start <= end
        ):
            raise ValueError("Invalid source frame interval")
        previous = intervals.setdefault(key, [])
        if any(
            start <= old_end and old_start <= end for old_start, old_end in previous
        ):
            raise ValueError("Overlapping source segments in aggregated data")
        previous.append((start, end))


def summarize(directory):
    directory = Path(directory)
    plan = json.loads((directory / "plan.json").read_text())
    attempts = json.loads((directory / "attempts.json").read_text())
    completed = [r for r in attempts if r.get("exit_code") == 0 and "score" in r]
    success = sum(r["score"]["passed"] for r in completed)
    return dict(
        planned_episodes=len(plan["episodes"]),
        attempted_episodes=len(attempts),
        completed_physics_episodes=len(completed),
        infrastructure_failures=sum(r.get("exit_code") != 0 for r in attempts),
        cancelled_before_start=sum(
            r.get("cancelled_before_start") is True for r in attempts
        ),
        all_planned_episodes_completed=len(completed) == len(plan["episodes"]),
        complete_task_successes=success,
        complete_task_success_rate=success / len(completed) if completed else None,
        goals=sum(r["score"]["whole_ball_in_goal"] for r in completed),
        falls=sum(r["score"]["fallen"] for r in completed),
        learner_only_successes=sum(
            r["score"]["policy_unassisted_success"] for r in completed
        ),
        sources=sorted({s for r in completed for s in r["score"]["control_sources"]}),
        note="Infrastructure failures are reported separately; assisted completion is not autonomous policy success.",
    )


def merge(inputs, output, output_split="train"):
    """Aggregate within one split; corrections may only enter training."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import merge_datasets

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if len(inputs) < 2:
        raise ValueError(
            "Provide original demonstrations and at least one added dataset"
        )
    if output_split not in {"train", "validation"}:
        raise ValueError("Aggregation output must be train or validation")
    allowed = {"train", "correction"} if output_split == "train" else {"validation"}
    contract, lineage, datasets, identities = None, [], [], set()
    for directory in map(Path, inputs):
        split = json.loads((directory / "split.json").read_text())["split"]
        if split not in allowed:
            raise ValueError("Cannot aggregate across training and validation splits")
        other = json.loads((directory / "control-contract.json").read_text())
        if contract is not None and other != contract:
            raise ValueError("Cannot aggregate different actuator bridges")
        contract = other
        records = json.loads((directory / "provenance.json").read_text())
        for record in records:
            if record.get("split") not in allowed:
                raise ValueError("Source lineage belongs to a different split")
            identity = (
                record["samples_sha256"],
                record["start_frame"],
                record["end_frame"],
                record.get("augmentation"),
            )
            if identity in identities:
                raise ValueError("Duplicate data segment in aggregation")
            identities.add(identity)
        lineage.extend(records)
        validate_segment_union(lineage)
        datasets.append(LeRobotDataset("local/" + directory.name, root=directory))
    merge_datasets(datasets, "local/" + output.name, output_dir=output)
    (output / "control-contract.json").write_text(json.dumps(contract, indent=2))
    (output / "provenance.json").write_text(json.dumps(lineage, indent=2))
    (output / "split.json").write_text(json.dumps(dict(split=output_split)))
    return dict(episodes=len(lineage), output=str(output))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("episode", type=Path)
    report = sub.add_parser("report")
    report.add_argument("collection", type=Path)
    ex = sub.add_parser("export")
    ex.add_argument("--manifest", type=Path, required=True)
    ex.add_argument("--output", type=Path, required=True)
    ex.add_argument("--source", choices=["human", "automatic_expert"], required=True)
    ex.add_argument("--corrections-only", action="store_true")
    mx = sub.add_parser("merge")
    mx.add_argument("--inputs", nargs="+", type=Path, required=True)
    mx.add_argument("--output", type=Path, required=True)
    mx.add_argument("--split", choices=["train", "validation"], default="train")
    a = p.parse_args()
    if a.command == "score":
        result = score_episode(a.episode)
    elif a.command == "report":
        result = summarize(a.collection)
    elif a.command == "merge":
        result = merge(a.inputs, a.output, a.split)
    else:
        result = export(a.manifest, a.output, a.source, a.corrections_only)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
