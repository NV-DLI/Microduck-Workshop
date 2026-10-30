"""Inspect and accept a bounded intervention segment before fine-tuning."""

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .data import read_rows, segments
    from .qualify_recovery import validate_qualification
except ImportError:
    from data import read_rows, segments
    from qualify_recovery import validate_qualification


def qualified_manifest(qualification, max_frames=None):
    """Select physically qualified automatic segments, with no human attribution."""
    if max_frames is not None and max_frames < 16:
        raise ValueError("Retain at least one complete16-frame action chunk")
    qualification = Path(qualification)
    evidence = json.loads(qualification.read_text())
    if evidence.get("passed") is not True:
        raise ValueError("Automatic recovery expert has not passed qualification")
    prospective = evidence.get("protocol") == "prospective_assisted_completion_v1"
    manifest = []
    for pair in evidence["episodes"] if prospective else evidence["pairs"]:
        if not (
            pair["assisted_success"]
            and (prospective or pair["matched_prefix"])
            and pair["intervention_frames"]
        ):
            continue
        episode = Path(pair["assisted_episode"])
        qualification_sha = validate_qualification(qualification, episode)
        _, groups = segments(episode, "automatic_expert", corrections_only=True)
        accepted = []
        for rows in groups:
            if len(rows) < 16:
                continue
            accepted.append(
                dict(
                    segment=rows[0]["segment"],
                    start_frame=rows[0]["frame"],
                    end_frame=rows[min(len(rows), max_frames or len(rows)) - 1][
                        "frame"
                    ],
                    reviewer="automatic_physics_qualification",
                    reason=(
                        "Full-task assisted completion after real learner execution under the prospectively qualified release-braking expert; not a counterfactual rescue claim."
                        if prospective
                        else "Matched learner prefix and full-task physical completion under the qualified release-braking expert."
                    ),
                    source_samples_sha256=pair["samples_sha256"],
                    source="automatic_expert",
                    qualification_sha256=qualification_sha,
                    qualification_path=str(qualification.resolve()),
                )
            )
        if accepted:
            manifest.append(
                dict(
                    path=str(episode.resolve()),
                    seed=pair["seed"],
                    split="correction",
                    accepted_segments=accepted,
                )
            )
    if not manifest:
        raise ValueError(
            "No qualified automatic correction segments of at least 16 frames"
        )
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    origin = p.add_mutually_exclusive_group(required=True)
    origin.add_argument("--episode", type=Path)
    origin.add_argument("--qualified-recoveries", type=Path)
    p.add_argument("--segment", type=int)
    p.add_argument("--start-frame", type=int)
    p.add_argument("--end-frame", type=int)
    p.add_argument("--reviewer")
    p.add_argument("--reason")
    p.add_argument("--output", type=Path)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--source", choices=["human", "automatic_expert"], default="human")
    p.add_argument("--qualification", type=Path)
    p.add_argument(
        "--max-frames",
        type=int,
        help="Cap each qualified automatic segment; at least16",
    )
    a = p.parse_args()
    if a.max_frames is not None and not a.qualified_recoveries:
        p.error("--max-frames applies only to --qualified-recoveries")
    if a.qualified_recoveries:
        if a.output is None:
            p.error("Qualified corrections require a new --output manifest")
        manifest = qualified_manifest(a.qualified_recoveries, a.max_frames)
        with a.output.open("x") as f:
            json.dump(manifest, f, indent=2)
        print(
            json.dumps(
                dict(
                    episodes=len(manifest),
                    source="automatic_expert",
                    manifest=str(a.output),
                ),
                indent=2,
            )
        )
        return
    samples = a.episode / "learning/samples.jsonl"
    rows = read_rows(samples)
    groups = {}
    for r in rows:
        if r["correction"]:
            groups.setdefault(r["segment"], []).append(r)
    print(
        json.dumps(
            [
                dict(
                    segment=k,
                    start_frame=v[0]["frame"],
                    end_frame=v[-1]["frame"],
                    source=v[0]["source"],
                    duration_s=len(v) * 0.08,
                )
                for k, v in groups.items()
            ],
            indent=2,
        )
    )
    if a.output is None:
        return
    if a.interactive:
        if not groups:
            p.error("No intervention segments were recorded")
        try:
            a.segment = int(input("Segment to review: "))
            a.start_frame = int(input("First accepted frame: "))
            a.end_frame = int(input("Last accepted frame: "))
        except ValueError:
            p.error("Use frame and segment numbers from the recorded episode")
        a.reviewer = a.reviewer or input("Reviewer name: ").strip()
        a.reason = (
            a.reason or input("Why are these corrective actions suitable? ").strip()
        )
    if not a.reviewer or not a.reason or a.segment not in groups:
        p.error("Accept a listed segment and provide reviewer/reason")
    selected = [
        r
        for r in groups[a.segment]
        if a.start_frame is not None
        and a.end_frame is not None
        and a.start_frame <= r["frame"] <= a.end_frame
    ]
    if (
        len(selected) < 16
        or selected[0]["frame"] != a.start_frame
        or selected[-1]["frame"] != a.end_frame
    ):
        p.error("Accepted range needs at least 16 contiguous recorded frames")
    if {r["source"] for r in selected} != {a.source}:
        p.error("Selected segment does not match the declared correction source")
    qualification_sha = None
    if a.source == "automatic_expert":
        if a.qualification is None:
            p.error("Automatic corrections require a successful recovery qualification")
        qualification_sha = validate_qualification(a.qualification, a.episode)
    report = json.loads((a.episode / "report.json").read_text())
    manifest = [
        dict(
            path=str(a.episode.resolve()),
            seed=report["seed"],
            split="correction",
            accepted_segments=[
                dict(
                    segment=a.segment,
                    start_frame=a.start_frame,
                    end_frame=a.end_frame,
                    reviewer=a.reviewer,
                    reason=a.reason,
                    source_samples_sha256=hashlib.sha256(
                        samples.read_bytes()
                    ).hexdigest(),
                    source=a.source,
                    qualification_sha256=qualification_sha,
                    qualification_path=(
                        str(a.qualification.resolve()) if a.qualification else None
                    ),
                )
            ],
        )
    ]
    with a.output.open("x") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
