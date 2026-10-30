"""Compare serial, matched learner-only and automatic-braking rollouts."""

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .data import read_rows, score_episode
except ImportError:
    from data import read_rows, score_episode


def qualify(baseline, assisted):
    baseline, assisted = Path(baseline), Path(assisted)
    bp = json.loads((baseline / "plan.json").read_text())
    ap = json.loads((assisted / "plan.json").read_text())
    if (
        bp["episodes"] != ap["episodes"]
        or bp["seconds"] != ap["seconds"]
        or bp["style"] != ap["style"]
        or bp.get("lighting", "daylight") != ap.get("lighting", "daylight")
    ):
        raise ValueError("Use matched seeds, starts, duration and scene")
    b = {x["seed"]: x for x in json.loads((baseline / "attempts.json").read_text())}
    a = {x["seed"]: x for x in json.loads((assisted / "attempts.json").read_text())}
    if set(b) != set(a) or set(b) != {x["seed"] for x in bp["episodes"]}:
        raise ValueError("Both planned collections must be complete")
    pairs = []
    expert_hashes = set()
    identities = set()
    mappings = set()
    for seed in sorted(a):
        if a[seed]["exit_code"] or b[seed]["exit_code"]:
            raise ValueError(
                "Infrastructure failures need resolution before qualification"
            )
        bd, ad = Path(b[seed]["path"]), Path(a[seed]["path"])
        b_scene = json.loads((bd / "report.json").read_text())["scene_profile"]
        a_scene = json.loads((ad / "report.json").read_text())["scene_profile"]
        if b_scene != a_scene or any(
            scene.get("lighting", "daylight") != plan.get("lighting", "daylight")
            for scene, plan in ((b_scene, bp), (a_scene, ap))
        ):
            raise ValueError(
                "Paired scenes or lighting differ from the collection plan"
            )
        b_schema = json.loads((bd / "learning/schema.json").read_text())
        a_schema = json.loads((ad / "learning/schema.json").read_text())
        if a_schema.get("configuration", {}).get("automatic_contact_completion"):
            raise ValueError(
                "Contact completion requires its own qualification protocol"
            )
        b_id = (b_schema.get("policy_identity") or {}).get("checkpoint_sha256")
        a_id = (a_schema.get("policy_identity") or {}).get("checkpoint_sha256")
        b_horizon = (b_schema.get("policy_identity") or {}).get(
            "native_horizon_contract"
        )
        a_horizon = (a_schema.get("policy_identity") or {}).get(
            "native_horizon_contract"
        )
        if (
            not b_id
            or b_id != a_id
            or b_schema["policy_identity"].get("inference_output_mapping")
            != a_schema["policy_identity"].get("inference_output_mapping")
            or b_schema["control_contract"] != a_schema["control_contract"]
            or not isinstance(b_horizon, dict)
            or not b_horizon
            or b_horizon != a_horizon
        ):
            raise ValueError(
                "Paired rollouts must use the same frozen model and actuator bridge"
            )
        mappings.add(b_schema["policy_identity"].get("inference_output_mapping"))
        bs, ass = score_episode(bd), score_episode(ad)
        identities.add(
            (
                b_id,
                json.dumps(b_schema["control_contract"], sort_keys=True),
                json.dumps(
                    b_schema["policy_identity"].get("native_horizon_contract"),
                    sort_keys=True,
                ),
            )
        )
        if bs["control_sources"] != ["learner"] or not set(ass["control_sources"]) <= {
            "learner",
            "automatic_expert",
        }:
            raise ValueError(
                "Qualification must exclude human control and baseline assistance"
            )
        rows = read_rows(ad / "learning/samples.jsonl")
        teacher = [
            r
            for r in rows
            if r["source"] == "automatic_expert" and r["correction"] is True
        ]
        if any(
            r.get("recovery_scope") not in (None, "released_ball_braking_only")
            for r in rows
        ):
            raise ValueError(
                "Paired braking qualification cannot accept another correction scope"
            )
        if any(r.get("action") != [0.0, 0.0, 0.0, 0.0] for r in teacher):
            raise ValueError(
                "Released-ball braking corrections must request zero action"
            )
        first = teacher[0]["timestamp_s"] if teacher else None
        before = read_rows(bd / "physical-evaluation.jsonl")
        after = read_rows(ad / "physical-evaluation.jsonl")
        errors = []
        if first is not None:
            if (
                not before
                or not after
                or min(before[-1]["timestamp_s"], after[-1]["timestamp_s"]) < first
            ):
                raise ValueError(
                    "Both physical traces must cover the intervention time"
                )
            before_prefix = [r for r in before if r["timestamp_s"] <= first]
            after_prefix = [r for r in after if r["timestamp_s"] <= first]
            if len(before_prefix) != len(after_prefix):
                raise ValueError("Pre-intervention physics frame counts differ")
            for br, ar in zip(before_prefix, after_prefix):
                if abs(br["timestamp_s"] - ar["timestamp_s"]) > 1e-6:
                    raise ValueError("Physics timestamps differ")
                errors.extend(
                    abs(x - y)
                    for key in ["robot_position", "ball_position"]
                    for x, y in zip(br[key], ar[key])
                )
        prefix_error = max(errors) if errors else None
        expert_hashes.add(
            hashlib.sha256(
                (ad / "source-snapshot/learning/recovery.py").read_bytes()
            ).hexdigest()
        )
        pairs.append(
            dict(
                seed=seed,
                intervention_frames=len(teacher),
                first_intervention_s=first,
                pre_intervention_max_position_error_m=prefix_error,
                matched_prefix=prefix_error is not None and prefix_error < 0.005,
                baseline_success=bs["policy_unassisted_success"],
                assisted_success=ass["physical_success"],
                rescued=(
                    prefix_error is not None
                    and prefix_error < 0.005
                    and ass["physical_success"]
                    and not bs["physical_success"]
                ),
                assisted_episode=str(ad.resolve()),
                samples_sha256=hashlib.sha256(
                    (ad / "learning/samples.jsonl").read_bytes()
                ).hexdigest(),
            )
        )
    if len(expert_hashes) != 1:
        raise ValueError("Expert implementation changed during collection")
    if len(identities) != 1 or len(mappings) != 1:
        raise ValueError(
            "All qualification episodes must use one frozen model and bridge"
        )
    # Short failed interventions still belong in the denominator. The 16-frame
    # minimum concerns later training export, never recovery success scoring.
    triggered = [r for r in pairs if r["intervention_frames"] > 0]
    successes = sum(r["assisted_success"] for r in triggered)
    passed = (
        len(triggered) >= 3
        and all(r["matched_prefix"] for r in triggered)
        and successes / len(triggered) >= 0.7
    )
    identity = identities.pop()
    mapping = mappings.pop()
    return dict(
        passed=passed,
        **({"inference_output_mapping": mapping} if mapping else {}),
        scope="released_ball_braking_only",
        expert_sha256=expert_hashes.pop(),
        policy_sha256=identity[0],
        native_horizon_contract=json.loads(identity[2]),
        baseline_collection=str(baseline.resolve()),
        assisted_collection=str(assisted.resolve()),
        triggered_pairs=len(triggered),
        triggered_matched_pairs=sum(r["matched_prefix"] for r in triggered),
        recovery_successes=successes,
        pairs=pairs,
        limitation="Qualification covers only the recorded near-ball, goal-directed release states; not arbitrary navigation or fall recovery.",
    )


def validate_qualification(path, episode, expected_sha=None):
    path, episode = Path(path), Path(episode)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha is not None and digest != expected_sha:
        raise ValueError("Recovery qualification changed after review")
    recorded = json.loads(path.read_text())
    if recorded.get("protocol") == "prospective_assisted_completion_v1":
        try:
            from .qualify_expert import qualify as qualify_expert
        except ImportError:
            from qualify_expert import qualify as qualify_expert
        recomputed = qualify_expert(recorded["collection"], recorded["protocol_plan"])
        if recorded != recomputed or recomputed["passed"] is not True:
            raise ValueError(
                "Expert qualification does not match complete raw evidence"
            )
        samples_sha = hashlib.sha256(
            (episode / "learning/samples.jsonl").read_bytes()
        ).hexdigest()
        if not any(
            r["assisted_episode"] == str(episode.resolve())
            and r["samples_sha256"] == samples_sha
            and r["assisted_success"] is True
            and r["intervention_frames"] > 0
            for r in recomputed["episodes"]
        ):
            raise ValueError("Episode is not a qualified successful intervention")
        return digest
    recomputed = qualify(
        recorded["baseline_collection"], recorded["assisted_collection"]
    )
    if recorded != recomputed or recomputed["passed"] is not True:
        raise ValueError("Recovery qualification does not match complete raw evidence")
    samples_sha = hashlib.sha256(
        (episode / "learning/samples.jsonl").read_bytes()
    ).hexdigest()
    matches = [
        r
        for r in recomputed["pairs"]
        if r["assisted_episode"] == str(episode.resolve())
        and r["samples_sha256"] == samples_sha
        and r["assisted_success"] is True
        and r["matched_prefix"] is True
    ]
    if not matches:
        raise ValueError("Episode is not a successful, matched qualified recovery")
    return digest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--assisted", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = qualify(a.baseline, a.assisted)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
