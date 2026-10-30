"""Qualify assisted completion on a prospectively fixed collection of learner runs.

This measures conditional expert competence, not counterfactual rescue of a
particular unassisted failure. Same-seed reruns need not share a physical prefix.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

try:
    from .data import read_rows, score_episode
except ImportError:
    from data import read_rows, score_episode

PROTOCOL = "prospective_assisted_completion_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_intervention(episode, rows):
    try:
        from .recovery import ReleaseBrakeExpert
    except ImportError:
        from recovery import ReleaseBrakeExpert
    truth = read_rows(episode / "learning/expert-truth.jsonl")
    physics = read_rows(episode / "physical-evaluation.jsonl")
    if len(truth) != len(rows):
        raise ValueError("Every control sample needs an aligned expert observation")
    expert = ReleaseBrakeExpert()
    fields = (
        "robot_position",
        "ball_position",
        "ball_velocity",
        "forward_velocity",
        "gravity_z",
        "angular_speed",
        "timestamp_s",
    )
    for index, (row, observed) in enumerate(zip(rows, truth, strict=True)):
        if not math.isclose(row["timestamp_s"], observed["timestamp_s"], abs_tol=1e-6):
            raise ValueError("Expert/control timestamps differ")
        if index:
            actual = physics[index * 16 - 1]
            if any(
                abs(x - y) > 1e-6
                for key in ("robot_position", "ball_position")
                for x, y in zip(actual[key], observed[key], strict=True)
            ):
                raise ValueError("Expert positions differ from the physical trajectory")
        was_active = expert.active
        proposal = row.get("learner_proposal")
        if was_active and proposal is not None:
            raise ValueError("Latched expert frames cannot retain a learner proposal")
        if (
            not isinstance(row["action"], list)
            or len(row["action"]) != 4
            or not all(math.isfinite(float(v)) for v in row["action"])
        ):
            raise ValueError("Four finite executed actions required")
        if not was_active and (not isinstance(proposal, list) or len(proposal) != 4):
            raise ValueError("Unassisted frames need the actual learner proposal")
        active = expert.update(
            proposal=proposal if not was_active else [0.0] * 4,
            **{k: observed[k] for k in fields},
        )
        if (
            observed["recovery_active"] is not active
            or observed["trigger"] != expert.trigger
        ):
            raise ValueError("Recorded expert trigger differs from gate replay")
        expected_source = "automatic_expert" if active else "learner"
        if row["source"] != expected_source or row["correction"] is not active:
            raise ValueError("Control attribution differs from replayed intervention")
        expected_action = [0.0] * 4 if active else proposal
        if len(row["action"]) != 4 or any(
            abs(x - y) > 1e-8
            for x, y in zip(row["action"], expected_action, strict=True)
        ):
            raise ValueError(
                "Expert must execute zero requests through the local brake"
            )


def make_plan(
    controller,
    seed,
    episodes,
    reference_episode,
    *,
    seconds=24.0,
    root_x=-0.30,
    root_y=0.045,
    position_jitter=0.002,
):
    cfg = json.loads(Path(controller).read_text())
    if cfg.get("mode") != "policy" or cfg.get("automatic_release_brake") is not True:
        raise ValueError("Use the learner with the automatic release-braking expert")
    if episodes < 3:
        raise ValueError("Plan at least three episodes before collecting")
    if (
        not all(math.isfinite(v) for v in (seconds, root_x, root_y, position_jitter))
        or seconds <= 0
        or position_jitter < 0
    ):
        raise ValueError(
            "Finite poses, positive duration and non-negative jitter required"
        )
    reference_episode = Path(reference_episode)
    schema = json.loads((reference_episode / "learning/schema.json").read_text())
    report = json.loads((reference_episode / "report.json").read_text())
    identity = schema["policy_identity"]
    fixtures = []
    for trial_seed in range(seed, seed + episodes):
        rng = random.Random(trial_seed)
        fixtures.append(
            dict(
                seed=trial_seed,
                root_x=root_x + rng.uniform(-position_jitter, position_jitter),
                root_y=root_y + rng.uniform(-position_jitter, position_jitter),
            )
        )
    return dict(
        protocol=PROTOCOL,
        created_at_unix=time.time(),
        seeds=list(range(seed, seed + episodes)),
        controller=cfg,
        fixtures=fixtures,
        seconds=seconds,
        style=report["scene_profile"]["mat_style"],
        lighting=report["scene_profile"].get("lighting", "daylight"),
        position_jitter_m=position_jitter,
        scene_profile=report["scene_profile"],
        learner_identity=dict(
            model=identity["checkpoint_sha256"],
            **(
                {"inference_output_mapping": identity["inference_output_mapping"]}
                if identity.get("inference_output_mapping")
                else {}
            ),
            horizon=identity["native_horizon_contract"],
            bridge=schema["control_contract"],
        ),
        expert_sha256=digest(Path(__file__).with_name("recovery.py")),
        minimum_triggered_episodes=3,
        minimum_assisted_success_fraction=0.7,
        stopping_rule="Complete every planned episode, regardless of observed outcomes.",
    )


def qualify(collection, protocol_plan):
    collection, protocol_plan = Path(collection), Path(protocol_plan)
    protocol = json.loads(protocol_plan.read_text())
    plan = json.loads((collection / "plan.json").read_text())
    protocol_sha = digest(protocol_plan)
    if (
        plan.get("expert_protocol_sha256") != protocol_sha
        or digest(collection / "expert-protocol.json") != protocol_sha
    ):
        raise ValueError("Collection is not bound to the registered expert protocol")
    if protocol["protocol"] != PROTOCOL:
        raise ValueError("Unknown recovery protocol")
    if digest(Path(__file__).with_name("recovery.py")) != protocol["expert_sha256"]:
        raise ValueError("Use the registered expert implementation for replay")
    if protocol["created_at_unix"] > (collection / "plan.json").stat().st_mtime:
        raise ValueError("Register the protocol before collecting the episodes")
    if (
        protocol["minimum_triggered_episodes"] != 3
        or protocol["minimum_assisted_success_fraction"] != 0.7
    ):
        raise ValueError("Qualification thresholds must remain fixed")
    seeds = protocol["seeds"]
    if len(set(seeds)) != len(seeds) or len(seeds) < 3:
        raise ValueError("Invalid prospective seed list")
    if [r["seed"] for r in plan["episodes"]] != seeds or plan["controller"] != protocol[
        "controller"
    ]:
        raise ValueError("Collection differs from the prospective plan")
    if (
        any(
            plan[key] != protocol[key]
            for key in ("seconds", "style", "position_jitter_m")
        )
        or plan.get("lighting", "daylight") != protocol.get("lighting", "daylight")
        or plan["episodes"] != protocol["fixtures"]
    ):
        raise ValueError(
            "Scene, duration or starting fixtures changed after registration"
        )
    attempts = json.loads((collection / "attempts.json").read_text())
    if len(attempts) != len(seeds) or {r["seed"] for r in attempts} != set(seeds):
        raise ValueError("Complete every planned episode before qualification")
    identities = set()
    seen_paths, seen_samples = set(), set()
    episodes = []
    for attempt in sorted(attempts, key=lambda x: x["seed"]):
        if attempt["exit_code"]:
            raise ValueError(
                "Resolve and retain infrastructure failures before qualification"
            )
        ep = Path(attempt["path"])
        registration = json.loads(
            (ep / "expert-protocol-registration.json").read_text()
        )
        launch_record = (
            collection / "launch-records" / f"episode-{attempt['seed']}.json"
        )
        if json.loads(launch_record.read_text()) != registration:
            raise ValueError("Episode registration differs from the pre-launch record")
        if (
            attempt.get("expert_protocol_sha256") != protocol_sha
            or registration
            != dict(
                expert_protocol_sha256=protocol_sha,
                started_at_unix=attempt.get("started_at_unix"),
                seed=attempt["seed"],
            )
            or attempt.get("started_at_unix", 0) < protocol["created_at_unix"]
        ):
            raise ValueError("Episode was not launched under the registered protocol")
        samples_sha = digest(ep / "learning/samples.jsonl")
        if ep.resolve() in seen_paths or samples_sha in seen_samples:
            raise ValueError(
                "Repeated episode path or sample trace cannot count as independent trials"
            )
        seen_paths.add(ep.resolve())
        seen_samples.add(samples_sha)
        if (ep / "learning/schema.json").stat().st_mtime < attempt["started_at_unix"]:
            raise ValueError("Episode was recorded before its registered launch")
        schema = json.loads((ep / "learning/schema.json").read_text())
        if schema["configuration"] != protocol["controller"]:
            raise ValueError("Controller changed during collection")
        identity = schema.get("policy_identity") or {}
        if not identity.get("checkpoint_sha256") or not identity.get(
            "native_horizon_contract"
        ):
            raise ValueError("Frozen learner identity and inference horizon required")
        identities.add(
            json.dumps(
                dict(
                    model=identity["checkpoint_sha256"],
                    **(
                        {
                            "inference_output_mapping": identity[
                                "inference_output_mapping"
                            ]
                        }
                        if identity.get("inference_output_mapping")
                        else {}
                    ),
                    horizon=identity["native_horizon_contract"],
                    bridge=schema["control_contract"],
                ),
                sort_keys=True,
            )
        )
        if (
            digest(ep / "source-snapshot/learning/recovery.py")
            != protocol["expert_sha256"]
        ):
            raise ValueError("Expert changed after protocol registration")
        score = score_episode(ep)
        if score["seed"] != attempt["seed"]:
            raise ValueError("Actual episode seed differs from its planned attempt")
        report = json.loads((ep / "report.json").read_text())
        fixture = next(r for r in protocol["fixtures"] if r["seed"] == score["seed"])
        if report["scene_profile"] != protocol["scene_profile"] or report["fixture"][
            "root_xy"
        ] != [fixture["root_x"], fixture["root_y"]]:
            raise ValueError("Actual scene or reset position differs from registration")
        if (
            not math.isclose(score["duration_s"], protocol["seconds"], abs_tol=1e-6)
            and not score["fallen"]
        ):
            raise ValueError("Incomplete episode without a scored fall")
        if not set(score["control_sources"]) <= {"learner", "automatic_expert"}:
            raise ValueError("Human control is outside this automatic protocol")
        rows = read_rows(ep / "learning/samples.jsonl")
        verify_intervention(ep, rows)
        teacher = [r for r in rows if r["source"] == "automatic_expert"]
        first = teacher[0]["frame"] if teacher else None
        if teacher:
            if first == 0 or any(
                r["source"] != "learner" or r["correction"] for r in rows[:first]
            ):
                raise ValueError("Intervention must follow genuine learner execution")
            if any(
                r["source"] != "automatic_expert" or r["correction"] is not True
                for r in rows[first:]
            ):
                raise ValueError("Release-braking intervention must remain latched")
        episodes.append(
            dict(
                seed=attempt["seed"],
                intervention_frames=len(teacher),
                first_intervention_s=teacher[0]["timestamp_s"] if teacher else None,
                assisted_success=score["physical_success"],
                assisted_episode=str(ep.resolve()),
                samples_sha256=samples_sha,
            )
        )
    if len(identities) != 1:
        raise ValueError("Use one frozen model, inference horizon and execution bridge")
    actual_identity = json.loads(next(iter(identities)))
    if actual_identity != protocol["learner_identity"]:
        raise ValueError("Learner or execution bridge differs from registration")
    triggered = [r for r in episodes if r["intervention_frames"]]
    completed = sum(r["assisted_success"] for r in triggered)
    return dict(
        protocol=PROTOCOL,
        protocol_plan=str(protocol_plan.resolve()),
        protocol_plan_sha256=digest(protocol_plan),
        collection=str(collection.resolve()),
        expert_sha256=protocol["expert_sha256"],
        learner_identity=json.loads(identities.pop()),
        planned_episodes=len(seeds),
        triggered_episodes=len(triggered),
        assisted_completions=completed,
        passed=len(triggered) >= 3 and completed / len(triggered) >= 0.7,
        episodes=episodes,
        scope="Full-task completion conditional on this fixed expert triggering in learner-visited release states.",
        limitation="No counterfactual rescue claim, no unassisted policy success claim, and no arbitrary navigation/fall recovery qualification.",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--controller", type=Path, required=True)
    plan.add_argument("--seed", type=int, required=True)
    plan.add_argument("--episodes", type=int, default=16)
    plan.add_argument("--reference-episode", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--seconds", type=float, default=24.0)
    plan.add_argument("--root-x", type=float, default=-0.30)
    plan.add_argument("--root-y", type=float, default=0.045)
    plan.add_argument("--position-jitter", type=float, default=0.002)
    run = sub.add_parser("qualify")
    run.add_argument("--collection", type=Path, required=True)
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = (
        make_plan(
            a.controller,
            a.seed,
            a.episodes,
            a.reference_episode,
            seconds=a.seconds,
            root_x=a.root_x,
            root_y=a.root_y,
            position_jitter=a.position_jitter,
        )
        if a.command == "plan"
        else qualify(a.collection, a.plan)
    )
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
