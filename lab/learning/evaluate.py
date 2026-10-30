"""Held-out action prediction; physical success requires a separate rollout."""

import argparse
import json
from pathlib import Path
import time

try:
    from .horizon import configure_prediction_horizon
except ImportError:
    from horizon import configure_prediction_horizon


def request_decision_metrics(records, threshold):
    """Measure errors at the actual propulsion/local-stop boundary."""
    pairs = [
        (target[0] > threshold, prediction[0] > threshold)
        for row in records
        for target, prediction in zip(row["target"], row["prediction"], strict=True)
    ]
    stop_targets = sum(not target for target, _ in pairs)
    propulsion_targets = sum(target for target, _ in pairs)
    unwanted = sum(not target and prediction for target, prediction in pairs)
    missed = sum(target and not prediction for target, prediction in pairs)
    return dict(
        propulsion_threshold_native=threshold,
        target_stop_steps=stop_targets,
        unwanted_propulsion_steps=unwanted,
        unwanted_propulsion_fraction=unwanted / stop_targets if stop_targets else None,
        target_propulsion_steps=propulsion_targets,
        missed_propulsion_steps=missed,
        missed_propulsion_fraction=(
            missed / propulsion_targets if propulsion_targets else None
        ),
        note="Offline action decisions on held-out observations; not physical task success.",
    )


def assert_disjoint(training, evaluation):
    train = json.loads((Path(training) / "provenance.json").read_text())
    test = json.loads((Path(evaluation) / "provenance.json").read_text())
    return assert_disjoint_records(train, test)


def assert_disjoint_records(train, test):
    for key in ("seed", "samples_sha256"):
        overlap = {r[key] for r in train} & {r[key] for r in test}
        if overlap:
            raise ValueError(f"Training/evaluation leakage in {key}: {overlap}")
    return dict(
        training_episodes=len({r["samples_sha256"] for r in train}),
        evaluation_episodes=len({r["samples_sha256"] for r in test}),
        training_lineage_records=len(train),
        evaluation_segments=len(test),
        disjoint=True,
    )


def main():
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.factory import make_pre_post_processors

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument(
        "--all-anchors",
        action="store_true",
        help="Evaluate every valid observation anchor, including action transitions",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--prediction-horizon", choices=["training", "checkpoint"], default="training"
    )
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    lineage = json.loads((a.checkpoint / "training-provenance.json").read_text())
    heldout = json.loads((a.dataset / "provenance.json").read_text())
    audit = assert_disjoint_records(lineage, heldout)
    contract = json.loads((a.checkpoint / "control-contract.json").read_text())
    if contract != json.loads((a.dataset / "control-contract.json").read_text()):
        raise ValueError("Evaluation actuator bridge differs")
    policy = GrootPolicy.from_pretrained(a.checkpoint).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(a.checkpoint)
    )
    horizon_contract = configure_prediction_horizon(policy, pre, a.prediction_horizon)
    dataset = LeRobotDataset(
        "local/" + a.dataset.name,
        root=a.dataset,
        delta_timestamps={"action": [i / 12.5 for i in range(16)]},
    )
    errors = []
    zero_errors = []
    grouped_errors = {}
    records = []
    if a.samples < 1:
        raise ValueError("Positive sample count required")
    indices = (
        np.arange(len(dataset))
        if a.all_anchors
        else np.unique(
            np.linspace(0, len(dataset) - 1, min(a.samples, len(dataset)), dtype=int)
        )
    )
    evaluation_start = time.monotonic()
    print(
        f"Evaluating {len(indices)} observation anchors; padded targets are skipped.",
        flush=True,
    )
    for index in indices:
        row = dataset[int(index)]
        if row.get("action_is_pad", torch.zeros(16, dtype=torch.bool)).any():
            continue
        batch = {
            k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
            for k, v in row.items()
            if k.startswith("observation.") or k == "task"
        }
        policy.reset()
        pre.reset()
        post.reset()
        # Keep each observation's random start independent of preceding anchors
        # and of the native prediction length in a paired inference comparison.
        torch.manual_seed(a.seed + int(index))
        start = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            chunk = post(policy.predict_action_chunk(pre(batch))).float().cpu()
        # LeRobot truncates predictions to n_action_steps (4 in this lab),
        # although the training target contains a 16-step horizon.
        horizon = min(policy.config.chunk_size, policy.config.n_action_steps)
        target = row["action"][:horizon].float().cpu()
        prediction = chunk[0]
        if prediction.shape != target.shape or not torch.isfinite(prediction).all():
            raise ValueError(
                f"Invalid action chunk: raw={tuple(chunk.shape)}, "
                f"prediction={tuple(prediction.shape)}, target={tuple(target.shape)}, "
                f"finite={bool(torch.isfinite(prediction).all())}"
            )
        error = (prediction - target).abs().mean(0).numpy()
        zero_error = target.abs().mean(0).numpy()
        group = (
            "propulsion"
            if target[:, 0].max() > 0.05
            else "gaze" if target[:, 3].max() > 0.1 else "hold_or_brake"
        )
        if contract.get("locomotion") == "walking":
            try:
                from .walking import request_group
            except ImportError:
                from walking import request_group
            group = request_group(target.tolist())
        errors.append(error)
        zero_errors.append(zero_error)
        grouped_errors.setdefault(group, []).append((error, zero_error))
        records.append(
            dict(
                index=int(index),
                inference_seed=a.seed + int(index),
                request_group=group,
                mae=error.tolist(),
                zero_request_mae=zero_error.tolist(),
                target=target.tolist(),
                prediction=prediction.tolist(),
                latency_s=time.monotonic() - start,
            )
        )
        if len(records) % 100 == 0:
            print(
                f"Evaluated {len(records)} valid anchors "
                f"in {time.monotonic() - evaluation_start:.1f}s",
                flush=True,
            )
    if not errors:
        raise ValueError("No complete held-out chunks")
    result = dict(
        **audit,
        metric="absolute high-level action error; not task success",
        evaluation_coverage=dict(
            selection="all_anchors" if a.all_anchors else "uniform_subsample",
            dataset_frames=len(dataset),
            attempted_anchors=len(indices),
            valid_anchors=len(records),
            unique_target_frames=len(
                {r["index"] + offset for r in records for offset in range(horizon)}
            ),
            note="Anchors with padded 16-step targets are excluded. Overlapping action windows are not independent trials.",
        ),
        prediction_horizon_steps=horizon,
        native_horizon_contract=horizon_contract,
        seed=a.seed,
        rng_mode="per_anchor_seed_plus_index",
        action_names=contract["action_names"],
        mean_absolute_error=np.mean(errors, axis=0).tolist(),
        zero_request_mean_absolute_error=np.mean(zero_errors, axis=0).tolist(),
        by_target_request={
            group: dict(
                samples=len(values),
                mean_absolute_error=np.mean([v[0] for v in values], axis=0).tolist(),
                zero_request_mean_absolute_error=np.mean(
                    [v[1] for v in values], axis=0
                ).tolist(),
            )
            for group, values in grouped_errors.items()
        },
        propulsion_decisions=(
            request_decision_metrics(records, contract["stop_threshold"])
            if contract.get("stop_threshold") is not None
            else None
        ),
        samples=records,
        checkpoint=str(a.checkpoint.resolve()),
        dataset=str(a.dataset.resolve()),
    )
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
