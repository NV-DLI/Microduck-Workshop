"""Fine-tune the real GR00T action head with an exactly frozen backbone cache.

Only suitable for a frozen VLM and deterministic image preprocessing. The cache
contains the same VLM features as a normal forward; labels never enter the VLM.
Direct/cached loss, gradient and serialization checks run before the main training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from walking_phase_sampling import load_sampling_weights
from verified_training_artifacts import load_verified_cache, compare_cache_row, verify_model_file, save_verified_model

import argparse
import hashlib
import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import shutil
import time

try:
    from .evaluate import assert_disjoint_records
    from .horizon import configure_prediction_horizon
    from .training_recipe import configure_training
except ImportError:
    from evaluate import assert_disjoint_records
    from horizon import configure_prediction_horizon
    from training_recipe import configure_training


def validate_new_module_gradients(records):
    for prefix in ("model.", "vlln.", "vl_self_attention."):
        matches = [r for r in records if r["parameter"].startswith(prefix)]
        if not matches or not all(
            math.isfinite(r[key]) and r[key] > 0
            for r in matches
            for key in ("direct_norm", "cached_norm")
        ):
            raise ValueError("Missing finite nonzero new-module gradient: " + prefix)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--validation-dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--balanced-action-groups", action="store_true")
    p.add_argument("--sampling-manifest", type=Path)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--backbone-cache-manifest", type=Path)
    p.add_argument(
        "--initialization",
        choices=["checkpoint", "official-base"],
        default="checkpoint",
    )
    p.add_argument(
        "--training-scope",
        choices=["checkpoint", "projectors", "action-head"],
        default="checkpoint",
    )
    p.add_argument(
        "--head-dropout",
        choices=["checkpoint", "base", "disabled"],
        default="checkpoint",
    )
    p.add_argument("--learning-rate", type=float)
    p.add_argument(
        "--base-revision", default="2fc962b973bccdd5d8ce4f67cc63b264d6886495"
    )
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.steps < 1 or a.batch_size < 1:
        raise ValueError("Positive budget required")
    if (
        a.initialization == "checkpoint"
        and not (a.checkpoint / "model.safetensors").exists()
    ):
        raise ValueError(
            "Processors alone contain no model weights; use --initialization official-base"
        )
    if json.loads((a.dataset / "split.json").read_text())["split"] != "train":
        raise ValueError("Training split required")
    lineage = json.loads(
        (a.checkpoint / "training-provenance.json").read_text()
    ) + json.loads((a.dataset / "provenance.json").read_text())
    assert_disjoint_records(
        lineage, json.loads((a.validation_dataset / "provenance.json").read_text())
    )
    contract = json.loads((a.checkpoint / "control-contract.json").read_text())
    if contract != json.loads((a.dataset / "control-contract.json").read_text()):
        raise ValueError("Contract mismatch")
    if a.sampling_manifest and a.balanced_action_groups:
        raise ValueError('Choose explicit phase sampling or equal action groups')
    phase_weights, phase_sampling = load_sampling_weights(a.sampling_manifest, a.dataset) if a.sampling_manifest else (None, None)
    a.output.mkdir()
    import torch

    torch.use_deterministic_algorithms(True)
    from transformers.feature_extraction_utils import BatchFeature
    from safetensors.torch import save_file
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.common.train_utils import save_training_state

    torch.manual_seed(a.seed)
    if a.initialization == "official-base":
        from huggingface_hub import snapshot_download
        from lerobot.policies.groot.configuration_groot import GrootConfig

        policy_config = GrootConfig.from_pretrained(a.checkpoint)
        policy_config.base_model_path = snapshot_download(
            "nvidia/GR00T-N1.7-3B", revision=a.base_revision
        )
        policy = GrootPolicy(policy_config).to("cuda")
    else:
        policy = GrootPolicy.from_pretrained(a.checkpoint).to("cuda")
    if a.initialization == "checkpoint":
        verify_model_file(policy, a.checkpoint / "model.safetensors")
    recipe = configure_training(
        policy, a.checkpoint, a.training_scope, a.head_dropout, a.learning_rate
    )
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(a.checkpoint)
    )
    horizon_contract = configure_prediction_horizon(policy, pre)
    model = policy._groot_model
    if any(v.requires_grad for v in model.backbone.parameters()):
        raise ValueError("Backbone is not entirely frozen")
    policy.train()
    dataset = LeRobotDataset(
        "local/" + a.dataset.name,
        root=a.dataset,
        delta_timestamps={"action": [i / 12.5 for i in range(16)]},
    )
    cache = {}
    class_ids = []
    equivalence = []
    gradient_equivalence = []
    processor_modes = [
        dict(
            name=type(step).__name__,
            training=getattr(step, "training", None),
            use_albumentations=getattr(step, "use_albumentations", None),
            state_dropout_prob=getattr(step, "state_dropout_prob", None),
        )
        for step in pre.steps
    ]
    if any(step["training"] is True for step in processor_modes):
        raise ValueError("Cache requires deterministic evaluation preprocessing")
    reused_cache = None
    cache_reuse = None
    original_experiment = None
    if a.backbone_cache_manifest:
        if a.initialization != 'checkpoint' or a.balanced_action_groups:
            raise ValueError('Cache reuse requires checkpoint initialization and explicit phase or uniform sampling')
        reused_cache, cache_reuse, original_experiment = load_verified_cache(
            a.backbone_cache_manifest, a.checkpoint, a.dataset, len(dataset))
    cache_start = time.monotonic()
    for index in ((0, 25, 63, 100) if reused_cache is not None else range(len(dataset))):
        row = dataset[index]
        # Padded targets retain the official action mask.
        batch = {
            k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
            for k, v in row.items()
        }
        filtered = policy._filter_groot_inputs(pre(batch), include_action=True)
        bb, act = model.prepare_input(filtered)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            features = model.backbone(bb)
        if reused_cache is not None:
            compare_cache_row(reused_cache, index, features, act)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if index in (0, 25, 63, 100):
                torch.manual_seed(a.seed + index)
                direct = model(filtered)["loss"].float()
                torch.manual_seed(a.seed + index)
                cached = model.action_head(BatchFeature(dict(features)), act)[
                    "loss"
                ].float()
                error = float((direct - cached).abs())
                if not torch.allclose(direct, cached, atol=1e-5, rtol=1e-5):
                    raise ValueError(f"Cache equivalence failed: {error}")
                equivalence.append(
                    dict(
                        index=index,
                        direct=float(direct),
                        cached=float(cached),
                        absolute_difference=error,
                    )
                )
        if index == 0:
            selected = []
            for prefix in (
                "state_encoder.",
                "action_encoder.",
                "action_decoder.",
                "position_embedding.",
                "model.",
                "vlln.",
                "vl_self_attention.",
            ):
                found = next(
                    (
                        (name, value)
                        for name, value in model.action_head.named_parameters()
                        if name.startswith(prefix) and value.requires_grad
                    ),
                    None,
                )
                if found is None:
                    continue
                selected.append(found)
            if not selected:
                raise ValueError("No trainable parameters for gradient checks")
            torch.manual_seed(a.seed + 9000)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                direct_loss = model(filtered)["loss"]
            direct_grad = [
                v.detach().cpu()
                for v in torch.autograd.grad(direct_loss, [v for _, v in selected])
            ]
            del direct_loss
            torch.manual_seed(a.seed + 9000)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cached_loss = model.action_head(BatchFeature(dict(features)), act)[
                    "loss"
                ]
            cached_grad = [
                v.detach().cpu()
                for v in torch.autograd.grad(cached_loss, [v for _, v in selected])
            ]
            del cached_loss
            for (name, original), direct, cached in zip(
                selected, direct_grad, cached_grad
            ):
                diff = float((direct - cached).abs().max())
                if not torch.allclose(direct, cached, atol=1e-5, rtol=1e-4):
                    raise ValueError("Gradient mismatch: " + name)
                updated = []
                for gradient in (direct, cached):
                    parameter = torch.nn.Parameter(original.detach().cpu().clone())
                    probe_optimizer = torch.optim.AdamW(
                        [parameter],
                        lr=policy.config.optimizer_lr,
                        betas=tuple(policy.config.optimizer_betas),
                        eps=policy.config.optimizer_eps,
                        weight_decay=0.0,
                    )
                    parameter.grad = gradient.clone()
                    probe_optimizer.step()
                    updated.append(parameter.detach().clone())
                    del probe_optimizer, parameter
                update_error = float((updated[0] - updated[1]).abs().max())
                if not torch.allclose(updated[0], updated[1], atol=1e-6, rtol=1e-5):
                    (a.output / "equivalence-failure.json").write_text(
                        json.dumps(
                            dict(
                                parameter=name,
                                gradient_max_difference=diff,
                                direct_gradient_norm=float(direct.norm()),
                                cached_gradient_norm=float(cached.norm()),
                                update_max_difference=update_error,
                            ),
                            indent=2,
                        )
                    )
                    raise ValueError("One-update mismatch: " + name)
                del updated
                gradient_equivalence.append(
                    dict(
                        parameter=name,
                        one_adam_update_max_difference=update_error,
                        max_absolute_difference=diff,
                        direct_norm=float(direct.norm()),
                        cached_norm=float(cached.norm()),
                    )
                )
            if not any(r["direct_norm"] > 0 for r in gradient_equivalence):
                raise ValueError("No learning gradient")
            # Fail before caching/training if a newly unfrozen module lacks a
            # finite, nonzero representative direct AND cached gradient.
            if a.training_scope == "action-head":
                validate_new_module_gradients(gradient_equivalence)
            del direct_grad, cached_grad, selected, direct, cached
            if a.probe_only:
                (a.output / "equivalence-probe.json").write_text(
                    json.dumps(
                        dict(
                            loss=equivalence,
                            gradients=gradient_equivalence,
                            processor_modes=processor_modes,
                            training_recipe=recipe,
                            initialization=a.initialization,
                            native_horizon_contract=horizon_contract,
                        ),
                        indent=2,
                    )
                )
                print("Equivalence probe passed", flush=True)
                return
        for key, value in features.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(key)
            cache.setdefault("features." + key, []).append(value.detach().cpu())
        for key in ("state", "state_mask", "action", "action_mask", "embodiment_id"):
            if key in act:
                cache.setdefault("action_inputs." + key, []).append(
                    act[key].detach().cpu()
                )
        raw = row["action"]
        if contract.get("locomotion") == "walking":
            try:
                from .walking import request_group
            except ImportError:
                from walking import request_group
            class_ids.append(['stand', 'observe', 'approach_and_align', 'kick_and_recover'].index(request_group(raw.tolist())))
        else:
            class_ids.append(
                2 if raw[:, 0].max() > 0.05 else 1 if raw[:, 3].max() > 0.1 else 0
            )
        if (index + 1) % 100 == 0:
            print(
                json.dumps(
                    dict(
                        stage="cache",
                        frames=index + 1,
                        total=len(dataset),
                        elapsed_s=time.monotonic() - cache_start,
                    )
                ),
                flush=True,
            )
    cache = reused_cache if reused_cache is not None else {k: torch.cat(v, dim=0).contiguous() for k, v in cache.items()}
    save_file(cache, str(a.output / "frozen-backbone-cache.safetensors"))
    manifest = dict(
        checkpoint=str(a.checkpoint.resolve()),
        dataset=str(a.dataset.resolve()),
        steps=a.steps,
        batch_size=a.batch_size,
        seed=a.seed,
        balanced_action_groups=a.balanced_action_groups,
        phase_sampling=phase_sampling,
        cache_elapsed_s=time.monotonic() - cache_start,
        cache_reuse=cache_reuse,
        equivalence=equivalence,
        gradient_equivalence=gradient_equivalence,
        processor_modes=processor_modes,
        cache_tensors={
            k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in cache.items()
        },
        optimizer="AdamW fresh state, official parameter groups",
        training_recipe=recipe,
        initialization=a.initialization,
        base_revision=a.base_revision if a.initialization == "official-base" else None,
        source_checkpoint_model_weights_used=a.initialization == "checkpoint",
        deterministic_algorithms=True,
        native_horizon_contract=horizon_contract,
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        warmup_steps=max(1, round(a.steps * 0.05)),
        learning_rate=policy.config.optimizer_lr,
        image_augmentation=False,
        backbone_frozen=True,
        method="Exact frozen-backbone feature reuse; trainable modules and regularization recorded in training_recipe",
        label_group_counts=(original_experiment["label_group_counts"] if reused_cache is not None else {str(g): class_ids.count(g) for g in set(class_ids)}),
    )
    for f in (
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        path = a.checkpoint / f
        if path.exists():
            with path.open("rb") as stream:
                manifest["source_checkpoint/" + f + "_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    (a.output / "experiment.json").write_text(json.dumps(manifest, indent=2))
    # Free only this process's frozen backbone GPU tensors; keep full weights for
    # normal complete-model checkpoint saving and later end-to-end evaluation.
    model.backbone.to("cpu")
    del bb, act, features, filtered, batch, row
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(
        policy.get_optim_params(),
        lr=policy.config.optimizer_lr,
        betas=tuple(policy.config.optimizer_betas),
        eps=policy.config.optimizer_eps,
        weight_decay=policy.config.optimizer_weight_decay,
    )
    warmup = manifest["warmup_steps"]

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, a.steps - warmup))
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
    groups = torch.tensor(class_ids)
    weights = torch.ones(len(dataset), dtype=torch.float64)
    if a.balanced_action_groups:
        for group in groups.unique():
            weights[groups == group] = 1 / int((groups == group).sum())
    if phase_weights is not None:
        if len(phase_weights) != len(dataset): raise ValueError('Phase weights differ from actual loaded dataset')
        weights = torch.tensor(phase_weights, dtype=torch.float64)
    rng = torch.Generator().manual_seed(a.seed)
    torch.manual_seed(a.seed)
    parameters = [v for v in policy.parameters() if v.requires_grad]
    start = time.monotonic()
    metrics = (a.output / "metrics.jsonl").open("x")
    for step in range(a.steps):
        indices = torch.multinomial(
            weights, a.batch_size, replacement=True, generator=rng
        )
        features = BatchFeature(
            {
                k.removeprefix("features."): v[indices].to("cuda")
                for k, v in cache.items()
                if k.startswith("features.")
            }
        )
        actions = BatchFeature(
            {
                k.removeprefix("action_inputs."): v[indices].to("cuda")
                for k, v in cache.items()
                if k.startswith("action_inputs.")
            }
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model.action_head(features, actions)["loss"]
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise ValueError("Non-finite gradient norm")
        optimizer.step()
        scheduler.step()
        if step == 0:
            # Before the longer run, verify a complete model+processor artifact
            # after one genuine update. This is a serialization probe, not a
            # claimed task-success result or an optimizer resume checkpoint.
            print("Checking full-model save/reload after first update", flush=True)
            state_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            model.backbone.to("cuda")
            policy.eval()
            probe_row = dataset[0]
            probe_batch = {
                k: (v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v])
                for k, v in probe_row.items()
                if k.startswith("observation.") or k == "task"
            }
            torch.manual_seed(a.seed + 10000)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                expected = (
                    post(policy.predict_action_chunk(pre(probe_batch)))
                    .detach()
                    .float()
                    .cpu()
                )
            probe_dir = a.output / "serialization-probe" / "pretrained_model"
            save_verified_model(policy, probe_dir)
            pre.save_pretrained(probe_dir)
            post.save_pretrained(probe_dir)
            (probe_dir / "training-recipe.json").write_text(
                json.dumps(recipe, indent=2)
            )
            shutil.copy2(
                a.checkpoint / "control-contract.json",
                probe_dir / "control-contract.json",
            )
            (probe_dir / "training-provenance.json").write_text(
                json.dumps(lineage, indent=2)
            )
            restored = GrootPolicy.from_pretrained(probe_dir).to("cuda").eval()
            rp, rq = make_pre_post_processors(
                restored.config, pretrained_path=str(probe_dir)
            )
            configure_prediction_horizon(restored, rp)
            torch.manual_seed(a.seed + 10000)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                actual = (
                    rq(restored.predict_action_chunk(rp(probe_batch)))
                    .detach()
                    .float()
                    .cpu()
                )
            if not torch.allclose(expected, actual, atol=1e-5, rtol=1e-5):
                raise ValueError("Serialized full-model output differs")
            (a.output / "serialization-check.json").write_text(
                json.dumps(
                    dict(
                        after_updates=1,
                        max_action_difference=float((expected - actual).abs().max()),
                        shape=list(actual.shape),
                        passed=True,
                    ),
                    indent=2,
                )
            )
            del restored, rp, rq, probe_row, probe_batch, expected, actual
            model.backbone.to("cpu")
            policy.train()
            pre.reset()
            post.reset()
            torch.set_rng_state(state_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
            import gc

            gc.collect()
            torch.cuda.empty_cache()
        record = dict(
            step=step + 1,
            loss=float(loss.detach()),
            gradient_norm=float(norm),
            lr=scheduler.get_last_lr()[0],
            elapsed_s=time.monotonic() - start,
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        )
        metrics.write(json.dumps(record) + "\n")
        metrics.flush()
        if (step + 1) % 10 == 0:
            print(json.dumps(record), flush=True)
    del cache, features, actions
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    dest = a.output / "checkpoints" / f"{a.steps:06d}" / "pretrained_model"
    print("Saving complete GR00T checkpoint", flush=True)
    save_verified_model(policy, dest)
    pre.save_pretrained(dest)
    post.save_pretrained(dest)
    (dest / "training-recipe.json").write_text(json.dumps(recipe, indent=2))
    save_training_state(
        dest.parent,
        a.steps,
        optimizer,
        scheduler,
        batch_size=a.batch_size,
        num_processes=1,
    )
    shutil.copy2(a.checkpoint / "control-contract.json", dest / "control-contract.json")
    (dest / "training-provenance.json").write_text(json.dumps(lineage, indent=2))
    (dest / "cached-training-experiment.json").write_text(
        json.dumps(manifest, indent=2)
    )
    (a.output / "completion.json").write_text(
        json.dumps(
            dict(
                checkpoint=str(dest),
                updates=a.steps,
                training_elapsed_s=time.monotonic() - start,
                complete=True,
            ),
            indent=2,
        )
    )
    print("Complete: independent full-model validation still required", flush=True)


if __name__ == "__main__":
    main()
