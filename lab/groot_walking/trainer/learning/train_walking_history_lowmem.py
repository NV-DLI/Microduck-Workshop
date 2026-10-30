"""Low-memory variant of train_walking_history.py (RTX 5090 32 GB, 24 GB host RAM).

Same model, data contract, trainable action head, frozen-feature cache checks,
loss/gradient equivalence probes, learning rate, warmup and update count as the
original trainer. Differences, all about memory:

* Each update still samples ``--batch-size`` rows (same sampler and seed) but
  runs them as micro-batches of ``--micro-batch-size`` with exact mask-weighted
  gradient accumulation (see ``lowmem_artifacts.micro_batch_weights``). The
  flow-matching noise and time draws are made per micro-batch, so the numbers
  are not bitwise identical to a single batch of 16.
* AdamW uses the single-tensor implementation (``foreach=False``), which has no
  model-sized temporaries.
* After the first update the complete training state (model, optimizer,
  scheduler, RNG, sampler) is saved, every GPU reference to the trained policy is
  released, the model is reloaded from disk and checked tensor by tensor and by
  its predicted actions, and training continues from the restored optimizer,
  scheduler and RNG state. There is never a second full model on the GPU.
* Checkpoints are streamed one tensor at a time; a reused feature cache is
  referenced by its verified manifest instead of being rewritten.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from walking_phase_sampling import load_sampling_weights
from verified_training_artifacts import load_verified_cache, compare_cache_row, verify_model_file
from lowmem_artifacts import (
    StreamingSafetensorsWriter, stream_save_verified_model, stream_save_training_state,
    compare_optimizer_with_file, comparable_param_groups, rng_snapshot, rng_compare,
    save_sampler_state, load_sampler_state, micro_batch_weights, memory_record, load_policy)

import argparse
import gc
import hashlib
import json
import math
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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

CHECK_ROWS = (0, 25, 63, 100)


def validate_new_module_gradients(records):
    for prefix in ("model.", "vlln.", "vl_self_attention."):
        matches = [r for r in records if r["parameter"].startswith(prefix)]
        if not matches or not all(
            math.isfinite(r[key]) and r[key] > 0
            for r in matches
            for key in ("direct_norm", "cached_norm")
        ):
            raise ValueError("Missing finite nonzero new-module gradient: " + prefix)


def trainable_names(policy):
    return [(n, list(v.shape)) for n, v in policy.named_parameters() if v.requires_grad]


def optimizer_layout(optimizer, policy):
    """Parameter name and moment shapes for every optimizer slot, in order."""
    names = {id(v): n for n, v in policy.named_parameters()}
    return [
        (names[id(p)], list(p.shape),
         {k: list(v.shape) for k, v in optimizer.state.get(p, {}).items() if k != "step"})
        for group in optimizer.param_groups for p in group["params"]
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--validation-dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--micro-batch-size", type=int, default=1,
                   help="Rows per forward/backward; must divide --batch-size")
    p.add_argument("--stop-after-updates", type=int,
                   help="Test probes only: stop after N updates of the --steps schedule "
                        "and save a partial checkpoint")
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
    if a.steps < 1 or a.batch_size < 1 or a.micro_batch_size < 1:
        raise ValueError("Positive budget required")
    if a.batch_size % a.micro_batch_size:
        raise ValueError("--micro-batch-size must divide --batch-size")
    updates = a.steps if a.stop_after_updates is None else a.stop_after_updates
    if not 1 <= updates <= a.steps:
        raise ValueError("--stop-after-updates must be between 1 and --steps")
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
    from safetensors.torch import load_file
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.common.train_utils import load_training_state

    memory_log = (a.output / "memory.jsonl").open("x")

    def note_memory(label, reset=True):
        record = memory_record(label)
        memory_log.write(json.dumps(record) + "\n")
        memory_log.flush()
        if reset and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return record

    torch.manual_seed(a.seed)
    base_binding = None
    if a.initialization == "official-base":
        from huggingface_hub import snapshot_download
        from lerobot.policies.groot.configuration_groot import GrootConfig

        policy_config = GrootConfig.from_pretrained(a.checkpoint)
        policy_config.base_model_path = snapshot_download(
            "nvidia/GR00T-N1.7-3B", revision=a.base_revision
        )
        policy = GrootPolicy(policy_config).to("cuda")
    else:
        policy, base_binding = load_policy(a.checkpoint, a.base_revision)
    if a.initialization == "checkpoint":
        verify_model_file(policy, a.checkpoint / "model.safetensors")
    note_memory("policy loaded and verified")
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
    trainable = trainable_names(policy)
    policy.train()
    dataset = LeRobotDataset(
        "local/" + a.dataset.name,
        root=a.dataset,
        delta_timestamps={"action": [i / 12.5 for i in range(16)]},
    )
    cache_writer = None
    cache_path = a.output / "frozen-backbone-cache.safetensors"
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
        # Memory-mapped: the 31.7 GB file stays in the page cache, not in RSS.
        reused_cache, cache_reuse, original_experiment = load_verified_cache(
            a.backbone_cache_manifest, a.checkpoint, a.dataset, len(dataset))
        note_memory("feature cache identities verified (memory-mapped)")
    cache_start = time.monotonic()
    for index in (CHECK_ROWS if reused_cache is not None else range(len(dataset))):
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
            if index in CHECK_ROWS:
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
            # Drop every name still bound to a GPU parameter of this policy.
            del direct_grad, cached_grad, selected, direct, cached, original, found
            note_memory("loss and gradient equivalence passed")
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
        if reused_cache is None:
            # Low-memory: rows go straight into a preallocated safetensors file
            # (same layout as save_file of the concatenated cache), not into RAM.
            fields = {"features." + k: v for k, v in features.items()}
            for key in ("state", "state_mask", "action", "action_mask", "embodiment_id"):
                if key in act:
                    fields["action_inputs." + key] = act[key]
            for key, value in fields.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(key)
            if cache_writer is None:
                cache_writer = StreamingSafetensorsWriter(cache_path, {
                    k: (v.dtype, (len(dataset),) + tuple(v.shape[1:])) for k, v in fields.items()})
            if set(fields) != set(cache_writer.specs):
                raise ValueError("Cache fields changed between rows")
            for key, value in fields.items():
                cache_writer.write_rows(key, index, value.detach().cpu())
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
    if reused_cache is not None:
        cache = reused_cache
        # The verified packaged cache is referenced, not copied (31.7 GB).
        (a.output / "frozen-backbone-cache-reference.json").write_text(json.dumps(dict(
            reused=True, cache=cache_reuse["cache"], cache_sha256=cache_reuse["cache_sha256"],
            manifest_sha256=cache_reuse["manifest_sha256"],
            checked_rows=list(CHECK_ROWS)), indent=2))
    else:
        cache_writer.close()
        cache = load_file(str(cache_path))
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
        base_model_binding=base_binding,
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
        low_memory_profile=dict(
            effective_batch_size=a.batch_size,
            micro_batch_size=a.micro_batch_size,
            gradient_accumulation_steps=a.batch_size // a.micro_batch_size,
            batch_sampling="same torch.multinomial draw of batch_size rows per update as the original trainer",
            loss_weighting="micro-batch loss x (valid action count of micro-batch + 1e-6) / (valid action count of batch + 1e-6)",
            optimizer_implementation="torch.optim.AdamW foreach=False (single-tensor)",
            serialization="streamed safetensors; checkpoint, free, reload and resume after update 1",
            not_bitwise_identical_to="the batch-16 trainer: flow-matching noise/time are drawn per micro-batch and AdamW uses another kernel path",
            stop_after_updates=a.stop_after_updates,
            pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        ),
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
    gc.collect()
    torch.cuda.empty_cache()
    note_memory("backbone moved to host; action head on GPU")
    warmup = manifest["warmup_steps"]

    def scale(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / max(1, a.steps - warmup))
        )

    def make_optimizer(current):
        built = torch.optim.AdamW(
            current.get_optim_params(),
            lr=current.config.optimizer_lr,
            betas=tuple(current.config.optimizer_betas),
            eps=current.config.optimizer_eps,
            weight_decay=current.config.optimizer_weight_decay,
            foreach=False,
        )
        return built, torch.optim.lr_scheduler.LambdaLR(built, scale)

    optimizer, scheduler = make_optimizer(policy)
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
    feature_keys = [k for k in cache if k.startswith("features.")]
    action_keys = [k for k in cache if k.startswith("action_inputs.")]

    def gather(rows):
        return (
            BatchFeature({k.removeprefix("features."): cache[k][rows].to("cuda") for k in feature_keys}),
            BatchFeature({k.removeprefix("action_inputs."): cache[k][rows].to("cuda") for k in action_keys}),
        )

    def continuity_loss(current_model):
        """Seeded training-mode loss on fixed rows; must survive save/reload."""
        torch.manual_seed(a.seed + 20000)
        features, actions = gather(torch.tensor(CHECK_ROWS))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            value = current_model.action_head(features, actions)["loss"].float().cpu()
        del features, actions
        return value

    start = time.monotonic()
    metrics = (a.output / "metrics.jsonl").open("x")
    for step in range(updates):
        indices = torch.multinomial(
            weights, a.batch_size, replacement=True, generator=rng
        )
        micro_weights = micro_batch_weights(
            cache["action_inputs.action_mask"][indices], a.micro_batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), device="cuda")
        for part, factor in zip(indices.split(a.micro_batch_size), micro_weights):
            features, actions = gather(part)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                micro = model.action_head(features, actions)["loss"]
            if not torch.isfinite(micro):
                raise ValueError("Non-finite training loss")
            (micro * factor).backward()
            loss += micro.detach().float() * factor
            del micro, features, actions
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise ValueError("Non-finite gradient norm")
        optimizer.step()
        scheduler.step()
        record = dict(
            step=step + 1,
            loss=float(loss),
            gradient_norm=float(norm),
            lr=scheduler.get_last_lr()[0],
            micro_batches=len(micro_weights),
            peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
        )
        del loss, norm
        if step == 0:
            note_memory("update 1 complete (training peak)")
            # Save the complete training state, release every GPU reference to
            # the trained policy, reload it from disk and continue from the
            # restored optimizer/scheduler/RNG. Not a task-success result.
            print("Checking full checkpoint save, reload and resume after first update", flush=True)
            optimizer.zero_grad(set_to_none=True)
            probe_root = a.output / "serialization-probe"
            probe_dir = probe_root / "pretrained_model"
            before_rng = rng_snapshot(rng)
            expected_groups = comparable_param_groups(optimizer)
            expected_layout = optimizer_layout(optimizer, policy)
            expected_scheduler = {k: v for k, v in scheduler.state_dict().items() if k != "lr_lambdas"}
            state_dir = stream_save_training_state(
                probe_root, 1, optimizer, scheduler, batch_size=a.batch_size)
            save_sampler_state(rng, state_dir)
            expected_loss = continuity_loss(model)
            del optimizer, scheduler, parameters
            gc.collect()
            torch.cuda.empty_cache()
            note_memory("training state saved; optimizer released")
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
            stream_save_verified_model(policy, probe_dir)
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
            del policy, model
            gc.collect()
            torch.cuda.empty_cache()
            released = note_memory("trained policy released before reload")
            if released["cuda_allocated_gib"] > 0.5:
                raise RuntimeError("GPU tensors of the trained policy are still referenced")
            restored, _ = load_policy(probe_dir, a.base_revision)
            restored.eval()
            restored_tensors = verify_model_file(restored, probe_dir / "model.safetensors")
            rp, rq = make_pre_post_processors(
                restored.config, pretrained_path=str(probe_dir)
            )
            if configure_prediction_horizon(restored, rp) != horizon_contract:
                raise ValueError("Restored prediction horizon differs")
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
            note_memory("restored policy verified")
            # Continue training with the restored policy.
            policy = restored
            restored_recipe = configure_training(
                policy, a.checkpoint, a.training_scope, a.head_dropout, a.learning_rate
            )
            if restored_recipe != recipe:
                raise ValueError("Restored training recipe differs")
            model = policy._groot_model
            if any(v.requires_grad for v in model.backbone.parameters()):
                raise ValueError("Backbone is not entirely frozen")
            if trainable_names(policy) != trainable:
                raise ValueError("Restored trainable parameters differ")
            model.backbone.to("cpu")
            policy.train()
            pre.reset()
            post.reset()
            gc.collect()
            torch.cuda.empty_cache()
            actual_loss = continuity_loss(model)
            if not torch.allclose(expected_loss, actual_loss, atol=1e-6, rtol=1e-6):
                raise ValueError("Restored training loss differs")
            optimizer, scheduler = make_optimizer(policy)
            resumed_step, optimizer, scheduler = load_training_state(probe_root, optimizer, scheduler)
            load_sampler_state(rng, state_dir)
            if resumed_step != 1:
                raise ValueError("Restored training step differs")
            optimizer_tensors = compare_optimizer_with_file(optimizer, state_dir)
            if comparable_param_groups(optimizer) != expected_groups:
                raise ValueError("Restored optimizer parameter groups differ")
            if optimizer_layout(optimizer, policy) != expected_layout:
                raise ValueError("Restored optimizer state is attached to different parameters")
            restored_scheduler = {k: v for k, v in scheduler.state_dict().items() if k != "lr_lambdas"}
            if restored_scheduler != expected_scheduler:
                raise ValueError("Restored scheduler state differs")
            rng_report = rng_compare(before_rng, rng_snapshot(rng))
            if not rng_report["restored_streams_equal"]:
                raise ValueError("Restored RNG state differs: %s" % rng_report)
            parameters = [v for v in policy.parameters() if v.requires_grad]
            (a.output / "serialization-check.json").write_text(
                json.dumps(
                    dict(
                        after_updates=1,
                        max_action_difference=float((expected - actual).abs().max()),
                        actions_bitwise_equal=bool(torch.equal(expected, actual)),
                        shape=list(actual.shape),
                        restored_model_tensors_bitwise_equal=restored_tensors,
                        training_loss_before=float(expected_loss),
                        training_loss_after=float(actual_loss),
                        training_loss_bitwise_equal=bool(torch.equal(expected_loss, actual_loss)),
                        optimizer_tensors_bitwise_equal=optimizer_tensors,
                        optimizer_param_groups_equal=True,
                        scheduler_state_equal=True,
                        rng=rng_report,
                        resumed_from=str(probe_root),
                        gpu_policies_resident_at_once=1,
                        passed=True,
                    ),
                    indent=2,
                )
            )
            del restored, rp, rq, probe_row, probe_batch, expected, actual
            gc.collect()
            torch.cuda.empty_cache()
            note_memory("training resumed from restored state")
        record["elapsed_s"] = time.monotonic() - start
        metrics.write(json.dumps(record) + "\n")
        metrics.flush()
        if step < 3:
            note_memory("after update %d" % (step + 1))
        if (step + 1) % 10 == 0 or step < 3:
            print(json.dumps(record), flush=True)
    optimizer.zero_grad(set_to_none=True)
    del cache, reused_cache
    gc.collect()
    torch.cuda.empty_cache()
    dest = a.output / "checkpoints" / f"{updates:06d}" / "pretrained_model"
    print("Saving complete GR00T checkpoint", flush=True)
    stream_save_verified_model(policy, dest)
    pre.save_pretrained(dest)
    post.save_pretrained(dest)
    (dest / "training-recipe.json").write_text(json.dumps(recipe, indent=2))
    state_dir = stream_save_training_state(
        dest.parent, updates, optimizer, scheduler, batch_size=a.batch_size)
    save_sampler_state(rng, state_dir)
    shutil.copy2(a.checkpoint / "control-contract.json", dest / "control-contract.json")
    (dest / "training-provenance.json").write_text(json.dumps(lineage, indent=2))
    (dest / "cached-training-experiment.json").write_text(
        json.dumps(manifest, indent=2)
    )
    note_memory("final checkpoint saved", reset=False)
    (a.output / "completion.json").write_text(
        json.dumps(
            dict(
                checkpoint=str(dest),
                updates=updates,
                scheduled_updates=a.steps,
                training_elapsed_s=time.monotonic() - start,
                complete=updates == a.steps,
            ),
            indent=2,
        )
    )
    print("Complete: independent full-model validation still required"
          if updates == a.steps else "Partial test run saved; not a trained course model", flush=True)


if __name__ == "__main__":
    main()
