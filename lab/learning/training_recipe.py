"""Record and reapply the trainable scope and regularization of a GR00T run."""

import json
from pathlib import Path


def checkpoint_dropout(checkpoint):
    checkpoint = Path(checkpoint)
    recipe = checkpoint / "training-recipe.json"
    if recipe.exists():
        return json.loads(recipe.read_text())["head_dropout"]
    experiment = checkpoint / "cached-training-experiment.json"
    if experiment.exists():
        changes = json.loads(experiment.read_text()).get("regularization_changes", [])
        if changes and all(item["new"] == 0 for item in changes):
            return "disabled"
    return "base"


def configure_training(policy, checkpoint, scope, dropout, learning_rate=None):
    import torch

    head = policy._groot_model.action_head
    if scope != "checkpoint":
        if scope not in ("projectors", "action-head"):
            raise ValueError("Unknown trainable scope")
        policy.config.tune_projector = True
        policy.config.tune_diffusion_model = scope == "action-head"
        policy.config.tune_vlln = scope == "action-head"
        head.set_trainable_parameters(
            policy.config.tune_projector,
            policy.config.tune_diffusion_model,
            policy.config.tune_vlln,
        )
    if dropout == "checkpoint":
        dropout = checkpoint_dropout(checkpoint)
    if dropout not in ("base", "disabled"):
        raise ValueError("Unknown head regularization setting")
    changes = []
    if dropout == "disabled":
        changes.append(
            dict(
                module="action_head",
                field="state_dropout_prob",
                old=head.state_dropout_prob,
                new=0.0,
            )
        )
        head.state_dropout_prob = 0.0
        for name, module in head.named_modules():
            if isinstance(module, torch.nn.Dropout):
                changes.append(dict(module=name, field="p", old=module.p, new=0.0))
                module.p = 0.0
            if hasattr(module, "dropout") and isinstance(module.dropout, (float, int)):
                changes.append(
                    dict(module=name, field="dropout", old=module.dropout, new=0.0)
                )
                module.dropout = 0.0
    if learning_rate is not None:
        if not 0 < learning_rate < 1:
            raise ValueError("Learning rate must be positive and below one")
        policy.config.optimizer_lr = learning_rate
    return dict(
        tune_projector=policy.config.tune_projector,
        tune_diffusion_model=policy.config.tune_diffusion_model,
        tune_vlln=policy.config.tune_vlln,
        head_dropout=dropout,
        regularization_changes=changes,
        learning_rate=policy.config.optimizer_lr,
        trainable_parameters=sum(
            p.numel() for p in head.parameters() if p.requires_grad
        ),
        optimizer="fresh AdamW; not optimizer-state resumption",
    )
