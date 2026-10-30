"""Match native GR00T denoising length to the serialized training processor.

Cropping returned actions does not change the DiT's self-attention context.
The base N1.7 model predicts 40 actions; this lab trains on 16-action chunks.
"""


def configure_prediction_horizon(policy, preprocessor, mode="training"):
    heads = [step for step in preprocessor.steps if hasattr(step, "action_horizon")]
    if len(heads) != 1:
        raise ValueError("Expected one serialized GR00T action packing step")
    training_horizon = int(heads[0].action_horizon)
    requested = int(policy.config.chunk_size)
    if training_horizon != requested:
        raise ValueError("Policy chunk size differs from serialized training horizon")
    model = policy._groot_model
    head = model.action_head
    original = int(head.config.action_horizon)
    if original != int(head.action_horizon):
        raise ValueError("Inconsistent native GR00T horizon fields")
    if not 1 <= int(policy.config.n_action_steps) <= training_horizon <= original:
        raise ValueError("Execution/training horizon exceeds the native model capacity")
    if mode not in ("training", "checkpoint"):
        raise ValueError("Unknown prediction horizon mode")
    if mode == "training":
        model.config.action_horizon = training_horizon
        head.config.action_horizon = training_horizon
        head.action_horizon = training_horizon
    return dict(
        mode=mode,
        original_native_horizon=original,
        serialized_training_horizon=training_horizon,
        native_prediction_horizon=int(head.action_horizon),
        execution_horizon=int(policy.config.n_action_steps),
    )
