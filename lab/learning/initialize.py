"""Fit GR00T input/output processors using only the training split.

This writes configuration and processors, not trained model weights. Use the
result with train_cached.py --initialization official-base.
"""

import argparse
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-camera-frame", action="store_true",
                        help="Walking: retain the complete camera field of view during training and inference")
    parser.add_argument("--action-execution-steps", type=int,
                        help="Walking: save the number of actions to execute before replanning (1–16)")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if json.loads((args.dataset / "split.json").read_text())["split"] != "train":
        raise ValueError("Fit input/output statistics on the training split only")
    contract = json.loads((args.dataset / "control-contract.json").read_text())
    if args.full_camera_frame and contract.get("locomotion") != "walking":
        raise ValueError("Full-camera initialization is supported only for walking")
    if args.action_execution_steps is not None and (
            contract.get("locomotion") != "walking" or not 1 <= args.action_execution_steps <= 16):
        raise ValueError("Walking execution steps must be between 1 and 16")

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.configs.types import FeatureType
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.feature_utils import dataset_to_policy_features
    from lerobot.utils.constants import IMAGENET_STATS
    import torch

    meta = LeRobotDatasetMetadata("local/" + args.dataset.name, root=args.dataset)
    # Match LeRobot's default dataset factory. The VLM applies its own image
    # transform; preserve the same serialized RGB statistics as normal training.
    for key in meta.camera_keys:
        if key not in meta.depth_keys:
            for name, values in IMAGENET_STATS.items():
                meta.stats[key][name] = torch.tensor(values, dtype=torch.float32)
    features = dataset_to_policy_features(meta.features)
    output_features = {
        k: v for k, v in features.items() if v.type is FeatureType.ACTION
    }
    config = GrootConfig(
        base_model_path="nvidia/GR00T-N1.7-3B",
        input_features={k: v for k, v in features.items() if k not in output_features},
        output_features=output_features,
        device="cpu",
        chunk_size=16,
        n_action_steps=4 if args.action_execution_steps is None else args.action_execution_steps,
        embodiment_tag="new_embodiment",
        use_relative_actions=False,
        use_bf16=True,
        use_flash_attention=False,
        tune_visual=False,
        tune_llm=False,
        tune_diffusion_model=False,
        tune_projector=True,
        tune_vlln=False,
        model_params_fp32=True,
    )
    set_metadata = getattr(config, "set_dataset_feature_metadata", None)
    if callable(set_metadata):
        set_metadata(meta.features)
    if hasattr(config, "action_feature_names"):
        config.action_feature_names = list(meta.features["action"]["names"])
    config._runtime_dataset_meta = meta
    pre, post = make_pre_post_processors(
        config, dataset_stats=meta.stats, dataset_meta=meta
    )
    if args.full_camera_frame:
        views = [step for step in pre.steps
                 if type(step).__name__ == "GrootN17VLMEncodeStep"]
        if len(views) != 1:
            raise ValueError("Expected one GR00T N1.7 image processor")
        # Near-foot objects can occupy the image border. Keep every camera
        # pixel before the existing resize; serialize this for deployment too.
        views[0].crop_fraction = 1.0
        views[0].image_crop_size = None
    args.output.mkdir(parents=True)
    config.save_pretrained(args.output)
    pre.save_pretrained(args.output)
    post.save_pretrained(args.output)
    shutil.copy2(
        args.dataset / "control-contract.json", args.output / "control-contract.json"
    )
    shutil.copy2(
        args.dataset / "provenance.json", args.output / "training-provenance.json"
    )
    (args.output / "initialization.json").write_text(
        json.dumps(
            dict(
                gradient_updates=0,
                contains_model_weights=False,
                camera_view="full_frame" if args.full_camera_frame else "base_preprocessing",
                action_execution_steps=config.n_action_steps,
                statistics_source=str(args.dataset.resolve()),
                next_step="train_cached.py --initialization official-base",
            ),
            indent=2,
        )
    )
    print("Training processors ready:", args.output)


if __name__ == "__main__":
    main()
