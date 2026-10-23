#!/usr/bin/env python3
"""Train new Beta ball/goal weights, validate held-out scenes, export ONNX.

Uses a separate detector environment. Never touches the Isaac Lab lockfile.
The final test split is evaluated only after best.pt is chosen on validation.
"""
import argparse, hashlib, json, os, time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("ULTRALYTICS_HUB", "false")


def deployment_scope():
    return {
        "rknn_architecture_support": "YOLO26 detection and RK3566 are listed in official Ultralytics documentation",
        "source": "https://docs.ultralytics.com/integrations/rockchip-rknn",
        "checked_date": "2026-10-19",
        "rknn_export_host_requirement": "x86 Linux; not this ARM64 Spark",
        "this_model_rknn_exported": False,
        "this_model_rk3566_tested": False,
        "rk3566_latency_ms": None,
        "rk3566_camera_to_action_latency_ms": None,
        "rk3566_thermal_endurance_tested": False,
        "scope": "Architecture support does not establish accuracy, speed or sustained operation of this course model on the robot",
    }


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate_dataset(folder):
    m = json.loads((folder / "manifest.json").read_text())
    if (
        m["schema"] != "microduck.beta-detector-dataset.v1"
        or m["runtime_id"] != "newton-bam-20261019"
        or m["hardware_profile_id"] != "beta-gc2093-l8cx-v1"
    ):
        raise ValueError("Only new Beta RTX datasets qualify")
    if sha(folder / "dataset.yaml") != m["dataset_yaml_sha256"]:
        raise ValueError("Dataset path/split YAML changed")
    scenes = {}
    seen = set()
    for r in m["images"]:
        if scenes.setdefault(r["scene_seed"], r["split"]) != r["split"]:
            raise ValueError("Scene leakage across splits")
        if r["image_sha256"] in seen:
            raise ValueError("Duplicate images")
        seen.add(r["image_sha256"])
        for kind in ("image", "label"):
            path = (folder / r[kind]).resolve()
            if (
                not path.is_relative_to(folder.resolve())
                or sha(path) != r[kind + "_sha256"]
            ):
                raise ValueError("Dataset file/hash mismatch")
    return m


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=973001)
    p.add_argument("--device", default="0")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Explicit optional general COCO model; default is random yolo26n.yaml",
    )
    a = p.parse_args()
    a.output = a.output.resolve()
    a.dataset = a.dataset.resolve()
    m = validate_dataset(a.dataset)
    if m["development_only"] and not a.smoke:
        raise ValueError("Incomplete development dataset cannot run full training")
    if a.output.exists():
        raise FileExistsError("Preserve every training run")
    a.output.mkdir(parents=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(a.output / "ultralytics-settings"))
    import torch, ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != "8.4.182":
        raise RuntimeError(
            "Use pinned independent detector environment: ultralytics8.4.182"
        )
    started = time.monotonic()
    initialization = (
        "random" if a.pretrained is None else "general-pretrained-fine-tune"
    )
    model = YOLO("yolo26n.yaml" if a.pretrained is None else str(a.pretrained))
    initial_sha = None if a.pretrained is None else sha(a.pretrained)
    initial_state = a.output / "initial-state.pt"
    ledger = {"optimizer_steps": 0, "epochs": []}
    epoch_start = [time.monotonic()]

    def attach_counter(trainer):
        # Trainer may rebuild the architecture after seeding; capture the exact
        # initialized model that will receive the first optimizer update.
        torch.save(trainer.model.state_dict(), initial_state)
        provenance["initial_state_sha256"] = sha(initial_state)
        (a.output / "training-report.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        original = trainer.optimizer_step

        def counted_step(*args, **kwargs):
            value = original(*args, **kwargs)
            ledger["optimizer_steps"] += 1
            return value

        trainer.optimizer_step = counted_step

    def epoch_end(trainer):
        now = time.monotonic()
        ledger["epochs"].append(
            {
                "epoch": int(trainer.epoch) + 1,
                "optimizer_steps_cumulative": ledger["optimizer_steps"],
                "elapsed_seconds": now - epoch_start[0],
            }
        )
        epoch_start[0] = now
        (a.output / "training-ledger.json").write_text(
            json.dumps(ledger, indent=2) + "\n"
        )

    model.add_callback("on_train_start", attach_counter)
    model.add_callback("on_train_epoch_end", epoch_end)
    provenance = {
        "schema": "microduck.beta-detector-training.v1",
        "runtime_id": "newton-bam-20261019",
        "hardware_profile_id": "beta-gc2093-l8cx-v1",
        "initialization": initialization,
        "initial_model_sha256": initial_sha,
        "initial_state_sha256": None,
        "dataset_manifest_sha256": sha(a.dataset / "manifest.json"),
        "seed": a.seed,
        "smoke": a.smoke,
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "device": a.device,
        "epochs_requested": 2 if a.smoke else a.epochs,
        "scene_split": m["split_unit"],
        "robot_task_success_claimed": False,
        "deployment_scope": deployment_scope(),
        "status": "training",
    }
    (a.output / "training-report.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    try:
        model.train(
            data=str((a.dataset / "dataset.yaml").resolve()),
            epochs=provenance["epochs_requested"],
            imgsz=640,
            batch=a.batch,
            seed=a.seed,
            device=a.device,
            workers=2,
            project=str(a.output),
            name="run",
            exist_ok=False,
            pretrained=a.pretrained is not None,
            patience=20,
            deterministic=True,
            amp=True,
            plots=True,
            save=True,
            cache=False,
            close_mosaic=10 if not a.smoke else 0,
        )
        best = Path(model.trainer.best)
        elapsed = time.monotonic() - started
        detector = YOLO(str(best))
        test = detector.val(
            data=str((a.dataset / "dataset.yaml").resolve()),
            split="test",
            imgsz=640,
            batch=a.batch,
            device=a.device,
            project=str(a.output),
            name="heldout-test",
            exist_ok=False,
            plots=True,
        )
        exported = Path(
            detector.export(
                format="onnx",
                imgsz=640,
                batch=1,
                dynamic=False,
                simplify=False,
                opset=17,
                device="cpu",
            )
        )
        provenance.update(
            status="completed",
            epochs_completed=int(model.trainer.epoch) + 1,
            training_wall_seconds=elapsed,
            total_wall_seconds=time.monotonic() - started,
            optimizer_updates=ledger["optimizer_steps"],
            best_checkpoint=str(best.resolve()),
            best_checkpoint_sha256=sha(best),
            onnx=str(exported.resolve()),
            onnx_sha256=sha(exported),
            heldout_test_metrics={k: float(v) for k, v in test.results_dict.items()},
            class_names=detector.names,
            validation_selection="Ultralytics best.pt chosen on val only; test evaluated after training",
            camera_input="RGB960x540, original-pixel boxes returned after internal letterbox640",
            qualification_claimed=False,
        )
        deploy = {
            "schema": "microduck.beta-detector-deployment.v1",
            "runtime_id": provenance["runtime_id"],
            "hardware_profile_id": provenance["hardware_profile_id"],
            "weights": str(best.resolve()),
            "weights_sha256": sha(best),
            "onnx": str(exported.resolve()),
            "onnx_sha256": sha(exported),
            "class_names": {"0": "ball", "1": "goal"},
            "input_raster": [960, 540],
            "network_image_size": 640,
            "training_dataset_manifest_sha256": provenance["dataset_manifest_sha256"],
            "smoke_only": a.smoke,
            "heldout_metrics": provenance["heldout_test_metrics"],
            "controller_input": "original-pixel detections only; no teacher files/labels/world poses",
            "physically_validated": False,
        }
        deploy["deployment_scope"] = deployment_scope()
        (a.output / "deployment.json").write_text(json.dumps(deploy, indent=2) + "\n")
    except BaseException as error:
        provenance.update(
            status="failed",
            error=repr(error),
            total_wall_seconds=time.monotonic() - started,
        )
        raise
    finally:
        (a.output / "training-report.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
