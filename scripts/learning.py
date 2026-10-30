#!/usr/bin/env python3
"""Run the Microduck data, GR00T and intervention labs in isolated environments."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import venv

from run import ROOT, config, environment, fresh


def setup(a):
    output = fresh(a.output)
    venv.EnvBuilder(with_pip=True).create(output)
    python = output / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "torch==2.11.0",
            "torchvision==0.26.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu130",
        ],
        check=True,
    )
    packages = (
        [
            "lerobot[training,groot] @ git+https://github.com/huggingface/lerobot.git@7e241bd630a3719a56157a497ce5d08f244784f1",
            "transformers==5.5.4",
        ]
        if a.variant == "groot"
        else [
            "diffusers @ git+https://github.com/huggingface/diffusers.git@aaf57d03a3edfbd9de457d2872417a92f149a91d",
            "transformers==5.5.4",
            "accelerate",
            "av",
            "cosmos_guardrail",
            "huggingface_hub",
            "imageio",
            "imageio-ffmpeg",
            "opencv-python-headless",
        ]
    )
    subprocess.run([str(python), "-m", "pip", "install", *packages], check=True)
    frozen = subprocess.check_output([str(python), "-m", "pip", "freeze"], text=True)
    (output / "installed-packages.txt").write_text(frozen)
    print("Environment ready:", python)


def collect(a):
    c = config()
    output = fresh(a.output)
    output.mkdir()
    controller = json.loads(a.controller.read_text())
    if a.diagnostic:
        if controller["mode"] != "expert":
            raise ValueError("Camera-free diagnostics require an expert controller")
        controller["diagnostic_no_rgb"] = True
    if a.headless and controller["mode"] == "human":
        raise ValueError("Human data requires the visible simulator")
    if a.wait_for_start and a.headless:
        raise ValueError("Waiting for Start requires the visible simulator")
    plan = []
    for i in range(a.episodes):
        seed = a.seed + i
        rng = random.Random(seed)
        plan.append(
            dict(
                seed=seed,
                root_x=a.root_x + rng.uniform(-a.position_jitter, a.position_jitter),
                root_y=a.root_y + rng.uniform(-a.position_jitter, a.position_jitter),
            )
        )
    collection_plan = dict(
        controller=controller,
        split=a.split,
        episodes=plan,
        seconds=a.seconds,
        style=a.style,
        lighting=a.lighting,
        position_jitter_m=a.position_jitter,
    )
    protocol_sha = None
    if a.expert_protocol:
        raw_protocol = a.expert_protocol.read_bytes()
        protocol = json.loads(raw_protocol)
        if (
            protocol["controller"] != controller
            or protocol.get("lighting", "daylight") != a.lighting
            or protocol["fixtures"] != plan
            or any(
                protocol[k] != collection_plan[k]
                for k in ("seconds", "style", "position_jitter_m")
            )
        ):
            raise ValueError("Collection must match the registered expert protocol")
        protocol_sha = hashlib.sha256(raw_protocol).hexdigest()
        collection_plan["expert_protocol_sha256"] = protocol_sha
        (output / "expert-protocol.json").write_bytes(raw_protocol)
    (output / "plan.json").write_text(json.dumps(collection_plan, indent=2))
    ctrl = output / "controller.json"
    ctrl.write_text(json.dumps(controller, indent=2))
    sys.path.insert(0, str(ROOT / "lab"))
    from learning.data import score_episode

    attempts = []
    for item in plan:
        episode = output / f"episode-{item['seed']}"
        cmd = [
            c["python"],
            str(ROOT / "lab/manual.py"),
            "--runtime",
            c["materialized"],
            "--models",
            str(ROOT / "models"),
            "--output",
            str(episode),
            "--seconds",
            str(a.seconds),
            "--seed",
            str(item["seed"]),
            "--locomotion",
            "rollers",
            "--style",
            a.style,
            "--lighting",
            a.lighting,
            "--root-x",
            str(item["root_x"]),
            "--root-y",
            str(item["root_y"]),
            "--learning-config",
            str(ctrl),
        ]
        if a.headless:
            cmd.append("--headless-check")
        if a.wait_for_start:
            cmd.append("--wait-for-start")
        started_at_unix = time.time()
        launch_record = None
        if protocol_sha:
            launches = output / "launch-records"
            launches.mkdir(exist_ok=True)
            launch_record = launches / f"episode-{item['seed']}.json"
            with launch_record.open("x") as registration:
                json.dump(
                    dict(
                        expert_protocol_sha256=protocol_sha,
                        started_at_unix=started_at_unix,
                        seed=item["seed"],
                    ),
                    registration,
                    indent=2,
                )
        with (output / f"episode-{item['seed']}.log").open("x") as log:
            result = subprocess.run(
                cmd, cwd=ROOT, env=environment(c), stdout=log, stderr=subprocess.STDOUT
            )
        record = dict(
            **item,
            path=str(episode.resolve()),
            split=a.split,
            exit_code=result.returncode,
        )
        if protocol_sha:
            record.update(
                expert_protocol_sha256=protocol_sha, started_at_unix=started_at_unix
            )
            if episode.exists():
                (episode / "expert-protocol-registration.json").write_bytes(
                    launch_record.read_bytes()
                )
        waiting = episode / "waiting-for-start.json"
        cancelled = (
            waiting.exists()
            and json.loads(waiting.read_text()).get("state") == "cancelled"
        )
        if cancelled:
            record["cancelled_before_start"] = True
        if result.returncode == 0 and not cancelled:
            record["score"] = score_episode(episode)
            (episode / "learning/score.json").write_text(
                json.dumps(record["score"], indent=2)
            )
        attempts.append(record)
        (output / "attempts.json").write_text(json.dumps(attempts, indent=2))
        accepted = [x for x in attempts if x.get("score", {}).get("passed")]
        (output / "accepted.json").write_text(json.dumps(accepted, indent=2))
        print(
            f"Episode {item['seed']}: {record.get('score',{}).get('passed',False)}; {len(accepted)}/{len(attempts)} complete successes",
            flush=True,
        )
        if result.returncode:
            raise RuntimeError("Simulation failed; inspect the saved episode log")
        if cancelled:
            print(
                "Cancelled before physics started; no training samples recorded.",
                flush=True,
            )
            break


def train(a):
    output = fresh(a.output)
    if not (a.dataset / "meta/info.json").exists():
        raise ValueError("Export a LeRobot dataset first")
    split = json.loads((a.dataset / "split.json").read_text())["split"]
    if split not in ("train", "correction"):
        raise ValueError("Training must not use validation or test data")
    sys.path.insert(0, str(ROOT / "lab"))
    from learning.evaluate import assert_disjoint_records

    lineage = json.loads((a.dataset / "provenance.json").read_text())
    if a.fine_tune_from:
        lineage = (
            json.loads((a.fine_tune_from / "training-provenance.json").read_text())
            + lineage
        )
    assert_disjoint_records(
        lineage, json.loads((a.validation_dataset / "provenance.json").read_text())
    )
    contract = json.loads((a.dataset / "control-contract.json").read_text())
    if (
        a.fine_tune_from
        and json.loads((a.fine_tune_from / "control-contract.json").read_text())
        != contract
    ):
        raise ValueError("Fine-tuning data uses a different actuator bridge")
    base = a.base_model
    policy = (
        [f"--policy.path={a.fine_tune_from}"]
        if a.fine_tune_from
        else [
            "--policy.type=groot",
            f"--policy.base_model_path={base}",
            "--policy.embodiment_tag=new_embodiment",
            "--policy.tune_visual=false",
            "--policy.tune_llm=false",
            "--policy.tune_diffusion_model=true",
            "--policy.tune_projector=true",
            "--policy.chunk_size=16",
            "--policy.n_action_steps=4",
            "--policy.use_relative_actions=false",
            "--policy.use_bf16=true",
            "--policy.use_flash_attention=false",
        ]
    )
    args = [
        str(a.python),
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id=local/{a.dataset.name}",
        f"--dataset.root={a.dataset.resolve()}",
        "--dataset.image_transforms.enable=false",
        *policy,
        # The pinned GR00T preset derives warmup from policy.max_steps, while
        # the outer trainer stops at --steps. Keep both budgets identical.
        f"--policy.max_steps={a.steps}",
        "--policy.device=cuda",
        "--policy.push_to_hub=false",
        f"--seed={a.seed}",
        f"--batch_size={a.batch_size}",
        f"--steps={a.steps}",
        f"--save_freq={a.save_freq}",
        "--num_workers=2",
        "--use_policy_training_preset=true",
        "--env_eval_freq=0",
        "--eval_steps=0",
        "--log_freq=10",
        f"--output_dir={output}",
        "--wandb.enable=false",
    ]
    # Save the exact invocation beside the immutable output, before training starts.
    output.with_suffix(".command.json").write_text(json.dumps(args, indent=2))
    result = subprocess.call(args, cwd=ROOT)
    for checkpoint in output.glob("checkpoints/*/pretrained_model"):
        (checkpoint / "training-provenance.json").write_text(
            json.dumps(lineage, indent=2)
        )
        shutil.copy2(
            a.dataset / "control-contract.json", checkpoint / "control-contract.json"
        )
    raise SystemExit(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    setup_parser = sub.add_parser("setup")
    setup_parser.add_argument("--output", type=Path, required=True)
    setup_parser.add_argument("--variant", choices=["groot", "cosmos"], required=True)
    c = sub.add_parser("collect")
    c.add_argument("--controller", type=Path, required=True)
    c.add_argument("--expert-protocol", type=Path)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--episodes", type=int, default=8)
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--seconds", type=float, default=16.0)
    c.add_argument("--style", choices=list("ABCD"), default="A")
    c.add_argument(
        "--lighting", choices=["daylight", "warm", "cool-dim"], default="daylight"
    )
    c.add_argument("--root-x", type=float, default=-0.22)
    c.add_argument("--root-y", type=float, default=0.045)
    c.add_argument("--position-jitter", type=float, default=0.002)
    c.add_argument(
        "--split", choices=["train", "validation", "test", "correction"], required=True
    )
    c.add_argument("--headless", action="store_true")
    c.add_argument("--wait-for-start", action="store_true")
    c.add_argument(
        "--diagnostic",
        action="store_true",
        help="Expert physics diagnostics without RGB; never export as training data",
    )
    t = sub.add_parser("train")
    t.add_argument("--python", type=Path, required=True)
    t.add_argument("--dataset", type=Path, required=True)
    t.add_argument("--validation-dataset", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--base-model", default="nvidia/GR00T-N1.7-3B")
    t.add_argument(
        "--fine-tune-from",
        type=Path,
        help="Start a new fine-tuning stage from saved policy weights; optimizer starts fresh",
    )
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--save-freq", type=int, default=500)
    t.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.command == "setup":
        setup(a)
    elif a.command == "collect":
        if a.episodes < 1 or not 0 <= a.position_jitter <= 0.03:
            p.error("Invalid collection range")
        collect(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
