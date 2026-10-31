#!/usr/bin/env python3
"""Train and validate the 8D high-level walking-shot GR00T policy (Sections 5.2–5.3).

Run inside the course container. The defaults reproduce the recipe and the
closed-loop check of the course checkpoint in course-data/groot-walking:

  train     fine-tune the GR00T action head on the provided demonstrations
  validate  run the walking-shot task in Isaac Lab / Newton with a checkpoint
  verify    check the course data package against its SHA-256 manifest

Training uses lab/groot_walking/trainer. On a GPU with enough memory for the
course recipe (about 47 GB at its peak) it runs the exact trainer of the course
checkpoint; on a 32 GB GPU it runs the low-memory variant of the same recipe:
every update still uses the same 16 sampled rows, computed as micro-batches with
exact gradient accumulation, so the result is a new model rather than a
bit-for-bit repeat of the course checkpoint. The choice follows the memory the
GPU actually reports; `--profile` overrides it.
Validation uses lab/groot_walking/student: the evaluation entry of
that checkpoint (GR00T server + simulated walking/kick/recovery executor) with
the course standing initialization and an Isaac Sim window.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "lab/groot_walking"))
from weight_page_cache import release_after_hash
DATA = ROOT / "course-data/groot-walking"
TRAINER = ROOT / "lab/groot_walking/trainer/learning/train_walking_history.py"
LOW_MEMORY_TRAINER = ROOT / "lab/groot_walking/trainer/learning/train_walking_history_lowmem.py"
ROLLOUT = ROOT / "lab/groot_walking/student/walking_student_rollout.py"
# Rebinds the recorded base-model path to this machine; checkpoints stay unchanged.
LAUNCH = ROOT / "lab/groot_walking/student/portable_launch.py"

# Recipe of the course checkpoint (2000 updates, batch 16, action head, frozen VLM).
RECIPE = dict(steps=2000, batch_size=16, learning_rate=5e-5, seed=8000001)
# Total GPU memory (bytes, as reported by CUDA) needed by each training profile.
# The course recipe peaked at about 47 GB; the low-memory profile holds 24.15 GiB
# of weights, gradients and AdamW state plus one micro-batch of activations.
FULL_PROFILE_MIN_BYTES = 56 * 2**30
LOW_MEMORY_MIN_BYTES = 30 * 2**30
# GPU on which the packaged feature cache was computed (DGX Spark, NVIDIA GB10).
PACKAGE_CACHE_CAPABILITY = "12.1"
# Feature caches recomputed on other GPU architectures, one folder each, next to
# the package: <name>/identity.json (+ "compute_capability"), the cache and its
# experiment.json. They pass the same checks as the packaged cache.
DEVICE_CACHES = ROOT / "course-data/groot-walking-feature-caches"

# Development episodes used for the course checkpoint: fixed field, nominal start.
VALIDATION = dict(seed=8000101, episodes=5, seconds=60.0, action_execution_steps=4)


def groot_python():
    python = Path(os.environ.get("MICRODUCK_GROOT_PYTHON", "/workspace/groot-env/bin/python"))
    if not python.is_file():
        raise SystemExit("GR00T environment not found. Run this inside the course container.")
    return python


def check_package(full=False):
    manifest = DATA / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(
            "GR00T course data package not found at course-data/groot-walking. Unpack it "
            "on the host (or set MICRODUCK_COURSE_DATA) and restart the container."
        )
    data = json.loads(manifest.read_text())
    problems = []
    for item in data["files"]:
        path = DATA / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"]:
            problems.append(item["path"])
            continue
        if full:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 24), b""):
                    digest.update(block)
                release_after_hash(stream)
            if digest.hexdigest() != item["sha256"]:
                problems.append(item["path"])
    if problems:
        raise SystemExit("Course data differs from its manifest: " + ", ".join(problems[:5]))
    return data


def fresh(path):
    path = path.resolve()
    if path.exists():
        raise SystemExit("Choose a new output directory to keep earlier results: " + str(path))
    if path.is_relative_to(ROOT / "course-data") or path.is_relative_to(ROOT / "models"):
        raise SystemExit("Outputs must not be written into course data or models")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def run(argv):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "lab"), OMP_NUM_THREADS="1",
               TOKENIZERS_PARALLELISM="false")
    print("Launching:", " ".join(map(str, argv)), flush=True)
    return subprocess.call(list(map(str, argv)), cwd=ROOT, env=env)


def gpu_info():
    """Name, compute capability and total memory of the GPU training will use."""
    probe = ("import json, torch\n"
             "assert torch.cuda.is_available(), 'no CUDA GPU visible'\n"
             "p = torch.cuda.get_device_properties(0)\n"
             "print(json.dumps(dict(name=p.name, capability=f'{p.major}.{p.minor}',"
             " total_memory=p.total_memory)))")
    result = subprocess.run([str(groot_python()), "-c", probe], capture_output=True, text=True)
    if result.returncode:
        raise SystemExit("Could not query the GPU:\n" + result.stderr.strip()[-800:])
    return json.loads(result.stdout.strip().splitlines()[-1])


def choose_profile(requested, gpu):
    """`full` = recipe of the course checkpoint, `low-memory` = same recipe in micro-batches."""
    memory = gpu["total_memory"]
    if requested != "auto":
        return requested
    if memory >= FULL_PROFILE_MIN_BYTES:
        return "full"
    if memory >= LOW_MEMORY_MIN_BYTES:
        return "low-memory"
    raise SystemExit(
        f"{gpu['name']} reports {memory / 2**30:.1f} GiB; GR00T training needs at least about "
        f"{LOW_MEMORY_MIN_BYTES / 2**30:.0f} GiB even with --profile low-memory.")


def cache_folder(gpu):
    """Feature cache to try first: one computed on this GPU architecture, else the package's."""
    if gpu["capability"] != PACKAGE_CACHE_CAPABILITY and DEVICE_CACHES.is_dir():
        for identity in sorted(DEVICE_CACHES.glob("*/identity.json")):
            if json.loads(identity.read_text()).get("compute_capability") == gpu["capability"]:
                return identity.parent
    return DATA / "backbone-cache"


def cache_manifest(output, folder):
    """Relocatable manifest for a frozen-backbone feature cache.

    The trainer recomputes every identity below (checkpoint and dataset trees, cache
    and experiment bytes) and four feature rows on the GPU before reusing the cache;
    any difference stops training instead of silently recomputing.
    """
    identity = folder / "identity.json"
    if not identity.is_file():
        return []
    record = json.loads(identity.read_text())
    manifest = output.with_name(output.name + "-cache-manifest.json")
    manifest.write_text(json.dumps(dict(
        checkpoint_tree_sha256=record["checkpoint_tree_sha256"],
        dataset_tree_sha256=record["dataset_tree_sha256"],
        cache=str(folder / "frozen-backbone-cache.safetensors"),
        cache_sha256=record["cache_sha256"],
        experiment=str(folder / "experiment.json"),
        experiment_sha256=record["experiment_sha256"],
    ), indent=2) + "\n")
    return ["--backbone-cache-manifest", manifest]


def train(a):
    check_package()
    gpu = gpu_info()
    profile = choose_profile(a.profile, gpu)
    output = fresh(a.output)
    print(f"GPU: {gpu['name']} (compute capability {gpu['capability']}, "
          f"{gpu['total_memory'] / 2**30:.1f} GiB) -> training profile: {profile}", flush=True)
    if a.feature_cache_manifest:
        cache = ["--backbone-cache-manifest", a.feature_cache_manifest.resolve()]
    elif a.recompute_features:
        cache = []
    else:
        folder = cache_folder(gpu)
        if gpu["capability"] != PACKAGE_CACHE_CAPABILITY and folder == DATA / "backbone-cache":
            print("The packaged feature cache was computed on a DGX Spark GPU. If the four-row check "
                  "fails on this GPU, rerun with --recompute-features.", flush=True)
        cache = cache_manifest(output, folder)
    trainer = [TRAINER]
    if profile == "low-memory":
        trainer = [LOW_MEMORY_TRAINER, "--micro-batch-size", a.micro_batch_size]
    return run([groot_python(), LAUNCH, *trainer, *cache,
                "--checkpoint", DATA / "initialization",
                "--dataset", DATA / "datasets/train-history127",
                "--validation-dataset", DATA / "datasets/validation-history127",
                "--output", output,
                "--initialization", "checkpoint", "--training-scope", "action-head",
                "--head-dropout", "disabled",
                "--sampling-manifest", DATA / "sampling.json",
                "--steps", a.steps, "--batch-size", a.batch_size,
                "--learning-rate", a.learning_rate, "--seed", a.seed])


def validate(a):
    checkpoint = a.checkpoint.resolve()
    if checkpoint == (DATA / "checkpoint").resolve():
        check_package()
    if not (checkpoint / "control-contract.json").is_file():
        raise SystemExit("Not a GR00T walking checkpoint folder: " + str(checkpoint))
    output = fresh(a.output)
    code = subprocess.call(
        [sys.executable, str(ROLLOUT), "rollout", "--python", str(groot_python()),
         "--checkpoint", str(checkpoint), "--output", str(output),
         "--seed", str(a.seed), "--episodes", str(a.episodes),
         "--seconds", str(a.seconds),
         "--action-execution-steps", str(VALIDATION["action_execution_steps"]),
         "--visualizer", "none" if a.headless else "kit"],
        cwd=ROOT)
    attempts = output / "attempts.json"
    if attempts.is_file():
        records = json.loads(attempts.read_text())
        passed = sum(r.get("score", {}).get("passed", False) for r in records)
        print(f"Complete walking shots (goal, no fall, final standing): {passed}/{len(records)}")
        for r in records:
            s = r.get("score", {})
            print(f"  seed {r['seed']}: contact={s.get('contact')} goal={s.get('goal')} "
                  f"fall={s.get('fall')} end={s.get('terminal')}")
    return code


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="Fine-tune the action head on the provided demonstrations")
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--steps", type=int, default=RECIPE["steps"])
    t.add_argument("--batch-size", type=int, default=RECIPE["batch_size"])
    t.add_argument("--learning-rate", type=float, default=RECIPE["learning_rate"])
    t.add_argument("--seed", type=int, default=RECIPE["seed"])
    t.add_argument("--recompute-features", action="store_true",
                   help="Ignore the packaged feature cache and recompute it (about 88 min on DGX Spark)")
    t.add_argument("--feature-cache-manifest", type=Path,
                   help="Reuse a cache recomputed earlier on this machine "
                        "(lab/groot_walking/trainer/learning/local_cache_manifest.py)")
    t.add_argument("--profile", choices=["auto", "full", "low-memory"], default="auto",
                   help="auto: chosen from the GPU's total memory (>= 56 GiB full, >= 30 GiB low-memory)")
    t.add_argument("--micro-batch-size", type=int, default=1,
                   help="Rows per forward/backward in the low-memory profile (must divide --batch-size)")
    v = sub.add_parser("validate", help="Run the walking-shot task with a checkpoint")
    v.add_argument("--checkpoint", type=Path, default=DATA / "checkpoint")
    v.add_argument("--output", type=Path, required=True)
    v.add_argument("--seed", type=int, default=VALIDATION["seed"])
    v.add_argument("--episodes", type=int, default=VALIDATION["episodes"])
    v.add_argument("--seconds", type=float, default=VALIDATION["seconds"])
    v.add_argument("--headless", action="store_true", help="Run without the Isaac Sim window")
    c = sub.add_parser("verify", help="Check the course data package (add --full for SHA-256)")
    c.add_argument("--full", action="store_true")
    a = p.parse_args()
    if a.command == "train":
        if a.steps < 1 or a.batch_size < 1 or not a.learning_rate > 0:
            p.error("Use positive steps, batch size and learning rate")
        if a.micro_batch_size < 1 or a.batch_size % a.micro_batch_size:
            p.error("--micro-batch-size must be positive and divide --batch-size")
        if a.feature_cache_manifest and a.recompute_features:
            p.error("Choose --recompute-features or --feature-cache-manifest")
        return train(a)
    if a.command == "validate":
        if a.episodes < 1 or not 0 < a.seconds <= 180:
            p.error("Use at least one episode and at most 180 seconds")
        return validate(a)
    data = check_package(full=a.full)
    print(f"OK {len(data['files'])} files, {data['total_bytes'] / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
