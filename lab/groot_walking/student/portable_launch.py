"""Run a GR00T walking entry (trainer or inference server) portably.

The checkpoint configs record base_model_path as the absolute Hugging Face cache
path of the machine that trained them. This launcher leaves the checkpoint files
and their hashes untouched and instead rebinds that path at load time to the same
pinned base revision on the current machine (MICRODUCK_GROOT_BASE, or the local
Hugging Face cache, downloading it if needed).

It also keeps the trainer's exact feature-cache check. If a fresh feature differs
from the packaged cache (possible on another GPU architecture), the differences
are written to <output>-cache-check.json before the original error stops training;
`--recompute-features` remains the way to train on such a device.

Usage: python portable_launch.py <entry.py> [entry arguments...]
"""
from pathlib import Path
import json
import os
import runpy
import sys

BASE_REPO = "nvidia/GR00T-N1.7-3B"
BASE_REVISION = "2fc962b973bccdd5d8ce4f67cc63b264d6886495"
REMAPS = []


def resolve_base():
    explicit = os.environ.get("MICRODUCK_GROOT_BASE")
    if explicit:
        return explicit
    from huggingface_hub import snapshot_download

    return snapshot_download(BASE_REPO, revision=BASE_REVISION)


def patch_config():
    from lerobot.policies.groot.configuration_groot import GrootConfig

    original = GrootConfig.__post_init__
    resolved = {}

    def post_init(self):
        path = self.base_model_path
        if (
            path
            and str(path).rstrip("/").endswith(BASE_REVISION)
            and "models--nvidia--GR00T-N1.7-3B" in str(path)
            and not Path(path).exists()
        ):
            if "path" not in resolved:
                resolved["path"] = resolve_base()
            self.base_model_path = resolved["path"]
            REMAPS.append(dict(recorded=str(path), resolved=resolved["path"]))
        original(self)

    GrootConfig.__post_init__ = post_init


def patch_cache_check(entry, argv):
    if "--backbone-cache-manifest" not in argv:
        return
    sys.path.insert(0, str(Path(entry).resolve().parent))
    import verified_training_artifacts as artifacts

    original = artifacts.compare_cache_row
    output = Path(argv[argv.index("--output") + 1])

    def compare(cache, index, features, actions):
        try:
            return original(cache, index, features, actions)
        except ValueError:
            import torch

            report = dict(index=index, device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          fields={})
            fresh = {"features." + k: v for k, v in features.items()}
            fresh.update({"action_inputs." + k: actions[k] for k in
                          ("state", "state_mask", "action", "action_mask", "embodiment_id") if k in actions})
            for key, value in fresh.items():
                if key not in cache:
                    report["fields"][key] = "missing in cache"
                    continue
                a = value.detach().cpu()
                b = cache[key][index:index + 1]
                if a.dtype.is_floating_point:
                    diff = (a.float() - b.float()).abs()
                    report["fields"][key] = dict(equal=bool(torch.equal(a, b)), max_abs=float(diff.max()),
                                                 max_rel=float((diff / b.float().abs().clamp_min(1e-6)).max()),
                                                 finite=bool(torch.isfinite(a).all()))
                else:
                    report["fields"][key] = dict(equal=bool(torch.equal(a, b)))
            path = output.with_name(output.name + "-cache-check.json")
            path.write_text(json.dumps(report, indent=2) + "\n")
            print("Packaged feature cache differs from features computed on this GPU; see", path,
                  "\nRe-run training with --recompute-features to compute the cache on this device.",
                  file=sys.stderr, flush=True)
            raise

    artifacts.compare_cache_row = compare


def main():
    entry, argv = sys.argv[1], sys.argv[2:]
    patch_config()
    patch_cache_check(entry, argv)
    sys.argv = [entry, *argv]
    try:
        runpy.run_path(entry, run_name="__main__")
    finally:
        if REMAPS and os.environ.get("MICRODUCK_GROOT_REMAP_LOG"):
            Path(os.environ["MICRODUCK_GROOT_REMAP_LOG"]).write_text(json.dumps(REMAPS[:1], indent=2) + "\n")


if __name__ == "__main__":
    main()
