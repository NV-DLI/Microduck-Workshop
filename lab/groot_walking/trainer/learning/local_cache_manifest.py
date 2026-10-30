"""Make a --backbone-cache-manifest for a feature cache computed on this machine.

If the packaged cache differs from features computed on this GPU (another GPU
architecture can round differently), train once with --recompute-features. That
run writes frozen-backbone-cache.safetensors and experiment.json into its output.
This tool records their identities so later runs reuse that cache with the same
checks as the packaged one (checkpoint and dataset tree hashes, cache and
experiment SHA-256, four rows recomputed on the GPU) instead of recomputing 31.7 GB.

Usage: python local_cache_manifest.py <recompute-run-output> <checkpoint> <dataset> <manifest.json>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verified_training_artifacts import digest, tree_identity  # noqa: E402


def main():
    run, checkpoint, dataset, out = (Path(p).resolve() for p in sys.argv[1:5])
    cache, experiment = run / 'frozen-backbone-cache.safetensors', run / 'experiment.json'
    record = json.loads(experiment.read_text())
    if record.get('cache_reuse') is not None:
        raise SystemExit('This run reused a cache; point to a --recompute-features run')
    if Path(record['checkpoint']) != checkpoint or Path(record['dataset']) != dataset:
        raise SystemExit('Run used another checkpoint or dataset')
    if out.exists():
        raise SystemExit('Choose a new manifest path: ' + str(out))
    out.write_text(json.dumps(dict(
        checkpoint_tree_sha256=tree_identity(checkpoint),
        dataset_tree_sha256=tree_identity(dataset),
        cache=str(cache), cache_sha256=digest(cache),
        experiment=str(experiment), experiment_sha256=digest(experiment),
    ), indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
