"""Extend a trained walking model with four recent camera views.

Fits new history statistics on the same training episodes. Existing state and
action normalization must match the parent. No model training happens here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def source_set(records):
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('Only declared training records are accepted')
    return {(str(Path(r['source']).resolve()), r['seed'], r['frames'],
             r['samples_sha256'], r['sensors_sha256']) for r in records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent, dataset, output = args.checkpoint.resolve(), args.dataset.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if not (parent/'model.safetensors').is_file():
        raise ValueError('A trained parent checkpoint is required')
    if json.loads((dataset/'split.json').read_text()).get('split') != 'train':
        raise ValueError('Fit history statistics on training data only')
    old = json.loads((parent/'control-contract.json').read_text())
    new = json.loads((dataset/'control-contract.json').read_text())
    if (old.get('locomotion') != 'walking' or len(old['state_names']) != 58
            or len(new['state_names']) != 127 or new['state_names'][:58] != old['state_names']
            or new['action_names'] != old['action_names']):
        raise ValueError('Expected the same walking action contract with added history')
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from learning.walking_visual_history import specification
    names = [name.removeprefix('q_relative/') for name in old['state_names'][6:20]]
    if new != specification(names):
        raise ValueError('History observation contract differs')
    if source_set(json.loads((parent/'training-provenance.json').read_text())) != source_set(json.loads((dataset/'provenance.json').read_text())):
        raise ValueError('History continuation must use the same training episodes as the parent')
    subprocess.run([sys.executable, str(Path(__file__).with_name('initialize.py')),
                    '--dataset', str(dataset), '--output', str(output),
                    '--full-camera-frame', '--action-execution-steps', '16'], check=True)
    from safetensors.torch import load_file
    import torch
    for name in ('policy_preprocessor_step_2_groot_n1_7_pack_inputs_v1.safetensors',
                 'policy_postprocessor_step_0_groot_action_unpack_unnormalize_v2.safetensors'):
        previous, extended = load_file(parent/name), load_file(output/name)
        for feature, width in (('observation.state', 58), ('action', 8)):
            for stat in ('min', 'max'):
                key = feature+'.'+stat
                if key in previous:
                    if key not in extended or not torch.equal(previous[key], extended[key][:width]):
                        raise ValueError('Parent normalization changed: '+key)
    config = json.loads((parent/'config.json').read_text())
    config['input_features'] = json.loads((output/'config.json').read_text())['input_features']
    (output/'config.json').write_text(json.dumps(config, indent=2))
    for name in ('training-provenance.json', 'training-recipe.json'):
        shutil.copy2(parent/name, output/name)
    (output/'model.safetensors').symlink_to(parent/'model.safetensors')
    digest = hashlib.sha256()
    for path in sorted(parent.iterdir()):
        if path.suffix in ('.json', '.safetensors'):
            with path.open('rb') as handle:
                value = hashlib.file_digest(handle, 'sha256').hexdigest()
            digest.update((path.name+':'+value+'\n').encode())
    (output/'initialization.json').write_text(json.dumps(dict(
        gradient_updates=0, contains_model_weights=True,
        parent_checkpoint=str(parent), parent_checkpoint_sha256=digest.hexdigest(),
        statistics_source=str(dataset),
        weights='Symbolic link to the preserved parent; training saves a separate checkpoint.',
        next_step='train_walking_history.py; compute a new four-view feature cache'), indent=2))
    print('History initialization ready:', output)


if __name__ == '__main__':
    main()
