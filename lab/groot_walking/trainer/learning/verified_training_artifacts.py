"""Verify frozen feature reuse and complete model serialization."""
from pathlib import Path
import hashlib
import json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tree_identity(root):
    h = hashlib.sha256()
    for path in sorted(Path(root).rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts and '.cache' not in path.parts:
            h.update((str(path.relative_to(root)) + ':' + digest(path) + '\n').encode())
    return h.hexdigest()


def load_verified_cache(manifest_path, checkpoint, dataset, size):
    from safetensors.torch import load_file
    record = json.loads(Path(manifest_path).read_text())
    for name, actual in [('checkpoint', checkpoint), ('dataset', dataset)]:
        if tree_identity(actual) != record[name + '_tree_sha256']:
            raise ValueError('Frozen cache ' + name + ' differs')
    if digest(record['cache']) != record['cache_sha256']:
        raise ValueError('Frozen cache bytes differ')
    if digest(record['experiment']) != record['experiment_sha256']:
        raise ValueError('Frozen cache experiment differs')
    original = json.loads(Path(record['experiment']).read_text())
    if original['image_augmentation'] or not original['backbone_frozen']:
        raise ValueError('Cache requires deterministic frozen backbone')
    cache = load_file(record['cache'])
    actual = {k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in cache.items()}
    if actual != original['cache_tensors'] or any(v.shape[0] != size for v in cache.values()):
        raise ValueError('Frozen cache shape/dtype/frame count differs')
    return cache, dict(manifest_sha256=digest(manifest_path), **record), original


def compare_cache_row(cache, index, features, actions):
    import torch
    expected = {'features.' + k: v for k, v in features.items()}
    expected.update({'action_inputs.' + k: actions[k] for k in
                    ('state', 'state_mask', 'action', 'action_mask', 'embodiment_id') if k in actions})
    if set(expected) != set(cache):
        raise ValueError('Fresh/cache field mismatch')
    for key, value in expected.items():
        if not torch.equal(value.detach().cpu(), cache[key][index:index + 1]):
            raise ValueError('Fresh/cache tensor mismatch: ' + key)


def verify_model_file(policy, path):
    """Exact full-weight check, in addition to the unchanged action check."""
    import torch
    from safetensors import safe_open
    torch.cuda.synchronize()
    state = policy.state_dict()
    with safe_open(str(path), framework='pt', device='cpu') as saved:
        aliases = {k: v for k, v in (saved.metadata() or {}).items()
                   if k in state and v in saved.keys()}
        if set(saved.keys()) | set(aliases) != set(state):
            raise ValueError('Serialized weight keys differ')
        for key, tensor in state.items():
            if not torch.equal(saved.get_tensor(aliases.get(key, key)), tensor.detach().to('cpu', copy=True)):
                raise ValueError('Serialized tensor differs: ' + key)
    return len(state)


def save_verified_model(policy, destination):
    """Use a synchronized, independently checked CPU snapshot for serialization."""
    import torch
    torch.cuda.synchronize()
    snapshot = {}
    shared = {}
    for key, tensor in policy.state_dict().items():
        identity = (tensor.device, tensor.untyped_storage().data_ptr(),
                    tensor.storage_offset(), tuple(tensor.shape),
                    tuple(tensor.stride()), tensor.dtype)
        if identity not in shared:
            shared[identity] = tensor.detach().to('cpu', copy=True).contiguous()
        snapshot[key] = shared[identity]
    torch.cuda.synchronize()
    for key, tensor in policy.state_dict().items():
        if not torch.equal(snapshot[key], tensor.detach().to('cpu', copy=True)):
            raise ValueError('CPU snapshot differs: ' + key)
    policy.save_pretrained(destination, state_dict=snapshot)
    del snapshot, shared
    count = verify_model_file(policy, Path(destination) / 'model.safetensors')
    (Path(destination) / 'weight-serialization-check.json').write_text(json.dumps(
        dict(passed=True, tensors=count, exact=True,
             method='Synchronized CPU snapshot, second copy comparison, serialized tensor comparison'), indent=2))
