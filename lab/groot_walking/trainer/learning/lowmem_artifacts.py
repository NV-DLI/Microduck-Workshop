"""Low-memory helpers for the GR00T walking action-head trainer.

Everything here keeps the on-disk formats of the original trainer:

* ``StreamingSafetensorsWriter`` writes a normal safetensors file one tensor at a
  time (same tensor order, header fields, padding and data bytes as
  ``safetensors.torch.save_file``), so a 12.6 GB model or a 13 GB optimizer
  state never needs a full CPU copy.
* ``stream_save_verified_model`` keeps every check of
  ``verified_training_artifacts.save_verified_model`` (two synchronized CPU
  copies per tensor must be equal, then the serialized file is compared with the
  live model tensor by tensor) but holds at most one tensor pair in RAM.
* ``stream_save_training_state`` writes the same ``training_state`` folder as
  ``lerobot.common.train_utils.save_training_state`` and is restored with the
  unchanged lerobot ``load_training_state``.
"""
from pathlib import Path
import json
import os
import struct

_ORDER_CACHE = {}


def _dtype_name(dtype):
    import torch
    names = {
        torch.bool: 'BOOL', torch.uint8: 'U8', torch.int8: 'I8', torch.int16: 'I16',
        torch.float16: 'F16', torch.bfloat16: 'BF16', torch.int32: 'I32',
        torch.float32: 'F32', torch.float64: 'F64', torch.int64: 'I64',
    }
    if dtype not in names:
        raise TypeError('Unsupported dtype for streaming safetensors: ' + str(dtype))
    return names[dtype]


def _dtype_rank(dtype):
    """Rank used by the installed safetensors to order tensors in the file.

    Derived from safetensors itself (tiny in-memory save) instead of a copied
    table, so a different safetensors release cannot silently change layout.
    """
    import torch
    from safetensors.torch import save
    if not _ORDER_CACHE:
        dtypes = [torch.bool, torch.uint8, torch.int8, torch.int16, torch.float16,
                  torch.bfloat16, torch.int32, torch.float32, torch.float64, torch.int64]
        # Same name for every dtype class would collide; use one name per dtype
        # that sorts identically so only the dtype decides the order.
        probe = {'t%02d' % (len(dtypes) - i): torch.zeros(1, dtype=d) for i, d in enumerate(dtypes)}
        raw = save(probe)
        n = struct.unpack('<Q', raw[:8])[0]
        header = json.loads(raw[8:8 + n])
        header.pop('__metadata__', None)
        by_offset = sorted(header.items(), key=lambda kv: kv[1]['data_offsets'][0])
        for position, (_, info) in enumerate(by_offset):
            _ORDER_CACHE[info['dtype']] = position
    return _ORDER_CACHE[_dtype_name(dtype)]


def safetensors_header(specs, metadata=None):
    """Header bytes, tensor order and byte layout for ``{name: (dtype, shape)}``.

    Matches safetensors: tensors sorted by dtype rank then name, compact JSON,
    ``__metadata__`` first, space padding to a multiple of 8. safetensors itself
    emits metadata keys in hash order, so with several metadata keys only the
    parsed header (not its byte order) is reproducible.
    """
    import torch
    specs = {k: (v[0], tuple(int(s) for s in v[1])) for k, v in specs.items()}
    order = sorted(specs, key=lambda k: (_dtype_rank(specs[k][0]), k))
    header = {}
    if metadata:
        header['__metadata__'] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    layout = {}
    for name in order:
        dtype, shape = specs[name]
        numel = 1
        for s in shape:
            numel *= s
        size = numel * torch.empty((), dtype=dtype).element_size()
        header[name] = dict(dtype=_dtype_name(dtype), shape=list(shape),
                            data_offsets=[offset, offset + size])
        layout[name] = (offset, size)
        offset += size
    text = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode()
    text += b' ' * (-len(text) % 8)
    return struct.pack('<Q', len(text)) + text, order, layout, offset


class StreamingSafetensorsWriter:
    """Write ``{name: (dtype, shape)}`` tensors into one safetensors file.

    Call ``write(name, tensor)`` for each tensor (any order) or ``write_rows``
    for row slices; ``close()`` checks every byte range was written once.
    """

    def __init__(self, path, specs, metadata=None):
        self.path = Path(path)
        self.specs = {k: (v[0], tuple(int(s) for s in v[1])) for k, v in specs.items()}
        head, self.order, self.layout, self.total = safetensors_header(self.specs, metadata)
        self.data_start = len(head)
        self._written = {}
        self._fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        self._pwrite(head, 0)
        os.ftruncate(self._fd, self.data_start + self.total)

    def _pwrite(self, data, position):
        view = memoryview(data)
        while view:
            done = os.pwrite(self._fd, view, position)
            view = view[done:]
            position += done

    @staticmethod
    def _bytes(tensor):
        import torch
        cpu = tensor.detach()
        if cpu.device.type != 'cpu':
            cpu = cpu.to('cpu', copy=True)
        cpu = cpu.contiguous()
        if cpu.numel() == 0:
            return b''
        return cpu.reshape(-1).view(torch.uint8).numpy()

    def write(self, name, tensor):
        dtype, shape = self.specs[name]
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise ValueError('Tensor spec differs: ' + name)
        if name in self._written:
            raise ValueError('Tensor already written: ' + name)
        offset, size = self.layout[name]
        data = self._bytes(tensor)
        if len(data) != size:
            raise ValueError('Byte length differs: ' + name)
        self._written[name] = 'all'
        self._pwrite(data, self.data_start + offset)

    def write_rows(self, name, start, tensor):
        """Write ``tensor`` as rows ``start:start+len(tensor)`` of ``name``."""
        dtype, shape = self.specs[name]
        if tensor.dtype != dtype or tuple(tensor.shape[1:]) != shape[1:]:
            raise ValueError('Row spec differs: ' + name)
        offset, size = self.layout[name]
        row = size // shape[0] if shape[0] else 0
        if start < 0 or start + tensor.shape[0] > shape[0]:
            raise IndexError(name)
        rows = self._written.setdefault(name, set())
        if rows == 'all':
            raise ValueError('Tensor already written whole: ' + name)
        new = set(range(start, start + tensor.shape[0]))
        if rows & new:
            raise ValueError('Rows written twice: %s%s' % (name, sorted(rows & new)[:3]))
        data = self._bytes(tensor)
        rows |= new
        self._pwrite(data, self.data_start + offset + start * row)

    def close(self):
        missing = []
        for name, (_, shape) in self.specs.items():
            got = self._written.get(name, set())
            if got == 'all':
                continue
            if not shape or len(got) != shape[0]:
                missing.append(name)
        os.fsync(self._fd)
        os.close(self._fd)
        if missing:
            raise ValueError('Incomplete safetensors file, missing: ' + ', '.join(missing[:5]))


def stream_save_file(tensors, path, metadata=None):
    """Drop-in for ``safetensors.torch.save_file`` copying one tensor at a time."""
    for k, v in tensors.items():
        if not v.is_contiguous():
            raise ValueError('Non-contiguous tensor: ' + k)
    writer = StreamingSafetensorsWriter(
        path, {k: (v.dtype, tuple(v.shape)) for k, v in tensors.items()}, metadata)
    for name in writer.order:
        writer.write(name, tensors[name])
    writer.close()


def model_state_for_safetensors(policy):
    """Names, aliases and metadata exactly as ``save_torch_state_dict`` writes them."""
    from huggingface_hub.serialization._torch import _clean_state_dict_for_safetensors
    state = dict(policy.state_dict())
    metadata = {}
    cleaned = _clean_state_dict_for_safetensors(dict(state), metadata, force_contiguous=False)
    file_metadata = {'format': 'pt'}
    file_metadata.update(metadata)
    return state, cleaned, file_metadata


def stream_save_verified_model(policy, destination):
    """Low-RAM equivalent of ``save_verified_model``.

    Same checks per tensor: synchronized CPU copy, a second independent copy that
    must be identical, then (after the file is closed) the unchanged
    ``verify_model_file`` comparison of every serialized tensor with the live
    model, including shared-tensor aliases.
    """
    import torch
    from verified_training_artifacts import verify_model_file
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    policy.config._save_pretrained(destination)
    torch.cuda.synchronize()
    state, cleaned, metadata = model_state_for_safetensors(policy)
    path = destination / 'model.safetensors'
    writer = StreamingSafetensorsWriter(
        path, {k: (v.dtype, tuple(v.shape)) for k, v in cleaned.items()}, metadata)
    for name in writer.order:
        tensor = cleaned[name].detach()
        first = tensor.to('cpu', copy=True).contiguous()
        torch.cuda.synchronize()
        second = tensor.to('cpu', copy=True)
        if not torch.equal(first, second):
            raise ValueError('CPU snapshot differs: ' + name)
        del second
        writer.write(name, first)
        del first
    writer.close()
    del state, cleaned
    count = verify_model_file(policy, path)
    (destination / 'weight-serialization-check.json').write_text(json.dumps(
        dict(passed=True, tensors=count, exact=True,
             method='Synchronized per-tensor CPU snapshot, second copy comparison, '
                    'streamed safetensors write, serialized tensor comparison'), indent=2))
    return count


def optimizer_flat_state(optimizer):
    """``state`` part of lerobot's optimizer file (references, no copies)."""
    from lerobot.utils.utils import flatten_dict
    state = optimizer.state_dict()
    param_groups = state.pop('param_groups')
    return flatten_dict(state), param_groups


def stream_save_training_state(checkpoint_dir, train_step, optimizer, scheduler,
                               batch_size, num_processes=1):
    """Same files as lerobot ``save_training_state``; optimizer tensors streamed.

    Returns the written training_state directory. Every optimizer tensor is
    compared with the file after writing.
    """
    import torch
    from safetensors import safe_open
    from lerobot.common.train_utils import save_training_step
    from lerobot.optim.schedulers import save_scheduler_state
    from lerobot.utils.random_utils import save_rng_state
    from lerobot.utils.constants import (TRAINING_STATE_DIR, OPTIMIZER_STATE,
                                         OPTIMIZER_PARAM_GROUPS)
    from lerobot.utils.io_utils import write_json
    save_dir = Path(checkpoint_dir) / TRAINING_STATE_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_training_step(train_step, save_dir, num_processes=num_processes, batch_size=batch_size)
    save_rng_state(save_dir)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    flat, param_groups = optimizer_flat_state(optimizer)
    stream_save_file(flat, save_dir / OPTIMIZER_STATE)
    write_json(param_groups, save_dir / OPTIMIZER_PARAM_GROUPS)
    with safe_open(str(save_dir / OPTIMIZER_STATE), framework='pt', device='cpu') as saved:
        if set(saved.keys()) != set(flat):
            raise ValueError('Optimizer state keys differ after save')
        for key, tensor in flat.items():
            if not torch.equal(saved.get_tensor(key), tensor.detach().to('cpu', copy=True)):
                raise ValueError('Optimizer state differs after save: ' + key)
    save_scheduler_state(scheduler, save_dir)
    return save_dir


def compare_optimizer_with_file(optimizer, training_state_dir):
    """Bitwise check of a restored optimizer against its saved file."""
    import torch
    from safetensors import safe_open
    from lerobot.utils.constants import OPTIMIZER_STATE
    flat, _ = optimizer_flat_state(optimizer)
    with safe_open(str(Path(training_state_dir) / OPTIMIZER_STATE), framework='pt', device='cpu') as saved:
        if set(saved.keys()) != set(flat):
            raise ValueError('Restored optimizer keys differ')
        for key, tensor in flat.items():
            if not torch.equal(saved.get_tensor(key), tensor.detach().to('cpu', copy=True)):
                raise ValueError('Restored optimizer tensor differs: ' + key)
    return len(flat)


def comparable_param_groups(optimizer):
    return [{k: (list(v) if isinstance(v, tuple) else v) for k, v in g.items() if k != 'params'}
            | dict(param_count=len(g['params'])) for g in optimizer.param_groups]


def rng_snapshot(generator=None):
    """Python, NumPy, torch CPU/CUDA RNG and the sampling generator state."""
    import random
    import numpy as np
    import torch
    snap = dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state().clone())
    if torch.cuda.is_available():
        snap['cuda'] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    if generator is not None:
        snap['sampler'] = generator.get_state().clone()
    return snap


def rng_compare(before, after):
    """Compare RNG streams before a checkpoint and after restoring it.

    lerobot's rng_state.safetensors stores the Mersenne-Twister state of Python
    and NumPy but not their cached second Gaussian (``random.gauss`` and NumPy
    normal draws); those caches are reported separately and do not count as
    restored. This trainer draws its samples and noise from torch generators.
    torch CPU, CUDA and the sampler generator must be equal.
    """
    import numpy as np
    import torch
    report = dict(
        torch_cpu=bool(torch.equal(before['torch'], after['torch'])),
        cuda=('cuda' not in before) or (len(before['cuda']) == len(after['cuda']) and
                                        all(torch.equal(x, y) for x, y in zip(before['cuda'], after['cuda']))),
        sampler=('sampler' not in before) or bool(torch.equal(before['sampler'], after['sampler'])),
        python_mersenne_twister=before['python'][:2] == after['python'][:2],
        numpy_mersenne_twister=(before['numpy'][0] == after['numpy'][0]
                                and bool(np.array_equal(before['numpy'][1], after['numpy'][1]))
                                and before['numpy'][2] == after['numpy'][2]),
        python_gauss_cache=dict(saved_by_lerobot=False, empty_before=before['python'][2] is None,
                                equal_after_restore=before['python'][2] == after['python'][2]),
        numpy_gauss_cache=dict(saved_by_lerobot=False, empty_before=not before['numpy'][3],
                               equal_after_restore=tuple(before['numpy'][3:]) == tuple(after['numpy'][3:])),
    )
    report['restored_streams_equal'] = all(report[k] for k in (
        'torch_cpu', 'cuda', 'sampler', 'python_mersenne_twister', 'numpy_mersenne_twister'))
    report['complete_rng_state_equal'] = (report['restored_streams_equal']
                                          and report['python_gauss_cache']['equal_after_restore']
                                          and report['numpy_gauss_cache']['equal_after_restore'])
    return report


def save_sampler_state(generator, training_state_dir):
    from safetensors.torch import save_file
    save_file({'sampler_generator_state': generator.get_state().clone()},
              str(Path(training_state_dir) / 'sampler_generator_state.safetensors'))


def load_sampler_state(generator, training_state_dir):
    from safetensors.torch import load_file
    generator.set_state(load_file(str(Path(training_state_dir) / 'sampler_generator_state.safetensors'))
                        ['sampler_generator_state'].clone())


def micro_batch_weights(action_mask, micro_batch_size):
    """Exact loss weights so that micro-batches add up to the batch loss.

    The GR00T flow-matching loss is ``sum(mask * mse) / (sum(mask) + 1e-6)``
    over the whole batch. Rows have different valid-action counts (episode ends
    are padded), so averaging micro-batch losses would reweight rows. Weighting
    micro-batch ``j`` by ``(m_j + 1e-6) / (M + 1e-6)`` gives
    ``sum_j s_j / (M + 1e-6)``: the batch loss and gradient formula.
    """
    import torch
    totals = action_mask.reshape(action_mask.shape[0], -1).double().sum(dim=1)
    whole = float(totals.sum())
    weights = []
    for start in range(0, action_mask.shape[0], micro_batch_size):
        part = float(totals[start:start + micro_batch_size].sum())
        weights.append((part + 1e-6) / (whole + 1e-6))
    return weights


def memory_record(label):
    """GPU allocator statistics plus process RSS/peak RSS (MiB)."""
    import torch
    record = dict(stage=label)
    try:
        with open('/proc/self/status') as status:
            for line in status:
                key = line.split(':')[0]
                if key in ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile'):
                    record[key + '_mib'] = int(line.split()[1]) // 1024
    except OSError:
        pass
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        record.update(
            cuda_allocated_gib=torch.cuda.memory_allocated() / 2**30,
            cuda_reserved_gib=torch.cuda.memory_reserved() / 2**30,
            cuda_peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            cuda_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            device_free_gib=free / 2**30, device_total_gib=total / 2**30)
    return record


def load_policy(path, revision, device='cuda'):
    """``GrootPolicy.from_pretrained`` as in the original trainer.

    The recorded base_model_path is rebound by student/portable_launch.py (the
    course launcher; no second override here). This only stops early with a
    clear message if the trainer is started without the launcher on a machine
    that lacks the recorded path, and checks the pinned revision.
    """
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    config = PreTrainedConfig.from_pretrained(path)
    base = str(config.base_model_path)
    if not Path(base).is_dir():
        raise SystemExit('GR00T base model not found at ' + base +
                         '; start training through lab/groot_walking/student/portable_launch.py')
    if '/snapshots/' in base and base.rstrip('/').split('/')[-1] != revision:
        raise SystemExit('GR00T base model revision differs from ' + revision + ': ' + base)
    policy = GrootPolicy.from_pretrained(path, config=config).to(device)
    return policy, dict(base_model_path=base)
