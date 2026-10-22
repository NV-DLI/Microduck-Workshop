"""Restart stochastic exploration inside its bounds, preserving policy means.

Scalar standard deviations inherited above a newly tightened clamp have zero
direct gradient. Reset only that parameter and its Adam moments. This is an
explicit training change, never an inference-model edit.
"""
import hashlib
import torch


def _mean_digest(actor):
    digest = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        if name == 'distribution.std_param':
            continue
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def reset_exploration(algorithm, initial_std=.06):
    actor = algorithm.actor
    distribution = actor.distribution
    parameter = distribution.std_param
    lower, upper = distribution.std_range
    if parameter.shape != (14,) or not parameter.requires_grad:
        raise ValueError('Expected fourteen learnable scalar noise parameters')
    if not lower < initial_std < upper:
        raise ValueError('Initial exploration must be strictly inside the clamp')
    before = _mean_digest(actor)
    old = parameter.detach().cpu().tolist()
    old_state = algorithm.optimizer.state.pop(parameter, None)
    with torch.no_grad():
        parameter.fill_(initial_std)
    after = _mean_digest(actor)
    if before != after:
        raise RuntimeError('Reset unexpectedly changed policy means or normalizer')
    probe = parameter.detach().clone().requires_grad_(True)
    probe.clamp(lower, upper).sum().backward()
    if not torch.equal(probe.grad, torch.ones_like(probe)):
        raise RuntimeError('Exploration gradient remains saturated')
    return {
        'old_stored_std_rad': old,
        'initial_std_rad': parameter.detach().cpu().tolist(),
        'range_rad': [lower, upper],
        'noise_optimizer_moments_cleared': old_state is not None,
        'actor_mean_and_normalizer_sha256_before': before,
        'actor_mean_and_normalizer_sha256_after': after,
        'all_non_noise_actor_tensors_unchanged': True,
        'noise_gradient_channels_nonzero': int(torch.count_nonzero(probe.grad)),
    }
