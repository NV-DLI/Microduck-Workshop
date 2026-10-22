"""Temporarily omit the last PPO Gaussian batch cache while copying an actor."""
from contextlib import contextmanager

@contextmanager
def suspend_gaussian_batch_cache(policy):
    distribution=policy.distribution
    if type(distribution).__name__!='GaussianDistribution':
        raise TypeError('Only the pinned RSL GaussianDistribution cache contract is supported')
    cached=distribution._distribution
    # This Normal contains the final optimizer minibatch graph. It is neither
    # a model parameter nor a buffer and is unused by deterministic_output().
    # Removing it only during export avoids deepcopy(non-leaf Tensor), without
    # altering saved weights, normalizer, learned std or any PPO computation.
    distribution._distribution=None
    try:yield
    finally:distribution._distribution=cached

