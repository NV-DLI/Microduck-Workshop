"""A public-command-only, smooth walk/stand policy supervisor.

Both policies receive the same current61D observation and mixed raw history.
This only blends requested motor actions; it cannot write physical state,
contact forces, sensor packets or normalization parameters.
"""
import torch


class DualPolicy:
    def __init__(self, walk, stand, worlds, device, transition_seconds=.4):
        if transition_seconds < .1:
            raise ValueError('A nonzero physical transition window is required')
        self.walk, self.stand = walk, stand
        self.alpha = torch.zeros(worlds, device=device)
        self.delta = .02 / transition_seconds
        self.last_walk = torch.zeros((worlds, 14), device=device)
        self.last_stand = torch.zeros_like(self.last_walk)

    def __call__(self, observation, world_ids):
        if observation.shape != (len(world_ids), 61):
            raise ValueError('Use the complete public observation without history resets')
        # Native61 order: gyro/gravity6, q/qd28, history14, body command3,
        # head command4, body pose6. No simulator velocity or position enters.
        stationary = observation[:, 48:51].abs().amax(dim=1) < 1e-6
        target = stationary.to(self.alpha.dtype)
        old = self.alpha[world_ids]
        weight = old + (target - old).clamp(-self.delta, self.delta)
        moving = self.walk(observation)
        standing = self.stand(observation)
        self.alpha[world_ids] = weight
        self.last_walk[world_ids] = moving
        self.last_stand[world_ids] = standing
        return moving + weight[:, None] * (standing - moving)

