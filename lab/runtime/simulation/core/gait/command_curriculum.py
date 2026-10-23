"""Axis-command precision curriculum with explicit sampled command readback.

The inherited official forward bucket has a0.3m/s minimum. Changing the range
alone does not train low speeds. This sampler replaces that bucket explicitly.
"""

import torch
from isaaclab_tasks.contrib.microduck.mdp.commands import MicroDuckVelocityCommand
from cycle_tracking import (
    configure as previous_configure,
    BoundedTrainingAction,
    filtered_velocity,
)

COMMAND_CASES = (
    (0.0, 0.0, 0.0),
    (-0.025, 0.0, 0.0),
    (0.025, 0.0, 0.0),
    (-0.06, 0.0, 0.0),
    (0.06, 0.0, 0.0),
    (-0.1, 0.0, 0.0),
    (0.1, 0.0, 0.0),
    (0.0, -0.025, 0.0),
    (0.0, 0.025, 0.0),
    (0.0, -0.06, 0.0),
    (0.0, 0.06, 0.0),
    (0.0, 0.0, -0.2),
    (0.0, 0.0, 0.2),
    (0.0, 0.0, -0.4),
    (0.0, 0.0, 0.4),
)


class AxisCommand(MicroDuckVelocityCommand):
    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if not len(env_ids):
            return
        table = torch.tensor(COMMAND_CASES, device=self.device)
        ids = torch.randint(len(table), (len(env_ids),), device=self.device)
        # An additional 15% explicit zero-command bucket teaches repeated stops.
        ids[torch.rand(len(ids), device=self.device) < 0.15] = 0
        self.vel_command_b[env_ids] = table[ids]
        self.is_standing_env[env_ids] = ids == 0
        if hasattr(self, "is_heading_env"):
            self.is_heading_env[env_ids] = False
        self.sampled_case_indices = getattr(
            self,
            "sampled_case_indices",
            torch.zeros(self.num_envs, device=self.device, dtype=torch.long),
        )
        self.sampled_case_indices[env_ids] = ids


def planar_score(mean, command):
    target = command[:, :2]
    actual = mean[:, :2]
    speed = target.norm(dim=-1)
    moving = speed > 1e-6
    direction = target / speed.clamp(min=1e-6)[:, None]
    progress = ((actual * direction).sum(-1) / speed.clamp(min=0.012)).clamp(0.0, 1.0)
    tolerance = (speed * 0.2).clamp(min=0.012)
    precision = 1.0 / (1.0 + ((actual - target).square().sum(-1) / tolerance.square()))
    standing = 12.0 / (1.0 + actual.square().sum(-1) / (0.012**2))
    return torch.where(moving, 8.0 * progress + 4.0 * precision, standing)


def yaw_score(mean, command):
    target = command[:, 2]
    actual = mean[:, 2]
    speed = target.abs()
    moving = speed > 1e-6
    progress = (actual * target.sign() / speed.clamp(min=0.04)).clamp(0.0, 1.0)
    tolerance = (speed * 0.2).clamp(min=0.04)
    precision = 1.0 / (1.0 + ((actual - target) / tolerance).square())
    standing = 12.0 / (1.0 + (actual / 0.07).square())
    return torch.where(moving, 8.0 * progress + 4.0 * precision, standing)


def planar(env):
    return planar_score(
        filtered_velocity(env), env.command_manager.get_command("base_velocity")
    )


def yaw(env):
    return yaw_score(
        filtered_velocity(env), env.command_manager.get_command("base_velocity")
    )


def configure(cfg):
    from isaaclab.managers import RewardTermCfg

    previous_configure(cfg)
    cfg.commands.base_velocity.class_type = AxisCommand
    cfg.commands.base_velocity.forward_min_speed = 0.0
    cfg.commands.base_velocity.rel_forward_envs = 0.0
    cfg.commands.base_velocity.rel_turn_in_place_envs = 0.0
    cfg.commands.base_velocity.resampling_time_range = (4.0, 8.0)
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.06, 0.06)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
    cfg.commands.head_pose.ranges = tuple((0.0, 0.0) for _ in range(4))
    cfg.commands.body_pose.ranges = tuple((0.0, 0.0) for _ in range(6))
    cfg.rewards.track_lin_vel = RewardTermCfg(func=planar, weight=1.0)
    cfg.rewards.track_ang_vel = RewardTermCfg(func=yaw, weight=1.0)
    cfg.rewards.camera_angular.weight = -0.01
    return cfg
