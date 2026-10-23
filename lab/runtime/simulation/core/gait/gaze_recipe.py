"""Gaze recipe: Microduck simulation and control."""

import json
from pathlib import Path
import torch
from isaaclab_tasks.contrib.microduck.mdp.commands import UniformPoseDeltaCommand
from exploration_curriculum import configure as previous_configure
from tracking_rewards import camera_angular_error


class GazeCommand(UniformPoseDeltaCommand):
    """Thirty verified IK targets, slewed through real joint command space."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        data = json.loads(
            Path(__file__).with_name("gaze-command-table.json").read_text()
        )
        if data["head_joint_order"] != [
            "neck_pitch",
            "head_pitch",
            "head_yaw",
            "head_roll",
        ]:
            raise ValueError("Explicit head command channel order required")
        self.table = torch.tensor(
            [r["head_pose_native_offsets_rad"] for r in data["rows"]],
            device=self.device,
        )
        self.desired = torch.zeros_like(self._command)
        self.delta_velocity = torch.zeros_like(self._command)
        self.period = float(env.step_dt)

    def _resample_command(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if not len(env_ids):
            return
        indices = torch.randint(len(self.table), (len(env_ids),), device=self.device)
        # Preserve an explicit neutral-head bucket, while remaining targets
        # include yaw±.4 and down-gaze through1.4rad, unlike the zero-only v5.
        indices[torch.rand(len(indices), device=self.device) < 0.25] = 2
        self.desired[env_ids] = self.table[indices]

    def _update_command(self):
        delta = (self.desired - self._command).clamp(
            -0.3 * self.period, 0.3 * self.period
        )
        self.delta_velocity = delta / self.period
        self._command += delta

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._command[ids] = 0.0
        self.desired[ids] = 0.0
        self.delta_velocity[ids] = 0.0
        return super().reset(env_ids)


def camera_error_when_gaze_settled(env, asset_cfg):
    term = env.command_manager.get_term("head_pose")
    # Intentional gaze changes must remain possible. Penalize world shake after
    # the requested gaze settles, not the commanded act of looking at the ball.
    settled = (term.delta_velocity.abs().amax(dim=-1) < 0.03).float()
    return camera_angular_error(env, asset_cfg) * settled


def configure(cfg):
    previous_configure(cfg)
    cfg.commands.head_pose.class_type = GazeCommand
    cfg.commands.head_pose.resampling_time_range = (5.0, 9.0)
    table = json.loads(Path(__file__).with_name("gaze-command-table.json").read_text())
    rows = [r["head_pose_native_offsets_rad"] for r in table["rows"]]
    cfg.commands.head_pose.ranges = tuple(
        (min(r[i] for r in rows), max(r[i] for r in rows)) for i in range(4)
    )
    cfg.rewards.head_pose_tracking.weight = 10.0
    cfg.rewards.head_pose_tracking.params["std"] = 0.20
    cfg.rewards.head_pose_bias.weight = -6.0
    cfg.rewards.camera_angular.func = camera_error_when_gaze_settled
    cfg.rewards.camera_angular.weight = -0.15
    return cfg


def configure_agent(agent):
    # Cap stochastic exploration while preserving the loaded actor tensors.
    # This is a declared refinement, not a random-restart training claim.
    agent.actor.distribution_cfg.std_range = (0.03, 0.10)
    agent.algorithm.entropy_coef = 0.0
    return agent
