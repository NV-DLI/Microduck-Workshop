"""Bounded tracking: Microduck simulation and control."""

import json, copy
from pathlib import Path
import torch
from isaaclab_tasks.contrib.microduck.mdp.actions import BiasedJointPositionAction
from bounded_actions import BoundedActions, load_limits, JOINT_NAMES
from tracking_rewards import camera_angular_error


def T(x):
    return getattr(x, "torch", x)


class BoundedTrainingAction(BiasedJointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.passive_ids, _ = self._asset.find_joints(
            ["passive_" + n + "_backlash" for n in JOINT_NAMES], preserve_order=True
        )
        self.limiter = BoundedActions(
            load_limits()["physical_output_limits_rad"],
            margin_rad=0.035,
            max_target_rate_rad_s=8.0,
        )
        self.target_excess = torch.zeros(self.num_envs, device=self.device)

    def process_actions(self, actions):
        q = T(self._asset.data.joint_pos)
        executed, diagnostic = self.limiter.apply(
            actions,
            scale=self._scale,
            offset=self._offset,
            encoder_bias=self.encoder_bias,
            q_servo=q[:, self._joint_ids],
            q_passive=q[:, self.passive_ids],
            servo_limits=T(self._asset.data.joint_pos_limits)[:, self._joint_ids],
        )
        self.target_excess = (
            (
                diagnostic["requested_output_target_rad"]
                - diagnostic["executed_output_target_rad"]
            )
            .square()
            .sum(-1)
        )
        super().process_actions(executed)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, "limiter") and self.limiter.previous is not None:
            ids = slice(None) if env_ids is None else env_ids
            q = T(self._asset.data.joint_pos)
            self.limiter.previous[ids] = (
                q[:, self._joint_ids] + q[:, self.passive_ids]
            )[ids]


def filtered_velocity(env):
    counter = env.common_step_counter
    if getattr(env, "_course_velocity_step", None) != counter:
        robot = env.scene["robot"]
        v = torch.cat(
            [T(robot.data.root_lin_vel_b)[:, :2], T(robot.data.root_ang_vel_b)[:, 2:3]],
            dim=-1,
        )
        if not hasattr(env, "_course_velocity_mean"):
            env._course_velocity_mean = v.clone()
        else:
            # alpha=1-exp(-20ms/200ms); do not average the raw evaluation trace.
            env._course_velocity_mean.lerp_(v, 0.09516258196404048)
            reset = env.episode_length_buf <= 1
            env._course_velocity_mean[reset] = v[reset]
        env._course_velocity_step = counter
    return env._course_velocity_mean


def track_planar(env, std=0.08):
    error = (
        (
            filtered_velocity(env)[:, :2]
            - env.command_manager.get_command("base_velocity")[:, :2]
        )
        .square()
        .sum(-1)
    )
    return torch.exp(-error / (std * std))


def track_yaw(env, std=0.2):
    error = (
        filtered_velocity(env)[:, 2]
        - env.command_manager.get_command("base_velocity")[:, 2]
    ).square()
    return torch.exp(-error / (std * std))


def raw_target_excess(env):
    return env.action_manager.get_term("joint_pos").target_excess


def configure(cfg):
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg
    import isaaclab.sim as sim_utils

    cfg.curriculum = None
    cfg.commands.base_velocity.rel_standing_envs = 0.2
    cfg.commands.base_velocity.rel_turn_in_place_envs = 0.25
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.25)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
    cfg.commands.base_velocity.resampling_time_range = (4.0, 8.0)
    cfg.commands.head_pose.ranges = (
        (-0.25, 0.25),
        (-0.45, 0.45),
        (-0.8, 0.8),
        (-0.15, 0.15),
    )
    cfg.actions.joint_pos.class_type = BoundedTrainingAction
    cfg.rewards.track_lin_vel = RewardTermCfg(
        func=track_planar, weight=8.0, params={"std": 0.08}
    )
    cfg.rewards.track_ang_vel = RewardTermCfg(
        func=track_yaw, weight=5.0, params={"std": 0.2}
    )
    cfg.rewards.camera_angular = RewardTermCfg(
        func=camera_angular_error,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["jaw_soft"])},
    )
    cfg.rewards.raw_target_excess = RewardTermCfg(func=raw_target_excess, weight=-20.0)
    cfg.rewards.body_ang_vel.weight = -0.05
    cfg.rewards.action_rate_l2.weight = -0.05
    cfg.rewards.dof_pos_limits.weight = -10.0
    cfg.rewards.head_pose_tracking.weight = 2.0
    cfg.rewards.head_pose_bias.weight = -3.0
    # Compiled mass matrices have been independently verified for these16
    # variants. Runtime CoM events are disabled, not falsely counted as applied.
    manifest = (
        Path(__file__).parent.parent
        / "skills/training/mass_properties/generated/robot_configuration/manifest.json"
    )
    variants = json.loads(manifest.read_text())["variants"]
    spawns = []
    for variant in variants:
        spawn = copy.deepcopy(cfg.scene.robot.spawn)
        spawn.usd_path = variant["usd"]
        spawns.append(spawn)
    cfg.scene.robot.spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=spawns)
    cfg.events.randomize_com = None
    cfg.events.randomize_head_com = None
    return cfg
