"""Course fine-tuning proposal: precise low-speed motion and a physical head.

These are training rewards, allowed to use simulator state. Policy observations
remain the same 61 sensor/command entries. No image stabilization is applied.
This recipe is processing until its independent physical tests pass.
"""

import torch


def camera_angular_error(env, asset_cfg):
    robot = env.scene[asset_cfg.name]
    omega = robot.data.body_ang_vel_w.torch[:, asset_cfg.body_ids[0], :]
    target = torch.zeros_like(omega)
    target[:, 2] = env.command_manager.get_command("base_velocity")[:, 2]
    # Angular velocity is the same at the camera optical site and its rigid head.
    return torch.sum((omega - target).square(), dim=1)


def planar_error(env):
    actual = env.scene["robot"].data.root_lin_vel_b.torch[:, :2]
    target = env.command_manager.get_command("base_velocity")[:, :2]
    return torch.sum((actual - target).square(), dim=1)


def configure(cfg):
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg

    cfg.curriculum = None
    cfg.commands.base_velocity.rel_standing_envs = 0.25
    cfg.commands.base_velocity.rel_turn_in_place_envs = 0.2
    cfg.commands.base_velocity.ranges.lin_vel_x = (-0.15, 0.25)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.7, 0.7)
    cfg.commands.base_velocity.resampling_time_range = (4.0, 8.0)
    cfg.commands.head_pose.ranges = (
        (-0.45, 0.45),
        (-0.6, 0.6),
        (-0.8, 0.8),
        (-0.15, 0.15),
    )
    cfg.rewards.track_lin_vel.weight = 8.0
    cfg.rewards.track_lin_vel.params["std"] = 0.08
    cfg.rewards.track_ang_vel.weight = 5.0
    cfg.rewards.track_ang_vel.params["std"] = 0.2
    cfg.rewards.precision_planar = RewardTermCfg(func=planar_error, weight=-8.0)
    cfg.rewards.body_ang_vel.weight = -0.3
    cfg.rewards.camera_angular = RewardTermCfg(
        func=camera_angular_error,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["jaw_soft"])},
    )
    cfg.rewards.dof_pos_limits.weight = -5.0
    cfg.rewards.action_rate_l2.weight = -0.5
    cfg.rewards.head_pose_tracking.weight = 2.0
    cfg.rewards.head_pose_bias.weight = -3.0
    # Retain the official final mass/CoM range, with the documented upstream
    # runtime-CoM recomputation limitation. No field-only DR claim is made.
    cfg.events.randomize_com.params["com_range"] = {
        axis: (-0.015, 0.015) for axis in "xyz"
    }
    cfg.events.randomize_head_com.params["com_range"] = {
        axis: (-0.01, 0.01) for axis in "xyz"
    }
    return cfg
