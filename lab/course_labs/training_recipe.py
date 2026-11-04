"""Keep the course reward and PPO recipe explicit when upstream defaults evolve.

Apply after resolving the pinned upstream task and before the lab's own overrides.
Framework classes, native BAM dynamics and observation/action implementations remain
those of the installed Isaac Lab revision.
"""
from copy import deepcopy
from math import sqrt

# Environment steps (24 per PPO rollout), matching the course's recorded recipe.
_SCHEDULES = {'action_rate_weight': [[12000, -0.2], [18000, -0.4], [24000, -0.6], [30000, -0.8], [36000, -1.0]],
 'head_pose_bias_weight': [[14400, -1.0], [24000, -2.0], [36000, -3.0]],
 'standing_envs': [[12000, 0.05], [18000, 0.1], [24000, 0.15], [36000, 0.2], [48000, 0.25]],
 'head_pose_range': [[12000, [[-0.17, 0.17], [-0.17, 0.17], [-0.21, 0.21], [-0.047, 0.047]]],
                     [24000, [[-0.39, 0.39], [-0.39, 0.39], [-0.49, 0.49], [-0.11, 0.11]]],
                     [36000, [[-0.72, 0.72], [-0.72, 0.72], [-0.91, 0.91], [-0.2, 0.2]]],
                     [48000, [[-1.1, 1.1], [-1.1, 1.1], [-1.4, 1.4], [-0.31, 0.31]]]],
 'com_range': [[12000, {'x': [-0.005, 0.005], 'y': [-0.005, 0.005], 'z': [-0.005, 0.005]}],
               [24000, {'x': [-0.01, 0.01], 'y': [-0.01, 0.01], 'z': [-0.01, 0.01]}],
               [36000, {'x': [-0.015, 0.015], 'y': [-0.015, 0.015], 'z': [-0.015, 0.015]}]],
 'head_com_range': [[12000, {'x': [-0.005, 0.005], 'y': [-0.005, 0.005], 'z': [-0.005, 0.005]}],
                    [24000, {'x': [-0.01, 0.01], 'y': [-0.01, 0.01], 'z': [-0.01, 0.01]}]]}


def apply_course_recipe(cfg, agent, *, rough=False):
    from isaaclab_tasks.contrib.microduck.mdp.rewards import (
        track_linear_velocity, track_angular_velocity,
    )
    cfg.rewards.track_lin_vel.func = track_linear_velocity
    cfg.rewards.track_lin_vel.weight = 2.0
    cfg.rewards.track_lin_vel.params["std"] = sqrt(0.1)
    cfg.rewards.track_ang_vel.func = track_angular_velocity
    cfg.rewards.track_ang_vel.weight = 2.0
    cfg.rewards.track_ang_vel.params["std"] = sqrt(0.5)
    cfg.rewards.pose.params["standing_only"] = False
    for name, stages in _SCHEDULES.items():
        getattr(cfg.curriculum, name).params["modify_params"]["stages"] = deepcopy(stages)
    agent.num_steps_per_env = 24
    agent.max_iterations = 50000
    agent.save_interval = 250
    agent.algorithm.symmetry_cfg = None
    agent.experiment_name = "microduck_velocity_rough_backlash" if rough else "microduck_velocity_flat_backlash"
    if rough:
        from isaaclab_tasks.contrib.microduck.mdp.commands import MicroDuckVelocityCommand
        from isaaclab_tasks.core.velocity.mdp.curriculums import terrain_levels_vel
        cfg.commands.base_velocity.class_type = MicroDuckVelocityCommand
        cfg.curriculum.terrain_levels.func = terrain_levels_vel
        cfg.scene.num_envs = 4096
    return cfg, agent
