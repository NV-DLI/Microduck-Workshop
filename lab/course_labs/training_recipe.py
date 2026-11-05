"""Keep the course reward and PPO recipe explicit when upstream defaults evolve.

Apply after resolving the pinned upstream task and before the lab's own overrides.
Framework classes, native BAM dynamics and observation/action implementations remain
those of the installed Isaac Lab revision.
"""
def apply_official_walking_recipe(cfg, agent):
    """Use the pinned official recipe, emphasizing forward, zero-yaw commands.

    Keep official head commands, head tracking/bias, posture gating and curricula.
    The only task additions are fall detection and more straight-command samples;
    symmetry uses the official optional augmentation, with no mirror loss.
    """
    from course_labs.training_safety import apply_training_safety
    from isaaclab_tasks.contrib.microduck.agents.rsl_rl_ppo_cfg import MicroDuckSymmetricPPORunnerCfg

    apply_training_safety(cfg)
    cfg.commands.base_velocity.rel_forward_envs = 0.6
    agent.algorithm.symmetry_cfg = MicroDuckSymmetricPPORunnerCfg().algorithm.symmetry_cfg
    agent.experiment_name = "microduck_official_straight_backlash"
    return cfg, agent
