"""Fresh native-action BODY50 experiment, without a learned initialization."""
import copy
import json
import math
from pathlib import Path
import torch

M = Path(__file__).resolve().parents[3]


def actual_fall(env):
    from environment_hook import T
    robot = env.scene['robot']
    return env.task_active_mask & ((T(robot.data.projected_gravity_b)[:, 2] > -math.cos(.8)) |
                                   (T(robot.data.root_pos_w)[:, 2] < .07))


def configure(cfg):
    from isaaclab.managers import TerminationTermCfg
    import isaaclab.sim as sim_utils
    from isaaclab_tasks.contrib.microduck.mdp.actions import BiasedJointPositionAction
    from environment_hook import startup_failed
    manifest = M/'skills/training/mass_properties/generated/robot_configuration/manifest.json'
    variants = json.loads(manifest.read_text())['variants']
    assets = []
    for variant in variants:
        spawn = copy.deepcopy(cfg.scene.robot.spawn)
        spawn.usd_path = variant['usd']
        assets.append(spawn)
    cfg.scene.robot.spawn = sim_utils.MultiAssetSpawnerCfg(assets_cfg=assets)
    cfg.events.randomize_com = cfg.events.randomize_head_com = None
    # CoM is already compiled into each mass matrix. Other upstream curricula
    # and reward schedules remain at their actual fresh-training iteration.
    cfg.curriculum.com_range = cfg.curriculum.head_com_range = None
    cfg.actions.joint_pos.class_type = BiasedJointPositionAction
    cfg.actions.joint_pos.scale = 1.
    cfg.actions.joint_pos.clip = None
    cfg.terminations.fell_over = TerminationTermCfg(func=actual_fall, time_out=False)
    cfg.terminations.beta_startup_failed = TerminationTermCfg(func=startup_failed, time_out=False)
    cfg.sim.dt = .005
    cfg.decimation = 4
    cfg.sim.render_interval = 4
    cfg.commands.base_velocity.debug_vis = False
    return cfg


def install_task_only_authority(hook):
    """Keep native raw-zero home targets until physical release is complete.

    Sensor readiness remains untouched inside the fixture. Only the public
    permission to invoke the new random actor is narrower during preparation.
    The same ready/task mask drives inference, PPO and normalizer selection.
    """
    def sync():
        hook.env.ready_mask = hook.fixture.ready & hook.fixture.task_active
        hook.env.task_active_mask = hook.fixture.task_active
    hook._sync_masks = sync
    sync()

