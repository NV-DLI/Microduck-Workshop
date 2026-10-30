"""Adapt the official roller (passive-wheel) actor to mixed flat and sloped terrain.

Course recipe for Section 3.3. It reuses the walking terrain trainer's terrain mix,
warm start and export (lab/train_terrain.py, unchanged) and swaps in the official
roller feet with configure_rollers(). configure_rollers() removes every walking
reward on purpose, because foot-height and air-time terms would measure the ankle
blocks; this module therefore defines its own small roller reward set:

  upright, action rate, joint limits, body angular velocity   (task terms, restored)
  roller_push_progress  forward body speed while a push is requested
  roller_coast_hold     forward/backward speed while no push is requested
  roller_lateral_drift  sideways body speed
  yaw_rate_tracking     zero turn rate (commands keep heading fixed)

The roller command means push amount (x > 0), coast (x = 0); it is not a speed in
m/s, so these rewards use body speed only as a progress signal. The course completed
one 1000-iteration run of this recipe; none of its five +5-degree wedge evaluations
passed. It is retained as a negative reward-design example, not a qualified slope policy.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))


def _speed(env):
    velocity = env.scene["robot"].data.root_link_lin_vel_b
    return getattr(velocity, "torch", velocity)


def roller_push_progress(env, limit=0.5):
    """Forward body speed (clipped) scaled by the requested push amount."""
    import torch

    push = env.command_manager.get_command("base_velocity")[:, 0].clamp(min=0.0)
    return push * torch.clamp(_speed(env)[:, 0], 0.0, limit) / limit


def roller_coast_hold(env):
    """Squared forward speed while no push is requested (controlled coasting on slopes)."""
    coast = env.command_manager.get_command("base_velocity")[:, 0].abs() < 1e-6
    return coast.float() * _speed(env)[:, 0].square()


def roller_lateral_drift(env):
    return _speed(env)[:, 1].square()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True, help="official roller ONNX actor to start from")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=47001)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import onnx
    import gymnasium as gym
    from onnx import numpy_helper
    from isaaclab.app import launch_simulation
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config
    from isaaclab.utils import to_dict
    from isaaclab.utils.io import dump_yaml
    from isaaclab.managers import RewardTermCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    import isaaclab.terrains as terrain_gen

    from train_terrain import compact_onnx, yaw_rate_tracking
    from roller_config import configure_rollers, check_roller_articulation

    sys.path.insert(0, str(a.runtime / "gait"))
    from batch_policy import BatchPolicy

    sys.path.insert(0, str(a.runtime / "skills/replay"))
    from export_cache import suspend_gaussian_batch_cache

    cfg, agent = resolve_task_config(
        "IsaacContrib-Velocity-Rough-MicroDuck",
        "rsl_rl_cfg_entry_point",
        overrides=["presets=backlash"],
    )
    # Keep the task's general stability terms; configure_rollers() clears all rewards.
    kept = {n: getattr(cfg.rewards, n) for n in ("upright", "action_rate_l2", "dof_pos_limits", "body_ang_vel")}
    roller_profile = configure_rollers(cfg)
    # The rough task adds two terrain rays under each walking sole (ankle_left/ankle_right)
    # for the walking critic and foot-clearance rewards. The roller body has ankle_l_v1 /
    # ankle_r_v1 with tyres instead, and configure_rollers() removed those consumers, so
    # the rays are dropped; they are sensors only and change no physics.
    cfg.scene.left_foot_height = None
    cfg.scene.right_foot_height = None
    # configure_rollers() also removes the walking critic group (sole heights and foot
    # bodies the roller body does not have). The fresh critic therefore reads the same
    # observation group as the actor.
    agent.obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    for name, term in kept.items():
        setattr(cfg.rewards, name, term)
    cfg.rewards.roller_push_progress = RewardTermCfg(func=roller_push_progress, weight=3.0)
    cfg.rewards.roller_coast_hold = RewardTermCfg(func=roller_coast_hold, weight=-1.0)
    cfg.rewards.roller_lateral_drift = RewardTermCfg(func=roller_lateral_drift, weight=-2.0)
    cfg.rewards.yaw_precision = RewardTermCfg(func=yaw_rate_tracking, weight=1.0, params={"std": 0.15})

    cfg.seed = agent.seed = a.seed
    cfg.scene.num_envs = a.num_envs
    cfg.sim.device = agent.device = "cuda:0"
    cfg.commands.base_velocity.debug_vis = False
    # Push amount in [0.3, 0.6] (0.6 is the course keyboard/terrain push); fixed heading.
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.6)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.rel_forward_envs = 0.9
    cfg.commands.base_velocity.rel_turn_in_place_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.1
    cfg.commands.head_pose.ranges = tuple((0.0, 0.0) for _ in range(4))
    cfg.commands.body_pose.ranges = tuple((0.0, 0.0) for _ in range(6))
    cfg.events.push_robot = None

    # Same terrain mix as the walking terrain lab; curriculum adapts difficulty.
    cfg.scene.terrain.terrain_generator = terrain_gen.TerrainGeneratorCfg(
        seed=a.seed,
        size=(4.0, 4.0),
        num_rows=5,
        num_cols=8,
        curriculum=True,
        border_width=2.0,
        horizontal_scale=0.025,
        vertical_scale=0.001,
        sub_terrains={
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
            "slopes": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.375, slope_range=(0.03, 0.22), platform_width=0.7
            ),
            "descending": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.375, slope_range=(0.03, 0.22), platform_width=0.7
            ),
        },
    )
    cfg.scene.terrain.max_init_terrain_level = 1
    agent.actor.distribution_cfg.init_std = 0.15
    agent.algorithm.learning_rate = 5e-5
    agent.algorithm.schedule = "fixed"
    agent.algorithm.entropy_coef = 0.001
    agent.save_interval = 250
    agent.logger = "tensorboard"
    cfg.log_dir = str(a.output)
    for name in ("action_rate_weight", "head_pose_bias_weight", "head_pose_range", "standing_envs"):
        if hasattr(cfg.curriculum, name):
            setattr(cfg.curriculum, name, None)

    digest = hashlib.sha256(a.model.read_bytes()).hexdigest()
    ledger = dict(
        status="starting",
        locomotion="rollers",
        recipe="course roller slope adaptation v1 (own reward set; not a reference result)",
        initialization="official roller ONNX actor; fresh critic and optimizer",
        critic_observations="policy group (same inputs as the actor)",
        parent_sha256=digest,
        seed=a.seed,
        num_envs=a.num_envs,
        iterations=a.iterations,
        training_slopes="pyramids, tan(angle)=0.03..0.22; 25% flat",
        roller_configuration=roller_profile,
        original_weights_modified=False,
    )

    def save_report():
        (a.output / "training-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")

    save_report()
    with launch_simulation(cfg, {"device": "cuda:0", "visualizer": []}):
        base = gym.make("IsaacContrib-Velocity-Rough-MicroDuck", cfg=cfg).unwrapped
        env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
        ledger["articulation"] = check_roller_articulation(
            base.scene["robot"], env.get_observations()["policy"]
        )
        runner = OnPolicyRunner(env, to_dict(agent), log_dir=str(a.output), device="cuda:0")
        actor = runner.alg.actor

        # Warm-start actor weights and observation normalization from the official ONNX.
        model = onnx.load(str(a.model))
        constants = {x.name: numpy_helper.to_array(x).copy() for x in model.graph.initializer}
        state = actor.state_dict()
        for name in state:
            if name in constants:
                state[name] = torch.from_numpy(constants[name]).to(state[name])
        divisor = next(constants[n.input[1]] for n in model.graph.node if n.op_type == "Div")
        state["obs_normalizer._std"] = torch.from_numpy(divisor - actor.obs_normalizer.eps).to(
            state["obs_normalizer._std"]
        )
        state["obs_normalizer._var"] = state["obs_normalizer._std"].square()
        actor.load_state_dict(state)
        actor.obs_normalizer.until = 0
        observations = env.get_observations()
        reference = BatchPolicy(a.model, digest, "cuda:0")
        actor.eval()
        with torch.no_grad():
            err = (actor(observations, stochastic_output=False) - reference(observations["policy"])).abs().max().item()
        if err > 1e-5:
            raise RuntimeError("Warm start numerical mismatch " + str(err))
        ledger["initial_actor_max_abs_error"] = err

        dump_yaml(str(a.output / "env.yaml"), cfg)
        dump_yaml(str(a.output / "agent.yaml"), agent)
        runner.save(str(a.output / "initial.pt"))
        start = time.monotonic()
        ledger["status"] = "training"
        save_report()

        def export(directory):
            from onnx.external_data_helper import convert_model_from_external_data

            with suspend_gaussian_batch_cache(actor):
                runner.export_policy_to_onnx(str(directory), "policy.onnx")
            path = directory / "policy.onnx"
            graph = onnx.load(str(path), load_external_data=True)
            convert_model_from_external_data(graph)
            compact_onnx(graph)
            onnx.save_model(graph, str(path), save_as_external_data=False)
            onnx.checker.check_model(str(path))

        original_save = runner.save

        def snapshot(path, infos=None):
            original_save(path, infos)
            export(a.output / ("export-" + Path(path).stem))
            ledger.update(latest_checkpoint=str(path), training_wall_seconds=time.monotonic() - start)
            save_report()

        runner.save = snapshot
        try:
            runner.learn(num_learning_iterations=a.iterations, init_at_random_ep_len=True)
            export(a.output / "exported")
            ledger.update(status="completed", training_wall_seconds=time.monotonic() - start)
        except BaseException as error:
            ledger.update(
                status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error=repr(error),
                training_wall_seconds=time.monotonic() - start,
            )
            raise
        finally:
            save_report()
            env.close()


if __name__ == "__main__":
    main()
