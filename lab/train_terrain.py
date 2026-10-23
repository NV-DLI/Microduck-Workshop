"""Adapt a preserved ONNX walking actor to mixed flat and sloped terrain."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time


def compact_onnx(message):
    """Remove exporter debug metadata without changing graph operations or tensors."""
    for key in ("doc_string", "metadata_props"):
        if key in message.DESCRIPTOR.fields_by_name:
            message.ClearField(key)
    for field, value in message.ListFields():
        if field.message_type is not None:
            if field.is_repeated:
                for child in value:
                    compact_onnx(child)
            else:
                compact_onnx(value)


def yaw_rate_tracking(env, std=0.15):
    """Reward turning accuracy independently of gait roll and pitch motion."""
    import torch

    velocity = env.scene["robot"].data.root_link_ang_vel_b
    velocity = getattr(velocity, "torch", velocity)
    target = env.command_manager.get_command("base_velocity")[:, 2]
    return torch.exp(-((velocity[:, 2] - target) / std).square())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--yaw-precision", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--refine-heading", action="store_true")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--seed", type=int, default=43001)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import onnx
    import gymnasium as gym
    import numpy as np
    from onnx import numpy_helper
    from isaaclab.app import launch_simulation
    import isaaclab_tasks
    from isaaclab_tasks.utils import resolve_task_config
    from isaaclab.utils import to_dict
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    import isaaclab.terrains as terrain_gen
    from tensordict import TensorDict

    sys.path.insert(0, str(a.runtime / "gait"))
    from batch_policy import BatchPolicy

    sys.path.insert(0, str(a.runtime / "skills/replay"))
    from export_cache import suspend_gaussian_batch_cache

    cfg, agent = resolve_task_config(
        "IsaacContrib-Velocity-Rough-MicroDuck",
        "rsl_rl_cfg_entry_point",
        overrides=["presets=backlash"],
    )
    cfg.seed = agent.seed = a.seed
    cfg.scene.num_envs = a.num_envs
    cfg.sim.device = agent.device = "cuda:0"
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.15, 0.3)
    cfg.commands.base_velocity.ranges.lin_vel_y = (-0.05, 0.05)
    cfg.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)
    cfg.commands.base_velocity.rel_forward_envs = 0.8
    cfg.commands.base_velocity.rel_turn_in_place_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 0.1
    cfg.commands.head_pose.ranges = tuple((0.0, 0.0) for _ in range(4))
    cfg.commands.body_pose.ranges = tuple((0.0, 0.0) for _ in range(6))
    cfg.events.push_robot = None

    # Narrow the experiment to direction control under nominal physical parameters.
    if a.refine_heading:
        cfg.rewards.track_ang_vel.weight = 4.0
        cfg.rewards.track_ang_vel.params["std"] = 0.2
        cfg.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        cfg.commands.base_velocity.rel_standing_envs = 0.2
        for name in [
            "foot_friction",
            "encoder_bias",
            "mass_inertia",
            "randomize_com",
            "randomize_head_com",
            "randomize_joint_friction",
            "randomize_armature",
        ]:
            setattr(cfg.events, name, None)
        cfg.scene.robot.actuators["servos"].vin_range = None
        cfg.scene.robot.actuators["servos"].vin_drop_gain_range = None
        cfg.scene.robot.actuators["servos"].min_delay = 4
        cfg.scene.robot.actuators["servos"].max_delay = 4
        cfg.observations.policy.enable_corruption = False
        cfg.observations.policy.imu.params.update(max_angle_deg=0, min_lag=0, max_lag=0)
        cfg.curriculum.com_range = None
        cfg.curriculum.head_com_range = None

    # Isolate body-z tracking from roll/pitch, while retaining the original stability terms.
    if a.yaw_precision:
        from isaaclab.managers import RewardTermCfg

        cfg.rewards.track_ang_vel.weight = 0.5
        cfg.rewards.track_ang_vel.params["std"] = 0.31622776601683794
        cfg.rewards.yaw_precision = RewardTermCfg(
            func=yaw_rate_tracking, weight=4.0, params={"std": 0.15}
        )
        cfg.rewards.track_lin_vel.weight = 3.0

    # Mix flat, ascending and descending tiles; curriculum controls their difficulty.
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
    # Keep the initial action-smoothness and head-command contract during adaptation.
    cfg.curriculum.action_rate_weight = None
    cfg.curriculum.head_pose_bias_weight = None
    cfg.curriculum.head_pose_range = None
    cfg.curriculum.standing_envs = None
    digest = hashlib.sha256(a.model.read_bytes()).hexdigest()
    ledger = dict(
        status="starting",
        initialization="official ONNX actor; fresh critic and optimizer",
        parent_sha256=digest,
        seed=a.seed,
        num_envs=a.num_envs,
        iterations=a.iterations,
        training_slopes="pyramids, tan(angle)=0.03..0.22; 25% flat",
        evaluation_seeds_excluded=[42001, 44001],
        original_weights_modified=False,
        yaw_precision=a.yaw_precision,
    )

    def save_report():
        (a.output / "training-ledger.json").write_text(
            json.dumps(ledger, indent=2) + "\n"
        )

    save_report()
    with launch_simulation(cfg, {"device": "cuda:0", "visualizer": []}):
        base = gym.make("IsaacContrib-Velocity-Rough-MicroDuck", cfg=cfg).unwrapped
        env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
        runner = OnPolicyRunner(
            env, to_dict(agent), log_dir=str(a.output), device="cuda:0"
        )
        actor = runner.alg.actor

        # Warm-start the actor and its normalization from the preserved ONNX file.
        # The critic and optimizer are new unless an explicit checkpoint is restored.
        constants = {
            x.name: numpy_helper.to_array(x).copy()
            for x in onnx.load(str(a.model)).graph.initializer
        }
        state = actor.state_dict()
        for name in state:
            if name in constants:
                state[name] = torch.from_numpy(constants[name]).to(state[name])
        divisor = next(
            constants[n.input[1]]
            for n in onnx.load(str(a.model)).graph.node
            if n.op_type == "Div"
        )
        state["obs_normalizer._std"] = torch.from_numpy(
            divisor - actor.obs_normalizer.eps
        ).to(state["obs_normalizer._std"])
        state["obs_normalizer._var"] = state["obs_normalizer._std"].square()
        actor.load_state_dict(state)
        actor.obs_normalizer.until = 0
        observations = env.get_observations()
        reference = BatchPolicy(a.model, digest, "cuda:0")
        actor.eval()
        with torch.no_grad():
            err = (
                (
                    actor(observations, stochastic_output=False)
                    - reference(observations["policy"])
                )
                .abs()
                .max()
                .item()
            )
        if err > 1e-5:
            raise RuntimeError("Warm start numerical mismatch " + str(err))
        ledger["initial_actor_max_abs_error"] = err

        # Continuation restores training state; never overwrite the source checkpoint.
        if a.resume:
            runner.load(str(a.resume))
            ledger.update(
                initialization="checkpoint continuation",
                parent_checkpoint_sha256=hashlib.sha256(
                    a.resume.read_bytes()
                ).hexdigest(),
                refine_heading=a.refine_heading,
            )
        actor.obs_normalizer.until = 0
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
            ledger.update(
                latest_checkpoint=str(path),
                training_wall_seconds=time.monotonic() - start,
            )
            save_report()

        runner.save = snapshot
        try:
            runner.learn(
                num_learning_iterations=a.iterations, init_at_random_ep_len=True
            )
            export(a.output / "exported")
            ledger.update(
                status="completed", training_wall_seconds=time.monotonic() - start
            )
        except BaseException as error:
            ledger.update(status="failed", error=repr(error))
            raise
        finally:
            save_report()
            env.close()


if __name__ == "__main__":
    main()
