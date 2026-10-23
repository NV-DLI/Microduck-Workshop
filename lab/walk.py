"""A bounded native-Newton walking observation lab. No automatic reset on a fall."""

from pathlib import Path
import argparse
import json
import hashlib
import time
import sys

# Keep each run in a new directory so comparisons retain their original evidence.
p = argparse.ArgumentParser()
p.add_argument("--runtime", type=Path, required=True)
p.add_argument("--model", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--velocity", nargs=3, type=float, required=True)
p.add_argument("--seconds", type=float, default=12)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--headless", action="store_true")
a = p.parse_args()
if not 0.02 <= a.seconds <= 180:
    p.error("seconds must be in [0.02,180]")
sys.path.insert(0, str(a.runtime / "gait"))
import numpy as np
import torch
import gymnasium as gym
from isaaclab.app import launch_simulation
from batch_policy import BatchPolicy

a.output.mkdir(parents=True, exist_ok=False)
digest = hashlib.sha256(a.model.read_bytes()).hexdigest()
report = dict(
    status="starting",
    seed=a.seed,
    model_sha256=digest,
    command=a.velocity,
    physics_hz=200,
    policy_hz=50,
    scope="nominal walking observation; not full-task evaluation",
)
rows = []
# GUI task imports must follow Kit startup; the headless path stays lightweight.
try:
    with launch_simulation(
        None,
        {
            "device": "cuda:0",
            "require_kit": not a.headless,
            "visualizer": [] if a.headless else ["kit"],
        },
    ):
        import isaaclab_tasks
        from isaaclab_tasks.utils import parse_env_cfg

        cfg = parse_env_cfg(
            "IsaacContrib-Velocity-Flat-MicroDuck",
            device="cuda:0",
            num_envs=1,
            overrides=["presets=backlash"],
        )
        cfg.seed = a.seed
        cfg.curriculum = None
        cfg.episode_length_s = a.seconds + 10
        if not a.headless:
            from isaaclab_visualizers.kit import KitVisualizerCfg

            cfg.sim.visualizer_cfgs = [KitVisualizerCfg()]
        cfg.scene.robot.spawn.usd_path = str(
            Path(__file__).parent / "assets/microduck_contact.usda"
        )

        from scene_style import configure_lighting

        configure_lighting(cfg)

        # Hold the requested command instead of sampling new commands during observation.
        cfg.commands.base_velocity.resampling_time_range = (1e8, 1e8)
        cfg.commands.base_velocity.debug_vis = False
        cfg.commands.head_pose.ranges = tuple((0.0, 0.0) for _ in range(4))
        cfg.commands.body_pose.ranges = tuple((0.0, 0.0) for _ in range(6))
        cfg.events.reset_base.params["pose_range"] = {}

        # Use nominal physics here; randomized training is a separate experiment.
        for name in [
            "foot_friction",
            "encoder_bias",
            "mass_inertia",
            "randomize_com",
            "randomize_head_com",
            "randomize_joint_friction",
            "randomize_armature",
            "push_robot",
        ]:
            setattr(cfg.events, name, None)
        cfg.scene.robot.actuators["servos"].vin_range = None
        cfg.scene.robot.actuators["servos"].vin_drop_gain_range = None
        cfg.scene.robot.actuators["servos"].min_delay = 4
        cfg.scene.robot.actuators["servos"].max_delay = 4
        cfg.observations.policy.enable_corruption = False
        cfg.observations.policy.imu.params.update(max_angle_deg=0, min_lag=0, max_lag=0)

        # Isaac Lab managers construct observations and translate policy actions.
        env = gym.make("IsaacContrib-Velocity-Flat-MicroDuck", cfg=cfg).unwrapped
        env.reset()
        robot = env.scene["robot"]
        manager = env.sim.backend_manager
        if manager.backend_name != "newton" or not env.sim.cfg.use_newton_actuators:
            raise RuntimeError("Native Newton actuators required")

        # Advance four physics ticks explicitly; avoid applying decimation twice.
        manager.set_decimation(1)
        env._physics_handles_decimation = False
        if not a.headless:
            env.sim.set_camera_view((1.2, -1.4, 0.9), (0.4, 0.0, 0.15))

        # Loading the ONNX actor performs inference only; no weights are updated.
        policy = BatchPolicy(a.model, digest, "cuda:0")
        cmd = env.command_manager.get_term("base_velocity")
        fallen = False
        with torch.inference_mode():
            for step in range(round(a.seconds / 0.02)):
                frame_start = time.monotonic()

                # Allow two seconds to settle before applying the requested velocity.
                cmd.is_standing_env[:] = False
                cmd.vel_command_b[:] = (
                    torch.tensor(a.velocity, device="cuda:0") if step >= 100 else 0
                )
                env.command_manager.get_term("head_pose").command[:] = 0
                env.command_manager.get_term("body_pose").command[:] = 0

                # One observation and action every 20 ms (50 Hz).
                action = policy(
                    env.observation_manager.compute(update_history=True)["policy"]
                )
                env.action_manager.process_action(action)

                # Hold that action through four 5 ms Newton steps (200 Hz).
                for tick in range(4):
                    env._sim_step_counter += 1
                    env.action_manager.apply_action()
                    env.scene.write_data_to_sim()
                    env.sim.step(render=not a.headless and tick == 3)
                    env.scene.update(0.005)
                env.episode_length_buf += 1
                env.common_step_counter += 1

                def T(x):
                    return getattr(x, "torch", x)

                # Stop on a fall rather than silently resetting a failed attempt.
                fallen = bool(
                    (T(robot.data.projected_gravity_b)[0, 2] > -0.6)
                    | (T(robot.data.root_pos_w)[0, 2] < 0.065)
                )
                if step % 5 == 0:
                    rows.append(
                        dict(
                            t=(step + 1) * 0.02,
                            position=T(robot.data.root_pos_w)[0].cpu().tolist(),
                            velocity=T(robot.data.root_lin_vel_b)[0].cpu().tolist(),
                            fell=fallen,
                        )
                    )
                if fallen:
                    break
                if not a.headless:
                    time.sleep(max(0.0, 0.02 - (time.monotonic() - frame_start)))

        # Save measured motion separately from the summary for later inspection.
        report.update(
            status="completed",
            fell=fallen,
            simulated_seconds=(step + 1) * 0.02,
            automatic_resets=False,
        )
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        (a.output / "trajectory.json").write_text(json.dumps(rows) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        env.close()
except BaseException as error:
    report.update(status="failed", error=repr(error))
    raise
finally:
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (a.output / "trajectory.json").write_text(json.dumps(rows) + "\n")
    print(json.dumps(report, indent=2))
