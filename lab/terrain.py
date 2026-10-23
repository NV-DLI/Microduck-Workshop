"""Evaluate a fixed walking policy on a flat approach, ramp and landing."""

from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import numpy as np


def ramp_mesh(difficulty, cfg):
    import trimesh

    # Spawn at (2,2); keep the ramp profile and extend the approach/landing
    # to a 100 m support surface so a failed policy cannot fall off a tiny tile.
    slope = math.tan(math.radians(cfg.slope_deg))
    xs = [-48.0, 2.35, 3.35, 52.0]
    zs = [0.0, 0.0, slope, slope]
    vertices = [(x, y, z) for x, z in zip(xs, zs) for y in [-48.0, 52.0]]
    faces = []
    for i in range(3):
        j = 2 * i
        faces.extend([(j, j + 2, j + 1), (j + 1, j + 2, j + 3)])
    return [trimesh.Trimesh(vertices=vertices, faces=faces, process=False)], np.array(
        [2.0, 2.0, 0.0]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--slope", type=float, default=5.0)
    p.add_argument("--num-envs", type=int, default=5)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--record", action="store_true")
    p.add_argument("--seed", type=int, default=42001)
    p.add_argument("--headless", action="store_true")
    a = p.parse_args()
    if (
        not -20 <= a.slope <= 20
        or not 1 <= a.num_envs <= 128
        or not 3 <= a.seconds <= 60
    ):
        p.error("slope: ±20°, envs: 1–128, seconds: 3–60")
    if a.record and a.num_envs != 1:
        p.error("Record one environment at a time: --num-envs 1")
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.runtime / "gait"))
    import torch
    import gymnasium as gym
    from isaaclab.app import launch_simulation
    from batch_policy import BatchPolicy

    digest = hashlib.sha256(a.model.read_bytes()).hexdigest()
    report = dict(
        status="starting",
        model_sha256=digest,
        slope_deg=a.slope,
        seed=a.seed,
        num_envs=a.num_envs,
        command_m_s=0.25,
        automatic_resets=False,
        support_surface_bounds_relative_m=[-50.0, 50.0, -50.0, 50.0],
        robot_contact_geometry="official allcollisions backlash",
    )
    (a.output / "report.json").write_text(json.dumps(report, indent=2))
    env = None
    video_process = None
    try:
        with launch_simulation(
            None,
            {
                "device": "cuda:0",
                "require_kit": not a.headless or a.record,
                "enable_cameras": a.record,
                "visualizer": [] if a.headless else ["kit"],
            },
        ):
            try:
                import isaaclab_tasks
                from isaaclab_tasks.utils import parse_env_cfg
                from isaaclab.terrains import TerrainGeneratorCfg, SubTerrainBaseCfg
                from isaaclab.utils import configclass
                from isaaclab_tasks.contrib.microduck.rough_env_cfg import (
                    MicroDuckTerrainImporter,
                )

                @configclass
                class RampCfg(SubTerrainBaseCfg):
                    function = ramp_mesh
                    slope_deg: float = a.slope

                cfg = parse_env_cfg(
                    "IsaacContrib-Velocity-Flat-MicroDuck",
                    device="cuda:0",
                    num_envs=a.num_envs,
                    overrides=["presets=backlash"],
                )
                if not a.headless:
                    from isaaclab_visualizers.kit import KitVisualizerCfg

                    cfg.sim.visualizer_cfgs = [KitVisualizerCfg()]
                cfg.seed = a.seed
                cfg.curriculum = None
                cfg.episode_length_s = a.seconds + 10
                cfg.scene.robot.spawn.usd_path = str(
                    Path(__file__).parent / "assets/microduck_contact.usda"
                )
                cfg.scene.terrain.terrain_type = "generator"
                cfg.scene.terrain.class_type = MicroDuckTerrainImporter
                cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
                    seed=a.seed,
                    size=(6.0, 4.0),
                    num_rows=1,
                    num_cols=1,
                    border_width=0.0,
                    use_cache=False,
                    sub_terrains={"ramp": RampCfg(proportion=1.0)},
                )
                if not a.headless or a.record:
                    from terrain_visuals import configure_scene

                    configure_scene(cfg)
                if a.record:
                    from terrain_visuals import camera_prestartup
                    from isaaclab.managers import EventTermCfg

                    cfg.events.terrain_camera = EventTermCfg(
                        func=camera_prestartup, mode="prestartup"
                    )
                cfg.sim.physics.solver_cfg.nconmax = 200
                cfg.sim.physics.solver_cfg.njmax = 1024
                cfg.commands.base_velocity.debug_vis = False
                cfg.commands.base_velocity.resampling_time_range = (1e8, 1e8)
                cfg.commands.head_pose.ranges = tuple((0.0, 0.0) for _ in range(4))
                cfg.commands.body_pose.ranges = tuple((0.0, 0.0) for _ in range(6))
                cfg.events.reset_base.params["pose_range"] = {
                    "y": (-0.025, 0.025),
                    "yaw": (-0.025, 0.025),
                }
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
                cfg.observations.policy.imu.params.update(
                    max_angle_deg=0, min_lag=0, max_lag=0
                )
                env = gym.make(
                    "IsaacContrib-Velocity-Flat-MicroDuck", cfg=cfg
                ).unwrapped
                env.reset()
                manager = env.sim.backend_manager
                assert manager.backend_name == "newton" and cfg.sim.use_newton_actuators
                manager.set_decimation(1)
                env._physics_handles_decimation = False
                if a.record:
                    from terrain_visuals import start_recording, capture

                    camera, video_process = start_recording(env, a.output)
                robot = env.scene["robot"]
                policy = BatchPolicy(a.model, digest, "cuda:0")
                cmd = env.command_manager.get_term("base_velocity")

                def T(x):
                    return getattr(x, "torch", x)

                origin = T(env.scene.env_origins).clone()
                fallen = torch.zeros(a.num_envs, dtype=torch.bool, device="cuda:0")
                reached = fallen.clone()
                rows = []
                start = time.monotonic()
                error = torch.zeros(a.num_envs, device="cuda:0")
                samples = 0
                upright_error = error.clone()
                upright_samples = error.clone()
                if not a.headless:
                    eye = (
                        (
                            origin[0]
                            + torch.tensor([1.35, -2.0, 0.85], device=env.device)
                        )
                        .cpu()
                        .tolist()
                    )
                    target = (
                        (origin[0] + torch.tensor([0.75, 0.0, 0.08], device=env.device))
                        .cpu()
                        .tolist()
                    )
                    env.sim.set_camera_view(eye, target)
                with torch.inference_mode():
                    for step in range(round(a.seconds / 0.02)):
                        frame_start = time.monotonic()
                        cmd.is_standing_env[:] = False
                        cmd.vel_command_b[:] = 0
                        if step >= 100:
                            cmd.vel_command_b[:, 0] = 0.25
                        env.command_manager.get_term("head_pose").command[:] = 0
                        env.command_manager.get_term("body_pose").command[:] = 0
                        action = policy(
                            env.observation_manager.compute(update_history=True)[
                                "policy"
                            ]
                        )
                        env.action_manager.process_action(action)
                        for tick in range(4):
                            env._sim_step_counter += 1
                            env.action_manager.apply_action()
                            env.scene.write_data_to_sim()
                            env.sim.step(render=not a.headless and tick == 3)
                            env.scene.update(0.005)
                        env.episode_length_buf += 1
                        env.common_step_counter += 1
                        if a.record and step % 2 == 0:
                            capture(env, camera, video_process)
                        pos = T(robot.data.root_pos_w) - origin
                        ground = (pos[:, 0] - 0.35).clamp(0, 1) * math.tan(
                            math.radians(a.slope)
                        )
                        fallen |= (T(robot.data.projected_gravity_b)[:, 2] > -0.6) | (
                            pos[:, 2] - ground < 0.065
                        )
                        reached |= (
                            (pos[:, 0] >= 1.45) & (pos[:, 1].abs() < 0.3) & ~fallen
                        )
                        if step >= 100:
                            squared_error = (
                                T(robot.data.root_lin_vel_b)[:, 0] - 0.25
                            ).square()
                            error += squared_error
                            samples += 1
                            upright_error += torch.where(~fallen, squared_error, 0.0)
                            upright_samples += (~fallen).float()
                        if step % 5 == 0:
                            rows.append(
                                dict(
                                    t=(step + 1) * 0.02,
                                    position=pos.cpu().tolist(),
                                    fell=fallen.cpu().tolist(),
                                )
                            )
                        if not a.headless:
                            time.sleep(
                                max(0.0, 0.02 - (time.monotonic() - frame_start))
                            )
                if a.record:
                    video_process.stdin.close()
                    if video_process.wait() != 0:
                        raise RuntimeError("Video encoding failed")
                report.update(
                    status="completed",
                    wall_seconds=time.monotonic() - start,
                    simulated_seconds=a.seconds,
                    physics_backend=manager.backend_name,
                    passed=(reached & ~fallen).cpu().tolist(),
                    fell=fallen.cpu().tolist(),
                    reached_landing=reached.cpu().tolist(),
                    velocity_rmse=(error / max(samples, 1)).sqrt().cpu().tolist(),
                    upright_velocity_rmse=(upright_error / upright_samples.clamp(min=1))
                    .sqrt()
                    .cpu()
                    .tolist(),
                    upright_velocity_samples=upright_samples.cpu().tolist(),
                    final_position=pos.cpu().tolist(),
                    criterion="Reach x>=1.45m within |y|<0.3m and remain upright for entire episode; no reset",
                )
                (a.output / "report.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
                (a.output / "trajectory.json").write_text(json.dumps(rows) + "\n")
                print(json.dumps(report), flush=True)
            finally:
                if video_process is not None:
                    import subprocess

                    try:
                        if video_process.stdin is not None:
                            video_process.stdin.close()
                    except BrokenPipeError:
                        pass
                    try:
                        video_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        video_process.kill()
                        video_process.wait()
                if env is not None:
                    env.close()
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
