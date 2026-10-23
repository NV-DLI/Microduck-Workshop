"""RTX illustrations of saved Newton states; no new task or physics result.

State restoration is permitted only in this offline renderer. The autonomous
controller never imports it. Original traces remain the authority for scoring.
"""

from pathlib import Path
import argparse, json, hashlib, sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--times", type=float, nargs="+", required=True)
    p.add_argument("--overview", action="store_true")
    p.add_argument("--style", choices=list("ABCD"))
    p.add_argument("--mat-gallery", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    M = a.runtime.resolve()
    sys.path[:0] = [
        str(M / "perception"),
        str(M / "football"),
        str(M / "sensors"),
        str(M / "gait"),
    ]
    manifest = json.loads((a.run / "manifest.json").read_text())
    scene = json.loads((a.run / "scene-contract.json").read_text())
    physics = [json.loads(x) for x in (a.run / "physics-trace.jsonl").open()]
    samples = [min(physics, key=lambda x: abs(x["timestamp_s"] - t)) for t in a.times]
    from isaaclab.app import launch_simulation

    with launch_simulation(
        None,
        {
            "device": "cuda:0",
            "require_kit": True,
            "enable_cameras": True,
            "visualizer": [],
        },
    ):
        import isaaclab_tasks
        import gymnasium as gym
        import torch, numpy as np
        from PIL import Image
        from pxr import Usd, UsdShade, Sdf
        from isaaclab.managers import EventTermCfg
        from isaac_execute import build_cfg
        from scene_style import configure_football, ASSETS
        from terrain_visuals import camera_prestartup, start_recording, capture

        setup = argparse.Namespace(
            device="cuda:0",
            seed=1,
            seconds=5.0,
            randomized=False,
            root_y=0.0,
            ball_xy=[0.0, 0.0],
            layout="empty",
            width=960,
            height=540,
            assumed_hfov=70,
            robot_usd=ASSETS / "microduck_contact.usda",
        )
        cfg, _ = build_cfg(setup)
        configure_football(cfg, style=a.style or scene["mat_style"])
        cfg.events.beta_sensors = None
        cfg.events.terrain_camera = EventTermCfg(
            func=camera_prestartup, mode="prestartup"
        )
        env = gym.make("IsaacContrib-Velocity-Flat-MicroDuck", cfg=cfg).unwrapped
        proc = None
        try:
            env.reset()
            robot = env.scene["robot"]
            ball = env.scene["ball"]
            if list(robot.joint_names) != manifest["joint_names"]:
                raise ValueError("Saved joint order differs")
            if env.sim.backend_manager.backend_name != "newton":
                raise ValueError("Newton required")
            records = []
            camera = None
            for i, row in enumerate(samples):
                root = [*row["root_position_world_m"], *row["root_quaternion_xyzw"]]
                q = torch.tensor(
                    [row["joint_positions_absolute_rad"]], device=env.device
                )
                robot.write_root_pose_to_sim_index(
                    root_pose=torch.tensor([root], device=env.device)
                )
                robot.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((1, 6), device=env.device)
                )
                robot.write_joint_position_to_sim_index(position=q)
                robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q))
                ball.write_root_pose_to_sim_index(
                    root_pose=torch.tensor(
                        [[*row["ball_position_world_m"], 0.0, 0.0, 0.0, 1.0]],
                        device=env.device,
                    )
                )
                ball.write_root_velocity_to_sim_index(
                    root_velocity=torch.zeros((1, 6), device=env.device)
                )
                env.scene.write_data_to_sim()
                env.sim.forward()
                env.scene.update(0.005)
                if camera is None:
                    eye = (1.4, -1.95, 1.2) if a.overview else (0.38, -0.69, 0.40)
                    target = (0.1, 0, 0.1) if a.overview else (-0.11, 0, 0.115)
                    camera, proc = start_recording(
                        env,
                        a.output,
                        eye_offset=eye,
                        target_offset=target,
                        filename="state-previews.mp4",
                        fps=15,
                    )
                styles = "ABCD" if a.mat_gallery else [a.style or scene["mat_style"]]
                for style in styles:
                    if a.mat_gallery:
                        rootprim = env.sim.stage.GetPrimAtPath(
                            "/World/envs/env_0/PitchVisual"
                        )
                        for prim in Usd.PrimRange(rootprim):
                            shader = UsdShade.Shader(prim)
                            if shader and shader.GetIdAttr().Get() == "UsdUVTexture":
                                shader.GetInput("file").Set(
                                    Sdf.AssetPath(str(ASSETS / f"pitch_{style}.png"))
                                )
                    for _ in range(40):
                        env.sim.render()
                    capture(env, camera, proc)
                    data = camera.data.output["rgb"]
                    data = getattr(data, "torch", data)
                    name = f"pose-{i+1}-{style}.png"
                    Image.fromarray(data[0, :, :, :3].cpu().numpy()).save(
                        a.output / name
                    )
                    records.append(
                        dict(
                            file=name,
                            source_timestamp_s=row["timestamp_s"],
                            mat_style=style,
                            physical_state_advanced=False,
                        )
                    )
            (a.output / "provenance.json").write_text(
                json.dumps(
                    dict(
                        mode="offline RTX state visualization; not new physical validation",
                        source_trace_sha256=hashlib.sha256(
                            (a.run / "physics-trace.jsonl").read_bytes()
                        ).hexdigest(),
                        records=records,
                    ),
                    indent=2,
                )
            )
        finally:
            if proc:
                proc.stdin.close()
                if proc.wait(timeout=30):
                    raise RuntimeError("Encoder failed")
            env.close()


if __name__ == "__main__":
    main()
