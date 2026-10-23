"""Run preserved actors in the dimensioned indoor scene, with an explicit new contract.
The frozen old task stays available. New scene results have a separate denominator.
"""

from pathlib import Path
import argparse, functools, hashlib, importlib.util, json, sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--runtime", type=Path, required=True)
    ap.add_argument("--style", choices=list("ABCD"), default="A")
    ap.add_argument("--record-hd", action="store_true")
    extra, remaining = ap.parse_known_args()
    if not any(x == "--root-x" or x.startswith("--root-x=") for x in remaining):
        remaining += ["--root-x", "-0.55"]
    sys.argv = [sys.argv[0], *remaining]
    M = extra.runtime.resolve()
    D = M / "evaluation/football"
    sys.path.insert(0, str(D))
    launcher = load("scene_frozen_launcher", D / "execute_media.py")
    launcher.verify()
    helper = load("scene_media_helper", M / "football/recording/execute.py")
    helper.verify()
    sys.path.insert(0, str(M / "football/recording"))
    entry = load("isaac_execute", D / "executor.py")
    wrapper = load("scene_preserved_controller", D / "execute.py")
    # Replace measured map/landmark constants, never observations or hidden state.
    import controller, visual_localization, nav2_bridge, contact_scoring, perception
    from scene_goal_points import extract_scene_goal_bases

    perception.extract_beta_goal_bases = extract_scene_goal_bases

    # Frozen legacy functions pass 7 cm explicitly. Override the physical-size
    # argument at this scene boundary without editing the preserved runtime.
    def sized_estimator(function):
        @functools.wraps(function)
        def estimate(*args, **kwargs):
            kwargs["radius_m"] = 0.05
            return function(*args, **kwargs)

        return estimate

    perception.estimate_box_sphere = sized_estimator(perception.estimate_box_sphere)
    perception.estimate_grounded_direction = sized_estimator(
        perception.estimate_grounded_direction
    )
    from processing.ground_estimator import geometry as grounded_geometry
    import scene_ball_geometry
    import task_navigation

    task_navigation.geometry.parent.refine_grounded_complete_ball = (
        scene_ball_geometry.refine_grounded_complete_ball
    )
    grounded_geometry.parent.refine_grounded_complete_ball = (
        scene_ball_geometry.refine_grounded_complete_ball
    )
    import startup_support
    from grounded_startup import GroundedStartup

    startup_support.StartupSupport = GroundedStartup
    from vision import known_goal_geometry, goal_geometry
    import scene_localization
    from scene_nav2 import map_contract, obstacle_contract, owner, client, contract
    from nav2_service import map_contract as old_map, contract as old_contract

    known_goal_geometry.POST_SEPARATION_M = 0.394
    goal_geometry.POST_CENTER_SEPARATION_M = 0.394
    visual_localization.fit_posts = scene_localization.fit_posts
    controller.fit_posts = scene_localization.fit_posts
    old_reset = controller.BetaNavigator.reset

    def reset(self):
        old_reset(self)
        self.goal_map[:] = [0.68, 0]

    controller.BetaNavigator.reset = reset
    original_crossing = contact_scoring.goal_crossing
    contact_scoring.goal_crossing = functools.partial(
        original_crossing, goal_x=0.68, width=0.388, height=0.244, radius=0.05
    )
    old_map.snapshot = map_contract.snapshot
    old_contract.snapshot = map_contract.snapshot
    nav2_bridge.prior = obstacle_contract.prior
    nav2_bridge.BridgeClient = client.BridgeClient

    class SceneOwner(owner.ServiceOwner):
        def __init__(self, source_dir, *args, **kwargs):
            super().__init__(HERE / "scene_nav2", *args, **kwargs)

    nav2_bridge.ServiceOwner = SceneOwner
    import coarse_track

    coarse_track.snapshot = map_contract.snapshot
    original_build = wrapper.base_build
    scene_record = {}
    recording = {}

    def build(a):
        a.robot_usd = HERE / "assets/microduck_contact.usda"
        cfg, fixture = original_build(a)
        from scene_style import configure_football

        scene_record.update(configure_football(cfg, style=extra.style))
        fixture.update(
            {
                k: scene_record[k]
                for k in [
                    "goal_plane_x_m",
                    "goal_opening_width_m",
                    "goal_opening_height_m",
                ]
            }
        )
        fixture["ball_pose_xyzw"][2] = 0.05
        fixture["ball_radius_m"] = 0.05
        fixture["ball_mass_kg"] = 0.03
        fixture["root_pose_xyzw"][2] = 0.12
        cfg.scene.robot.init_state.pos = (*cfg.scene.robot.init_state.pos[:2], 0.12)
        fixture["scene_profile"] = scene_record["profile"]
        fixture["scope"] = (
            "New dimensioned indoor fixture; privileged initialization and evaluation only"
        )
        original_sensor = cfg.events.beta_sensors.func

        def scene_sensor(env, env_ids, width, height, hfov, seed):
            original_sensor(
                env, env_ids, width=width, height=height, hfov=hfov, seed=seed
            )
            env.course_visual_geometry.update(
                pitch_visual_top_z_m=0.00005,
                sphere_radius_m=0.05,
                sphere_mass_kg=0.03,
                sphere_visual="official black/white UV mesh",
            )

        cfg.events.beta_sensors.func = scene_sensor
        if extra.record_hd:

            def sensor(env, env_ids, width, height, hfov, seed):
                scene_sensor(
                    env, env_ids, width=width, height=height, hfov=hfov, seed=seed
                )
                env.course_visual_geometry.update(
                    pitch_visual_top_z_m=0.00005,
                    sphere_radius_m=0.05,
                    sphere_mass_kg=0.03,
                    sphere_visual="official black/white UV mesh",
                )
                from terrain_visuals import camera_prestartup, start_recording, capture

                camera_prestartup(env, env_ids)
                original_capture = env.beta_rig.capture

                def recorded_capture(*args, **kwargs):
                    result = original_capture(*args, **kwargs)
                    if not recording:
                        camera, proc = start_recording(
                            env,
                            a.output,
                            eye_offset=(1.40, -1.95, 1.20),
                            target_offset=(0.10, 0, 0.10),
                            filename="scene-hd.mp4",
                            fps=15,
                        )
                        recording.update(camera=camera, process=proc, rows=[])
                    capture(env, recording["camera"], recording["process"])

                    def values(value):
                        return getattr(value, "torch", value)[0].detach().cpu().tolist()

                    robot_data = env.scene["robot"].data
                    ball_data = env.scene["ball"].data
                    recording["rows"].append(
                        {
                            "timestamp_s": kwargs["timestamp_s"],
                            "frame": len(recording["rows"]),
                            "root_position_world_m": values(robot_data.root_pos_w),
                            "root_quaternion_xyzw": values(robot_data.root_quat_w),
                            "joint_positions_absolute_rad": values(
                                robot_data.joint_pos
                            ),
                            "ball_position_world_m": values(ball_data.root_pos_w),
                            "ball_quaternion_xyzw": values(ball_data.root_quat_w),
                        }
                    )
                    return result

                env.beta_rig.capture = recorded_capture

            cfg.events.beta_sensors.func = sensor
        scene_record.update(
            locomotion_and_kick_models_retrained=False,
            perception_weights="explicit --detector-deployment; hash recorded in manifest.detector",
            known_post_center_separation_m=0.394,
            map_bounds_xy_m=[-0.7, 0.9, -0.5, 0.5],
            new_scene_requires_independent_validation=True,
        )
        return cfg, fixture

    wrapper.base_build = build
    output = Path(remaining[remaining.index("--output") + 1])
    files = [
        HERE / "football_scene.py",
        HERE / "scene_style.py",
        HERE / "grounded_startup.py",
        HERE / "scene_ball_geometry.py",
        HERE / "scene_localization.py",
        HERE / "scene_goal_points.py",
        HERE / "terrain_visuals.py",
        *sorted((HERE / "assets").glob("*")),
        *sorted((HERE / "scene_nav2").glob("*")),
    ]
    hashes = {str(p): sha(p) for p in files if p.is_file()}
    launch_path = output.with_name(output.name + "-scene-launcher.json")
    launch_path.write_text(
        json.dumps(
            dict(source_sha256=hashes, argv=remaining, style=extra.style), indent=2
        )
    )
    original_run = wrapper.run

    def run_and_finalize(*args, **kwargs):
        try:
            return original_run(*args, **kwargs)
        finally:
            code = 0
            if recording:
                proc = recording["process"]
                proc.stdin.close()
                code = proc.wait(timeout=30)
                (output / "scene-hd-ledger.json").write_text(
                    json.dumps(
                        {
                            "frames": recording["rows"],
                            "fps": 15,
                            "dimensions": [1920, 1080],
                            "encoder_exit_code": code,
                        },
                        indent=2,
                    )
                )
            if output.exists():
                scene_record.update(
                    source_sha256=hashes,
                    source_changed=[p for p, h in hashes.items() if sha(p) != h],
                )
                (output / "scene-contract.json").write_text(
                    json.dumps(scene_record, indent=2)
                )
            launcher.verify()
            helper.verify()
            if scene_record.get("source_changed"):
                raise RuntimeError("Scene source changed during execution")
            if code:
                raise RuntimeError("HD encoder failed; partial video retained")

    # Kit shutdown may exit the process; finalize inside its simulation context.
    wrapper.run = run_and_finalize
    wrapper.main()


if __name__ == "__main__":
    main()
