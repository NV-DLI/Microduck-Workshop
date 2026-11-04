"""Keyboard football with the two Pollen Robotics ONNX policies."""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--headless-check", action="store_true")
    p.add_argument("--style", choices=list("ABCD"), default="A")
    p.add_argument(
        "--lighting", choices=["daylight", "warm", "cool-dim"], default="daylight"
    )
    p.add_argument("--record", action="store_true")
    p.add_argument(
        "--check-motion", choices=["stand", "drive", "kick"], default="stand"
    )
    p.add_argument("--root-x", type=float, default=-0.55)
    p.add_argument("--root-y", type=float, default=0.0)
    p.add_argument("--ball-x", type=float, default=0.0)
    p.add_argument("--ball-y", type=float, default=0.0)
    p.add_argument("--walking-model", type=Path)
    p.add_argument("--grounded-start", action="store_true",
                   help="Settle in the HOME pose before the first policy action")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    import shutil

    snapshot = a.output / "source-snapshot"
    snapshot.mkdir()
    for source in [
        Path(__file__),
        Path(__file__).with_name("scene_style.py"),
    ]:
        shutil.copy2(source, snapshot / source.name)
    sys.path.insert(0, str(a.runtime / "gait"))
    sys.path.insert(0, str(a.runtime / "football"))
    from scoring import goal_crossing
    import torch
    import gymnasium as gym
    from isaaclab.app import launch_simulation
    from batch_policy import BatchPolicy

    sys.path.insert(0, str(Path(__file__).parent / "course_labs"))
    from kit_settings import kit_args, log_settings, trace_fabric_paths, trace_kit_viewer
    from realtime_view import Pacer, apply_viewport_profile, render_every

    # Keyboard sessions use the live display profile (960x540 viewport texture, redraw every
    # 2nd control step, cumulative real-time pacing).
    live_profile = not a.headless_check
    every = render_every() if live_profile else 1

    # Start Kit before importing scene and rendering components.
    with launch_simulation(
        None,
        {
            "device": "cuda:0",
            "require_kit": True,
            "kit_args": kit_args(),
            "enable_cameras": a.record,
            "visualizer": [] if a.headless_check else ["kit"],
        },
    ):
        log_settings("kit started, before the scene")
        trace_kit_viewer()
        trace_fabric_paths()
        import isaaclab_tasks
        import isaaclab.sim as sim_utils
        from isaaclab_tasks.utils import parse_env_cfg
        from isaaclab.assets import RigidObjectCfg
        from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg

        cfg = parse_env_cfg(
            "IsaacContrib-Velocity-Flat-MicroDuck",
            device="cuda:0",
            num_envs=1,
            overrides=["presets=backlash"],
        )
        if not a.headless_check:
            from isaaclab_visualizers.kit import KitVisualizerCfg

            cfg.sim.visualizer_cfgs = [KitVisualizerCfg()]
        cfg.seed = a.seed
        cfg.curriculum = None
        cfg.episode_length_s = a.seconds + 10
        cfg.scene.robot.spawn.usd_path = str(
            Path(__file__).parent / "assets/microduck_contact.usda"
        )
        cfg.scene.robot.init_state.pos = (a.root_x, a.root_y, 0.12)

        # Physics and visuals share the same ball size and initial position.
        cfg.scene.ball = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Ball",
            spawn=sim_utils.SphereCfg(
                radius=0.05,
                rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.91, 0.88), roughness=0.75
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(a.ball_x, a.ball_y, 0.05)),
        )

        from scene_style import configure_football

        scene_profile = configure_football(cfg, style=a.style, lighting=a.lighting)
        scene_profile["robot_asset_sha256"] = hashlib.sha256(
            Path(cfg.scene.robot.spawn.usd_path).read_bytes()
        ).hexdigest()
        if a.record:
            from isaaclab.sensors import ContactSensorCfg

            cfg.scene.ball_contact = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle",
                filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball"],
                history_length=1,
                update_period=0.005,
            )
        if a.record:
            from terrain_visuals import camera_prestartup
            from isaaclab.managers import EventTermCfg

            cfg.events.terrain_camera = EventTermCfg(
                func=camera_prestartup, mode="prestartup"
            )
        cfg.sim.physics.solver_cfg.nconmax = 300
        cfg.sim.physics.solver_cfg.njmax = 1500
        cfg.commands.base_velocity.resampling_time_range = (1e8, 1e8)
        cfg.commands.base_velocity.debug_vis = False
        cfg.commands.head_pose.ranges = tuple((0.0, 0.0) for _ in range(4))
        cfg.commands.body_pose.ranges = tuple((0.0, 0.0) for _ in range(6))
        cfg.events.reset_base.params["pose_range"] = {}
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
        env = gym.make("IsaacContrib-Velocity-Flat-MicroDuck", cfg=cfg).unwrapped
        env.reset()
        env.sim.set_camera_view((1.40, -1.95, 1.20), (0.10, 0, 0.10))
        log_settings("scene created and reset")
        manager = env.sim.backend_manager
        assert manager.backend_name == "newton" and cfg.sim.use_newton_actuators
        manager.set_decimation(1)
        env._physics_handles_decimation = False
        hashes = {
            f: hashlib.sha256((a.models / f).read_bytes()).hexdigest()
            for f in ["walk.onnx", "kick-teacher.onnx"]
        }
        locomotion_path = a.walking_model or a.models / "walk.onnx"
        hashes["locomotion"] = hashlib.sha256(locomotion_path.read_bytes()).hexdigest()
        walk = BatchPolicy(locomotion_path, hashes["locomotion"], "cuda:0")
        video = None
        hd_rows = []
        if a.record:
            from terrain_visuals import start_recording, capture

            camera, video = start_recording(
                env,
                a.output,
                eye_offset=(1.4, -1.95, 1.2),
                target_offset=(0.1, 0, 0.1),
                fps=25,
            )

        kick = BatchPolicy(
            a.models / "kick-teacher.onnx", hashes["kick-teacher.onnx"], "cuda:0"
        )
        held = set()
        flags = dict(reset=False, kick=False, quit=False)
        events = []
        kick_start = None
        fallen = False
        scored = False
        resets = 0
        if not a.headless_check:
            import carb.input
            import omni.appwindow
            import omni.ui as ui
            from omni.kit.hotkeys.core import get_hotkey_registry

            registry = get_hotkey_registry()
            for hotkey in registry.get_all_hotkeys():
                combo = hotkey.key_combination
                if (
                    combo
                    and combo.modifiers == 0
                    and str(getattr(combo.key, "name", combo.key)).upper()
                    in (
                        "W",
                        "A",
                        "S",
                        "D",
                        "Q",
                        "E",
                        "SPACE",
                        "R",
                        "ESCAPE",
                    )
                ):
                    registry.deregister_hotkey(hotkey)
            inp = carb.input.acquire_input_interface()
            window = omni.appwindow.get_default_app_window()

            def key(event, *_):
                name = str(getattr(event.input, "name", event.input)).upper()
                if event.type == carb.input.KeyboardEventType.KEY_REPEAT:
                    if name in ("W", "A", "S", "D", "Q", "E"):
                        held.add(name)
                elif event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    held.add(name)
                    if name == "R":
                        flags["reset"] = True
                    elif name == "SPACE":
                        flags["kick"] = True
                    elif name == "ESCAPE":
                        flags["quit"] = True
                    events.append({"key": name, "at": time.monotonic()})
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    held.discard(name)
                return True

            subscription = inp.subscribe_to_keyboard_events(window.get_keyboard(), key)
            # 1920 x 1080 UI points: on a HiDPI desktop (UI scale 2) the window doubles in
            # pixels, so the help panel covers the same share of the scene as at scale 1.
            scale = getattr(ui.Workspace, "get_dpi_scale", lambda: 1.0)() or 1.0
            window.resize(round(1920 * scale), round(1080 * scale))
            viewport_record = apply_viewport_profile()
            events.append({"viewport": viewport_record, "at": time.monotonic()})
            print("COURSE_VIEWPORT " + json.dumps(viewport_record), flush=True)
            panel = ui.Window("Microduck • Keyboard football", width=480, height=330)
            panel.position_x = 20
            panel.position_y = 50
            # Dock the help as the active tab of the right-hand Property column (as Kit's
            # own panels do), so the viewport and the robot stay uncovered.
            panel.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
            with panel.frame:
                with ui.VStack(spacing=8):
                    status = ui.Label("Waiting to start", height=32)
                    ui.Label("W/S: forward/back   A/D: step sideways")
                    ui.Label("Q/E: turn   Space: right-foot kick")
                    ui.Label("Release keys to stand. R: restart. Esc: exit.")
                    ui.Label(
                        "After restart, release and press movement keys again."
                    )
                    ui.Label("Official walking + kicking ONNX policies")
                    ui.Button(
                        "Restart scene",
                        clicked_fn=lambda: flags.update(reset=True),
                    )
            print("MANUAL_FOOTBALL_READY", flush=True)
        cmd = env.command_manager.get_term("base_velocity")
        robot = env.scene["robot"]
        rows = []

        def T(x):
            return getattr(x, "torch", x)

        kick_issued = False
        touch = False
        early_touch = False
        kick_at = None
        contact_at = None
        recovered_since = None
        recovered_duration = 0.0
        physics_rows = []
        action_rows = []
        shot_end_s = 2.5
        performance_rows = []
        with torch.inference_mode():
            # Course keyboard sessions use the standing61 start-up: settle on the
            # ground with temporarily locked joints, release before the first action.
            start_brake = None
            if a.grounded_start:
                sys.path.insert(0, str(Path(__file__).parent / "course_labs"))
                from grounded_start import settle as _settle, StartupNotSettled

                def settle(env, render):
                    try:
                        return _settle(
                            env,
                            pose="home",
                            render=render,
                        )
                    except StartupNotSettled as error:
                        (a.output / "startup-failed.json").write_text(
                            json.dumps(error.record, indent=2)
                        )
                        raise

                start_brake, startup = settle(env, render=True)
                events.append({"startup": startup, "at": time.monotonic()})
            pacer = Pacer()
            for step in range(round(a.seconds / 0.02)):
                tick_start = time.monotonic()
                if flags["quit"]:
                    break
                if flags["reset"]:
                    # R can arrive while settle() is rendering the locked start-up.
                    # Restore those limits before the next brake snapshots them.
                    if start_brake is not None and start_brake.active:
                        start_brake.release(step * 0.02)
                    env.reset()
                    flags.update(reset=False, kick=False)
                    held.clear()
                    kick_start = None
                    fallen = False
                    scored = False
                    resets += 1
                    if a.grounded_start:
                        start_brake, startup = settle(env, render=True)
                        events.append({"startup": startup, "at": time.monotonic()})
                if a.headless_check:
                    held.clear()
                    if a.check_motion == "drive":
                        if 100 <= step < 250:
                            held.add("W")
                        elif 350 <= step < 450:
                            held.add("Q")
                    if a.check_motion == "kick" and step == 150:
                        flags["kick"] = True
                if not a.headless_check:
                    # Read current key state so a missed release cannot leave motion latched.
                    for name in ("W", "A", "S", "D", "Q", "E"):
                        if inp.get_keyboard_value(window.get_keyboard(), getattr(carb.input.KeyboardInput, name)) > 0:
                            held.add(name)
                        else:
                            held.discard(name)
                if flags["kick"] and kick_start is None and not fallen:
                    kick_start = step
                    kick_at = step * 0.02
                    kick_issued = True
                    flags["kick"] = False
                cmd.is_standing_env[:] = False
                cmd.vel_command_b[:] = 0
                if kick_start is None and not fallen:
                    cmd.vel_command_b[0] = torch.tensor(
                        [
                            0.0 if ("W" in held) == ("S" in held) else (0.25 if "W" in held else -0.40),
                            0.15 * ("A" in held) - 0.15 * ("D" in held),
                            0.45 * (("Q" in held) - ("E" in held)),
                        ],
                        device="cuda:0",
                    )
                env.command_manager.get_term("head_pose").command[:] = 0
                env.command_manager.get_term("body_pose").command[:] = 0
                obs = env.observation_manager.compute(update_history=True)["policy"]
                action = walk(obs)
                if kick_start is not None:
                    phase = (step - kick_start) * 0.02
                    # A bounded official kick, then a smooth handoff to standing.
                    weight = (
                        1.0 if phase < 0.14 else max(0.0, 1.0 - (phase - 0.14) / 0.12)
                    )
                    if weight:
                        action = torch.lerp(action, kick(obs), weight)
                    if phase >= shot_end_s:
                        kick_start = None
                        flags["kick"] = False
                action_rows.append(
                    dict(
                        timestamp_s=step * 0.02,
                        observation=obs[0].cpu().tolist(),
                        raw_action=action[0].cpu().tolist(),
                        executed_action=action[0].cpu().tolist(),
                        command=cmd.vel_command_b[0].cpu().tolist(),
                    )
                )
                if start_brake is not None and start_brake.active:
                    start_brake.release(step * 0.02)
                    print(
                        "COURSE_POLICY_TAKEOVER "
                        + json.dumps(dict(t=step * 0.02, locomotion="walking")),
                        flush=True,
                    )
                env.action_manager.process_action(action)
                # Telemetry is read with one GPU->CPU copy per physics tick (packed below);
                # the ball's position before tick n+1 is its position after tick n.
                before_ball = T(env.scene["ball"].data.root_pos_w)[0].cpu().tolist()
                need_force = a.record
                for tick in range(4):
                    env._sim_step_counter += 1
                    env.action_manager.apply_action()
                    env.scene.write_data_to_sim()
                    env.sim.step(
                        render=not a.headless_check
                        and tick == 3
                        and step % every == 0
                    )
                    env.scene.update(0.005)
                    parts = [
                        T(env.scene["ball"].data.root_pos_w)[0],
                        T(robot.data.root_pos_w)[0],
                        T(robot.data.root_lin_vel_b)[0].norm()[None],
                        T(robot.data.root_ang_vel_b)[0].norm()[None],
                        T(robot.data.projected_gravity_b)[0, 2][None],
                    ]
                    if need_force:
                        parts.append(
                            T(env.scene["ball_contact"].data.force_matrix_w)
                            .norm(dim=-1)
                            .max()[None]
                        )
                    packed = torch.cat([v.reshape(-1).float() for v in parts]).cpu().tolist()
                    after_ball, robot_position = packed[0:3], packed[3:6]
                    speed, angular, gz = packed[6], packed[7], packed[8]
                    force = 0.0
                    if need_force:
                        force = packed[-1]
                        if force > 0.05:
                            in_window = (
                                kick_at is not None
                                and 0
                                <= (step * 0.02 + (tick + 1) * 0.005) - kick_at
                                <= shot_end_s
                            )
                            touch = touch or in_window
                            if in_window and contact_at is None:
                                contact_at = step * 0.02 + (tick + 1) * 0.005
                            early_touch = early_touch or kick_at is None
                    scored = scored or (
                        touch
                        and goal_crossing(
                            before_ball,
                            after_ball,
                            goal_x=scene_profile["goal_plane_x_m"],
                            width=scene_profile["goal_opening_width_m"],
                            height=scene_profile["goal_opening_height_m"],
                            radius=scene_profile["ball_diameter_m"] / 2,
                        )
                    )
                    stamp = step * 0.02 + (tick + 1) * 0.005
                    quiet = (
                        kick_at is not None
                        and stamp >= kick_at + shot_end_s
                        and speed < 0.02
                        and angular < 0.3
                        and gz < -0.94
                    )
                    recovered_since = (
                        (stamp if recovered_since is None else recovered_since)
                        if quiet
                        else None
                    )
                    recovered_duration = (
                        0.0 if recovered_since is None else stamp - recovered_since
                    )
                    physics_rows.append(
                        dict(
                            timestamp_s=stamp,
                            ball_position=after_ball,
                            ball_contact_force_n=force,
                            robot_position=robot_position,
                            linear_speed=speed,
                            angular_speed=angular,
                            gravity_z=gz,
                            kick_at=kick_at,
                            scored=scored,
                            recovered_duration_s=recovered_duration,
                        )
                    )
                    before_ball = after_ball
                if step % 2 == 0:
                    if video is not None:
                        capture(env, camera, video)

                    # One GPU->CPU copy for the five pose fields of this row.
                    fields = [
                        T(robot.data.root_pos_w)[0],
                        T(robot.data.root_quat_w)[0],
                        T(robot.data.joint_pos)[0],
                        T(env.scene["ball"].data.root_pos_w)[0],
                        T(env.scene["ball"].data.root_quat_w)[0],
                    ]
                    flat = torch.cat([v.reshape(-1).float() for v in fields]).cpu().tolist()
                    cuts, start = [], 0
                    for v in fields:
                        cuts.append(flat[start:start + v.numel()])
                        start += v.numel()
                    hd_rows.append(
                        dict(
                            timestamp_s=(step + 1) * 0.02,
                            frame=len(hd_rows),
                            root_position_world_m=cuts[0],
                            root_quaternion_xyzw=cuts[1],
                            joint_positions_absolute_rad=cuts[2],
                            ball_position_world_m=cuts[3],
                            ball_quaternion_xyzw=cuts[4],
                        )
                    )
                env.episode_length_buf += 1
                env.common_step_counter += 1
                fallen = fallen or bool(
                    (T(robot.data.projected_gravity_b)[0, 2] > -0.6)
                    | (T(robot.data.root_pos_w)[0, 2] < 0.065)
                )
                ball = T(env.scene["ball"].data.root_pos_w)[0]
                if step % 5 == 0:
                    rows.append(
                        dict(
                            t=(step + 1) * 0.02,
                            position=T(robot.data.root_pos_w)[0].cpu().tolist(),
                            ball=ball.cpu().tolist(),
                            fell=fallen,
                            kicking=kick_start is not None,
                            command=cmd.vel_command_b[0].cpu().tolist(),
                            velocity=T(robot.data.root_lin_vel_b)[0].cpu().tolist(),
                            gravity=T(robot.data.projected_gravity_b)[0].cpu().tolist(),
                            joint_positions=T(robot.data.joint_pos)[0].cpu().tolist(),
                            joint_velocities=T(robot.data.joint_vel)[0].cpu().tolist(),
                        )
                    )
                if not a.headless_check:
                    text = (
                        "Fallen: press R to restart."
                        if fallen
                        else (
                            "Goal! Press R to try again."
                            if scored
                            else "Position the ball beside the right foot; release keys; Space."
                        )
                    )
                    if status.text != text:  # a Kit UI update only when the text changes
                        status.text = text
                    pacer.wait()
                performance_rows.append(
                    dict(
                        step=step,
                        wall_s=time.monotonic() - tick_start,
                    )
                )
        if not a.headless_check:
            inp.unsubscribe_to_keyboard_events(window.get_keyboard(), subscription)
            panel.visible = False
            panel.destroy()
        (a.output / "performance.json").write_text(
            json.dumps(dict(steps=performance_rows), indent=2)
        )
        if video is not None:
            video.stdin.close()
            if video.wait(timeout=30):
                raise RuntimeError("Recording failed")
        report = dict(
            status="completed",
            locomotion="walking",
            seed=a.seed,
            check_motion=a.check_motion,
            fixture=dict(root_xy=[a.root_x, a.root_y], ball_xy=[a.ball_x, a.ball_y]),
            task_result=dict(
                kick_issued=kick_issued,
                foot_ball_contact=touch,
                early_contact=early_touch,
                whole_ball_in_goal=scored,
                fallen=fallen,
                recovered=recovered_duration >= 1.0,
                recovery_continuous_seconds=recovered_duration,
                kick_at_s=kick_at,
                contact_at_s=contact_at,
                contact_window_s=shot_end_s,
            ),
            joint_names=robot.joint_names,
            models=hashes,
            physics_backend=manager.backend_name,
            physics_hz=200,
            policy_hz=50,
            resets=resets,
            fell=fallen,
            goal_observed=scored,
            scene_profile=scene_profile,
            events=events,
            scope=(
                "keyboard skill exploration; goal position observation is not autonomous-task certification"
            ),
        )
        (a.output / "report.json").write_text(json.dumps(report, indent=2))
        (a.output / "scene-hd-ledger.json").write_text(
            json.dumps(
                dict(
                    frames=hd_rows,
                    fps=25,
                    dimensions=[1920, 1080],
                    encoder_exit_code=0 if video is not None else None,
                    live_hd_recorded=video is not None,
                )
            )
        )
        (a.output / "trajectory.json").write_text(json.dumps(rows))
        (a.output / "physical-evaluation.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in physics_rows)
        )
        (a.output / "action-trace.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in action_rows)
        )
        env.close()


if __name__ == "__main__":
    main()
