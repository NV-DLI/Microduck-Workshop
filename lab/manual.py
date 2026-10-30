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
    p.add_argument("--locomotion", choices=["walking", "rollers"], default="walking")
    p.add_argument("--record", action="store_true")
    p.add_argument(
        "--check-motion", choices=["stand", "drive", "kick"], default="stand"
    )
    p.add_argument("--root-x", type=float, default=-0.55)
    p.add_argument("--root-y", type=float, default=0.0)
    p.add_argument("--ball-x", type=float, default=0.0)
    p.add_argument("--ball-y", type=float, default=0.0)
    p.add_argument("--shot-push-seconds", type=float, default=2.0)
    p.add_argument("--walking-model", type=Path)
    p.add_argument("--roller-shot", choices=["glide"], default="glide")
    p.add_argument("--autonomous", action="store_true")
    p.add_argument("--detector-deployment", type=Path)
    p.add_argument("--learning-config", type=Path)
    p.add_argument("--wait-for-start", action="store_true")
    p.add_argument("--ui-benchmark", choices=["full", "fast", "physics"])
    p.add_argument("--takeover-toggle", action="store_true")
    p.add_argument(
        "--mode-switch",
        action="store_true",
        help="Press M to switch walking and roller feet without leaving Isaac Sim",
    )
    a = p.parse_args()
    if a.mode_switch and (a.learning_config or a.autonomous or a.headless_check):
        p.error("Mode switching is for the visible keyboard session")
    if a.ui_benchmark and (not a.learning_config or a.headless_check):
        p.error("UI benchmarks require a visible learning session")
    if a.wait_for_start and (not a.learning_config or a.headless_check):
        p.error("Waiting for Start requires a visible learning session")
    if a.learning_config and (a.locomotion != "rollers" or a.autonomous):
        p.error("Learning experiments use rollers and a separate high-level controller")
    if a.autonomous and (a.locomotion != "rollers" or a.detector_deployment is None):
        p.error("Experimental visual servo requires rollers and detector deployment")
    if a.learning_config and a.headless_check:
        if json.loads(a.learning_config.read_text())["mode"] == "human":
            p.error("Human demonstrations require the interactive window")
    a.output.mkdir(parents=True, exist_ok=False)
    import shutil

    snapshot = a.output / "source-snapshot"
    snapshot.mkdir()
    for source in [
        Path(__file__),
        Path(__file__).with_name("roller_vision.py"),
        Path(__file__).with_name("roller_config.py"),
        Path(__file__).with_name("roller_imu.py"),
        Path(__file__).with_name("scene_style.py"),
    ]:
        shutil.copy2(source, snapshot / source.name)
    if a.learning_config:
        shutil.copytree(
            Path(__file__).parent / "learning",
            snapshot / "learning",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy2(a.learning_config, snapshot / "learning-config.json")
    sys.path.insert(0, str(a.runtime / "gait"))
    sys.path.insert(0, str(a.runtime / "football"))
    sys.path.insert(0, str(a.runtime / "sensors"))
    from scoring import goal_crossing
    import torch
    import gymnasium as gym
    from isaaclab.app import launch_simulation
    from batch_policy import BatchPolicy

    sys.path.insert(0, str(Path(__file__).parent / "course_labs"))
    from kit_settings import kit_args, log_settings, trace_fabric_paths, trace_kit_viewer
    from realtime_view import Pacer, apply_viewport_profile, render_every

    # Keyboard sessions use the live display profile (960x540 viewport texture, redraw every
    # 2nd control step, cumulative real-time pacing); learning sessions keep their settings.
    live_profile = not a.learning_config and not a.headless_check
    every = render_every() if live_profile else 1

    # Start Kit before importing scene and rendering components.
    with launch_simulation(
        None,
        {
            "device": "cuda:0",
            "require_kit": True,
            "kit_args": kit_args(),
            "enable_cameras": a.record or a.autonomous or bool(a.learning_config),
            "visualizer": [] if a.headless_check else ["kit"],
        },
    ):
        log_settings("kit started, before the scene")
        trace_kit_viewer()
        trace_fabric_paths()
        base_output = a.output
        session = 0
        while True:
            # With --mode-switch, M closes this scene and rebuilds the same pitch with
            # the other robot configuration (walking feet / roller feet) in this window.
            # Each scene keeps its own records in a numbered subfolder.
            if a.mode_switch:
                a.output = base_output / f"{session:02d}-{a.locomotion}"
                a.output.mkdir()
            import isaaclab_tasks
            import isaaclab.sim as sim_utils
            from isaaclab_tasks.utils import parse_env_cfg
            from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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
            if a.locomotion == "rollers":
                from roller_config import configure_rollers

                scene_profile["locomotion"] = configure_rollers(cfg)
            scene_profile["robot_asset_sha256"] = hashlib.sha256(
                Path(cfg.scene.robot.spawn.usd_path).read_bytes()
            ).hexdigest()
            if a.autonomous or a.record or a.locomotion == "rollers":
                from isaaclab.managers import EventTermCfg
                from isaaclab.sensors import ContactSensorCfg
                from roller_vision import create_sensors

                if a.autonomous or a.learning_config:
                    cfg.events.beta_sensors = EventTermCfg(
                        func=create_sensors, mode="prestartup"
                    )
                cfg.scene.ball_contact = ContactSensorCfg(
                    prim_path=(
                        "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/tire(_[2-4])?"
                        if a.locomotion == "rollers"
                        else "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle"
                    ),
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
            locomotion_path = (
                a.models / "official-browser/rollers.onnx"
                if a.locomotion == "rollers"
                else a.walking_model or a.models / "walk.onnx"
            )
            hashes["locomotion"] = hashlib.sha256(locomotion_path.read_bytes()).hexdigest()
            walk = BatchPolicy(locomotion_path, hashes["locomotion"], "cuda:0")
            if a.locomotion == "rollers":
                from roller_config import check_roller_articulation

                scene_profile["articulation"] = check_roller_articulation(
                    env.scene["robot"], env.observation_manager.compute()["policy"]
                )
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
            flags = dict(
                reset=False,
                kick=False,
                quit=False,
                started=not a.wait_for_start,
                takeover=False,
                switch=False,
                switch_armed=False,
            )
            events = []
            kick_start = None
            fallen = False
            scored = False
            resets = 0
            if not a.headless_check:
                import carb.input
                import omni.appwindow
                import omni.ui as ui
                import omni.kit.viewport.utility as vu
                from omni.kit.hotkeys.core import get_hotkey_registry

                registry = get_hotkey_registry()
                suspended = []
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
                            "H",
                            "I",
                            "K",
                            "ENTER",
                            "M",
                        )
                    ):
                        if registry.deregister_hotkey(hotkey):
                            suspended.append(hotkey)
                inp = carb.input.acquire_input_interface()
                window = omni.appwindow.get_default_app_window()

                def key(event, *_):
                    name = str(getattr(event.input, "name", event.input)).upper()
                    if a.ui_benchmark:
                        if name == "ESCAPE":
                            flags["quit"] = True
                        return True
                    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                        if name == "H" and a.takeover_toggle and name not in held:
                            flags["takeover"] = not flags["takeover"]
                        # One switch per fresh M press, and only once this scene's policy has
                        # taken over: a press that is still held, auto-repeated or queued
                        # while the next scene loads must not switch the feet back again.
                        fresh_m = name == "M" and name not in held and flags["switch_armed"]
                        held.add(name)
                        if a.learning_config and name == "H":
                            status.text = (
                                "HUMAN CONTROL"
                                if (flags["takeover"] if a.takeover_toggle else True)
                                else "GR00T CONTROL"
                            )
                        if name == "R":
                            flags["quit" if a.learning_config else "reset"] = True
                        elif name == "SPACE" and not a.learning_config:
                            flags["kick"] = True
                        elif name == "ESCAPE":
                            flags["quit"] = True
                        elif name == "M" and a.mode_switch:
                            if fresh_m:
                                flags.update(switch=True, quit=True, switch_armed=False)
                        elif name == "ENTER" and a.wait_for_start:
                            flags["started"] = True
                        events.append({"key": name, "at": time.monotonic()})
                    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                        held.discard(name)
                    return True

                subscription = inp.subscribe_to_keyboard_events(window.get_keyboard(), key)
                # 1920 x 1080 UI points: on a HiDPI desktop (UI scale 2) the window doubles in
                # pixels, so the help panel covers the same share of the scene as at scale 1.
                scale = getattr(ui.Workspace, "get_dpi_scale", lambda: 1.0)() or 1.0
                window.resize(round(1920 * scale), round(1080 * scale))
                if live_profile:
                    viewport_record = apply_viewport_profile()
                    events.append({"viewport": viewport_record, "at": time.monotonic()})
                    print("COURSE_VIEWPORT " + json.dumps(viewport_record), flush=True)
                else:
                    vu.get_active_viewport().set_texture_resolution(
                        (1280, 720) if a.ui_benchmark in ("fast", "physics") else (1920, 1080)
                    )
                panel = ui.Window("Microduck • Keyboard football", width=480, height=330)
                panel.position_x = 20
                panel.position_y = 50
                # Dock the help as the active tab of the right-hand Property column (as Kit's
                # own panels do), so the viewport and the robot stay uncovered.
                panel.deferred_dock_in("Property", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
                with panel.frame:
                    with ui.VStack(spacing=8):
                        status = ui.Label("Waiting to start", height=32)
                        if a.learning_config:
                            if a.wait_for_start:
                                ui.Button(
                                    "Start episode (Enter)",
                                    clicked_fn=lambda: flags.update(started=True),
                                )
                            ui.Label(
                                "Press H: toggle HUMAN / GR00T"
                                if a.takeover_toggle
                                else "Hold H: take over; release H: return to policy"
                            )
                            ui.Label("W/S: propel/brake; Q/E: turn; I/K: look down/up")
                            ui.Label(
                                "Esc: end this episode. Restart with a new output directory."
                            )
                            ui.Label("Space is disabled; R ends this recording.")
                            ui.Label("Held H with no movement keys: zero forward request.")
                        else:
                            ui.Label(
                                "W: push  S: brake  A/D: unused"
                                if a.locomotion == "rollers"
                                else "W/S: forward/back   A/D: step sideways"
                            )
                            ui.Label(
                                "Q/E: turn   Space: gliding shot"
                                if a.locomotion == "rollers"
                                else "Q/E: turn   Space: right-foot kick"
                            )
                            ui.Label(
                                "Release keys to coast; S to brake. R: restart."
                                if a.locomotion == "rollers"
                                else "Release keys to stand. R: restart. Esc: exit."
                            )
                            ui.Label(
                                "After restart, release and press movement keys again."
                            )
                        ui.Label(
                            "Official passive-wheel policy"
                            if a.locomotion == "rollers"
                            else "Official walking + kicking ONNX policies"
                        )
                        if a.mode_switch:
                            ui.Label(
                                "M: switch to "
                                + ("walking feet" if a.locomotion == "rollers" else "roller feet")
                                + " (scene reloads)"
                            )
                        ui.Button(
                            "End recording" if a.learning_config else "Restart scene",
                            clicked_fn=lambda: (
                                flags.update(quit=True)
                                if a.learning_config
                                else flags.update(reset=True)
                            ),
                        )
                print("MANUAL_FOOTBALL_READY", flush=True)
            cmd = env.command_manager.get_term("base_velocity")
            robot = env.scene["robot"]
            rows = []
            start = time.monotonic()

            def T(x):
                return getattr(x, "torch", x)

            imu = None
            imu_rows = []
            if a.locomotion == "rollers":
                from roller_imu import (
                    IdealTrunkImu,
                    InertialVelocity,
                    RollerBrake,
                    braking_command,
                    turn_command,
                )

                imu = IdealTrunkImu(T(robot.data.root_lin_vel_w)[0].cpu().tolist())
                inertial = InertialVelocity()
                brake = RollerBrake()
            vision = None
            kick_issued = False
            touch = False
            early_touch = False
            kick_at = None
            contact_at = None
            recovered_since = None
            recovered_duration = 0.0
            physics_rows = []
            action_rows = []
            planned_push_seconds = a.shot_push_seconds
            shot_end_s = max(1.0, planned_push_seconds + 0.5)
            shot_heading_error = 0.0
            prepare_at = None
            prepare_quiet_at = None
            if a.autonomous:
                from roller_vision import RollerVision

                vision = RollerVision(env, a.detector_deployment, a.output)
                from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES

                servo_ids, _ = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)
                play_ids, _ = robot.find_joints(
                    ["passive_" + n + "_backlash" for n in MICRODUCK_JOINT_NAMES],
                    preserve_order=True,
                )
            learner = None
            if a.learning_config:
                from learning.runtime import LearningController

                learner = LearningController(
                    env, a.learning_config, a.output, hashes["locomotion"]
                )
                if a.ui_benchmark and not learner.cfg.get("performance_diagnostic"):
                    raise ValueError(
                        "UI benchmarks must be explicitly excluded from training"
                    )
                if a.ui_benchmark == "fast":
                    learner.render_subframes = 1
                elif a.ui_benchmark == "physics":
                    learner.diagnostic_no_rgb = True
            performance_rows = []
            live_physics = None
            if learner is not None:
                live_physics = (a.output / "physical-evaluation-live.jsonl").open(
                    "x", buffering=1
                )
            with torch.inference_mode():
                if a.wait_for_start:
                    status.text = "Paused. Click Start or press Enter when ready."
                    (a.output / "waiting-for-start.json").write_text(
                        json.dumps({"state": "waiting", "physics_steps": 0})
                    )
                    print("LEARNING_WAITING_FOR_START", flush=True)
                    while not flags["started"] and not flags["quit"]:
                        env.sim.render()
                        time.sleep(0.02)
                    (a.output / "waiting-for-start.json").write_text(
                        json.dumps(
                            {
                                "state": (
                                    "started"
                                    if flags["started"] and not flags["quit"]
                                    else "cancelled"
                                ),
                                "wall_time": time.time(),
                            }
                        )
                    )
                # Course keyboard sessions use the standing61 start-up: settle on the
                # ground with temporarily locked joints, release before the first action.
                start_brake = None
                if a.mode_switch:
                    sys.path.insert(0, str(Path(__file__).parent / "course_labs"))
                    from grounded_start import settle as _settle, StartupNotSettled

                    def settle(env, render):
                        try:
                            return _settle(
                                env,
                                pose="roller-rest" if a.locomotion == "rollers" else "home",
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
                        if learner is not None:
                            break
                        env.reset()
                        if imu is not None:
                            imu = IdealTrunkImu(
                                T(robot.data.root_lin_vel_w)[0].cpu().tolist()
                            )
                            inertial = InertialVelocity()
                            brake = RollerBrake()
                        flags.update(reset=False, kick=False)
                        held.clear()
                        kick_start = None
                        fallen = False
                        scored = False
                        resets += 1
                        if a.mode_switch:
                            start_brake, startup = settle(env, render=True)
                            events.append({"startup": startup, "at": time.monotonic()})
                    if a.headless_check:
                        held.clear()
                        if a.check_motion == "drive":
                            if 100 <= step < 250:
                                held.add("W")
                            elif 250 <= step < 350 and a.locomotion == "rollers":
                                held.add("S")
                            elif 350 <= step < 450:
                                held.add("Q")
                        if a.check_motion == "kick" and step == 150:
                            flags["kick"] = True
                    visual_command = None
                    if vision is not None and step > 0:
                        observation = dict(
                            timestamp_s=step * 0.02,
                            joint_names=list(MICRODUCK_JOINT_NAMES),
                            q_rad=(
                                T(robot.data.joint_pos)[0, servo_ids]
                                + T(robot.data.joint_pos)[0, play_ids]
                            )
                            .cpu()
                            .tolist(),
                            qd_rad_s=(
                                T(robot.data.joint_vel)[0, servo_ids]
                                + T(robot.data.joint_vel)[0, play_ids]
                            )
                            .cpu()
                            .tolist(),
                            gyro_body_rad_s=T(robot.data.root_ang_vel_b)[0].cpu().tolist(),
                            gravity_body=T(robot.data.projected_gravity_b)[0]
                            .cpu()
                            .tolist(),
                        )
                        visual_command, head_offsets, request = vision.step(
                            step * 0.02, observation, inertial.velocity.tolist()
                        )
                        if request and prepare_at is None:
                            prepare_at = step * 0.02
                            planned_push_seconds = a.shot_push_seconds + max(
                                0.0, min(0.5, (float(vision.ball[0]) - 0.18) / 0.35)
                            )
                            shot_end_s = planned_push_seconds + 0.5
                        if prepare_at is not None and not kick_issued:
                            visual_command = [
                                0.0,
                                0.0,
                                turn_command(
                                    vision.goal_heading or 0.0,
                                    observation["gyro_body_rad_s"][2],
                                ),
                            ]
                            quiet = (
                                sum(v * v for v in observation["gyro_body_rad_s"]) < 0.25**2
                            )
                            prepare_quiet_at = (
                                (
                                    step * 0.02
                                    if prepare_quiet_at is None
                                    else prepare_quiet_at
                                )
                                if quiet
                                else None
                            )
                            elapsed = step * 0.02 - prepare_at
                            if (
                                2.0 <= elapsed <= 4.0
                                and abs(vision.goal_heading or 0.0) < 0.04
                                and prepare_quiet_at is not None
                                and step * 0.02 - prepare_quiet_at >= 0.4
                            ):
                                flags["kick"] = True
                            elif elapsed > 4.0:
                                vision.failure = "neutral_head_handoff_not_quiet"
                                break
                    if flags["kick"] and kick_start is None and not fallen:
                        kick_start = step
                        kick_at = step * 0.02
                        shot_heading_error = (
                            float(vision.goal_heading or 0.0) if vision is not None else 0.0
                        )
                        kick_issued = True
                        flags["kick"] = False
                    brake_requested = False
                    cmd.is_standing_env[:] = False
                    cmd.vel_command_b[:] = 0
                    if kick_start is None and not fallen:
                        cmd.vel_command_b[0] = torch.tensor(
                            [
                                (0.6 if a.locomotion == "rollers" else 0.25) * ("W" in held)
                                - (0.5 if a.locomotion == "rollers" else 0.18)
                                * ("S" in held),
                                (
                                    0.0
                                    if a.locomotion == "rollers"
                                    else 0.15 * ("A" in held) - 0.15 * ("D" in held)
                                ),
                                (0.3 if a.locomotion == "rollers" else 0.45)
                                * (("Q" in held) - ("E" in held)),
                            ],
                            device="cuda:0",
                        )
                    if (
                        kick_issued
                        and kick_start is None
                        and a.locomotion == "rollers"
                        and not held
                    ):
                        brake_requested = True
                    if (
                        visual_command is not None
                        and kick_start is None
                        and not fallen
                        and not kick_issued
                    ):
                        cmd.vel_command_b[0] = torch.tensor(
                            visual_command, device=env.device
                        )
                        if visual_command[0] <= 0.0:
                            brake_requested = True
                    if (
                        kick_start is not None
                        and a.locomotion == "rollers"
                        and a.roller_shot == "glide"
                    ):
                        shot_phase = (step - kick_start) * 0.02
                        # A new gliding strike: propulsion then a brief braking pulse.
                        # Wheels remain passive; no walking foot or teacher substitution.
                        cmd.vel_command_b[0, 0] = (
                            0.6 if shot_phase < planned_push_seconds else 0.0
                        )
                        brake_requested = shot_phase >= planned_push_seconds
                        # Relative heading from the same ideal body gyro already
                        # declared by this lab; no world yaw or ball pose input.
                        shot_heading_error -= (
                            T(robot.data.root_ang_vel_b)[0, 2].item() * 0.02
                        )
                        cmd.vel_command_b[0, 2] = (
                            0.0  # Let the official actor stabilize its own heading during contact.
                        )
                    if imu is not None:
                        if brake_requested:
                            if kick_issued:
                                brake.reset()
                                cmd.vel_command_b[0, 0] = braking_command(
                                    inertial.velocity[0]
                                )
                            else:
                                cmd.vel_command_b[0, 0] = brake.update(inertial.velocity[0])
                        else:
                            brake.reset()
                    learning_action = None
                    learning_started = time.monotonic()
                    if learner is not None:
                        control_held = set(held)
                        if a.takeover_toggle:
                            control_held.discard("H")
                            if flags["takeover"]:
                                control_held.add("H")
                        learning_action = learner.step(
                            step, inertial.velocity.tolist(), control_held
                        )
                        cmd.vel_command_b[0] = torch.tensor(
                            learner.motor_command(inertial.velocity.tolist()),
                            device=env.device,
                        )
                    learning_elapsed = time.monotonic() - learning_started
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
                        if weight and (
                            a.locomotion != "rollers" or a.roller_shot == "teacher"
                        ):
                            action = torch.lerp(action, kick(obs), weight)
                        if phase >= shot_end_s:
                            kick_start = None
                            flags["kick"] = False
                    raw_action = action.clone()
                    if vision is not None and step * 0.02 >= 2.0:
                        # Explicit head controller, recorded in the executed action history.
                        # Roller policy was not trained for walking gaze commands.
                        gaze_weight = (
                            1.0
                            if prepare_at is None
                            else max(0.0, 1.0 - (step * 0.02 - prepare_at) / 0.4)
                        )
                        action[:, 5:9] = torch.lerp(
                            action[:, 5:9],
                            torch.tensor([vision.head_offsets], device=env.device),
                            gaze_weight,
                        )
                    if learning_action is not None:
                        learner.apply_head(action)
                    action_rows.append(
                        dict(
                            timestamp_s=step * 0.02,
                            observation=obs[0].cpu().tolist(),
                            raw_action=raw_action[0].cpu().tolist(),
                            executed_action=action[0].cpu().tolist(),
                            command=cmd.vel_command_b[0].cpu().tolist(),
                            inertial_velocity_body_m_s=(
                                None if imu is None else inertial.velocity.tolist()
                            ),
                            brake_active=brake_requested,
                            brake_integral_m=brake.integral if imu is not None else None,
                        )
                    )
                    if start_brake is not None and start_brake.active:
                        start_brake.release(step * 0.02)
                        flags["switch_armed"] = True
                        print(
                            "COURSE_POLICY_TAKEOVER "
                            + json.dumps(dict(t=step * 0.02, locomotion=a.locomotion)),
                            flush=True,
                        )
                    env.action_manager.process_action(action)
                    # Telemetry is read with one GPU->CPU copy per physics tick (packed below);
                    # the ball's position before tick n+1 is its position after tick n.
                    before_ball = T(env.scene["ball"].data.root_pos_w)[0].cpu().tolist()
                    need_force = a.record or vision is not None or a.locomotion == "rollers"
                    for tick in range(4):
                        env._sim_step_counter += 1
                        env.action_manager.apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(
                            render=not a.headless_check
                            and tick == 3
                            and (a.ui_benchmark not in ("fast", "physics") or step % 2 == 0)
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
                        if imu is not None:
                            parts += [
                                T(robot.data.root_lin_vel_w)[0],
                                T(robot.data.root_quat_w)[0],
                                T(robot.data.root_ang_vel_b)[0],
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
                        if imu is not None:
                            packet = imu.sample(
                                packed[9:12],
                                packed[12:16],
                                packed[16:19],
                                step * 0.02 + (tick + 1) * 0.005,
                            )
                            estimate = inertial.update(packet)
                            imu_rows.append(
                                dict(
                                    **packet, estimated_velocity_body_m_s=estimate.tolist()
                                )
                            )
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
                        if live_physics is not None:
                            live_physics.write(json.dumps(physics_rows[-1]) + "\n")
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
                    if learner is not None and fallen:
                        break
                    if vision is not None and (
                        fallen
                        or early_touch
                        or (
                            kick_at is not None
                            and (
                                (scored and recovered_duration >= 1.0)
                                or (step + 1) * 0.02 >= kick_at + shot_end_s + 8
                            )
                        )
                    ):
                        break
                    if not a.headless_check:
                        text = (
                            f"{'HUMAN' if learner.last_gate else 'GR00T'} | {(step + 1) * 0.02:.1f} s | {learner.sequence} frames"
                            if learner is not None
                            else (
                                "Fallen — press R to restart."
                                if fallen
                                else (
                                    "Goal! Press R to try again."
                                    if scored
                                    else "Position the ball beside the right foot; release keys; Space."
                                )
                            )
                        )
                        if status.text != text:  # a Kit UI update only when the text changes
                            status.text = text
                        if live_profile:
                            pacer.wait()
                        else:
                            time.sleep(max(0, 0.02 - (time.monotonic() - tick_start)))
                    performance_rows.append(
                        dict(
                            step=step,
                            wall_s=time.monotonic() - tick_start,
                            learning_s=learning_elapsed,
                        )
                    )
            if not a.headless_check:
                inp.unsubscribe_to_keyboard_events(window.get_keyboard(), subscription)
                panel.visible = False
                panel.destroy()
            if live_physics is not None:
                live_physics.close()
            (a.output / "performance.json").write_text(
                json.dumps(
                    dict(ui_benchmark=a.ui_benchmark, steps=performance_rows), indent=2
                )
            )
            if learner is not None:
                learner.close()
            if vision is not None:
                vision.close()
            if video is not None:
                video.stdin.close()
                if video.wait(timeout=30):
                    raise RuntimeError("Recording failed")
            report = dict(
                status="completed",
                locomotion=a.locomotion,
                seed=a.seed,
                check_motion=a.check_motion,
                roller_shot=a.roller_shot,
                fixture=dict(
                    root_xy=[a.root_x, a.root_y],
                    ball_xy=[a.ball_x, a.ball_y],
                    shot_push_seconds=a.shot_push_seconds,
                    planned_push_seconds=planned_push_seconds,
                ),
                autonomous=a.autonomous,
                autonomous_scope="RGB/ToF visual servo with ideal trunk IMU and joint encoders; no Nav2 or walking sole odometry",
                task_result=dict(
                    kick_issued=kick_issued,
                    foot_ball_contact=touch,
                    early_contact=early_touch,
                    whole_ball_in_goal=scored,
                    fallen=fallen,
                    recovered=recovered_duration >= 1.0,
                    recovery_continuous_seconds=recovered_duration,
                    visual_request_at_s=prepare_at,
                    kick_at_s=kick_at,
                    contact_at_s=contact_at,
                    contact_window_s=shot_end_s,
                    passed=bool(
                        a.autonomous
                        and kick_issued
                        and touch
                        and not early_touch
                        and scored
                        and not fallen
                        and recovered_duration >= 1.0
                    ),
                ),
                joint_names=robot.joint_names,
                models=hashes,
                physics_backend=manager.backend_name,
                brake_controller=(
                    dict(
                        kind="PI_wait_P_stop",
                        kp=3.0,
                        ki=4.0,
                        limits=[-0.5, 0.35],
                        integral_limits=[-0.15, 0.15],
                        dt_s=0.02,
                    )
                    if imu is not None
                    else None
                ),
                imu_velocity_estimator=(
                    "ideal specific-force integration from zero reset; no bias/noise calibration"
                    if imu is not None
                    else None
                ),
                physics_hz=200,
                policy_hz=50,
                resets=resets,
                fell=fallen,
                goal_observed=scored,
                scene_profile=scene_profile,
                events=events,
                scope=(
                    "autonomous visual gliding experiment; complete success requires independent scoring"
                    if a.autonomous
                    else "keyboard skill exploration; goal position observation is not autonomous-task certification"
                ),
            )
            (a.output / "imu.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in imu_rows)
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
            if not flags["switch"]:
                break
            a.locomotion = "rollers" if a.locomotion == "walking" else "walking"
            session += 1
            sim_utils.create_new_stage()


if __name__ == "__main__":
    main()
