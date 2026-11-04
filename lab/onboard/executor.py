#!/usr/bin/env python3
"""Executor: Microduck simulation and control."""
from __future__ import annotations
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

ONBOARD_DIR = Path(__file__).resolve().parent
ENTRY_DIR = ONBOARD_DIR.parent / "groot_walking/student"
sys.path.insert(0, str(ONBOARD_DIR))
from schedule import Schedule
SCHEDULE = Schedule(100, 50, 15)
SCHEDULE.after_tick = lambda tick, env: None
SESSION = None
sys.path.insert(0, str(ENTRY_DIR))
# v121: explicit roots; never infer M from the location of a prepared copy.
from paths import M, K10_DEPLOYMENT
# Official kick and standing teachers; no residual model is loaded.
OFFICIAL_KICK_RUNTIME = ONBOARD_DIR / "official_kick_runtime.py"
sys.path.insert(0, str(OFFICIAL_KICK_RUNTIME.parent))

ROOT = M / "football"
MEDIA_DIR = ROOT / "recording"
sys.path[:0] = [
    str(ENTRY_DIR),
    str(M / "evaluation"),
    str(M / "gait/training/body_imu"),
    str(M / "evaluation/body50_training"),
]
if str(MEDIA_DIR) not in sys.path:
    sys.path.insert(0, str(MEDIA_DIR))
for folder in (ROOT, M / "sensors", M / "gait"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arguments():
    from isaaclab.app import add_launcher_args

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument(
        "--trained-recipe-dir",
        type=Path,
        help="Frozen training run for exact internal8 action class/raw history; mutually exclusive with external bounded-actions",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--development",
        action="store_true",
        required=True,
        help="This entry only runs development episodes",
    )
    p.add_argument("--seconds", type=float, default=0.0)
    p.add_argument("--experiment-plan", type=Path)
    p.add_argument("--seed", type=int, default=971001)
    p.add_argument("--layout", choices=["empty", "one-obstacle"], default="empty")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--assumed-hfov", type=float, default=70.0)
    p.add_argument("--render-subframes", type=int, default=1)
    p.add_argument(
        "--shared-imu-readback",
        action="store_true",
        help="Experimental transport only: one private head/body GPU readback per 100 Hz tick; unchanged sensor samples/decoders",
    )
    p.add_argument(
        "--packed-physics-readback",
        action="store_true",
        help="Experimental lossless grouped200Hz diagnostic/encoder readback; same physical samples and scorer",
    )
    p.add_argument(
        "--verify-readback",
        action="store_true",
        help="Development only: compare every grouped field and delivered IMU packet with the original producer on the same physical tick",
    )
    p.add_argument(
        "--calibration-command",
        type=float,
        nargs=3,
        help="Development only: fixed raw gait command after fixture release, no navigation/kick",
    )
    p.add_argument(
        "--calibration-gaze",
        type=float,
        nargs=2,
        default=[0.0, 0.0],
        help="Development only: fixed body-relative yaw/pitch goal, explicitly slewed at .6rad/s; requires calibration-command",
    )
    p.add_argument("--detector-backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--detector-device", choices=["cpu", "cuda:0"], default="cpu")
    p.add_argument(
        "--kick-model",
        type=Path,
        help="Legacy argument rejected by this fixed K10 staged entry",
    )
    p.add_argument("--kick-after-s", type=float, default=1.0)
    p.add_argument("--kick-duration-s", type=float, default=0.5)
    p.add_argument(
        "--max-target-rate",
        type=float,
        default=3.0,
        help="Explicit development limiter assumption; part of the model adapter",
    )
    p.add_argument(
        "--bounded-actions",
        action="store_true",
        help="Apply explicit official14joint target margins and slew limit, retain raw/executed action trace",
    )
    p.add_argument(
        "--body-added-delay-s",
        type=float,
        default=0.0,
        help="Explicit unmeasured extra read-to-policy transport delay; median3 already models decoder phase.0 is zero-added-delay baseline, .020 retained conservative v9 assumption",
    )
    p.add_argument(
        "--body-fusion-tau-s",
        type=float,
        default=0.5,
        help="Unmeasured virtual complementary-fusion time constant; explicit sensitivity parameter, not ST SFLP configuration",
    )
    p.add_argument(
        "--policy-imu",
        choices=["body50", "head-fifo"],
        default="body50",
        help="Official body50 control IMU; head-fifo preserves the prior failed development architecture",
    )
    p.add_argument(
        "--navigation-ground",
        choices=["legacy-imu-sole", "tof-virtual-floor"],
        default="tof-virtual-floor",
        help="Explicit64zone virtual forward model; not calibrated ST target-selection firmware",
    )
    p.add_argument(
        "--randomized",
        action="store_true",
        help="Retain official DR; independently report this setting",
    )
    p.add_argument(
        "--root-y",
        type=float,
        default=None,
        help="Explicit development fixture, otherwise seeded uniform[-.08,.08]",
    )
    p.add_argument(
        "--root-x",
        type=float,
        default=-0.66,
        help="Explicit development initialization only; formal layout always starts at -.66",
    )
    p.add_argument("--root-yaw-rad", type=float, default=0.0,
                   help="Additional world-Z yaw of the standing61 pose, within +/-pi/6")
    p.add_argument(
        "--ball-xy",
        type=float,
        nargs=2,
        default=None,
        help="Explicit development fixture, otherwise seeded prescribed range",
    )
    p.add_argument(
        "--robot-usd",
        type=Path,
        default=M / "visuals/assets/microduck_white_orange.usda",
    )
    p.add_argument(
        "--recovery-approval",
        type=Path,
        required=True,
        help="v121: root-written approval manifest binding Stage A backward calibration, Stage B near-ball no-contact evidence, calibrated parameters and these exact sources; test-only manifests are refused",
    )
    detector = p.add_mutually_exclusive_group(required=True)
    detector.add_argument(
        "--detector-deployment",
        type=Path,
        help="Actual new Beta ONNX detector deployment manifest; independent CPU process",
    )
    detector.add_argument(
        "--development-contour",
        action="store_true",
        help="Explicit image-only development baseline; never report as YOLO validation",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="Explicit developer-only diagnostic; frames and traces are still retained",
    )
    add_launcher_args(p)
    p.add_argument(
        "--course-onboard",
        action="store_true",
        help="Onboard route (YOLO detector, straight path, continuous approach) with the course joint-brake start-up",
    )
    p.add_argument("--interactive-reset", action="store_true", help="Keep the Isaac window open; R starts another small-offset episode")
    p.add_argument("--session-episodes", type=int, default=0, help="Optional session limit; 0 has no limit")
    a = p.parse_args()
    a.experiment = None
    if a.experiment_plan:
        from experiment_plan import load_plan
        a.experiment = load_plan(a.experiment_plan)
        plan = a.experiment['document']
        if not a.interactive_reset or a.seconds != 30 or not a.no_video:
            p.error('Trial plans require 30 seconds, no video and an interactive session')
        if a.session_episodes not in (0, len(plan['episodes'])):
            p.error('Session length must match the frozen experiment plan')
        a.session_episodes = len(plan['episodes'])
    if a.session_episodes < 0:
        p.error("Session limit must be nonnegative")
    if a.interactive_reset and "kit" not in (getattr(a, "visualizer", None) or []):
        p.error("Interactive reset requires the Kit window")
    if not a.course_onboard:
        p.error("Use the course onboard entry")
    if a.detector_deployment is None:
        p.error("The onboard route requires the trained detector deployment")
    a.enable_cameras = True
    if a.device is None:
        a.device = "cuda:0"
    if not math.isfinite(a.seconds) or not 0 <= a.seconds <= 180:
        p.error("Duration must be in [0,180]; zero waits for manual reset")
    if a.output.exists():
        p.error("Use a new output directory; never overwrite a failed episode")
    if not a.model.is_file() or not a.robot_usd.is_file():
        p.error("Model and current backlash USD must exist")
    if not a.no_video and shutil.which("ffmpeg") is None:
        p.error("ffmpeg is required for browser-readable dual-view video")
    if a.detector_deployment is not None and (a.width, a.height) != (960, 540):
        p.error("Trained Beta detector uses actual960x540 RGB")
    if not math.isfinite(a.max_target_rate) or a.max_target_rate <= 0:
        p.error("Finite positive target rate required")
    if (
        not math.isfinite(a.body_added_delay_s)
        or not 0 <= a.body_added_delay_s <= 0.060
    ):
        p.error("Extra body transport delay must be in[0,.060]")
    if not math.isfinite(a.body_fusion_tau_s) or not 0.1 <= a.body_fusion_tau_s <= 10.0:
        p.error("Declared virtual fusion time constant must be in[.1,10]")
    if a.render_subframes < 1:
        p.error("At least one actual render subframe is required")
    if a.verify_readback and not (a.shared_imu_readback and a.packed_physics_readback):
        p.error("Readback comparison requires both explicit processing transports")
    if a.calibration_command is not None and not all(
        math.isfinite(v) for v in a.calibration_command
    ):
        p.error("Finite calibration command required")
    if a.trained_recipe_dir is not None and (
        a.bounded_actions or a.policy_imu != "body50"
    ):
        p.error(
            "Trained internal adapter requires body50 and forbids duplicate external target limiting"
        )
    if a.trained_recipe_dir is not None and not a.trained_recipe_dir.is_dir():
        p.error("Frozen trained recipe directory missing")
    if a.kick_model is not None and (
        a.calibration_command != [0.0, 0.0, 0.0]
        or not (a.bounded_actions or a.trained_recipe_dir is not None)
    ):
        p.error(
            "Kick development requires explicit zero-command calibration and one explicit bounded adapter"
        )
    if a.kick_model is not None and not a.kick_model.is_file():
        p.error("Kick model file missing")
    if a.root_y is not None and not math.isfinite(a.root_y):
        p.error("Finite root-y required")
    if not math.isfinite(a.root_x):
        p.error("Finite root-x required")
    if not math.isfinite(a.root_yaw_rad) or abs(a.root_yaw_rad) > math.pi / 6 + 1e-12:
        p.error("Finite root-yaw-rad within +/-pi/6 required")
    if not all(math.isfinite(v) for v in a.calibration_gaze) or (
        a.calibration_gaze != [0.0, 0.0] and a.calibration_command is None
    ):
        p.error("Finite calibration gaze requires fixed command mode")
    if a.ball_xy is not None and not all(math.isfinite(v) for v in a.ball_xy):
        p.error("Finite ball-xy required")
    if a.calibration_command is not None:
        p.error(
            "v121 recovery entry is not a calibration executor; use the frozen v117 calibration mode"
        )
    from recovery_config import verify_approval, ApprovalError

    try:
        # The opt-in VLA entry loads this newer executor beside the installed
        # frozen runtime. This check still binds the preserved recovery helper;
        # it is not approval or qualification of the new learned controller.
        # Both course routes run from this student folder, so both bind the installed one.
        recovery_dir = M / "evaluation/football"
        a.recovery_approval_verified = verify_approval(
            a.recovery_approval, parameters_dir=recovery_dir, allow_test_only=False
        )
    except ApprovalError as error:
        p.error("v121 retreat recovery refused: " + str(error))
    return a


class Video:
    def __init__(self, path, width, height):
        self.path = path
        self.log = (path.with_suffix(".ffmpeg.log")).open("w")
        self.proc = subprocess.Popen(
            [
                "ffmpeg",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                "15",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=self.log,
        )
        self.count = 0

    def add(self, rgb):
        self.proc.stdin.write(rgb.tobytes())
        self.count += 1

    def close(self):
        if self.proc.stdin is not None:
            self.proc.stdin.close()
            self.proc.stdin = None
        code = self.proc.wait(timeout=60)
        self.log.close()
        return {
            "path": self.path.name,
            "frames": self.count,
            "nominal_fps": 15,
            "exit_code": code,
            "sha256": sha(self.path) if self.path.exists() else None,
        }


def configure_visual_assets(env):
    """Refine only the visible USD sphere; Newton sphere collision stays analytic."""
    from pxr import Sdf, Usd, UsdGeom

    ball = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Ball")
    paths = []
    for prim in Usd.PrimRange(ball):
        if prim.IsA(UsdGeom.Sphere):
            prim.CreateAttribute(
                "refinementEnableOverride", Sdf.ValueTypeNames.Bool, custom=True
            ).Set(True)
            prim.CreateAttribute(
                "refinementLevel", Sdf.ValueTypeNames.Int, custom=True
            ).Set(2)
            paths.append(str(prim.GetPath()))
    if not paths:
        raise RuntimeError("Expected the actual USD sphere for visual refinement")
    env.course_visual_geometry = {
        "sphere_refinement_level": 2,
        "sphere_prims": paths,
        "collision_changed": False,
        "sphere_radius_m": 0.035,
        "sphere_mass_kg": 0.045,
        "pitch_visual_top_z_m": 0.0002,
        "pitch_has_collision": False,
    }


def sensor_prestartup(env, env_ids, *, width, height, hfov, seed):
    """Called by the environment before the first PHYSICS_READY event."""
    from isaac_beta_sensor_rig import IsaacBetaSensors
    from beta_sensor_profile import BetaCameraProfile
    from isaaclab.sensors.camera import Camera, CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    import isaaclab.sim as sim_utils

    from performance import pin_main_thread
    pin_main_thread()
    configure_visual_assets(env)
    env.beta_rig = IsaacBetaSensors(
        env.sim,
        env.scene["robot"],
        mount_profile="rl-jaw-soft-20261012",
        camera_profile=BetaCameraProfile(
            width_px=width, height_px=height, assumed_horizontal_fov_deg=hfov
        ),
        seed=seed,
    )
    focal = width / (2 * math.tan(math.radians(60) / 2))
    K = [focal, 0, width / 2, 0, focal, height / 2, 0, 0, 1]
    env.course_overhead = Camera(
        CameraCfg(
            prim_path="/World/envs/env_0/BetaSensors/Overhead",
            width=width,
            height=height,
            update_period=0.0,
            data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                K, width, height, clipping_range=(0.03, 20.0), focal_length=20.0
            ),
        )
    )


def build_cfg(a):
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg, AssetBaseCfg
    from isaaclab.managers import EventTermCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from onboard_geometry.obstacle_contract import OBSTACLE

    cfg = parse_env_cfg(
        "IsaacContrib-Velocity-Flat-MicroDuck",
        device=a.device,
        num_envs=1,
        overrides=["presets=backlash"],
    )
    if getattr(a, "trained_recipe_dir", None) is not None:
        import importlib

        a.trained_recipe_dir = a.trained_recipe_dir.resolve()
        if any(
            name in sys.modules
            for name in ("bounded_actions", "tracking_rewards", "cycle_tracking")
        ):
            raise RuntimeError("Frozen trained action import contaminated")
        sys.path.insert(0, str(a.trained_recipe_dir))
        cfg.actions.joint_pos.class_type = importlib.import_module(
            "cycle_tracking"
        ).BoundedTrainingAction
        metadata = json.loads(a.model.with_suffix(".json").read_text())
        if (
            metadata["sha256"] != sha(a.model)
            or Path(metadata["checkpoint"]).resolve().parent != a.trained_recipe_dir
            or sha(metadata["checkpoint"]) != metadata["checkpoint_sha256"]
        ):
            raise ValueError("Export does not bind this frozen training run")
    cfg.seed = a.seed
    cfg.episode_length_s = a.seconds + 5.0 if a.seconds else 1e12
    cfg.curriculum = None
    cfg.sim.dt = SCHEDULE.physics_dt
    cfg.decimation = SCHEDULE.decimation or SCHEDULE.P // SCHEDULE.C
    cfg.sim.render_interval = SCHEDULE.decimation or SCHEDULE.P // SCHEDULE.C
    # Extra ball/goal contacts need more capacity than walking alone.
    cfg.sim.physics.solver_cfg.nconmax = 64
    cfg.sim.physics.solver_cfg.njmax = 256
    cfg.scene.robot.spawn.usd_path = str(a.robot_usd.resolve())
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = (1e8, 1e8)
    cfg.commands.head_pose.resampling_time_range = (1e8, 1e8)
    cfg.commands.head_pose.ranges = tuple((0.0, 0.0) for _ in range(4))
    cfg.commands.body_pose.resampling_time_range = (1e8, 1e8)
    cfg.commands.body_pose.ranges = tuple((0.0, 0.0) for _ in range(6))
    cfg.events.reset_base.params["pose_range"] = {}
    if not a.randomized:
        for name in (
            "foot_friction",
            "encoder_bias",
            "mass_inertia",
            "randomize_com",
            "randomize_head_com",
            "randomize_joint_friction",
            "randomize_armature",
            "push_robot",
        ):
            setattr(cfg.events, name, None)
        cfg.scene.robot.actuators["servos"].vin_range = None
        cfg.scene.robot.actuators["servos"].vin_drop_gain_range = None
        cfg.scene.robot.actuators["servos"].min_delay = SCHEDULE.delay_steps
        cfg.scene.robot.actuators["servos"].max_delay = SCHEDULE.delay_steps
        cfg.observations.policy.enable_corruption = False
        cfg.observations.policy.imu.params.update(max_angle_deg=0, min_lag=0, max_lag=0)
    cfg.events.beta_sensors = EventTermCfg(
        func=sensor_prestartup,
        mode="prestartup",
        params={
            "width": a.width,
            "height": a.height,
            "hfov": a.assumed_hfov,
            "seed": a.seed,
        },
    )
    rng = np.random.default_rng(a.seed)
    root_y = float(rng.uniform(-0.08, 0.08)) if a.root_y is None else a.root_y
    ball_xy = (
        [float(rng.uniform(-0.08, 0.12)), float(rng.uniform(-0.08, 0.08))]
        if a.ball_xy is None
        else a.ball_xy
    )
    root_x = getattr(a, "root_x", -0.66)
    cfg.scene.robot.init_state.pos = (root_x, root_y, 0.125)
    cfg.scene.ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.035,
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.045),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.9, 0.91, 0.88), roughness=0.75
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(*ball_xy, 0.035)),
    )
    # Opening is .24m: post centres +/- .1325, each .025m thick.
    for name, size, pos in [
        ("goal_left", (0.025, 0.025, 0.1925), (0.93, 0.1325, 0.09625)),
        ("goal_right", (0.025, 0.025, 0.1925), (0.93, -0.1325, 0.09625)),
        ("goal_bar", (0.025, 0.29, 0.025), (0.93, 0.0, 0.1925)),
    ]:
        setattr(
            cfg.scene,
            name,
            AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/" + name,
                spawn=sim_utils.CuboidCfg(
                    size=size,
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.9, 0.92, 0.88), roughness=0.7
                    ),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=pos),
            ),
        )
    cfg.scene.pitch = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PitchVisual",
        spawn=sim_utils.CuboidCfg(
            size=(2.0, 1.0, 0.003),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.32, 0.16), roughness=0.85
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.0013)),
    )
    cfg.scene.sky_light.spawn.intensity = 1800.0
    # Sensor body is the ball; partner order is left then right. These are
    # forces ON the ball from the individual feet, not net ground contacts.
    feet = [
        "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_" + side
        for side in ("left", "right")
    ]
    cfg.scene.football_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        filter_prim_paths_expr=feet,
        history_length=1,
        update_period=SCHEDULE.physics_dt,
    )
    if a.layout == "one-obstacle":
        pos = (*OBSTACLE["center_xy_m"], OBSTACLE["height_m"] / 2)
        size = (*OBSTACLE["size_xy_m"], OBSTACLE["height_m"])
        cfg.scene.obstacle = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/KnownObstacle",
            spawn=sim_utils.CuboidCfg(
                size=size,
                rigid_props=UsdPhysicsRigidBodyCfg(
                    rigid_body_enabled=True, kinematic_enabled=True
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                activate_contact_sensors=True,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 0.2, 0.65), roughness=0.8
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        )
        cfg.scene.obstacle_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/KnownObstacle",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/Geometry/.*"],
            history_length=1,
            update_period=SCHEDULE.physics_dt,
        )
    from ground_contact_evidence import configure_ground_filter

    configure_ground_filter(cfg)
    # The default rendered ground asset also has a visual mesh. Select only its actual collision plane.
    cfg.scene.contact_forces.filter_shape_prim_expr = [
        cfg.scene.terrain.prim_path + "/terrain/GroundPlane/CollisionPlane"
    ]
    return cfg, {
        "root_pose_xyzw": [root_x, root_y, 0.125, 0.0, 0.0, 0.0, 1.0],
        "ball_pose_xyzw": [*ball_xy, 0.035, 0.0, 0.0, 0.0, 1.0],
        "goal_plane_x_m": 0.93,
        "goal_opening_width_m": 0.24,
        "goal_opening_height_m": 0.18,
        "ball_radius_m": 0.035,
        "ball_mass_kg": 0.045,
        "seed": a.seed,
        "layout": a.layout,
        "scope": "Initialization and independent evaluation only; never controller input",
    }


def run(a, cfg, fixture):
    import numpy as np, torch, gymnasium as gym
    from PIL import Image, ImageDraw
    from isaac_beta_sensor_rig import array
    from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES
    from batch_policy import BatchPolicy
    from controller import BetaNavigator, ControllerConfig
    from path_request import quaternion_xyzw
    from head_adapter import HeadAdapter
    from imu_predictor import DeliveredBodyImu
    from body_imu_input import BodyImuInput
    from isaac_beta_body_imu import IsaacBodyImu
    from beta_body_imu import BetaBodyImuProfile
    from kick_control import KickPulse, validate_actor
    from mechanical_scoring import PhysicalGoalScorerV2
    from bounded_actions import load_limits
    from startup_support import NewtonSupportAdapter
    # Course initialization (standing61): temporary joint brake at HOME on the measured
    # grounded root pose, free root, zero external wrench; released before the first
    # walking policy output is applied.
    from joint_brake import NoSupportStartup as StartupSupport, JointBrake

    output = a.output.resolve()
    from initial_pose import configure_grounded_pose, yaw_xyzw
    pose_file = ENTRY_DIR / "grounded-pose.json"
    rest = json.loads(pose_file.read_text())
    # Outer scene wrappers set their own default height. Finalize after them,
    # before gym.make, so scene creation, initialization and audit share one pose.
    configure_grounded_pose(cfg, fixture, rest, a.root_yaw_rad)
    output.mkdir(parents=True, exist_ok=False)
    (output / "frames").mkdir()
    started = time.monotonic()
    env = None
    nav = None
    detector = None
    media = None
    logs = {}
    report = {
        "schema": "microduck.beta-football-development.v1",
        "runtime_id": "newton-bam-20261019",
        "hardware_profile_id": "beta-gc2093-l8cx-v1",
        "qualification_claimed": False,
        "status": "starting",
        "model": str(a.model.resolve()),
        "model_sha256": sha(a.model),
        "robot_usd_sha256": sha(a.robot_usd),
        "fixture": fixture,
        "physics_hz": SCHEDULE.P,
        "policy_hz": SCHEDULE.C,
        "sensor_hz": SCHEDULE.S,
        "schedule": SCHEDULE.record(),
        "velocity_mapping": "identity, not calibrated for v8 dead zone",
        "kick_enabled": True,
        "autonomous_kick_enabled": True,
        "uncalibrated_point_estimate_development": True,
        "failures_retained": True,
        "automatic_resets": False,
        "randomized": a.randomized,
        "experiment": getattr(a, "experiment_case", None),
    }
    timings = {
        key: 0.0
        for key in (
            "startup_feedback",
            "policy_50hz",
            "physics_apply_step_scene_update",
            "state_readback_IMU_scoring_trace_100hz",
            "RTX_capture",
            "capture_readback_and_archive_submit",
            "YOLO_request",
            "navigation_and_gaze",
            "trace_write_subset",
        )
    }
    source_snapshot = {}
    from shot_completion import ShotCompletion
    from navigation_retry import restart_head_feedback
    shot = ShotCompletion(a.seconds)
    unlimited = a.seconds == 0
    retry_at = None
    retry_events = []
    navigation_epoch = 0
    blind = None
    initialization_failure = None
    pose_helper = ONBOARD_DIR / "initial_pose.py"
    pose_target = output / "source-snapshot/onboard/initial_pose.py"
    pose_target.parent.mkdir(parents=True, exist_ok=True)
    pose_target.write_bytes(pose_helper.read_bytes())
    source_snapshot[str(pose_helper)] = dict(
        sha256=sha(pose_target), snapshot=str(pose_target.relative_to(output)))
    for file in [
        *ONBOARD_DIR.glob("*.py"),
        *MEDIA_DIR.glob("*.py"),
        MEDIA_DIR / "release.json",
        *ROOT.glob("*.py"),
        *ROOT.glob("vision/*.py"),
        *(M / "sensors").glob("*.py"),
        M / "gait/batch_policy.py",
        M / "gait/bounded_actions.py",
        M / "gait/bounded-action-limits.json",
        M / "perception/detector.py",
        M / "perception/serve_detector.py",
    ]:
        target = output / "source-snapshot" / (Path("onboard") / file.name if file.parent == ONBOARD_DIR else file.relative_to(M))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
        source_snapshot[str(file)] = {
            "sha256": sha(target),
            "snapshot": str(target.relative_to(output)),
        }
    if a.shared_imu_readback:
        file = M / "sensors/processing/shared_imu_snapshot.py"
        target = output / "source-snapshot" / file.relative_to(M)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
        source_snapshot[str(file)] = {
            "sha256": sha(target),
            "snapshot": str(target.relative_to(output)),
        }
    if a.packed_physics_readback:
        file = M / "sensors/processing/packed_readback.py"
        target = output / "source-snapshot" / file.relative_to(M)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
        source_snapshot[str(file)] = {
            "sha256": sha(target),
            "snapshot": str(target.relative_to(output)),
        }
    if a.trained_recipe_dir is not None:
        dependencies = [
            "bounded_actions.py",
            "bounded-action-limits.json",
            "tracking_rewards.py",
            "bounded_tracking.py",
            "cycle_tracking.py",
            "command_curriculum.py",
            "env.yaml",
        ]
        for name in dependencies:
            file = a.trained_recipe_dir / name
            target = output / "source-snapshot/trained-action" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(file.read_bytes())
            source_snapshot[str(file)] = {
                "sha256": sha(target),
                "snapshot": str(target.relative_to(output)),
            }
        report["trained_adapter"] = {
            "kind": "frozen BoundedTrainingAction,internal8 margin.035",
            "recipe_directory": str(a.trained_recipe_dir),
            "network_export": json.loads(a.model.with_suffix(".json").read_text()),
            "last_action": "raw",
            "duplicate_external_adapter": False,
        }
    report["source_snapshot"] = source_snapshot
    write_json(output / "manifest.json", report)
    scorer = PhysicalGoalScorerV2(MICRODUCK_JOINT_NAMES)
    physical_bounds = np.asarray(load_limits()["physical_output_limits_rad"])
    head = HeadAdapter()
    last_body_imu = None
    last_control = None
    last_action = None
    head_result = head.solve([0.0, 0.0])
    capture_count = 0
    last_capture_s = 0.0
    policy_count = 0
    fell = False
    control_failures = []
    support = None
    try:
        env = gym.make("IsaacContrib-Velocity-Flat-MicroDuck", cfg=cfg).unwrapped
        robot = env.scene["robot"]
        ball = env.scene["ball"]
        sim = env.sim
        sim.backend_manager.set_decimation(1)
        env._physics_handles_decimation = False
        if (
            sim.backend_manager.backend_name != "newton"
            or not sim.cfg.use_newton_actuators
        ):
            raise RuntimeError("Native Newton/BAM is mandatory")
        if len(robot.joint_names) != 28:
            raise RuntimeError("Expected14servo+14passive backlash joints")
        ids, names = robot.find_joints(MICRODUCK_JOINT_NAMES, preserve_order=True)
        passive_ids, passive_names = robot.find_joints(
            ["passive_" + n + "_backlash" for n in MICRODUCK_JOINT_NAMES],
            preserve_order=True,
        )
        if list(names) != list(MICRODUCK_JOINT_NAMES) or len(passive_ids) != 14:
            raise RuntimeError("Explicit effective encoder order mismatch")
        imu_predictor = DeliveredBodyImu(head, names)
        body_input = BodyImuInput()
        body_sensor = IsaacBodyImu(
            robot,
            seed=a.seed + 1,
            profile=BetaBodyImuProfile(
                pipeline_delay_s=a.body_added_delay_s,
                orientation_correction_tau_s=a.body_fusion_tau_s,
            ),
        )
        report["body_fusion_sensitivity"] = {
            "orientation_correction_tau_s": a.body_fusion_tau_s,
            "default_tau_s": 0.5,
            "is_measured": False,
            "is_ST_SFLP_setting": False,
            "scope": "Declared virtual complementary-filter assumption; no truth initialization/correction",
        }
        report["policy_IMU_selection"] = a.policy_imu
        report["IMU_delivery_adapter"] = {
            "hardware_sample_hz": 60,
            "hardware_fifo_delivery_hz": 15,
            "policy_prediction_hz": 50,
            "maximum_prediction_age_s": 0.08,
            "method": "same-time encoder FK of delivered FIFO, body-gravity Exp(-omega*age), bodygyro zero-order hold",
            "unreceived_imu_latest_used_for_control": False,
            "prediction_is_measurement": False,
        }
        rig = env.beta_rig
        overhead = env.course_overhead
        report["visual_geometry"] = env.course_visual_geometry
        obs, _ = env.reset()

        def frame_pitch(label):
            """Isaac Sim viewport only: frame the 2 m x 1 m pitch from the same viewpoint as the
            Section 2.1 keyboard scene. This window uses Kit's own viewport (no Isaac Lab
            visualizer is configured here), so the active viewport camera is moved directly,
            as Isaac Lab's Kit viewer does. The robot's head camera, overhead sensor, policy
            inputs and physics do not use this view. Applied after reset and again after the
            policy takeover; the camera path and read-back position are logged."""
            if "kit" not in (getattr(a, "visualizer", None) or []):
                return
            o = env.scene.env_origins[0].tolist()
            eye = (o[0] + 1.40, o[1] - 1.95, o[2] + 1.20)
            target = (o[0] + 0.10, o[1], o[2] + 0.10)
            record = dict(at=label, eye=eye, target=target)
            try:
                from pxr import Gf
                import omni.kit.viewport.utility as viewport_utility
                from omni.kit.viewport.utility.camera_state import ViewportCameraState

                viewport = viewport_utility.get_active_viewport()
                if viewport is None:
                    record["error"] = "no active viewport"
                else:
                    camera = str(viewport.get_active_camera() or "/OmniverseKit_Persp")
                    state = ViewportCameraState(camera, viewport)
                    # rotate=False first: a fresh default camera has no centre of interest yet.
                    state.set_position_world(Gf.Vec3d(*eye), False)
                    state.set_target_world(Gf.Vec3d(*target), True)
                    record.update(camera=camera,
                                  eye_read=[round(float(v), 4) for v in state.position_world],
                                  target_read=[round(float(v), 4) for v in state.target_world])
                    # Live display profile: the window's render texture at 960x540 (as in the
                    # other course windows). Every capture's sim.render() also redraws this
                    # viewport, so a smaller texture makes each render cheaper; the head camera,
                    # ToF, overhead sensor, render subframes and policy inputs are unchanged.
                    import os
                    from realtime_view import VIEWPORT_RESOLUTION, parse_resolution, viewport_resolution

                    record["viewport"] = viewport_resolution(
                        parse_resolution(os.environ.get("MICRODUCK_VIEWPORT_RESOLUTION", VIEWPORT_RESOLUTION)))
            except Exception as error:  # display only: never stop the episode for the view
                record["error"] = repr(error)
            print("COURSE_VIEW_CAMERA " + json.dumps(record), flush=True)

        frame_pitch("after reset")
        rig.reset()
        overhead.reset()
        body_sensor.reset()
        shared_imu = None
        if a.shared_imu_readback:
            from processing.shared_imu_snapshot import SharedIsaacImuProducer

            shared_imu = SharedIsaacImuProducer(body_sensor, rig)
        reference_body = reference_rig = None
        readback_checks = {
            "physical_fields_compared": 0,
            "head_packets_compared": 0,
            "body_packets_compared": 0,
            "mismatches": 0,
        }
        if a.verify_readback:
            import copy

            reference_body = IsaacBodyImu(
                robot, seed=a.seed + 1, profile=body_sensor.profile
            )
            reference_rig = copy.copy(rig)
            reference_rig.imu = copy.deepcopy(rig.imu)
        if a.policy_imu == "body50":
            report["IMU_delivery_adapter"] = {
                "control_read_hz": 50,
                "source": "separate body powerboard LSM6DSV16X",
                "decoder": "official median3 gyro/gravity,25live quaternion readiness",
                "extra_read_to_policy_delay_s": a.body_added_delay_s,
                "extra_delay_is_measured": False,
                "median3_phase_delay_included": True,
                "timing_source": "official main loop read→inference→apply within same tick; additional queue is an explicit sensitivity assumption",
                "extra_prediction": False,
                "head_FIFO_used_for_locomotion": False,
                "encoder_timing": "same50Hz sync_read as body IMU,quantized4096positioncounts and.229rpm velocitycounts",
                "last_action": "previous raw network output before target adapter",
                "hardware_description_correction": "official body IMU was previously omitted; camera/head FIFO settings unchanged",
            }
        contact_names = list(env.scene["football_contact"].filter_object_names)
        if len(contact_names) != 2 or set(contact_names) != {
            "ankle_left",
            "ankle_right",
        }:
            raise RuntimeError(
                "Contact partners must resolve exactly to the two ankle bodies"
            )
        report["paired_contact_order"] = contact_names
        if a.detector_deployment is not None:
            from detector_client import DetectorSession

            detector = DetectorSession(
                a.detector_deployment,
                output / "detector",
                backend=a.detector_backend,
                device=a.detector_device,
            )
            report["detector"] = {
                "mode": "actual-RGB-YOLO26n-"
                + a.detector_backend
                + "-"
                + a.detector_device,
                "deployment": str(a.detector_deployment.resolve()),
                "deployment_sha256": sha(a.detector_deployment),
                **detector.health,
                "clock_scope": "Synchronous measured inference; simulation pauses during request, real robot latency is not established",
            }
        else:
            report["detector"] = {
                "mode": "explicit-development-contour",
                "trained_model": False,
            }
        # The only explicit state writes are initialization, before any recorded
        # substep. All later motion is native BAM contact physics.
        root = torch.tensor(
            [fixture["root_pose_xyzw"]], device=env.device, dtype=torch.float32
        )
        bp = torch.tensor(
            [fixture["ball_pose_xyzw"]], device=env.device, dtype=torch.float32
        )
        # Record the exact float32 pose sent to physics, also used by startup.
        fixture["root_pose_xyzw"] = root[0].detach().cpu().tolist()
        fixture["root_yaw_rad"] = yaw_xyzw(fixture["root_pose_xyzw"][3:])
        robot.write_root_pose_to_sim_index(root_pose=root)
        robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=env.device)
        )
        robot.reset()
        ball.write_root_pose_to_sim_index(root_pose=bp)
        ball.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros((1, 6), device=env.device)
        )
        ball.reset()
        env.scene.update(SCHEDULE.physics_dt)
        initialization_brake = JointBrake(robot)
        initialization_brake.lock()
        support = StartupSupport(
            fixture["root_pose_xyzw"], float(array(robot.data.body_mass).sum())
        )
        onboard = None
        from onboard_startup import OnboardBrakeStartup

        # Same brake and stationary check as the GR00T route; the onboard actor's
        # first output additionally waits for delivered RGB, ToF and BODY50 data.
        support = onboard = OnboardBrakeStartup(support)
        report["initialization_reference"] = dict(
            file=str(pose_file),
            sha256=sha(pose_file),
            case=rest["case"],
            recorded_root_xyz=rest["root"],
            applied_root_z=fixture["root_pose_xyzw"][2],
            recorded_quaternion_xyzw=rest["quat"],
            requested_yaw_offset_rad=a.root_yaw_rad,
            applied_yaw_rad=fixture["root_yaw_rad"],
            applied_quaternion_xyzw=fixture["root_pose_xyzw"][3:],
            composition="world-Z XYZW quaternion left-multiplied onto grounded reference",
            fixture_root_xy=fixture["root_pose_xyzw"][:2],
            fixture_minus_recorded_xy_m=[
                fixture["root_pose_xyzw"][i] - rest["root"][i] for i in range(2)
            ],
            ground="flat football pitch; the recorded grounded height is valid only on this plane",
        )
        support_adapter = NewtonSupportAdapter(robot)
        report["startup_support"] = support.metadata()
        eye = np.array([1.3, -1.7, 1.4])
        target = np.array([0.05, 0.0, 0.03])
        z = target - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, [0, 0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        rig._place(
            overhead,
            "/World/envs/env_0/BetaSensors/Overhead",
            eye,
            quaternion_xyzw(np.column_stack([x, y, z])),
        )
        kick_model = None
        kick_pulse = None
        active_policy = "walk"
        kick_requested = False
        previous_raw_action = torch.zeros((1, 14), device=env.device)
        if a.kick_model is not None:
            report["kick_control"] = validate_actor(a.kick_model, names)
            kick_model = BatchPolicy(
                a.kick_model, report["kick_control"]["sha256"], env.device
            )
            kick_pulse = KickPulse(a.kick_after_s, a.kick_duration_s)
            report["kick_control"].update(
                start_after_release_s=a.kick_after_s,
                duration_s=a.kick_duration_s,
                not_autonomous_approach=True,
            )
        from official_kick_runtime import OfficialTeacherKickRuntime

        # Frozen official standing + kick teachers only; the course residual
        # network, checkpoint, training ledger and K10 deployment are not loaded.
        k10 = OfficialTeacherKickRuntime.from_installed(M)
        import official_kick_runtime

        route = dict(
            kick_runtime=str(Path(official_kick_runtime.__file__).resolve()),
            kick_composition=k10.MODE,
            navigator=BetaNavigator.__name__,
            walking_actor=str(a.model.resolve()),
        )
        if (
            route["kick_runtime"] != str(OFFICIAL_KICK_RUNTIME.resolve())
            or route["kick_composition"] != "official_teacher_only_no_residual"
            or "kick_runtime" in sys.modules
            or route["navigator"] != "BallReacquisitionNavigator"
            or a.model.resolve() != (M / "models/reference/velstand.onnx").resolve()
        ):
            raise RuntimeError("Onboard route must use the installed controller: " + json.dumps(route))
        report["onboard_route"] = route
        report["official_teacher_kick_runtime"] = {
            "identity": k10.identity,
            "course_residual_used": False,
            "source_sha256": k10.source_sha256,
            "development_point_estimate_only": True,
            "formal_qualification_claimed": False,
        }
        from ground_contact_evidence import GroundContactEvidence

        ground_ids, _ = env.scene["contact_forces"].find_bodies(
            ["ankle_left", "ankle_right"], preserve_order=True
        )
        # Read-only inventory records every actual compiled ground shape, including visual geometry.
        ground_model = env.sim.backend_manager.get_model()
        ground_bodies = ground_model.shape_body.numpy()
        ground_types = ground_model.shape_type.numpy()
        ground_flags = ground_model.shape_flags.numpy()
        ground_records = [
            dict(
                index=i,
                label=label,
                body=int(ground_bodies[i]),
                type=int(ground_types[i]),
                flags=int(ground_flags[i]),
            )
            for i, label in enumerate(ground_model.shape_label)
            if label.startswith(cfg.scene.terrain.prim_path + "/")
        ]
        (output / "actual-ground-shape-inventory.json").write_text(
            json.dumps(
                dict(
                    filter_shape_prim_expr=cfg.scene.contact_forces.filter_shape_prim_expr,
                    shapes=ground_records,
                    control_input=False,
                ),
                indent=2,
            )
            + "\n"
        )
        ground_evidence = GroundContactEvidence(env, ground_ids, output)
        report["ground_contact_evidence"] = ground_evidence.report()
        model = BatchPolicy(a.model, report["model_sha256"], env.device)
        obs = env.observation_manager.compute()
        report["onnx_parity"] = model.parity(obs["policy"], tolerance=1e-5)
        if kick_model is not None:
            report["kick_control"]["onnx_parity"] = kick_model.parity(
                obs["policy"], tolerance=1e-5
            )
        if tuple(obs["policy"].shape) != (1, 61):
            raise RuntimeError("Expected native61-wide policy observation")
        from direct_path import DirectPath
        nav = DirectPath(output / "direct-path")
        report["path_planner"] = dict(kind="direct", external_service_started=False,
                                      coarse_follower="continuous_public_feedback",
                                      scope="Straight reference path with public ball/heading feedback and near-ball adjustment")
        report["onboard_route"]["actual_planner_instance"] = type(nav).__name__
        from approach_control import install_duration_support, install as install_approach
        install_duration_support(BetaNavigator)
        report["approach_control"] = install_approach(BetaNavigator)
        controller = BetaNavigator(
            nav,
            camera=rig.profile,
            config=ControllerConfig(
                layout=a.layout,
                duration_s=0 if unlimited else a.seconds,
                gait_id=report["model_sha256"],
                ground_mode=a.navigation_ground,
            ),
        )
        report["navigation_ground_mode"] = a.navigation_ground
        from fine_kick import (
            HeadHandoffV65,
            FineStepV65,
            standoff_entry,
            route_at_step_end,
        )
        from contact_odometry import SoleContactLegOdometry
        from recovery_handoff import (
            HeadHandoffV121,
            recovery_reacquire_head_allowed,
            rear_path_from_controller,
        )

        recovery_approval = a.recovery_approval_verified
        import head_handoff as handoff_module
        from kick_handoff import install as install_handoff
        report["final_handoff_tolerance"] = install_handoff(handoff_module)
        handoff = HeadHandoffV121(
            SoleContactLegOdometry(controller.vision.leg),
            names,
            params=recovery_approval["parameters"],
            task_limit_s=180.0 if unlimited else a.seconds,
        )
        from blind_arrival import BlindArrivalSupervisor, PROFILE, PARAMETERS as BLIND_PARAMETERS
        blind = BlindArrivalSupervisor()
        handoff.blind_arrival = blind
        report["blind_arrival"] = dict(profile=PROFILE, parameters=BLIND_PARAMETERS, validated_kick_domain=False)
        report["retreat_recovery_approval"] = {
            k: (v.as_dict() if k == "parameters" else v)
            for k, v in recovery_approval.items()
        }
        report["retreat_recovery_scope"] = {
            "reasons": [
                "propagated_point_outside_training_box",
                "goal_heading_outside_development_gate",
            ],
            "max_recoveries": 3,
            "reserve_s": 5.0,
            "task_limit_s": None if unlimited else a.seconds,
            "layout": a.layout,
            "kick_authority": False,
            "motion": "exact fixed-duration public-heading-feedback helper, 80 ticks, no post-stop authority",
            "plan_distance_m": 0.28,
            "max_public_backward_m": 0.24,
            "min_public_backward_m": 0.10,
            "rear_source": "original surveyed empty-field prior + original 10cm check_path; not rear sensing",
            "memory_ttl_s": 6.5,
            "truth_used": False,
        }
        handoff_state = handoff.status(0.0)
        from neutral_fine_step import NeutralFineStep
        from settled_entry import (
            settled_entry,
            neutral_turn_entry,
            neutral_forward_entry,
            continuous_track_entry,
        )

        fine_route = None
        handoff.adapter.memory.maximum_age_s = 6.5
        fine_step = FineStepV65()
        fine_state = fine_step.status(0.0, handoff.adapter.memory.status(0.0))
        actual_head = [0.0, 0.0, 0.0, 0.0]
        last_sent_velocity = [0.0, 0.0, 0.0]
        from head_feedback import ObservationHeadFeedback, head_context_allowed
        from fine_feedback_permission import fine_reacquire_feedback_allowed

        head_feedback = ObservationHeadFeedback()
        head_feedback_trace = None
        runtime_trace = None
        limiter = None
        if a.bounded_actions:
            from bounded_actions import BoundedActions, load_limits

            bounded_source = load_limits()
            limiter = BoundedActions(
                bounded_source["physical_output_limits_rad"],
                max_target_rate_rad_s=a.max_target_rate,
            )
            report["bounded_action_recipe"] = {
                "margin_rad": 0.035,
                "max_target_rate_rad_s": a.max_target_rate,
                "control_dt": 0.02,
                "limits": bounded_source,
                "source_sha256": sha(M / "gait/bounded_actions.py"),
            }
        if a.trained_recipe_dir is not None:
            term = env.action_manager.get_term("joint_pos")
            original_apply = term.limiter.apply

            def retain_internal_diagnostics(*args, **kwargs):
                executed, diagnostic = original_apply(*args, **kwargs)
                term.course_last_diagnostics = diagnostic
                return executed, diagnostic

            term.limiter.apply = retain_internal_diagnostics
        from gaze_tracking import GazeSlew

        calibration_gaze = GazeSlew()
        report["calibration_command"] = a.calibration_command
        report["calibration_gaze_body_yaw_pitch_rad"] = a.calibration_gaze
        commands = env.command_manager.get_term("base_velocity")
        head_cmd = env.command_manager.get_term("head_pose")
        body_cmd = env.command_manager.get_term("body_pose")
        for name in (
            "physics-trace",
            "control",
            "policy-trace",
            "startup-support",
            "detections",
            "imu-fifo",
            "body-imu",
            "public-memory-readback",
            "recovery-trace",
        ):
            logs[name] = (output / (name + ".jsonl")).open("x")
        from media_archive import ArchiveSink, ArchiveError

        media = ArchiveSink(output, capacity=16, compression=1, no_video=a.no_video)

        def log(name, row):
            writing = time.monotonic()
            logs[name].write(
                json.dumps(dict(row, navigation_epoch=navigation_epoch), ensure_ascii=False, allow_nan=False) + "\n"
            )
            logs[name].flush()
            timings["trace_write_subset"] += time.monotonic() - writing

        packed_state = None
        packed_reader = None
        if a.packed_physics_readback:
            from processing.packed_readback import PackedReadback

            packed_reader = PackedReadback()

        def physics_value(name, original):
            return packed_state[name] if packed_state is not None else array(original())

        def encoders():
            q = physics_value("q", lambda: robot.data.joint_pos)[0]
            qd = physics_value("qd", lambda: robot.data.joint_vel)[0]
            bias = physics_value(
                "encoder_bias",
                lambda: env.action_manager.get_term("joint_pos").encoder_bias,
            )[0]
            return q[ids] + q[passive_ids] + bias, qd[ids] + qd[passive_ids]

        native = sim.backend_manager.get_model()
        drives = [x for x in native.actuators if type(x.drive).__name__ == "DriveBam"]
        if not drives:
            raise RuntimeError("Native BAM telemetry missing")
        report["native_bam_drives"] = [
            {
                "pos_indices": x.pos_indices.numpy().tolist(),
                "min_delay": x.drive.min_delay,
                "max_delay": x.drive.max_delay,
            }
            for x in drives
        ]
        origin = sim.get_physics_step_count()
        steps = math.ceil(
            (
                a.seconds
                + support.cfg.calibration_timeout_s
                + support.cfg.supported_stand_s
                + support.cfg.release_duration_s
                + 0.1
            )
            / SCHEDULE.physics_dt
        )
        next_capture = 1
        report["setup_wall_seconds_including_Kit"] = time.monotonic() - globals().get(
            "PROGRAM_STARTED", started
        )
        t = 0.0
        task_t = None
        loop_started = time.monotonic()
        with torch.inference_mode():
            import itertools
            for tick in itertools.count() if unlimited else range(steps):
                if SESSION is not None and SESSION.should_interrupt():
                    report["terminal_reason"] = "user_reset" if SESSION.requested else "window_closed"
                    break
                section = time.monotonic()
                t = SCHEDULE.time(tick)
                last_body_imu = (
                    body_input.observe(t)
                    if a.policy_imu == "body50"
                    else imu_predictor.predict(t)
                )
                # World feedback is confined to this temporary initialization
                # fixture. No value from it enters controller.step/proprio.
                support_row = support.update(
                    t,
                    position_world=array(robot.data.root_pos_w)[0],
                    quaternion_xyzw=array(robot.data.root_quat_w)[0],
                    velocity_world=array(robot.data.root_lin_vel_w)[0],
                    angular_velocity_world=array(robot.data.root_ang_vel_w)[0],
                    imu_orientation_valid=last_body_imu is not None,
                )
                timings["startup_feedback"] += time.monotonic() - section
                if initialization_brake.active and support_row["score_enabled"]:
                    # Navigation stays enabled so the first GR00T request is made, but
                    # while the joints are still locked nothing is scored or timed as
                    # model control; the task clock restarts at the brake release.
                    support_row = dict(
                        support_row, score_enabled=False, phase="JOINT_BRAKE_AWAITING_GR00T"
                    )
                if support_row["failure"] and initialization_failure is None:
                    initialization_failure = support_row["failure"]
                    report["initialization_failure_reason"] = initialization_failure
                    log("startup-support", support_row)
                if SCHEDULE.control_at(tick):
                    section = time.monotonic()
                    task_clock = (
                        None
                        if support.task_started_at is None
                        else t - support.task_started_at
                    )
                    kick_requested = (
                        kick_pulse is not None
                        and kick_pulse.active(task_clock)
                        and last_body_imu is not None
                    )
                    sent_head_before = list(actual_head)
                    if blind is not None:
                        blind.control(epoch=navigation_epoch, fine=fine_step,
                                      actual_head=actual_head, actual_velocity=last_sent_velocity)
                    handoff_state = handoff.policy(
                        now_s=t,
                        packet=last_body_imu,
                        actual_sent_head=sent_head_before,
                        task_s=task_clock,
                    )
                    fine_before = fine_step.state
                    fine_state = fine_step.policy(
                        t=t,
                        packet=last_body_imu,
                        memory=handoff_state["memory"],
                        actual_head=actual_head,
                    )
                    if (
                        fine_before in ("STEP", "TRACK", "MACRO")
                        and fine_step.state == "SETTLE"
                    ):
                        fine_route = route_at_step_end(
                            now_s=t,
                            fine=fine_step,
                            handoff=handoff,
                            memory=handoff.adapter.memory.status(t),
                            goal_heading=fine_step.goal_heading,
                            goal_fix_s=fine_step.goal_s,
                        )
                        handoff_state = handoff.status(t)
                        fine_state = fine_step.status(t, handoff_state["memory"])
                    # v121: recovery owns the body (FinalApproach/coarse budgets held, counters unchanged);
                    # it owns the head except in REACQUIRE (original attention owner, deadlines continue).
                    controller.external_fine_step_active = (
                        fine_step.active or handoff.recovery_owns_body
                    )
                    controller.external_fine_step_head_owned = (
                        fine_step.hold_navigation_head or handoff.recovery_owns_head
                    )
                    log(
                        "public-memory-readback",
                        {
                            "timestamp_s": t,
                            "readback": handoff.latest_readback,
                            "handoff": handoff_state,
                        },
                    )
                    kick_requested = handoff_state["kick_runtime_authorized"]
                    active_policy = "kick_right" if kick_requested else "walk"
                    runtime_trace = None
                    velocity = (
                        [0.0, 0.0, 0.0]
                        if last_control is None or not support_row["navigation_enabled"]
                        else last_control["velocity_body_xyw"]
                    )
                    pulse_trace = None
                    if (
                        last_control is not None
                        and support_row["navigation_enabled"]
                        and handoff_state["state"] == "APPROACH"
                        and not fine_step.active
                    ):
                        velocity, pulse_trace = (
                            controller.final_approach.policy_command(
                                t=t, packet=last_body_imu, velocity=velocity
                            )
                        )
                    if handoff_state["override_body_zero"]:
                        velocity = [0.0, 0.0, 0.0]
                    if fine_state["override"]:
                        velocity = fine_state["velocity"]
                    recovery_velocity = handoff.recovery_velocity(t)
                    if recovery_velocity is not None:
                        velocity = recovery_velocity
                    actual_head = (
                        (
                            handoff_state["head_offsets"]
                            if handoff_state["override_body_zero"]
                            and handoff_state["head_offsets"] is not None
                            else head_result["head_pose_native_offsets_rad"]
                        )
                        if support_row["navigation_enabled"]
                        else [0.0, 0.0, 0.0, 0.0]
                    )
                    if fine_state["head_offsets"] is not None:
                        actual_head = fine_state["head_offsets"]
                    recovery_head_permission = None
                    if (
                        handoff_state["state"] == "RECOVERY"
                        and handoff_state["head_offsets"] is None
                    ):
                        recovery_head_permission = recovery_reacquire_head_allowed(
                            now_s=t, capture_s=last_capture_s, control=last_control
                        )
                        if not recovery_head_permission["allowed"]:
                            actual_head = sent_head_before
                    requested_head = list(actual_head)
                    head_allowed = head_context_allowed(
                        now_s=t,
                        capture_s=last_capture_s,
                        control=last_control,
                        geometry=controller.last_geometry,
                        too_close=controller.refresh_too_close,
                        ball_context=controller.ball_recovery_context_valid,
                        goal_context=controller.goal_attention_context_valid,
                    )
                    fine_feedback_allowed = fine_reacquire_feedback_allowed(
                        now_s=t,
                        capture_s=last_capture_s,
                        control=last_control,
                        fine_status=fine_state,
                        handoff_state=handoff_state["state"],
                        head_motion_allowed=head_allowed,
                        velocity=velocity,
                        kick_requested=kick_requested,
                    )
                    actual_head, head_feedback_trace = head_feedback.update(
                        now_s=t,
                        nominal=actual_head,
                        packet=last_body_imu,
                        velocity=velocity,
                        state=(
                            ""
                            if last_control is None
                            else last_control.get("state", "")
                        ),
                        handoff_state=(
                            "NEUTRAL_FINE_STEP"
                            if fine_step.active
                            else handoff_state["state"]
                        ),
                        navigation_enabled=support_row["navigation_enabled"],
                        head_motion_allowed=head_allowed,
                        fine_reacquire_head_only=fine_feedback_allowed,
                        macro_neutral_return=bool(
                            fine_step.active
                            and fine_step.state == "RETURN"
                            and fine_step.motion_kind in ("MACRO", "TRACK")
                            and not kick_requested
                        ),
                    )
                    if head_feedback_trace["body_zero_for_withdrawal"]:
                        velocity = [0.0, 0.0, 0.0]
                    velocity, recovery_row = handoff.recovery_after_head(
                        now_s=t,
                        task_s=task_clock,
                        requested_velocity=recovery_velocity,
                        velocity=velocity,
                        requested_head=requested_head,
                        sent_head=actual_head,
                        head_feedback=head_feedback_trace,
                        head_permission=recovery_head_permission,
                        packet=last_body_imu,
                    )
                    if recovery_row is not None:
                        handoff_state = handoff.status(t)
                        log("recovery-trace", recovery_row)
                    if head_feedback.failure:
                        kick_requested = False
                        active_policy = "walk"
                    # Onboard route: while the joints are locked, HOME is held and no network
                    # runs. The first actor output is computed once the locked robot has
                    # settled and RGB, ToF and BODY50 data are ready; the brake is released
                    # after that output exists and before it is applied.
                    release_now = (
                        onboard is not None
                        and initialization_brake.active
                        and initialization_failure is None
                        and not kick_requested
                        and onboard.ready(t, last_body_imu)
                    )
                    hold_home = (
                        onboard is not None and initialization_brake.active and not release_now
                    )
                    commands.is_standing_env[:] = False
                    commands.vel_command_b[:] = torch.tensor(
                        [velocity], device=env.device
                    )
                    head_cmd.command[:] = torch.tensor([actual_head], device=env.device)
                    body_cmd.command[:] = 0
                    obs = env.observation_manager.compute(update_history=True)
                    policy_obs = obs["policy"].clone()
                    # The gait consumes the official BODY control IMU. Head FIFO
                    # transport is retained only as an explicit historical A/B.
                    # Until the selected actual sensor is ready, hold home.
                    if last_body_imu is None or hold_home:
                        last_action = torch.zeros(
                            (1, 14), device=env.device, dtype=torch.float32
                        )
                    else:
                        policy_obs[:, 0:3] = torch.tensor(
                            [last_body_imu["gyro_body_rad_s"]], device=env.device
                        )
                        policy_obs[:, 3:6] = torch.tensor(
                            [last_body_imu["gravity_body"]], device=env.device
                        )
                        if a.policy_imu == "body50":
                            policy_obs[:, 6:20] = (
                                torch.tensor(
                                    [last_body_imu["encoder_q_rad"]], device=env.device
                                )
                                - robot.data.default_joint_pos.torch[:, ids]
                            )
                            policy_obs[:, 20:34] = torch.tensor(
                                [last_body_imu["encoder_qd_rad_s"]], device=env.device
                            )
                            policy_obs[:, 34:48] = previous_raw_action
                        if kick_requested:
                            runtime_trace = k10.step(
                                policy_obs[0].cpu().numpy(),
                                previous_raw14=previous_raw_action[0].cpu().numpy(),
                                phase_s=handoff_state["kick_phase_s"],
                                ready=True,
                                authorized=True,
                                head_return_complete=handoff_state[
                                    "head_return_complete"
                                ],
                            )
                            last_action = torch.tensor(
                                runtime_trace["raw14"][None], device=env.device
                            )
                            runtime_trace = {
                                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in runtime_trace.items()
                            }
                        else:
                            last_action = model(policy_obs)
                    if not torch.isfinite(last_action).all():
                        raise RuntimeError("Policy produced nonfinite action")
                    if release_now:
                        if not torch.isfinite(policy_obs).all():
                            raise RuntimeError("Nonfinite first onboard observation")
                        # Restores and checks every original joint limit and physical field;
                        # the task clock starts only after that succeeds.
                        initialization_brake.release(t)
                        onboard.mark_released(t)
                        report["initialization_brake"] = dict(
                            initialization_brake.metadata(),
                            first_policy_source="onboard walking actor",
                            first_policy_output_s=t,
                            settled_at_s=onboard.settled_at,
                            locked_after_settled_s=t - onboard.settled_at,
                            ready_check=onboard.last_check,
                            capture_check=onboard.capture,
                            release_command=dict(
                                velocity_body_xyw=list(velocity), head_offsets_rad=list(actual_head)
                            ),
                            task_clock_starts_at_release=True,
                        )
                        print(
                            "COURSE_POLICY_TAKEOVER "
                            + json.dumps(dict(t=t, source="onboard walking actor")),
                            flush=True,
                        )
                        frame_pitch("after takeover")
                        (output / "initialization-brake.json").write_text(
                            json.dumps(report["initialization_brake"], indent=2)
                        )
                    raw_action = last_action.clone()
                    previous_raw_action = raw_action.clone()
                    bound_trace = None
                    if limiter is not None:
                        term = env.action_manager.get_term("joint_pos")
                        q_now = torch.as_tensor(
                            array(robot.data.joint_pos), device=env.device
                        )
                        limits_now = torch.as_tensor(
                            array(robot.data.joint_pos_limits), device=env.device
                        )
                        last_action, diagnostics = limiter.apply(
                            raw_action,
                            scale=term._scale,
                            offset=term._offset,
                            encoder_bias=term.encoder_bias,
                            q_servo=q_now[:, ids],
                            q_passive=q_now[:, passive_ids],
                            servo_limits=limits_now[:, ids],
                        )
                        bound_trace = {
                            key: (
                                value[0].cpu().tolist()
                                if hasattr(value, "cpu")
                                else value
                            )
                            for key, value in diagnostics.items()
                        }
                    env.action_manager.process_action(last_action)
                    policy_count += 1
                    if a.trained_recipe_dir is not None:
                        term = env.action_manager.get_term("joint_pos")
                        last_action = term.raw_actions.clone()
                        bound_trace = {
                            key: (
                                value[0].cpu().tolist()
                                if hasattr(value, "cpu")
                                else value
                            )
                            for key, value in term.course_last_diagnostics.items()
                        }
                        if not torch.equal(env.action_manager.action, raw_action):
                            raise RuntimeError(
                                "Internal trained adapter must retain raw network action history"
                            )
                    log(
                        "policy-trace",
                        {
                            "timestamp_s": t,
                            "policy": active_policy,
                            "kick_requested": kick_requested,
                            "observation": policy_obs[0].cpu().tolist(),
                            "action": last_action[0].cpu().tolist(),
                            "raw_action": raw_action[0].cpu().tolist(),
                            "bounded_action_trace": bound_trace,
                            "head_mapping": head_result,
                            "IMU_prediction": last_body_imu,
                            "IMU_source": (
                                (
                                    "delivered BODY50 register block with official median3 and pipeline age"
                                    if a.policy_imu == "body50"
                                    else "delivered Beta head FIFO + same-time encoder FK prediction"
                                )
                                if last_body_imu and not hold_home
                                else (
                                    "initialization joint brake; HOME held; no network execution"
                                    if hold_home
                                    else "initial hold; no network execution"
                                )
                            ),
                            "velocity_mapping": "identity-development",
                            "kick_enabled": True,
                            "native_k10": runtime_trace,
                            "public_handoff": handoff_state,
                            "actual_head_command_rad": actual_head,
                            "head_feedback": head_feedback_trace,
                            "actual_velocity_command": velocity,
                            "final_pulse_policy": pulse_trace,
                            "public_neutral_fine_step": fine_state,
                            "blind_arrival": None if blind is None else blind.status(),
                        },
                    )
                    last_sent_velocity = list(velocity)
                    timings["policy_50hz"] += time.monotonic() - section
                section = time.monotonic()
                env.action_manager.apply_action()
                support_adapter.submit(support_row)
                env.scene.write_data_to_sim()
                support_row["actual_external_wrench_peak"] = (
                    support_adapter.check_after_write(support_row)
                )
                if onboard is not None:
                    onboard.check_wrench(support_row["actual_external_wrench_peak"])
                log("startup-support", support_row)
                sim.step(render=False)
                env.scene.update(SCHEDULE.physics_dt)
                env._sim_step_counter += 1
                timings["physics_apply_step_scene_update"] += time.monotonic() - section
                section = time.monotonic()
                measured_count = sim.get_physics_step_count() - origin
                if measured_count != tick + 1:
                    raise RuntimeError(
                        "Physics calls no longer correspond to individual200Hz steps"
                    )
                if packed_reader is not None:
                    # World values stay in this executor's privileged scorer.
                    # Only encoder channels are selected for sensor transport.
                    fields = {
                        "q": robot.data.joint_pos,
                        "qd": robot.data.joint_vel,
                        "limits": robot.data.joint_pos_limits,
                        "encoder_bias": env.action_manager.get_term(
                            "joint_pos"
                        ).encoder_bias,
                        "root_xyz": robot.data.root_pos_w,
                        "root_quat": robot.data.root_quat_w,
                        "root_ang_b": robot.data.root_ang_vel_b,
                        "root_lin_b": robot.data.root_lin_vel_b,
                        "gravity": robot.data.projected_gravity_b,
                        "ball_xyz": ball.data.root_pos_w,
                        "forces": env.scene["football_contact"].data.force_matrix_w,
                    }
                    if a.layout == "one-obstacle":
                        fields["obstacle_forces"] = env.scene[
                            "obstacle_contact"
                        ].data.force_matrix_w
                    for i, drive in enumerate(drives):
                        fields[f"bam_{i}_torque"] = drive.drive.motor_torque
                        fields[f"bam_{i}_voltage"] = drive.drive.effective_vin
                        fields[f"bam_{i}_target"] = drive.drive._delayed_target
                    packed_state = packed_reader.read(fields)
                    if a.verify_readback:
                        from processing.packed_readback import as_tensor

                        for name, value in fields.items():
                            reference = as_tensor(value).detach().cpu().numpy()
                            if reference.dtype != packed_state[
                                name
                            ].dtype or not np.array_equal(
                                reference, packed_state[name]
                            ):
                                readback_checks["mismatches"] += 1
                                raise RuntimeError("Packed field changed:" + name)
                            readback_checks["physical_fields_compared"] += 1
                t = SCHEDULE.time(tick + 1)
                q_sample, qd_sample = encoders()
                imu_predictor.record_encoders(t, q_sample, qd_sample)
                body_input.record_encoders(t, q_sample.tolist(), qd_sample.tolist())
                if shared_imu is None:
                    fifo = rig.update_imu(timestamp_s=t)
                    body_packets = body_sensor.update(timestamp_s=t)
                else:
                    delivered = shared_imu.update(timestamp_s=t)
                    fifo = delivered["head"]
                    body_packets = delivered["body"]
                if a.verify_readback:
                    reference_head_packets = reference_rig.update_imu(timestamp_s=t)
                    reference_body_packets = reference_body.update(timestamp_s=t)
                    if (
                        fifo != reference_head_packets
                        or body_packets != reference_body_packets
                    ):
                        readback_checks["mismatches"] += 1
                        raise RuntimeError(
                            "Shared IMU packet/timing differs from original producer"
                        )
                    readback_checks["head_packets_compared"] += len(fifo)
                    readback_checks["body_packets_compared"] += len(body_packets)
                transported = imu_predictor.ingest(fifo, now_s=t)
                body_input.ingest(body_packets, now_s=t)
                if body_packets:
                    log(
                        "body-imu",
                        {
                            "delivered_timestamp_s": t,
                            "packets": body_packets,
                            "source": "separate body control IMU,50Hz register reads; producer includes pipeline age",
                        },
                    )
                if fifo:
                    log(
                        "imu-fifo",
                        {
                            "delivered_timestamp_s": t,
                            "raw_head_samples": fifo,
                            "body_samples": transported,
                            "source": "actual15Hz delivery; same-acquisition-time encoders, no root pose",
                        },
                    )
                # Privileged values below have one-way flow into raw trace and
                # independent scoring. They never enter controller.step().
                root_xyz = physics_value("root_xyz", lambda: robot.data.root_pos_w)[0]
                root_q = physics_value("root_quat", lambda: robot.data.root_quat_w)[0]
                ball_xyz = physics_value("ball_xyz", lambda: ball.data.root_pos_w)[0]
                forces = physics_value(
                    "forces", lambda: env.scene["football_contact"].data.force_matrix_w
                )
                if forces.shape != (1, 1, 2, 3):
                    raise RuntimeError(
                        "Expected ball→left/right paired contact matrix [1,1,2,3], got "
                        + str(forces.shape)
                    )
                obstacle_force = 0.0
                if a.layout == "one-obstacle":
                    obstacle_force = float(
                        np.linalg.norm(
                            physics_value(
                                "obstacle_forces",
                                lambda: env.scene[
                                    "obstacle_contact"
                                ].data.force_matrix_w,
                            ).reshape(-1, 3),
                            axis=1,
                        ).sum()
                    )
                q = physics_value("q", lambda: robot.data.joint_pos)[0]
                limits = physics_value("limits", lambda: robot.data.joint_pos_limits)[0]
                violation = float(
                    np.maximum(np.maximum(limits[:, 0] - q, q - limits[:, 1]), 0).max()
                )
                gravity = physics_value(
                    "gravity", lambda: robot.data.projected_gravity_b
                )[0]
                fell |= bool(root_xyz[2] < 0.07 or gravity[2] > -math.cos(0.8))
                actual_ground_forces = ground_evidence.read()[0].cpu().numpy()
                task_t = (
                    None
                    if support.task_started_at is None
                    else t - support.task_started_at
                )
                score = (
                    scorer.update(
                        timestamp_s=task_t,
                        ball_world_xyz=ball_xyz.tolist(),
                        paired_foot_forces_world_n=forces[0, 0].tolist(),
                        foot_names=contact_names,
                        active_policy=active_policy,
                        kick_requested=kick_requested,
                        fell=fell,
                        obstacle_force_n=obstacle_force,
                        joint_violation_rad=violation,
                        servo_coordinate_violation_rad=np.maximum(
                            np.maximum(
                                limits[ids, 0] - q[ids], q[ids] - limits[ids, 1]
                            ),
                            0,
                        ).tolist(),
                        physical_output_violation_rad=np.maximum(
                            np.maximum(
                                physical_bounds[:, 0] - (q[ids] + q[passive_ids]),
                                q[ids] + q[passive_ids] - physical_bounds[:, 1],
                            ),
                            0,
                        ).tolist(),
                        passive_soft_limit_residual_rad=np.maximum(
                            np.maximum(
                                limits[passive_ids, 0] - q[passive_ids],
                                q[passive_ids] - limits[passive_ids, 1],
                            ),
                            0,
                        ).tolist(),
                    )
                    if support_row["score_enabled"]
                    else {}
                )
                bam = (
                    [
                        {
                            "motor_torque_nm": packed_state[f"bam_{i}_torque"].tolist(),
                            "effective_supply_v": packed_state[
                                f"bam_{i}_voltage"
                            ].tolist(),
                            "delayed_target_rad": packed_state[
                                f"bam_{i}_target"
                            ].tolist(),
                        }
                        for i in range(len(drives))
                    ]
                    if packed_state is not None
                    else [
                        {
                            "motor_torque_nm": x.drive.motor_torque.numpy().tolist(),
                            "effective_supply_v": x.drive.effective_vin.numpy().tolist(),
                            "delayed_target_rad": x.drive._delayed_target.numpy().tolist(),
                        }
                        for x in drives
                    ]
                )
                log(
                    "physics-trace",
                    {
                        "timestamp_s": t,
                        "task_time_s": task_t,
                        "startup_phase": support_row["phase"],
                        "score_enabled": support_row["score_enabled"],
                        "calibration_only": a.calibration_command is not None,
                        "support_force_world_n": support_row["force_world_n"],
                        "support_torque_world_nm": support_row["torque_world_nm"],
                        "root_position_world_m": root_xyz.tolist(),
                        "root_quaternion_xyzw": root_q.tolist(),
                        "root_angular_velocity_body_rad_s": physics_value(
                            "root_ang_b", lambda: robot.data.root_ang_vel_b
                        )[0].tolist(),
                        "root_linear_velocity_body_m_s": physics_value(
                            "root_lin_b", lambda: robot.data.root_lin_vel_b
                        )[0].tolist(),
                        "projected_gravity_body": gravity.tolist(),
                        "BAM_telemetry": bam,
                        "ball_position_world_m": ball_xyz.tolist(),
                        "joint_positions_absolute_rad": q.tolist(),
                        "joint_velocities_rad_s": physics_value(
                            "qd", lambda: robot.data.joint_vel
                        )[0].tolist(),
                        "joint_limit_violation_rad": violation,
                        "fell": fell,
                        "obstacle_contact_force_n": obstacle_force,
                        "raw_paired_foot_ball_forces_n": forces[0, 0].tolist(),
                        "actual_ground_paired_forces_n": actual_ground_forces.tolist(),
                        "kick_onset_s": handoff.kick_start,
                        **score,
                        "timestamp_s": t,
                        "scope": "privileged evaluation only; startup fixture excluded from task score",
                    },
                )
                timings["state_readback_IMU_scoring_trace_100hz"] += (
                    time.monotonic() - section
                )
                if SCHEDULE.control_period_end(tick):
                    env.episode_length_buf += 1
                    env.common_step_counter += 1
                    if "interval" in env.event_manager.active_terms:
                        env.event_manager.apply(mode="interval", dt=SCHEDULE.control_period_s(tick))
                SCHEDULE.after_tick(tick, env)
                if SCHEDULE.capture_after(tick):
                    next_capture += 1
                    capture_wall_start = time.monotonic()
                    capture = rig.capture(
                        timestamp_s=t, render_subframes=a.render_subframes
                    )
                    overhead.update(t - last_capture_s, force_recompute=True)
                    capture_wall_s = time.monotonic() - capture_wall_start
                    last_capture_s = t
                    timings["RTX_capture"] += capture_wall_s
                    section = time.monotonic()
                    third = array(overhead.data.output["rgb"])[0, :, :, :3].copy()
                    rgb = capture["rgb_raw"]
                    if onboard is not None:
                        onboard.note_capture(t, rgb, capture["tof"])
                    q_effective, qd_effective = encoders()
                    sample = imu_predictor.latest_head_sample
                    body_imu = (
                        body_input.observe(t)
                        if a.policy_imu == "body50"
                        else imu_predictor.predict(t)
                    )
                    imu_valid = body_imu is not None
                    last_body_imu = body_imu
                    proprio = (
                        {
                            "timestamp_s": t,
                            "joint_names": list(names),
                            "q_rad": q_effective.tolist(),
                            "qd_rad_s": qd_effective.tolist(),
                            "gyro_body_rad_s": body_imu["gyro_body_rad_s"],
                            "gravity_body": body_imu["gravity_body"],
                        }
                        if imu_valid
                        else None
                    )
                    stem = f"{capture_count:06d}"
                    row = {
                        "timestamp_s": t,
                        "capture_wall_seconds": capture_wall_s,
                        "render_generation": sim.render_generation,
                        "physics_step": sim.get_physics_step_count(),
                        "render_subframes": a.render_subframes,
                        "rgb": f"frames/{stem}-rgb.png",
                        "overhead": f"frames/{stem}-overhead.png",
                        "rgb_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                        "tof_range": f"frames/{stem}-tof-range.npy",
                        "tof_frame": capture["tof"],
                        "proprio": proprio,
                        "head_imu": sample,
                        "body_imu": body_input.latest,
                        "policy_imu_source": a.policy_imu,
                        "imu_transport": body_imu,
                        "simulated_mount_poses_excluded": True,
                    }
                    row["capture_submit_unix_s"] = time.time()
                    media.capture(
                        row,
                        rgb,
                        third,
                        capture["tof_source_ray_range"],
                        phase=support_row["phase"],
                        policy=active_policy,
                    )
                    timings["capture_readback_and_archive_submit"] += (
                        time.monotonic() - section
                    )
                    decision_started = time.monotonic()
                    detector_before = timings["YOLO_request"]
                    navigation_before = timings["navigation_and_gaze"]
                    fine_latch_diag = None
                    try:
                        detected = None
                        if (
                            imu_valid
                            and support_row["navigation_enabled"]
                            and detector is not None
                                                    ):
                            section = time.monotonic()
                            detector_result = detector.detect(rgb)
                            detected = detector_result["detections"]
                            log("detections", {"timestamp_s": t, **detector_result})
                            timings["YOLO_request"] += time.monotonic() - section
                        section = time.monotonic()
                        if not imu_valid:
                            # Free fall and startup impact do not define gravity.
                            # Keep actual raw samples and home PD targets while
                            # the sensor waits for its measured stationary gate.
                            last_control = {
                                "state": "IMU_WARMUP",
                                "velocity_body_xyw": [0.0, 0.0, 0.0],
                                "kick_request": False,
                                "reason": "Waiting for stable measured gravity; no policy/control inference",
                                "gaze_body_yaw_pitch_rad": [0.0, 0.0],
                            }
                        elif not support_row["navigation_enabled"]:
                            last_control = {
                                "state": support_row["phase"],
                                "velocity_body_xyw": [0.0, 0.0, 0.0],
                                "kick_request": False,
                                "reason": (
                                    "Joint brake at HOME; no network runs before the first walking-actor output"
                                    if onboard is not None
                                    else "Measured IMU valid; zero-command policy with explicitly fading physical startup support"
                                ),
                                "gaze_body_yaw_pitch_rad": [0.0, 0.0],
                            }
                        elif a.calibration_command is not None:
                            gaze = calibration_gaze.update(a.calibration_gaze, t)
                            last_control = {
                                "state": "GAIT_CALIBRATION",
                                "velocity_body_xyw": a.calibration_command,
                                "kick_request": False,
                                "reason": "Explicit current-gait raw command and body-relative gaze calibration; no navigation/task claim",
                                "gaze_body_yaw_pitch_rad": gaze,
                                "gaze_tracking": calibration_gaze.last,
                            }
                            head_result = head.solve(gaze)
                            if not head_result["accepted"]:
                                raise ValueError(
                                    "Calibration gaze IK residual exceeds bound"
                                )
                        else:
                            last_control = controller.step(
                                rgb,
                                proprio,
                                capture["tof"],
                                now_s=t,
                                detections=detected,
                            )
                            head_result = head.solve(
                                last_control["gaze_body_yaw_pitch_rad"]
                            )
                            if not head_result["accepted"]:
                                raise ValueError("Gaze IK residual exceeds bound")
                            if handoff.state != "APPROACH" or fine_step.active:
                                last_control["velocity_body_xyw"] = [0.0, 0.0, 0.0]
                            goal_heading = float("nan")
                            memory_now = handoff.adapter.memory.status(t)
                            g = controller.last_geometry
                            if (
                                controller.pose is not None
                                and np.isfinite(controller.pose).all()
                            ):
                                from visual_localization import rotate

                                goal_relative = np.asarray(
                                    rotate(
                                        -controller.pose[2],
                                        controller.goal_map - controller.pose[:2],
                                    )
                                )
                                point = (
                                    memory_now["point_level_m"]
                                    if memory_now["memory_available"]
                                    else last_control.get("final_approach", {}).get(
                                        "point_level_m"
                                    )
                                )
                                if point is not None:
                                    goal_heading = math.atan2(
                                        goal_relative[1] - point[1],
                                        goal_relative[0] - point[0],
                                    )
                            if not fine_step.active:
                                handoff_state = handoff.capture(
                                    now_s=t,
                                    geometry=g,
                                    head_offsets=actual_head,
                                    parameters=last_control.get(
                                        "final_approach", {}
                                    ).get("final_alignment_parameters", False),
                                    goal_heading=goal_heading,
                                    goal_fix_s=controller.goal_time,
                                )
                            # Complete current geometry grants a new anchor only while
                            # ordinary approach owns control. Fine-step memory is frozen.
                            fine_guard = False
                            rear_inputs = None
                            complete = bool(
                                g
                                and g.get("timestamp_s") == t
                                and (g.get("ball") or {}).get("accepted")
                                and g.get("floor_observation", {}).get("accepted")
                            )
                            if (
                                g
                                and g.get("timestamp_s") == t
                                and g.get("floor_observation", {}).get("accepted")
                                and controller.pose is not None
                                and np.isfinite(controller.pose).all()
                            ):
                                from perception import level_rotation

                                fine_R = level_rotation(
                                    (-np.asarray(g["ground"]["normal"])).tolist()
                                )
                                fine_point = (
                                    fine_R
                                    @ np.asarray(
                                        g["ball"]["geometry"]["point_reference_m"]
                                    )
                                    if complete
                                    else None
                                )
                                _, fine_hazard = controller._tof_sample(
                                    capture["tof"],
                                    proprio,
                                    fine_R,
                                    g["height"]["conditional_root_height_m"],
                                    fine_point,
                                )
                                fine_reason = last_control.get(
                                    "motion_authorization", {}
                                ).get("upstream_reason", last_control.get("reason", ""))
                                fine_unsafe = fine_reason in {
                                    "stale_proprioception",
                                    "control_gap_requires_visual_relocalization",
                                    "actual_ToF_floor_unavailable",
                                    "actual_ToF_near_obstacle_or_too_close",
                                    "180s_teaching_limit",
                                    "local_segment_hits_surveyed_obstacle",
                                    "ball_goal_direction_degenerate",
                                } or fine_reason.startswith("planning_failed:")
                                fine_guard = bool(
                                    not fine_hazard
                                    and not controller.refresh_too_close
                                    and not fine_unsafe
                                    and last_control.get("state") != "CONTROL_ERROR"
                                )
                                rear_inputs = (
                                    fine_R,
                                    g["height"]["conditional_root_height_m"],
                                )
                            final_diag = last_control.get("final_approach", {})
                            fine_memory = handoff.adapter.memory.status(t)
                            from approach_control import transfer_continuous_track

                            last_control["continuous_transfer"] = transfer_continuous_track(
                                fine_step, t=t, memory=fine_memory,
                                heading=goal_heading, goal_s=controller.goal_time,
                                guard=fine_guard,
                            )
                            if getattr(
                                fine_step, "coarse_track", False
                            ) and fine_step.state in ("RETURN", "TRACK", "TURN"):
                                from coarse_track import path_authority

                                coarse_active_path = {
                                    "allowed": False,
                                    "capture_s": t,
                                    "reason": "fresh_public_inputs_required",
                                }
                                if fine_guard and fine_memory.get("memory_available"):
                                    coarse_active_path = path_authority(
                                        controller,
                                        capture["tof"],
                                        proprio,
                                        fine_R,
                                        g["height"]["conditional_root_height_m"],
                                        fine_memory["point_level_m"],
                                    )
                                fine_guard = bool(
                                    fine_guard and coarse_active_path["allowed"]
                                )
                                last_control["coarse_active_path"] = coarse_active_path
                            if blind is not None:
                                blind.capture(t=t, epoch=navigation_epoch, complete=complete,
                                              guard_safe=fine_guard, goal_heading=goal_heading,
                                              goal_s=controller.goal_time)
                            handoff.note_capture(
                                capture_s=t, guard_safe=fine_guard, fresh_floor=True
                            )
                            fine_trigger, entry_reason = settled_entry(
                                t=t,
                                handoff_state=handoff.state,
                                fine_active=fine_step.active,
                                fine_state=fine_step.state,
                                final_diag=final_diag,
                                velocity_body_xyw=last_control["velocity_body_xyw"],
                                complete=complete,
                                guard_safe=fine_guard,
                                memory=fine_memory,
                                goal_heading=goal_heading,
                                goal_fix_s=controller.goal_time,
                                actual_head=actual_head,
                            )
                            fine_latch_diag = {
                                "trigger": fine_trigger,
                                "reason": entry_reason,
                                "route": fine_route,
                                "truth_used": False,
                            }
                            attention_permission = (
                                last_control.get("public_attention_head_permission")
                                or {}
                            )
                            turn_trigger, turn_entry_reason = neutral_turn_entry(
                                t=t,
                                handoff_state=handoff.state,
                                fine_active=fine_step.active,
                                fine_state=fine_step.state,
                                final_diag=final_diag,
                                velocity_body_xyw=last_control["velocity_body_xyw"],
                                complete=complete,
                                guard_safe=fine_guard,
                                memory=fine_memory,
                                goal_heading=goal_heading,
                                goal_fix_s=controller.goal_time,
                                actual_head=actual_head,
                            )
                            forward_trigger, forward_entry_reason = (
                                neutral_forward_entry(
                                    t=t,
                                    handoff_state=handoff.state,
                                    fine_active=fine_step.active,
                                    fine_state=fine_step.state,
                                    final_diag=final_diag,
                                    velocity_body_xyw=last_control["velocity_body_xyw"],
                                    complete=complete,
                                    guard_safe=fine_guard,
                                    memory=fine_memory,
                                    goal_heading=goal_heading,
                                    goal_fix_s=controller.goal_time,
                                    actual_head=actual_head,
                                )
                            )
                            fine_motion_kind = (
                                "TURN"
                                if turn_trigger
                                else "FAR_STEP" if forward_trigger else "STEP"
                            )
                            if forward_trigger:
                                fine_trigger = True
                                entry_reason = forward_entry_reason
                            elif (
                                final_diag.get("phase") == "ADVANCE"
                                and final_diag.get("staging_phase") == "GOAL_ALIGN"
                            ):
                                last_control["velocity_body_xyw"] = [0.0, 0.0, 0.0]
                                controller.final_approach.invalidate(
                                    t, "neutral_forward_entry_wait"
                                )
                            if turn_trigger:
                                fine_trigger = True
                                entry_reason = turn_entry_reason
                            elif (
                                final_diag.get("phase") == "TURN"
                                and controller.final_approach.turn_goal
                            ):
                                last_control["velocity_body_xyw"] = [0.0, 0.0, 0.0]
                                controller.final_approach.invalidate(
                                    t, "neutral_turn_entry_wait"
                                )
                            track_trigger, track_reason = continuous_track_entry(
                                t=t,
                                handoff_state=handoff.state,
                                fine_active=fine_step.active,
                                fine_state=fine_step.state,
                                final_diag=final_diag,
                                velocity_body_xyw=last_control["velocity_body_xyw"],
                                complete=complete,
                                guard_safe=fine_guard,
                                memory=fine_memory,
                                goal_heading=goal_heading,
                                goal_fix_s=controller.goal_time,
                                actual_head=actual_head,
                            )
                            if track_trigger:
                                fine_trigger = True
                                fine_motion_kind = "TRACK"
                                entry_reason = track_reason
                            from short_primitive import entry as primitive_entry

                            primitive_trigger, primitive_reason = primitive_entry(
                                t=t,
                                handoff_state=handoff.state,
                                fine_active=fine_step.active,
                                fine_state=fine_step.state,
                                final_diag=final_diag,
                                velocity_body_xyw=last_control["velocity_body_xyw"],
                                complete=complete,
                                guard_safe=fine_guard,
                                memory=fine_memory,
                                goal_heading=goal_heading,
                                goal_fix_s=controller.goal_time,
                                actual_head=actual_head,
                            )
                            if primitive_trigger:
                                fine_trigger = True
                                fine_motion_kind = "MACRO"
                                entry_reason = primitive_reason
                            from coarse_track import coarse_entry

                            coarse_trigger, coarse_reason = coarse_entry(
                                path=last_control.get("coarse_track_path"),
                                t=t,
                                handoff_state=handoff.state,
                                fine_active=fine_step.active,
                                fine_state=fine_step.state,
                                final_diag=final_diag,
                                velocity_body_xyw=last_control["velocity_body_xyw"],
                                complete=complete,
                                guard_safe=fine_guard,
                                memory=fine_memory,
                                goal_heading=goal_heading,
                                goal_fix_s=controller.goal_time,
                                actual_head=actual_head,
                            )
                            if coarse_trigger:
                                fine_trigger = True
                                fine_motion_kind = "COARSE_TRACK"
                                entry_reason = coarse_reason
                            fine_latch_diag = {
                                "trigger": fine_trigger,
                                "reason": entry_reason,
                                "route": fine_route,
                                "primitive_parameters_reason": primitive_reason,
                                "coarse_parameters_reason": coarse_reason,
                                "truth_used": False,
                            }
                            fine_head_only = bool(
                                fine_step.state == "REACQUIRE"
                                and attention_permission.get("accepted") is True
                                and attention_permission.get("capture_s") == t
                                and attention_permission.get("owner")
                                in ("GOAL_REFRESH", "BALL_REACQUIRE")
                                and last_control.get("public_attention_owner", {}).get(
                                    "owner"
                                )
                                == attention_permission.get("owner")
                                and last_control["velocity_body_xyw"] == [0.0, 0.0, 0.0]
                                and last_control["kick_request"] is False
                            )
                            from approach_control import remembered_kick_gate, PARAMETERS
                            rear = dict(allowed=False, capture_s=t, reason="missing_current_floor")
                            if rear_inputs is not None:
                                rear = rear_path_from_controller(controller, capture["tof"], proprio,
                                    *rear_inputs, distance_m=.16, fixed_target=None)
                            fine_step.parking_path = rear
                            p = np.asarray(fine_memory.get("point_level_m"), float)
                            parking_entry = bool(handoff.state == "APPROACH" and fine_step.state == "IDLE"
                                and complete and fine_guard and rear.get("allowed")
                                and p.shape == (3,) and np.isfinite(p).all()
                                and .13 <= p[0] <= .35 and np.linalg.norm(p[:2]) <= .4
                                and abs(p[1]) <= .3 and math.isfinite(goal_heading)
                                and PARAMETERS["parking_angle_rad"] <= abs(goal_heading) <= 1.2
                                and 0 <= t-controller.goal_time <= 2.)
                            if parking_entry:
                                fine_trigger, fine_motion_kind = True, "TURN"
                                entry_reason = "reverse_arc_for_shot_alignment"
                            gate = remembered_kick_gate(t=t, memory=fine_memory, heading=goal_heading,
                                goal_s=controller.goal_time, guard=fine_guard,
                                moving=fine_step.state not in ("IDLE", "SETTLE", "REACQUIRE"),
                                kicked=handoff.kick_start is not None)
                            last_control["remembered_kick_gate"] = gate
                            if gate["allowed"] and handoff.state == "APPROACH":
                                fine_step.state = "IDLE"
                                fine_trigger = False
                                handoff.enter_from_fine(now_s=t, goal_heading=goal_heading,
                                                       goal_fix_s=controller.goal_time)
                                handoff.head = np.asarray(actual_head, float).copy()
                                handoff_state = handoff.status(t)
                                last_control["velocity_body_xyw"] = [0., 0., 0.]
                                last_control["remembered_kick_gate"]["entered_handoff"] = True
                            fine_state = fine_step.capture(
                                t=t,
                                guard_safe=fine_guard,
                                complete=complete,
                                trigger=fine_trigger,
                                motion_kind=fine_motion_kind,
                                actual_head=actual_head,
                                memory=fine_memory,
                                goal_heading=goal_heading,
                                goal_fix_s=controller.goal_time,
                                head_only_safe=fine_head_only,
                            )
                            if handoff.state == "RECOVERY":
                                rear_path = {
                                    "allowed": False,
                                    "capture_s": t,
                                    "layout": a.layout,
                                    "reason": "fresh_public_guard_inputs_required",
                                    "truth_used": False,
                                }
                                if (
                                    handoff.recovery.state in ("AUTHORIZE", "RETREAT")
                                    and rear_inputs is not None
                                ):
                                    rear_path = rear_path_from_controller(
                                        controller,
                                        capture["tof"],
                                        proprio,
                                        *rear_inputs,
                                        distance_m=handoff.recovery.distance,
                                        fixed_target=handoff.recovery.fixed_target,
                                    )
                                recovery_row = handoff.recovery_capture(
                                    now_s=t,
                                    task_s=task_t,
                                    guard_safe=fine_guard,
                                    complete=complete,
                                    rear_path=rear_path,
                                    geometry=g,
                                    control=last_control,
                                    controller_pose=(
                                        None
                                        if controller.pose is None
                                        else np.asarray(controller.pose, float).tolist()
                                    ),
                                    goal_fix_s=controller.goal_time,
                                    approach_counters={
                                        "final_turns": controller.final_approach.turns,
                                        "final_advances": controller.final_approach.advances,
                                        "fine_attempts": fine_step.attempts,
                                    },
                                )
                                handoff_state = handoff.status(t)
                                if recovery_row["released"]:
                                    # No stale route/pulse survives release; the original guards must re-observe.
                                    last_control["velocity_body_xyw"] = [0.0, 0.0, 0.0]
                                    controller.final_approach.invalidate(
                                        t,
                                        "recovery_release_requires_new_original_observation",
                                    )
                                log("recovery-trace", recovery_row)
                            controller.external_fine_step_active = (
                                fine_step.active or handoff.recovery_owns_body
                            )
                            controller.external_fine_step_head_owned = (
                                fine_step.hold_navigation_head
                                or handoff.recovery_owns_head
                            )
                            if fine_step.active:
                                last_control["velocity_body_xyw"] = [0.0, 0.0, 0.0]
                                last_control["state"] = (
                                    "NEUTRAL_FINE_" + fine_step.state
                                )
                            last_control["public_neutral_fine_step"] = fine_state
                            last_control["public_handoff"] = handoff_state
                            if handoff_state["override_body_zero"]:
                                last_control["velocity_body_xyw"] = [0.0, 0.0, 0.0]
                                last_control["state"] = handoff_state["state"]
                        if last_control["kick_request"]:
                            # Do not borrow the old kick: retain this request as
                            # an explicit integration gap, keep current walk zero.
                            last_control["velocity_body_xyw"] = [0.0, 0.0, 0.0]
                            last_control["kick_execution"] = (
                                "disabled_pending_new_v8_contract_and_physical_validation"
                            )
                            control_failures.append(
                                {"t": t, "reason": "new_kick_model_not_enabled"}
                            )
                        timings["navigation_and_gaze"] += time.monotonic() - section
                    except (ValueError, TimeoutError, OSError) as error:
                        if fine_step.active:
                            fine_step.fail("neutral_step_capture_error")
                        if blind is not None:
                            blind.capture(t=t, epoch=navigation_epoch, complete=False,
                                          guard_safe=False, goal_heading=None, goal_s=None)
                        handoff.note_capture(
                            capture_s=t, guard_safe=False, fresh_floor=False
                        )
                        recovery_row = handoff.recovery_capture_unavailable(
                            now_s=t, task_s=task_t, reason="capture_control_exception"
                        )
                        if recovery_row is not None:
                            log("recovery-trace", recovery_row)
                        fine_latch_diag = {
                            "trigger": False,
                            "reason": "control_exception",
                        }
                        control_failures.append({"t": t, "reason": str(error)})
                        last_control = {
                            "state": "CONTROL_ERROR",
                            "velocity_body_xyw": [0.0, 0.0, 0.0],
                            "kick_request": False,
                            "reason": str(error),
                            "gaze_body_yaw_pitch_rad": [0.0, 0.0],
                        }
                    if fine_latch_diag is None:
                        handoff.note_capture(
                            capture_s=t, guard_safe=False, fresh_floor=False
                        )
                        recovery_row = handoff.recovery_capture_unavailable(
                            now_s=t,
                            task_s=task_t,
                            reason="capture_without_navigation_evaluation",
                        )
                        if recovery_row is not None:
                            log("recovery-trace", recovery_row)
                        fine_latch_diag = {
                            "trigger": False,
                            "reason": "capture_without_navigation_evaluation",
                        }
                    last_control["processing_wall_ms"] = dict(
                        decision=(time.monotonic() - decision_started) * 1000,
                        detector=(timings["YOLO_request"] - detector_before) * 1000,
                        navigation=(timings["navigation_and_gaze"] - navigation_before) * 1000,
                        scope="capture ready to command ready; excludes rendering, archive and actor; synchronous host time")
                    log(
                        "control",
                        {
                            "timestamp_s": t,
                            **last_control,
                            "active_policy": active_policy,
                            "development_kick_requested": kick_requested,
                            "head_mapping": head_result,
                            "head_feedback": head_feedback_trace,
                            "public_fine_entry": fine_latch_diag,
                            "blind_arrival": None if blind is None else blind.status(),
                        },
                    )
                    capture_count += 1  # All evidence and video frames stay in the bounded archive queue.
                    print(
                        json.dumps(
                            {
                                "sim_s": t,
                                "state": last_control["state"],
                                "camera_frames": capture_count,
                                "fell": fell,
                                "wall_s": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
                # Privileged contact and ball displacement decide only when the
                # episode ends. They never authorize a kick or navigation motion.
                reason = shot.update(
                    now_s=t, task_s=t if task_t is None else task_t,
                    ball_xyz=ball_xyz.tolist(),
                    kick_completed=bool(runtime_trace and runtime_trace.get("active")
                                        and runtime_trace.get("kick_weight") == 0.0
                                        and runtime_trace.get("phase_s", -1) >= .26 - 1e-8),
                    kick_contact=scorer.contact.kick_contact_observed,
                )
                if reason is not None and not unlimited:
                    report["terminal_reason"] = reason
                    break
                # Local failures stop the current maneuver, not the task clock.
                # Before any kick, discard stale observations and reacquire using
                # the same real sensors. No pose, scorer, motor history or world
                # state is reset. After a kick, keep observing until rest/timeout.
                # A valid blind handoff owns observation waiting, while actual
                # head/sensor/motion failures still preempt it immediately.
                local_failure = (
                    head_feedback.failure
                    or (fine_step.reason if fine_step.state == "FAILED" else None)
                    or (controller.ball_search.failed if blind is None else blind.filter_observation_failure(controller.ball_search.failed, handoff))
                    or (controller.final_approach.failed if blind is None else blind.filter_observation_failure(controller.final_approach.failed, handoff))
                    or (handoff.reason if handoff.state == "FAILED" else None)
                )
                if local_failure and handoff.kick_start is None and task_t is not None:
                    if retry_at is None:
                        retry_at = t + 1.0
                        retry_events.append(dict(timestamp_s=t, task_s=task_t, epoch=navigation_epoch,
                            reason=str(local_failure), handoff=handoff.status(t),
                            recovery=handoff.recovery_report(), retry_at_s=retry_at))
                        handoff.fail("pause_before_reacquisition")
                        fine_step.fail("pause_before_reacquisition")
                    if last_control is not None:
                        last_control["velocity_body_xyw"] = [0., 0., 0.]
                    if t >= retry_at - 1e-9:
                        gaze = controller.gaze.current.copy()
                        controller.reset()
                        controller.gaze.current = gaze
                        handoff = HeadHandoffV121(
                            SoleContactLegOdometry(controller.vision.leg), names,
                            params=recovery_approval["parameters"], task_limit_s=180.0 if unlimited else a.seconds)
                        handoff.adapter.memory.maximum_age_s = 6.5
                        fine_step = FineStepV65()
                        blind = BlindArrivalSupervisor()
                        handoff.blind_arrival = blind
                        head_feedback = restart_head_feedback(head_feedback)
                        fine_route = None
                        last_control = None
                        retry_events[-1]["restarted_at_s"] = t
                        navigation_epoch += 1
                        retry_events[-1]["next_epoch"] = navigation_epoch
                        retry_at = None
        report["shot_completion"] = shot.status()
        report["navigation_retries"] = retry_events
        report["loop_wall_seconds"] = time.monotonic() - loop_started
        report["processing_shared_imu_transport"] = (
            None if shared_imu is None else shared_imu.metadata()
        )
        report["processing_packed_physics_transport"] = (
            None if packed_reader is None else packed_reader.metadata()
        )
        report["readback_equivalence"] = dict(
            enabled=a.verify_readback, **readback_checks
        )
        if "handoff" in locals() and "retreat_recovery" not in report:
            report["retreat_recovery"] = handoff.recovery_report()
        report.update(
            status="development_episode_finished",
            simulated_seconds=t,
            task_seconds=(
                None if support.task_started_at is None else t - support.task_started_at
            ),
            startup_support=support.metadata(),
            camera_frames=capture_count,
            policy_updates=policy_count,
            physics_samples=sim.get_physics_step_count() - origin,
            scoring_physics_samples=scorer.contact.samples,
            scorer=scorer.summary(),
            control_failures=control_failures,
            sensors=rig.metadata(),
            body_control_imu=body_sensor.metadata(),
            joint_names=list(robot.joint_names),
            policy_joint_names=list(names),
            passive_joint_names=list(passive_names),
            backend={
                "name": sim.backend_manager.backend_name,
                "native_bam": sim.cfg.use_newton_actuators,
                "physics_decimation": 1,
                "policy_decimation": SCHEDULE.decimation,
            },
            terminal_reason=report.get("terminal_reason", "initialization_time_budget"),
            wall_seconds=time.monotonic() - started,
        )
    except BaseException as error:
        report.update(
            status="failed",
            error=repr(error),
            traceback=traceback.format_exc(),
            scorer=scorer.summary(),
            camera_frames=capture_count,
            control_failures=control_failures,
            wall_seconds=time.monotonic() - started,
        )
        raise
    finally:
        if "shared_imu" in locals():
            report["processing_shared_imu_transport"] = (
                None if shared_imu is None else shared_imu.metadata()
            )
        if "packed_reader" in locals():
            report["processing_packed_physics_transport"] = (
                None if packed_reader is None else packed_reader.metadata()
            )
        if "readback_checks" in locals():
            report["readback_equivalence"] = dict(
                enabled=a.verify_readback, **readback_checks
            )
        report["timing_wall_seconds"] = {
            **timings,
            "direct_path_request_subset": (
                controller.planning_wall_seconds if "controller" in locals() else 0.
            ),
        }
        report["timing_scope"] = (
            "Host wall elapsed; capture_readback_and_archive_submit includes original readback plus immutable packet copy and bounded queue wait, not worker PNG/video cost. Media worker and post-episode drain times are separate. Path/trace times are subsets; GPU async work is charged at synchronization."
        )
        report["program_wall_before_cleanup_s"] = time.monotonic() - globals().get(
            "PROGRAM_STARTED", started
        )
        if support is not None:
            report["startup_support"] = support.metadata()
        if "initialization_brake" not in report and "initialization_brake" in locals():
            # No GR00T prediction (or onboard first output) arrived: the joints were never
            # released and no part of this episode counts as model control.
            report["initialization_brake"] = dict(
                initialization_brake.metadata(),
                **(
                    dict(first_policy_source="onboard walking actor", first_policy_output_s=None)
                ),
            )
        for log_file in logs.values():
            log_file.close()
        report["videos"] = {}
        media_error = None
        if media is not None:
            try:
                report["media_archive"] = (SESSION.drain_media(media) if SESSION is not None else media.close())
                report["videos"] = report["media_archive"]["videos"]
            except BaseException as error:
                media_error = error
                report["media_archive"] = getattr(
                    media,
                    "result",
                    {
                        "evidenceComplete": False,
                        "status": "failed",
                        "error": "Media result unavailable",
                    },
                )
                report["evidence_status"] = "incomplete"
                report["status"] = "failed"
                report["media_error"] = repr(error)
        report["wall_seconds_including_media_drain"] = time.monotonic() - started
        if detector is not None:
            try:
                report["detector_cleanup"] = detector.close()
            except Exception as error:
                report["detector_cleanup"] = {"error": repr(error)}
        if nav is not None:
            try:
                report["path_cleanup"] = nav.close()
            except Exception as error:
                report["path_cleanup"] = {"error": repr(error)}
        report["source_changed_during_run"] = [
            str(p) for p, v in source_snapshot.items() if sha(p) != v["sha256"]
        ]
        report["source_sha256"] = {
            str(p.relative_to(ROOT)): sha(p)
            for p in [*ROOT.glob("*.py"), *ROOT.glob("vision/*.py")]
        }
        report["trace_sha256"] = {p.name: sha(p) for p in output.glob("*.jsonl")}
        try:
            write_json(output / "manifest.json", report)
        finally:
            if env is not None:
                try:
                    if SESSION is not None and report.get("status") != "failed":
                        SESSION.hold(env, report)
                finally:
                    try:
                        if "initialization_brake" in locals() and initialization_brake.active:
                            initialization_brake.release(float(locals().get("t", 0.0)))
                            report["initialization_cleanup"] = dict(joint_limits_restored=True, policy_never_started=True)
                            write_json(output / "manifest.json", report)
                    finally:
                        from session import close_environment_window
                        try:
                            close_environment_window(env)
                        finally:
                            env.close()
        if media_error is not None and sys.exc_info()[0] is None:
            raise media_error


def main():
    global PROGRAM_STARTED
    PROGRAM_STARTED = time.monotonic()
    a = arguments()
    from isaaclab.app import launch_simulation

    # Kit must own USD imports before task registration/configuration imports
    # pxr. Importing the pip USD first creates incompatible libusd symbols.
    sys.path.insert(0, str(ONBOARD_DIR.parent / "course_labs"))
    from kit_settings import kit_args

    with launch_simulation(None, {**vars(a), "require_kit": True,
                                  "kit_args": kit_args(getattr(a, "kit_args", "") or "")}):
        import isaaclab_tasks

        from performance import install, pin_main_thread
        install()
        if a.interactive_reset:
            global SESSION
            from session import ResetSession
            SESSION = ResetSession(a)
            try:
                while SESSION.next_episode():
                    phase = "build"
                    try:
                        cfg, fixture = build_cfg(a)
                        phase = "run_or_close"
                        run(a, cfg, fixture)
                        phase = "finalized"
                        SESSION.finalized()
                    except Exception as error:
                        SESSION.record_technical_error(error, phase)
                        raise
            finally:
                SESSION.close()
        else:
            cfg, fixture = build_cfg(a)
            run(a, cfg, fixture)


if __name__ == "__main__":
    main()
