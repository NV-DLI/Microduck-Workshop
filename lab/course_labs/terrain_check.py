"""Course slope check: walk (or skate) up onto, or down from, a finite wedge.

Derived from lab/terrain.py, with three differences: the terrain is a finite wedge with
a raised platform on a flat floor (ramp_mesh below) instead of an endless step-up ramp;
the test is reach-then-hold (the forward request stops at the goal and the robot must
stay there); and every robot first settles at its start with joints temporarily locked
(course_labs/grounded_start.py), released right before the first policy action.
"""

from pathlib import Path
import argparse
import os

LAB = Path(__file__).resolve().parents[1]
import hashlib
import json
import math
import sys
import time
import numpy as np


# Finite plywood wedge on a flat floor (metres, relative to the spawn point of the
# uphill test): the ramp starts 0.35 m ahead, rises over 0.5 m to the platform height
# tan(slope) * 0.5 m, and the flat top platform is 0.6 m long; the wedge is 0.6 m wide
# with vertical sides and a vertical back face.
WEDGE = dict(foot_x=0.35, ramp_length=0.5, platform_length=0.6, half_width=0.3)
# Goal margins: the root must be 0.15 m onto the platform (uphill) or 0.25 m in front of
# the ramp foot (downhill) and within 0.15 m of the centre line; both ankle bodies must be
# 0.06 m inside the goal side of the edge (soles reach ~0.02 m, roller wheels ~0.055 m
# behind the ankle), so the whole feet stand on the goal surface.
GOAL = dict(up_margin=0.15, down_margin=0.25, side_margin=0.15, foot_clearance=0.06)


def wedge_height(slope_deg):
    return WEDGE["ramp_length"] * math.tan(math.radians(abs(slope_deg)))


def ramp_mesh(difficulty, cfg):
    """Terrain for the slope test: a 100 m flat floor plus a wedge resting on it (ramp,
    platform, sides, back). The spawn point is tile (2, 2); for 0 deg only the floor is built."""
    import trimesh

    cx, cy = 2.0, 2.0
    floor = trimesh.Trimesh(
        vertices=[(-48.0, -48.0, 0.0), (52.0, -48.0, 0.0), (52.0, 52.0, 0.0), (-48.0, 52.0, 0.0)],
        faces=[(0, 1, 2), (0, 2, 3)],
        process=False,
    )
    h = wedge_height(cfg.slope_deg)
    if h < 1e-3:
        return [floor], np.array([cx, cy, 0.0])
    x0 = cx + WEDGE["foot_x"]
    x1 = x0 + WEDGE["ramp_length"]
    x2 = x1 + WEDGE["platform_length"]
    y0, y1 = cy - WEDGE["half_width"], cy + WEDGE["half_width"]
    v = [(x0, y0, 0.0), (x0, y1, 0.0), (x1, y0, h), (x1, y1, h), (x2, y0, h), (x2, y1, h),
         (x1, y0, 0.0), (x1, y1, 0.0), (x2, y0, 0.0), (x2, y1, 0.0)]
    A, B, C, D, E, F, G, H, I, J = range(10)
    faces = [
        (A, C, D), (A, D, B),  # sloping top
        (C, E, F), (C, F, D),  # platform top
        (E, I, J), (E, J, F),  # back face (+x)
        (A, G, C), (G, I, E), (G, E, C),  # side -y
        (B, D, H), (H, D, F), (H, F, J),  # side +y
    ]
    wedge = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    return [trimesh.util.concatenate([floor, wedge])], np.array([cx, cy, 0.0])


def surface_height(x, y, slope_deg):
    """Height of the walkable surface under (x, y), relative to the spawn point (torch)."""
    import torch

    h = wedge_height(slope_deg)
    foot, top = WEDGE["foot_x"], WEDGE["foot_x"] + WEDGE["ramp_length"]
    end = top + WEDGE["platform_length"]
    on = (y.abs() <= WEDGE["half_width"]) & (x >= foot) & (x <= end)
    ramp = (x - foot).clamp(0, WEDGE["ramp_length"]) / WEDGE["ramp_length"] * h
    return torch.where(on, torch.where(x <= top, ramp, torch.full_like(x, h)), torch.zeros_like(x))


# Head body and its rest orientation (x, y, z, w) in both robot assets (all joints at 0,
# trunk level). R(head) R(rest)^-1 x equals the head camera's optical axis (-z of jaw_soft,
# lab/runtime/simulation/core/sensors/beta_mounts.py), which is level at rest.
HEAD_BODY = "jaw_soft"
HEAD_REST_XYZW = (0.0, -0.7071067811865476, 0.0, 0.7071067811865476)


def _quat_mul(a, b):
    import torch

    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack((aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz), -1)


def _quat_rotate(q, v):
    import torch

    u, w = q[..., :3], q[..., 3:]
    return v + 2 * torch.cross(u, torch.cross(u, v, dim=-1) + w * v, dim=-1)


def head_gaze_down_deg(head_q, trunk_q):
    """How far the head looks down (deg, + = down): in the world, and relative to the trunk."""
    import torch

    rest_inv = torch.tensor(HEAD_REST_XYZW, device=head_q.device) * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=head_q.device)
    forward = _quat_rotate(_quat_mul(head_q, rest_inv.expand_as(head_q)), torch.tensor([1.0, 0.0, 0.0], device=head_q.device).expand(head_q.shape[0], 3))
    trunk_inv = trunk_q * torch.tensor([-1.0, -1.0, -1.0, 1.0], device=head_q.device)
    local = _quat_rotate(trunk_inv, forward)
    world = torch.rad2deg(torch.asin((-forward[:, 2]).clamp(-1, 1)))
    relative = torch.rad2deg(torch.asin((-local[:, 2]).clamp(-1, 1)))
    return world, relative


# Course head hold: standing still (the first 2 s, and after the goal) the trained policy
# tilts its head ~67 deg down, so the four head servos are given their
# HOME targets instead, through the normal servo action (same PD servos, no extra force or
# joint lock). Leg actions always stay the policy's. The switch blends over 0.5 s.
# Under the hold, gravity and gearbox play left the head ~0.12 rad nose-down of HOME, so the
# head_pitch target sits HEAD_PITCH_HOLD_OFFSET below HOME (negative = nose up); the offset
# eases in from 0 over the first 0.5 s after takeover, so the start pose is HOME.
HEAD_JOINTS = ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
HEAD_HOLD_BLEND_S = 0.5
HEAD_PITCH_HOLD_OFFSET = -0.20


class HeadHold:
    """Replace the head slots of the policy action with HOME-head action values (blended)."""

    def __init__(self, env, term_name="joint_pos", pitch_offset=HEAD_PITCH_HOLD_OFFSET):
        import torch

        term = env.action_manager.get_term(term_name)
        robot = env.scene["robot"]
        names = list(term._joint_names)
        self.slots = torch.tensor([names.index(n) for n in HEAD_JOINTS], device=env.device)
        joints = [robot.joint_names.index(n) for n in HEAD_JOINTS]

        def T(x):
            return getattr(x, "torch", x)

        home = T(robot.data.default_joint_pos)[:, joints]
        limits = T(robot.data.soft_joint_pos_limits)[:, joints]
        home = home.clamp(limits[..., 0], limits[..., 1])
        target = home.clone()
        target[:, HEAD_JOINTS.index("head_pitch")] += pitch_offset
        target = target.clamp(limits[..., 0], limits[..., 1])
        offset, scale = term._offset, term._scale
        if isinstance(offset, torch.Tensor):
            offset = offset[:, self.slots]
        if isinstance(scale, torch.Tensor):
            scale = scale[:, self.slots]
        # The servo target is raw * scale + offset (clipped), minus the encoder bias.
        bias = getattr(term, "encoder_bias", None)
        bias = bias[:, self.slots] if bias is not None else 0.0
        self.hold_home = (home + bias - offset) / scale
        self.hold_final = (target + bias - offset) / scale
        self.hold = self.hold_home
        self.home_rad = home
        self.target_rad = target
        self.pitch_offset = pitch_offset
        self.weight = torch.ones(env.num_envs, device=env.device)
        self.rate = 0.02 / HEAD_HOLD_BLEND_S
        self.ramp_steps = round(HEAD_HOLD_BLEND_S / 0.02)
        self.steps = 0

    def update(self, want):
        """Move each weight toward 1 (want) or 0 by at most one blend step."""
        import torch

        self.weight = (self.weight + (want.float() * 2 - 1) * self.rate).clamp(0.0, 1.0)
        # Hold target: HOME at takeover, eased to the offset target over the first 0.5 s.
        if self.steps >= self.ramp_steps:
            self.hold = self.hold_final
        elif self.steps > 0:
            self.hold = torch.lerp(self.hold_home, self.hold_final, self.steps / self.ramp_steps)
        self.steps += 1

    def __call__(self, action):
        return blend_head(action, self.slots, self.hold, self.weight)


def blend_head(action, slots, hold, weight):
    """Action with the head slots at weight 1 = hold, 0 = policy (exact at both ends)."""
    import torch

    out = action.clone()
    w = weight[:, None]
    policy = action[:, slots]
    mixed = torch.where(w >= 1, hold.expand_as(policy), torch.where(w <= 0, policy, policy + w * (hold - policy)))
    out[:, slots] = mixed
    return out


# Interactive viewport render-texture size. Only the image Kit renders for the viewport
# shrinks; the window and the full Isaac UI keep their size and the image is scaled into it.
# Display only: camera sensors, recorded video and policy inputs do not use this texture.
# MICRODUCK_VIEWPORT_RESOLUTION="WxH" overrides (e.g. 960x540, or 1920x1080 for a sharper
# view); "native" leaves Kit's own setting.
VIEWPORT_RESOLUTION = "960x540"
# Recorded video camera (--record; RECORD_SIZE, 52 deg, RECORD_FPS of simulated time), offsets from
# the spawn point: low front-left three-quarter view that keeps the whole v3 wedge and the
# robot at the uphill start/goal and the downhill start/goal in the frame.
RECORD_EYE_OFFSET = (-0.55, -1.75, 0.60)
RECORD_TARGET_OFFSET = (0.75, 0.0, 0.08)
# Recorded video: the same live renderer and look as the window, one frame per 20 ms control
# step (50 fps of simulated time), rendered once per frame. Not a separate high-quality render.
RECORD_SIZE = (960, 540)
RECORD_FPS = 50


def parse_resolution(text):
    """'WxH' -> (W, H); 'native' -> None."""
    text = text.strip().lower()
    if text == "native":
        return None
    w, h = (int(v) for v in text.split("x"))
    if not (160 <= w <= 7680 and 90 <= h <= 4320):
        raise ValueError(f"viewport resolution out of range: {text}")
    return w, h


def viewport_resolution(requested=None):
    """Set (when requested) and read the active Kit viewport's render resolution."""
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
    except Exception as error:  # report only; the evaluation does not depend on it
        return dict(status="unavailable", error=repr(error))
    if viewport is None:
        return dict(status="no active viewport")
    before = viewport.full_resolution
    if requested:
        viewport.resolution = requested  # ViewportAPI.resolution (set_texture_resolution is its deprecated alias)
    return dict(
        status="ok",
        requested=list(requested) if requested else "native",
        before=list(before) if before else None,
        full_resolution=list(viewport.full_resolution),
        render_resolution=list(viewport.resolution),
        resolution_scale=viewport.resolution_scale,
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
    p.add_argument("--locomotion", choices=["walking", "rollers"], default="walking")
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
    sys.path[:0] = [str(LAB), str(Path(__file__).resolve().parent)]
    import torch
    import gymnasium as gym
    from isaaclab.app import launch_simulation
    from batch_policy import BatchPolicy
    from kit_settings import kit_args

    digest = hashlib.sha256(a.model.read_bytes()).hexdigest()
    report = dict(
        status="starting",
        model_sha256=digest,
        slope_deg=a.slope,
        seed=a.seed,
        num_envs=a.num_envs,
        command_m_s=0.25 if a.locomotion == "walking" else None,
        locomotion=a.locomotion,
        roller_push_request=0.6 if a.locomotion == "rollers" else None,
        automatic_resets=False,
        support_surface_bounds_relative_m=[-50.0, 50.0, -50.0, 50.0],
        geometry=dict(
            kind="finite wedge on a flat floor (v3, half the v2 size); v2 was twice as large, older reports used an endless step-up ramp",
            direction="uphill" if a.slope > 0 else "downhill" if a.slope < 0 else "flat",
            platform_height_m=wedge_height(a.slope),
            **WEDGE,
        ),
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
                "kit_args": kit_args(),
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

                    # All environments share one ramp tile (same origin, no contact between
                    # copies), so drawing them all would overlay the ducks. The window shows
                    # environment 0 only; every environment is still simulated and reported.
                    cfg.sim.visualizer_cfgs = [KitVisualizerCfg(visible_env_indices=[0], max_visible_envs=1)]
                cfg.seed = a.seed
                cfg.curriculum = None
                cfg.episode_length_s = a.seconds + 10
                cfg.scene.robot.spawn.usd_path = str(
                    LAB / "assets/microduck_contact.usda"
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
                    # Studio look (wood ramp, light floor); MICRODUCK_TERRAIN_LOOK=legacy keeps
                    # the earlier appearance, for a before/after physics audit only.
                    if os.environ.get("MICRODUCK_TERRAIN_LOOK") == "legacy":
                        from terrain_visuals import configure_scene
                    else:
                        from terrain_studio import configure_scene

                    configure_scene(cfg)
                if a.record:
                    from terrain_visuals import camera_prestartup
                    from isaaclab.managers import EventTermCfg

                    cfg.events.terrain_camera = EventTermCfg(
                        func=camera_prestartup, mode="prestartup",
                        params=dict(width=RECORD_SIZE[0], height=RECORD_SIZE[1]),
                    )
                if a.locomotion == "rollers":
                    from roller_config import configure_rollers

                    report["roller_configuration"] = configure_rollers(cfg)
                if a.slope < 0:
                    # Downhill: start in the middle of the high platform, facing the ramp (-x).
                    x, y, z = cfg.scene.robot.init_state.pos
                    cfg.scene.robot.init_state.pos = (
                        WEDGE["foot_x"] + WEDGE["ramp_length"] + WEDGE["platform_length"] / 2,
                        y,
                        z + wedge_height(a.slope),
                    )
                    cfg.scene.robot.init_state.rot = (0.0, 0.0, 1.0, 0.0)  # yaw 180 deg (x, y, z, w)
                report["robot_asset_sha256"] = hashlib.sha256(
                    Path(cfg.scene.robot.spawn.usd_path).read_bytes()
                ).hexdigest()
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
                from physics_audit import write_snapshot

                write_snapshot(manager, "terrain_check after build")  # MICRODUCK_PHYSICS_AUDIT only
                if (not a.headless or a.record) and os.environ.get("MICRODUCK_TERRAIN_LOOK") != "legacy":
                    from terrain_studio import wood_alignment

                    # Display skin vs the physical wedge in the finalized Newton model.
                    try:
                        report["wood_skin_alignment"] = wood_alignment(env, manager)
                    except Exception as error:  # diagnosis only; never stops the evaluation
                        report["wood_skin_alignment"] = dict(status="error", error=repr(error))
                    print("COURSE_WOOD_ALIGNMENT " + json.dumps(report["wood_skin_alignment"]), flush=True)
                if a.record:
                    from terrain_visuals import start_recording, capture

                    camera, video_process = start_recording(
                        env, a.output, eye_offset=RECORD_EYE_OFFSET, target_offset=RECORD_TARGET_OFFSET, fps=RECORD_FPS
                    )
                    import zlib

                    recorded = dict(frames=0, duplicate_consecutive_frames=0, last_crc=None)
                robot = env.scene["robot"]
                policy = BatchPolicy(a.model, digest, "cuda:0")
                from grounded_start import settle, StartupNotSettled

                try:
                    brake, report["startup"] = settle(
                        env,
                        pose="roller-rest" if a.locomotion == "rollers" else "home",
                        render=not a.headless,
                    )
                except StartupNotSettled as error:
                    report["startup"] = error.record
                    raise
                cmd = env.command_manager.get_term("base_velocity")

                def T(x):
                    return getattr(x, "torch", x)

                origin = T(env.scene.env_origins).clone()
                if a.locomotion == "rollers":
                    report["initial_body_names"] = list(robot.body_names)
                    report["initial_body_position_relative_m"] = (
                        (T(robot.data.body_pos_w)[0] - origin[0]).cpu().tolist()
                    )
                fallen = torch.zeros(a.num_envs, dtype=torch.bool, device="cuda:0")
                reached = fallen.clone()
                reached_time = torch.full((a.num_envs,), float("nan"), device="cuda:0")
                left_goal = torch.zeros_like(fallen)
                fell_off_platform = torch.zeros_like(fallen)
                hold_seconds_required = 2.0
                brake_integral = torch.zeros(a.num_envs, device="cuda:0")
                h = wedge_height(a.slope)
                foot = WEDGE["foot_x"]
                top = foot + WEDGE["ramp_length"]
                end = top + WEDGE["platform_length"]
                inside = WEDGE["half_width"] - GOAL["side_margin"]
                feet = [i for i, name in enumerate(robot.body_names)
                        if name in ("ankle_left", "ankle_right", "ankle_l_v1", "ankle_r_v1")]
                if len(feet) != 2:
                    raise RuntimeError(f"expected two ankle bodies, found {[robot.body_names[i] for i in feet]}")
                report["goal"] = dict(GOAL, foot_bodies=[robot.body_names[i] for i in feet])
                head = robot.body_names.index(HEAD_BODY)
                trunk = robot.body_names.index("trunk_base")
                head_joints = [robot.joint_names.index(n) for n in HEAD_JOINTS]
                # Head pose command (offsets from the default head joint angles, the policy's
                # trained head interface). MICRODUCK_HEAD_COMMAND="n,h,y,r" is a diagnostic override.
                head_command = [float(v) for v in os.environ.get("MICRODUCK_HEAD_COMMAND", "0,0,0,0").split(",")]
                report["head_command_rad"] = head_command

                def head_state():
                    bq = T(robot.data.body_quat_w)
                    world, relative = head_gaze_down_deg(bq[:, head], bq[:, trunk])
                    offset = (T(robot.data.joint_pos) - T(robot.data.default_joint_pos))[:, head_joints]
                    return dict(gaze_down_world_deg=world.cpu().tolist(), gaze_down_vs_trunk_deg=relative.cpu().tolist(),
                                head_joint_offset_rad=offset.cpu().tolist())

                report["head_after_settle"] = head_state()
                # MICRODUCK_HEAD_HOLD=0 turns the course head hold off (diagnostic: raw policy head).
                # Built after the start-up brake is released (the brake narrows the joint limits).
                hold_enabled = os.environ.get("MICRODUCK_HEAD_HOLD", "1") != "0"
                # MICRODUCK_HEAD_PITCH_HOLD_OFFSET (rad) overrides the calibration for diagnosis.
                head_pitch_offset = float(os.environ.get("MICRODUCK_HEAD_PITCH_HOLD_OFFSET", HEAD_PITCH_HOLD_OFFSET))
                head_hold = None
                report["head_hold"] = dict(
                    enabled=hold_enabled,
                    kind="course head hold controller added to the policy: head servos only, legs always policy",
                    joints=list(HEAD_JOINTS),
                    when="velocity request zero: before the walk starts (first 2 s) and after the goal is reached",
                    target="HOME head joint angles (task default pose, level gaze relative to the trunk), with the "
                    "head_pitch target offset below; neck_pitch, head_yaw, head_roll at HOME",
                    head_pitch_target_offset_rad=head_pitch_offset,
                    head_pitch_offset_ease_in_seconds=HEAD_HOLD_BLEND_S,
                    offset_reason="calibration for the residual nose-down sag (gravity, gearbox play) measured under "
                    "the HOME hold; a fixed configured offset, not a closed-loop gaze controller",
                    mechanism="the four head slots of the 14-servo action are set to the action value whose servo "
                    "target is HOME (raw = (HOME + encoder bias - offset) / scale); same PD servos and gains, no joint "
                    "lock, no external force; the applied action is the observed previous action",
                    blend_seconds=HEAD_HOLD_BLEND_S,
                    note="results reflect policy + course head hold, not the policy's own head control",
                )
                timing = dict(policy=[], physics=[], render=[], step=[])
                # MICRODUCK_PROFILE=1 splits each step into policy/physics/render with CUDA
                # synchronisation (diagnostic: the syncs add their own overhead).
                profile = not a.headless and os.environ.get("MICRODUCK_PROFILE") == "1"
                report["profile_split_with_cuda_sync"] = profile
                # Viewport refresh interval in control steps (display only: physics 200 Hz and
                # policy 50 Hz are unchanged): every 2nd step = 25 Hz. MICRODUCK_RENDER_EVERY overrides.
                render_every = max(1, int(os.environ.get("MICRODUCK_RENDER_EVERY", "2")))
                # While recording with a window every control step is drawn (one render per
                # recorded frame), so the effective display interval is 1 (50 Hz).
                display_every = 1 if a.record else render_every
                report["render_every_control_steps"] = display_every
                report["render_every_configured"] = render_every
                start_pos = T(robot.data.root_pos_w) - origin
                q = T(robot.data.root_quat_w)  # (x, y, z, w)
                start_yaw = torch.atan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
                start_height = start_pos[:, 2] - surface_height(start_pos[:, 0], start_pos[:, 1], a.slope)
                report["start_pose_after_settle"] = dict(
                    position_relative_m=start_pos.cpu().tolist(),
                    yaw_deg=torch.rad2deg(start_yaw).cpu().tolist(),
                    root_height_above_surface_m=start_height.cpu().tolist(),
                )
                if a.slope < 0:
                    on_platform = (start_pos[:, 0] >= top) & (start_pos[:, 0] <= end) & (start_pos[:, 1].abs() < WEDGE["half_width"])
                    standing = (start_height > 0.06) & (start_height < 0.2)
                    facing_ramp = torch.cos(start_yaw) < -0.95
                    if not bool((on_platform & standing & facing_ramp).all()):
                        raise RuntimeError("Downhill start is not standing on the platform facing the ramp: "
                                           + json.dumps(report["start_pose_after_settle"]))
                rows = []
                start = time.monotonic()
                error = torch.zeros(a.num_envs, device="cuda:0")
                upright_error = error.clone()
                upright_samples = error.clone()
                moving_samples = error.clone()
                if not a.headless:
                    # Low three-quarter view from the front left: the whole wedge (ramp face,
                    # side, platform) and the robot at either start stay in the picture.
                    eye = (
                        (
                            origin[0]
                            + torch.tensor([-0.35, -1.45, 0.55], device=env.device)
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
                    report["viewport"] = viewport_resolution(
                        parse_resolution(os.environ.get("MICRODUCK_VIEWPORT_RESOLUTION", VIEWPORT_RESOLUTION))
                    )
                    print("COURSE_VIEWPORT " + json.dumps(report["viewport"]), flush=True)
                # Real-time pacing against a cumulative deadline: a slow step is followed by
                # steps without waiting until the run is back on schedule (lag beyond 0.1 s
                # is dropped rather than caught up).
                deadline = loop_start = time.monotonic()
                rendered_frames = 0
                goal_printed = False
                with torch.inference_mode():
                    for step in range(round(a.seconds / 0.02)):
                        frame_start = time.monotonic()
                        cmd.is_standing_env[:] = False
                        cmd.vel_command_b[:] = 0
                        if step >= 100:
                            forward = 0.6 if a.locomotion == "rollers" else 0.25
                            cmd.vel_command_b[:, 0] = torch.where(reached, 0.0, forward)
                            if a.locomotion == "rollers":
                                # After the goal: hold still with the course roller brake
                                # (lab/roller_imu.RollerBrake, vectorised; zero would coast).
                                vx = T(robot.data.root_lin_vel_b)[:, 0]
                                candidate = (brake_integral + vx * 0.02).clamp(-0.15, 0.15)
                                requested = -3.0 * vx - 4.0 * candidate
                                keep = (requested >= -0.5) & (requested <= 0.35) & reached
                                brake_integral = torch.where(keep, candidate, brake_integral)
                                hold = (-3.0 * vx - 4.0 * brake_integral).clamp(-0.5, 0.35)
                                cmd.vel_command_b[:, 0] = torch.where(reached, hold, forward)
                        env.command_manager.get_term("head_pose").command[:] = torch.tensor(head_command, device=env.device)
                        env.command_manager.get_term("body_pose").command[:] = 0
                        action = policy(
                            env.observation_manager.compute(update_history=True)[
                                "policy"
                            ]
                        )
                        if brake.active:
                            # First policy output exists: release the temporary joint
                            # limits before this action is applied.
                            brake.release(step * 0.02)
                            report["startup"]["brake"] = brake.metadata()
                            print(
                                "COURSE_POLICY_TAKEOVER "
                                + json.dumps(dict(t=step * 0.02, step=step, envs=a.num_envs)),
                                flush=True,
                            )
                        if hold_enabled and head_hold is None:
                            head_hold = HeadHold(env, pitch_offset=head_pitch_offset)
                            report["head_hold"].update(hold_action=head_hold.hold_final.cpu().tolist(),
                                                       hold_target_rad=head_hold.target_rad.cpu().tolist(),
                                                       home_rad=head_hold.home_rad.cpu().tolist())
                        if head_hold is not None:
                            # Hold before the walk starts and after the goal; the policy's head while moving.
                            head_hold.update(reached | (step < 100))
                            action = head_hold(action)
                        env.action_manager.process_action(action)
                        if profile:
                            torch.cuda.synchronize()
                            t_policy = time.monotonic()
                        for tick in range(4):
                            env._sim_step_counter += 1
                            env.action_manager.apply_action()
                            env.scene.write_data_to_sim()
                            if profile and tick == 3:
                                torch.cuda.synchronize()
                                t_physics = time.monotonic()
                            # While recording with a window, every control step renders once (the
                            # recorded frame); headless, the camera update renders it instead.
                            draw = not a.headless and tick == 3 and (a.record or step % render_every == 0)
                            env.sim.step(render=draw)
                            rendered_frames += draw
                            env.scene.update(0.005)
                        if profile:
                            torch.cuda.synchronize()
                            t_render = time.monotonic()
                            timing["policy"].append(t_policy - frame_start)
                            timing["physics"].append(t_physics - t_policy)
                            timing["render"].append(t_render - t_physics)
                        env.episode_length_buf += 1
                        env.common_step_counter += 1
                        if a.record:
                            frame = capture(env, camera, video_process, renders=0)
                            crc = zlib.crc32(frame)
                            recorded["duplicate_consecutive_frames"] += crc == recorded["last_crc"]
                            recorded["frames"] += 1
                            recorded["last_crc"] = crc
                        pos = T(robot.data.root_pos_w) - origin
                        feet_x = T(robot.data.body_pos_w)[:, feet, 0] - origin[:, :1]
                        ground = surface_height(pos[:, 0], pos[:, 1], a.slope)
                        fallen |= (T(robot.data.projected_gravity_b)[:, 2] > -0.6) | (
                            pos[:, 2] - ground < 0.065
                        )
                        above = pos[:, 2] - ground  # root height above the surface under it
                        if a.slope >= 0:
                            # Uphill (or flat): supported by the platform, root 0.15 m and both feet onto it.
                            on_platform = (pos[:, 0] >= top) & (pos[:, 0] <= end) & (pos[:, 1].abs() < WEDGE["half_width"])
                            feet_in = (feet_x >= top + GOAL["foot_clearance"]).all(dim=1)
                            goal = (on_platform & (pos[:, 0] >= top + GOAL["up_margin"]) & feet_in
                                    & (pos[:, 1].abs() < inside) & (above > 0.06))
                            holding = on_platform & (above > 0.06)
                            fell_off_platform |= reached & ~on_platform
                        else:
                            # Downhill: supported by the floor, root 0.25 m and both feet beyond the ramp foot.
                            on_floor = pos[:, 0] < foot
                            feet_in = (feet_x <= foot - GOAL["foot_clearance"]).all(dim=1)
                            goal = ((pos[:, 0] <= foot - GOAL["down_margin"]) & feet_in
                                    & (pos[:, 1].abs() < inside) & (above > 0.06))
                            holding = on_floor & (above > 0.06)
                        # Holding must be continuous from arrival on: leaving and coming back fails.
                        left_goal |= reached & ~holding
                        newly = goal & ~fallen & ~reached
                        reached_time = torch.where(newly, torch.full_like(reached_time, (step + 1) * 0.02), reached_time)
                        reached |= newly
                        # No per-step GPU->CPU reads here (they stall the pipeline): environments
                        # that have arrived contribute zero, and the goal marker is printed at the
                        # next 0.1 s trajectory row with the exact arrival time.
                        if step >= 100:
                            moving = ~reached
                            squared_error = torch.where(
                                moving, (T(robot.data.root_lin_vel_b)[:, 0] - 0.25).square(), 0.0
                            )
                            error += squared_error
                            moving_samples += moving.float()
                            upright_error += torch.where(~fallen, squared_error, 0.0)
                            upright_samples += (~fallen & moving).float()
                        if step % 5 == 0:
                            if not goal_printed and bool(reached[0]):  # environment 0 is the one the window shows
                                goal_printed = True
                                print("COURSE_SLOPE_GOAL " + json.dumps(dict(env=0, sim_time_s=float(reached_time[0]))), flush=True)
                            rows.append(
                                dict(
                                    t=(step + 1) * 0.02,
                                    position=pos.cpu().tolist(),
                                    fell=fallen.cpu().tolist(),
                                    head_hold_weight=head_hold.weight.cpu().tolist() if head_hold else None,
                                    **head_state(),
                                )
                            )
                        if not a.headless:
                            timing["step"].append(time.monotonic() - frame_start)
                            deadline = max(deadline + 0.02, time.monotonic() - 0.1)
                            time.sleep(max(0.0, deadline - time.monotonic()))
                if not goal_printed and bool(reached[0]):  # arrived after the last trajectory row
                    print("COURSE_SLOPE_GOAL " + json.dumps(dict(env=0, sim_time_s=float(reached_time[0]))), flush=True)
                if a.record:
                    video_process.stdin.close()
                    if video_process.wait() != 0:
                        raise RuntimeError("Video encoding failed")
                    report["recording"] = dict(
                        width=int(camera.cfg.width),
                        height=int(camera.cfg.height),
                        fps_simulated_time=RECORD_FPS,
                        frames=recorded["frames"],
                        expected_frames=round(a.seconds / 0.02),
                        video_seconds=recorded["frames"] / RECORD_FPS,
                        simulated_seconds=a.seconds,
                        wall_seconds=time.monotonic() - start,
                        duplicate_consecutive_frames=recorded["duplicate_consecutive_frames"],
                        renders_per_frame=1,
                        render_path="window: sim.step(render=True) at the 4th physics step of each control step, "
                        "then the camera reads that render; headless: the camera update pumps one Kit update",
                        renderer="same live RTX settings as the window (no extra samples, no offline quality pass)",
                        frames_written="one per 20 ms control step, in order; no duplication, interpolation or speed change",
                        start_warmup_renders=8,
                    )
                held_seconds = torch.nan_to_num(a.seconds - reached_time, nan=0.0)
                report["head_at_end"] = head_state()
                if timing["step"]:
                    # Per control step (20 ms of simulated time): policy = observation + ONNX,
                    # physics = 3 physics steps (+ sync), render = 4th physics step + viewport render.
                    stats = {}
                    for key, values in timing.items():
                        if not values:
                            continue
                        ms = sorted(v * 1000 for v in values)
                        stats[key] = dict(mean_ms=sum(ms) / len(ms), p50_ms=ms[len(ms) // 2], p95_ms=ms[int(len(ms) * 0.95)])
                    loop_wall = time.monotonic() - loop_start
                    stats["real_time_factor"] = (len(timing["step"]) * 0.02) / loop_wall
                    stats["loop_wall_seconds"] = loop_wall
                    stats["loop_simulated_seconds"] = len(timing["step"]) * 0.02
                    stats["display"] = dict(
                        render_every_control_steps=display_every,
                        render_every_configured=render_every,
                        recording=a.record,
                        nominal_hz=50 / display_every,
                        rendered_frames=rendered_frames,
                        measured_frames_per_wall_second=rendered_frames / loop_wall,
                    )
                    report["timing"] = stats
                    if "viewport" in report:
                        report["viewport"]["at_end"] = viewport_resolution()
                    print("COURSE_TIMING " + json.dumps(stats), flush=True)
                goal_at_end = reached & ~left_goal & (held_seconds >= hold_seconds_required)
                report.update(
                    status="completed",
                    wall_seconds=time.monotonic() - start,
                    simulated_seconds=a.seconds,
                    physics_backend=manager.backend_name,
                    passed=(reached & ~fallen & goal_at_end).cpu().tolist(),
                    fell=fallen.cpu().tolist(),
                    reached_goal=reached.cpu().tolist(),
                    reached_goal_time_s=reached_time.cpu().tolist(),
                    held_goal_until_end=(reached & ~fallen & ~left_goal).cpu().tolist(),
                    held_seconds=held_seconds.cpu().tolist(),
                    left_goal_after_arrival=left_goal.cpu().tolist(),
                    fell_off_platform=fell_off_platform.cpu().tolist() if a.slope >= 0 else None,
                    hold_command=(
                        "walking: zero velocity request after arrival (the walking policy stands)"
                        if a.locomotion == "walking"
                        else "rollers: after arrival an external course brake controller (lab/roller_imu.RollerBrake, "
                        "vectorised) sets the push request from body speed; the same controller is used for every "
                        "roller model, so holding reflects policy + this controller, not the policy alone"
                    ),
                    velocity_reference_m_s=0.25 if a.locomotion == "walking" else None,
                    velocity_rmse=(
                        (error / moving_samples.clamp(min=1)).sqrt().cpu().tolist()
                        if a.locomotion == "walking"
                        else None
                    ),
                    upright_velocity_rmse=(
                        (upright_error / upright_samples.clamp(min=1))
                        .sqrt()
                        .cpu()
                        .tolist()
                        if a.locomotion == "walking"
                        else None
                    ),
                    upright_velocity_samples=upright_samples.cpu().tolist(),
                    final_position=pos.cpu().tolist(),
                    criterion=(
                        "Wedge v3 (ramp 0.5 m, platform 0.6 m, width 0.6 m). Uphill: reach the platform (supported by "
                        "it, root at least 0.15 m and both ankle bodies at least 0.06 m past the ramp top, "
                        "|y| < 0.15 m); the forward request then stops (walking 0, rollers course brake) and the robot "
                        "must stay on the platform continuously, upright, for at least 2 s and until the end. "
                        "Downhill: start verified standing on the platform facing the ramp; reach the floor at least "
                        "0.25 m (both ankle bodies 0.06 m) beyond the ramp foot (|y| < 0.15 m); stop; stay on the floor in front of the ramp "
                        "continuously, upright, for at least 2 s and until the end. Heights are measured above the "
                        "surface under the robot. A fall at any time fails; no reset."
                    ),
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
