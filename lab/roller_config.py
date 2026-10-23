"""Official passive-wheel geometry with the course's native BAM motor model."""

from pathlib import Path
import json
import hashlib


def reset_roller_stance(env, env_ids):
    """Reset physical joints without changing the actor home/normalization pose."""
    import torch

    robot = env.scene["robot"]
    ids = (
        env_ids
        if env_ids is not None
        else torch.arange(env.num_envs, device=env.device)
    )
    pose = json.loads(
        (Path(__file__).parent / "assets/rollers/rest_pose.json").read_text()
    )["joint_positions_rad"]
    q = robot.data.default_joint_pos[ids].clone()
    for i, name in enumerate(robot.joint_names):
        if name in pose:
            q[:, i] = pose[name]
    robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=ids)


def configure_rollers(cfg):
    cfg.scene.robot.spawn.usd_path = str(
        Path(__file__).parent / "assets/rollers/microduck_rollers_analytic.usdc"
    )
    cfg.scene.robot.init_state.pos = (*cfg.scene.robot.init_state.pos[:2], 0.122)
    # The four tyres are independent rigid bodies; ankle blocks are not soles.
    cfg.scene.contact_forces.prim_path = (
        "{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/.*/tire(_[2-4])?)?"
    )
    cfg.scene.self_collision = None
    cfg.observations.critic = None
    # This is a deployment environment. Walking rewards/sole heights are not a
    # roller training recipe and must not silently measure ankle blocks as feet.
    for name in vars(cfg.rewards):
        if not name.startswith("_"):
            setattr(cfg.rewards, name, None)
    cfg.events.foot_friction = None
    cfg.events.reset_robot_joints.func = reset_roller_stance
    cfg.events.reset_robot_joints.params = {}
    cfg.scene.robot.init_state.joint_pos = {
        "left_hip_yaw": 0.0,
        "left_hip_roll": -0.0873,
        "left_hip_pitch": -0.4579,
        "left_knee": -0.0049,
        "left_ankle": 0.4530,
        "neck_pitch": 0.3491,
        "head_pitch": 0.3491,
        "head_yaw": 0.0,
        "head_roll": 0.0,
        "right_hip_yaw": 0.0,
        "right_hip_roll": 0.0873,
        "right_hip_pitch": 0.4579,
        "right_knee": 0.0049,
        "right_ankle": -0.4530,
        "passive_.*": 0.0,
    }
    return dict(
        kind="official four passive wheels with native BAM and gearbox play",
        powered_joints=14,
        passive_wheels=4,
        passive_backlash_hinges=14,
        tyre_collision="analytic cylinder: radius 15 mm, width 7.6 mm; rounded shoulders approximated",
        tyre_visual_and_inertia="official original mesh and inertia retained",
        initial_root_height_m=0.122,
        reset_pose_sha256=hashlib.sha256(
            (Path(__file__).parent / "assets/rollers/rest_pose.json").read_bytes()
        ).hexdigest(),
        command_x="positive push, zero coast, negative brake",
        command_y="unused",
        command_z="relative heading request, not calibrated yaw rate",
        shooting="course gliding ball contact; roller model remains active, no kick actor",
    )


def check_roller_articulation(robot, observation):
    names = robot.joint_names
    wheels = [n for n in names if n.startswith("passive_") and n.endswith("wheel")]
    play = [n for n in names if n.endswith("_backlash")]
    servos = [n for n in names if not n.startswith("passive_")]
    if (
        len(wheels) != 4
        or len(play) != 14
        or len(servos) != 14
        or observation.shape[-1] != 61
    ):
        raise RuntimeError(
            f"Invalid roller articulation/actor contract: {names}, {observation.shape}"
        )
    return dict(
        servo_names=servos,
        wheel_names=wheels,
        backlash_names=play,
        observation_width=61,
    )
