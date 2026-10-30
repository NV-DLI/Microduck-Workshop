"""Course start-up protocol for policy verification runs (standing61).

The robot starts in a named joint pose with every joint temporarily locked by a
joint-limit brake; the root is free and no external force supports it. Physics
runs until every body has settled on its actual ground contact (so the grounded
height comes from the scene being simulated, flat or sloped). If it does not
settle in time, the run stops with the joints still locked and nothing is scored.
After settling, the caller asks the policy for its first action and releases the
brake before that action is applied; the brake restores and checks every
original joint limit and physical parameter.

Training runs do not use this: their episode resets and randomization are part of
the training objective.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "groot_walking/student"))
from joint_brake import JointBrake  # noqa: E402


class StartupNotSettled(RuntimeError):
    def __init__(self, record):
        super().__init__("Start-up did not settle; joints stay locked and the run is not scored: "
                         + record["failure"])
        self.record = record


def _t(x):
    return getattr(x, "torch", x)


def settle(env, *, pose="home", render=False, max_s=3.0, stable_s=0.5, physics_dt=0.005):
    """Lock joints in `pose`, let the free root settle, and return (brake, record).

    pose="home" writes the task's default (HOME) joint pose explicitly;
    pose="roller-rest" keeps the stance written by the roller reset event
    (assets/rollers/rest_pose.json). The brake is still active on return; call
    brake.release(t) after the first policy output and before applying it.
    Raises StartupNotSettled (with the record) if the bodies do not settle.
    """
    import torch

    robot = env.scene["robot"]
    if pose == "home":
        q = _t(robot.data.default_joint_pos).clone()
    elif pose == "roller-rest":
        q = _t(robot.data.joint_pos).clone()
    else:
        raise ValueError("pose must be 'home' or 'roller-rest'")
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    env.scene.write_data_to_sim()
    brake = JointBrake(robot)
    brake.lock()
    stable_since = None
    steps = 0
    t = 0.0
    last = {}
    while t < max_s:
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step(render=render and steps % 4 == 3)
        env.scene.update(physics_dt)
        steps += 1
        t = steps * physics_dt
        gravity = _t(robot.data.projected_gravity_b)
        speed = _t(robot.data.root_lin_vel_w).norm(dim=-1)
        spin = _t(robot.data.root_ang_vel_w).norm(dim=-1)
        last = dict(upright=bool((gravity[:, 2] < -0.99).all()), max_speed_m_s=float(speed.max()),
                    max_spin_rad_s=float(spin.max()))
        still = last["upright"] and last["max_speed_m_s"] < 0.01 and last["max_spin_rad_s"] < 0.05
        stable_since = (stable_since if stable_since is not None else t) if still else None
        if stable_since is not None and t - stable_since >= stable_s:
            break
    settled = stable_since is not None and t - stable_since >= stable_s
    origins = _t(env.scene.env_origins)
    record = dict(
        protocol="standing61: named joint pose locked by temporary joint limits, free root, no external support",
        pose=pose,
        joint_pose_rad=[[float(v) for v in row] for row in q.cpu()],
        locked_joint_axes=int(q.shape[1]),
        settle_seconds=t,
        settle_physics_steps=steps,
        criteria=dict(upright="projected gravity z < -0.99", max_speed_m_s=0.01, max_spin_rad_s=0.05,
                      stable_s=stable_s, max_s=max_s),
        last_state=last,
        settled=settled,
        root_position_relative_m=[[float(v) for v in row] for row in (_t(robot.data.root_pos_w) - origins).cpu()],
        root_quaternion=[[float(v) for v in row] for row in _t(robot.data.root_quat_w).cpu()],
        release="before the first policy action is applied",
    )
    if not settled:
        record["failure"] = f"not stable within {max_s} s (last state {last})"
        raise StartupNotSettled(record)
    return brake, record
