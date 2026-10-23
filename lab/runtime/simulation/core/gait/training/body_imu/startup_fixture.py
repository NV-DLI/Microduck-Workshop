"""Batched version of the declared physical startup fixture; no state writes."""

import torch


def norm_limit(v, maximum):
    return v * (
        maximum / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-12)
    ).clamp(max=1.0)


def rotation_error(target, current):
    # target * inverse(current), XYZW. Shortest rotation vector.
    xyz = (
        -target[:, 3:] * current[:, :3]
        + current[:, 3:] * target[:, :3]
        - torch.cross(target[:, :3], current[:, :3], dim=-1)
    )
    w = (target * current).sum(-1, keepdim=True)
    q = torch.cat((xyz, w), dim=-1)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[:, 3:] < 0, -q, q)
    sine = torch.linalg.vector_norm(q[:, :3], dim=-1, keepdim=True)
    return q[:, :3] * torch.where(
        sine < 1e-9, 2.0, 2.0 * torch.atan2(sine, q[:, 3:]) / sine.clamp_min(1e-12)
    )


class StartupFixture:
    def __init__(self, num_envs, device):
        self.n = num_envs
        self.device = device
        self.reference = torch.zeros((num_envs, 7), device=device)
        self.reference[:, 6] = 1
        self.mass = torch.ones(num_envs, device=device)
        self.valid_at = torch.full((num_envs,), -1.0, device=device)
        self.task_started = self.valid_at.clone()
        self.failed = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.ready = self.failed.clone()
        self.task_active = self.failed.clone()
        self.gain = torch.ones(num_envs, device=device)
        self.force = torch.zeros((num_envs, 3), device=device)
        self.torque = self.force.clone()

    def reset(self, ids, *, reference_pose_xyzw, actual_mass_kg):
        ids = slice(None) if ids is None else ids
        self.reference[ids] = reference_pose_xyzw[ids]
        self.mass[ids] = actual_mass_kg[ids]
        self.valid_at[ids] = -1
        self.task_started[ids] = -1
        self.failed[ids] = False
        self.ready[ids] = False
        self.task_active[ids] = False
        self.gain[ids] = 1
        self.force[ids] = 0
        self.torque[ids] = 0

    def update(
        self,
        t,
        *,
        position,
        quaternion,
        linear_velocity,
        angular_velocity,
        orientation_valid
    ):
        t = torch.as_tensor(t, device=self.device, dtype=self.reference.dtype)
        position, quaternion, linear_velocity, angular_velocity = [
            value.to(device=self.device, dtype=self.reference.dtype)
            for value in (position, quaternion, linear_velocity, angular_velocity)
        ]
        first_valid = (self.valid_at < 0) & orientation_valid & ~self.failed
        self.valid_at = torch.where(first_valid, t, self.valid_at)
        invalidated = (self.valid_at >= 0) & ~orientation_valid & ~self.task_active
        timeout = (self.valid_at < 0) & (t >= 5.0)
        self.failed |= invalidated | timeout
        release_u = (t - self.valid_at - 0.5) / 1.0
        release_done = (self.valid_at >= 0) & (release_u >= 1.0 - 1e-7) & ~self.failed
        self.task_started = torch.where(
            release_done & ~self.task_active, t, self.task_started
        )
        self.task_active |= release_done
        self.ready = orientation_valid & ~self.failed
        release_gain = 0.5 * (1.0 + torch.cos(torch.pi * release_u.clamp(0, 1)))
        self.gain = torch.where(self.valid_at < 0, 1.0, release_gain)
        self.gain = torch.where(self.task_active | self.failed, 0.0, self.gain)
        f = 150.0 * (self.reference[:, :3] - position) - 12.0 * linear_velocity
        f[:, 2] += self.mass * 9.80665
        torque = (
            2.0 * rotation_error(self.reference[:, 3:], quaternion)
            - 0.16 * angular_velocity
        )
        self.force = norm_limit(f, 12.0) * self.gain[:, None]
        self.torque = norm_limit(torque, 0.6) * self.gain[:, None]
        return self.force, self.torque
