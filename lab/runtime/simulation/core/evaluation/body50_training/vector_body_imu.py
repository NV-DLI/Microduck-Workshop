"""Batched virtual BODY50 sensor, not a privileged policy observation.

Matches the explicitly assumed BetaBodyImu zero-extra-delay pipeline. World
kinematics enter this sensor boundary only. Outputs contain decoded local
measurements and readiness/timestamps, never absolute pose. Use float64 inside
fusion/quantization; cast the decoded policy observation at the final boundary.
The default tau2 is the declared football sensitivity, not ST SFLP firmware.
"""

from dataclasses import dataclass
import math
import torch

G = 9.80665
GYRO_LSB = math.radians(0.0175)
ACCEL_LSB = 0.000122 * G


def norm(v):
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(1e-30)


def cross(a, b):
    return torch.linalg.cross(a, b, dim=-1)


def inverse(q):
    q = norm(q)
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def rotate(q, v):
    q = norm(q)
    v = torch.broadcast_to(v, q.shape[:-1] + (3,))
    t = 2 * cross(q[..., :3], v)
    return v + q[..., 3:] * t + cross(q[..., :3], t)


def qmul(q, r):
    a, w = q[..., :3], q[..., 3:]
    b, z = r[..., :3], r[..., 3:]
    return torch.cat(
        (w * b + z * a + cross(a, b), w * z - (a * b).sum(-1, keepdim=True)), -1
    )


def between(a, b):
    a = norm(a)
    b = norm(torch.broadcast_to(b, a.shape))
    dot = (a * b).sum(-1, keepdim=True)
    q = norm(torch.cat((cross(a, b), 1 + dot), -1))
    x = torch.zeros_like(a)
    x[..., 0] = 1
    y = torch.zeros_like(a)
    y[..., 1] = 1
    axis = cross(a, x)
    axis = torch.where((axis * axis).sum(-1, keepdim=True) < 1e-12, cross(a, y), axis)
    flip = torch.cat((norm(axis), torch.zeros_like(dot)), -1)
    return torch.where(dot < -0.999999, flip, q)


@dataclass(frozen=True)
class VectorBodyProfile:
    simulated_sample_hz: float = 120.0
    read_hz: float = 50.0
    orientation_correction_tau_s: float = 2.0
    gyro_noise_std_rad_s: float = 0.003
    accel_noise_std_m_s2: float = 0.03
    gyro_bias_rad_s: tuple = (0.0, 0.0, 0.002)
    pipeline_delay_s: float = 0.0
    additional_transport_delay_s: float = 0.0


class VectorBodyImu:
    """Poll once after every200Hz physics update; reset selected envs separately.

    `timestamp_s` is each env's elapsed sensor time since reset (scalar or[N]).
    q_world_from_trunk_xyzw, angular_velocity_world_rad_s and linear_acceleration_
    world_m_s2 are private producer inputs. Only observe()/poll() outputs reach RL.
    The first two finite-difference ticks belong in a producer adapter, not here.
    Optional noise_standard_normal[N,6] permits exact CPU-reference testing; default
    Torch Gaussian draws use an explicit independent generator and same distribution.
    """

    def __init__(
        self,
        num_envs,
        *,
        device="cpu",
        profile=None,
        seed=38112,
        strict_validation=None
    ):
        self.n = num_envs
        self.device = torch.device(device)
        self.dtype = torch.float64
        self.profile = profile or VectorBodyProfile()
        p = self.profile
        if p.pipeline_delay_s or p.additional_transport_delay_s:
            raise ValueError(
                "This frozen contract supports explicit zero additional queue only"
            )
        if (
            p.read_hz != 50
            or p.simulated_sample_hz != 120
            or p.orientation_correction_tau_s <= 0
        ):
            raise ValueError("Pinned120/50Hz and positive tau required")
        self.strict_validation = (
            (self.device.type == "cpu")
            if strict_validation is None
            else strict_validation
        )
        self.window = math.ceil(0.2 * p.simulated_sample_hz) + 1
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self.up = torch.tensor([0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        self.down = -self.up
        self.bias = torch.tensor(
            p.gyro_bias_rad_s, dtype=self.dtype, device=self.device
        )

        def f(*shape):
            return torch.zeros((num_envs, *shape), device=self.device, dtype=self.dtype)

        def i():
            return torch.zeros(num_envs, device=self.device, dtype=torch.int64)

        def b():
            return torch.zeros(num_envs, device=self.device, dtype=torch.bool)

        self.orientation = f(4)
        self.initialized = b()
        self.last_sample = f()
        self.last_poll = f()
        self.next_sample = i()
        self.next_read = i()
        self.acquired_seq = i()
        self.read_seq = i()
        self.has_pending = b()
        self.initial_count = i()
        self.initial_cursor = i()
        self.initial_accel = f(self.window, 3)
        self.initial_times = f(self.window)
        self.latest_gyro = f(3)
        self.latest_accel = f(3)
        self.latest_sample_t = f()
        self.latest_gyro_clipped = torch.zeros(
            (num_envs, 3), device=self.device, dtype=torch.bool
        )
        self.latest_accel_clipped = self.latest_gyro_clipped.clone()
        self.decoder_quat = f(4)
        self.live_reads = i()
        self.gyro_history = f(2, 3)
        self.gravity_history = f(2, 3)
        self.delivered_gyro = f(3)
        self.delivered_gravity = f(3)
        self.delivered_quat = f(4)
        self.delivered_valid = b()
        self.delivered_present = b()
        self.delivered_sample_t = f()
        self.delivered_read_t = f()
        self.delivered_seq = i()
        self.delivered_sample_seq = i()
        self.gyro_counts = torch.zeros(
            (num_envs, 3), device=self.device, dtype=torch.int16
        )
        self.quat_half_bits = self.gyro_counts.clone()
        self.quaternion_updated = b()
        self.delivered_gyro_clipped = self.latest_gyro_clipped.clone()
        self.delivered_accel_clipped = self.latest_accel_clipped.clone()
        self.reset()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name, value in vars(self).items():
            if (
                isinstance(value, torch.Tensor)
                and value.shape[:1] == (self.n,)
                and name not in ("up", "down", "bias")
            ):
                value[ids] = 0
        self.orientation[ids, 3] = 1.0
        self.decoder_quat[ids, 3] = 1.0
        self.delivered_quat[ids, 3] = 1.0
        self.gravity_history[ids, :, 2] = -1.0
        self.last_sample[ids] = -torch.inf
        self.last_poll[ids] = -torch.inf

    def _put(self, target, value, mask):
        target.copy_(
            torch.where(
                mask.reshape((self.n,) + (1,) * (target.ndim - 1)), value, target
            )
        )

    def poll(
        self,
        timestamp_s,
        *,
        q_world_from_trunk_xyzw,
        angular_velocity_world_rad_s,
        linear_acceleration_world_m_s2,
        noise_standard_normal=None,
        enabled_mask=None
    ):
        p = self.profile
        t = torch.as_tensor(timestamp_s, device=self.device, dtype=self.dtype).expand(
            self.n
        )
        q = torch.as_tensor(
            q_world_from_trunk_xyzw, device=self.device, dtype=self.dtype
        )
        w = torch.as_tensor(
            angular_velocity_world_rad_s, device=self.device, dtype=self.dtype
        )
        a = torch.as_tensor(
            linear_acceleration_world_m_s2, device=self.device, dtype=self.dtype
        )
        if q.shape != (self.n, 4) or w.shape != (self.n, 3) or a.shape != (self.n, 3):
            raise ValueError("Private kinematic batch dimensions do not match")
        if self.strict_validation and (
            not torch.isfinite(t).all()
            or (t < 0).any()
            or (t < self.last_poll).any()
            or not torch.isfinite(q).all()
            or not torch.isfinite(w).all()
            or not torch.isfinite(a).all()
            or (torch.linalg.vector_norm(q, dim=-1) < 1e-12).any()
        ):
            raise ValueError(
                "Finite monotone sensor time and finite valid kinematics required"
            )
        self.last_poll.copy_(t)
        enabled = (
            torch.ones(self.n, device=self.device, dtype=torch.bool)
            if enabled_mask is None
            else enabled_mask
        )
        sample_index = torch.floor(t * 120 + 1e-9).long()
        acquire = (sample_index >= self.next_sample) & enabled
        self._put(self.next_sample, sample_index + 1, acquire)
        noise = (
            torch.randn(
                (self.n, 6),
                generator=self.generator,
                device=self.device,
                dtype=self.dtype,
            )
            if noise_standard_normal is None
            else torch.as_tensor(
                noise_standard_normal, device=self.device, dtype=self.dtype
            )
        )
        gi = rotate(inverse(q), w) + self.bias + noise[:, :3] * p.gyro_noise_std_rad_s
        ai = (
            rotate(inverse(q), a - self.down * G)
            + noise[:, 3:] * p.accel_noise_std_m_s2
        )
        gc = abs(gi) > math.radians(500)
        ac = abs(ai) > 4 * G
        gyro = (
            torch.round(gi.clamp(-math.radians(500), math.radians(500)) / GYRO_LSB)
            * GYRO_LSB
        )
        accel = torch.round(ai.clamp(-4 * G, 4 * G) / ACCEL_LSB) * ACCEL_LSB
        an = torch.linalg.vector_norm(accel, dim=-1)
        gap = torch.isfinite(self.last_sample) & ((t - self.last_sample) > 0.04)
        lost = acquire & (gap | gc.any(-1))
        self.initialized &= ~lost
        self._put(self.initial_count, torch.zeros_like(self.initial_count), lost)
        old_initialized = self.initialized.clone()
        need = acquire & ~old_initialized
        plausible = (
            ~(gc.any(-1) | ac.any(-1))
            & ((an - G).abs() <= 1.18)
            & (torch.linalg.vector_norm(gyro, dim=-1) <= 0.25)
        )
        self._put(
            self.initial_count, torch.zeros_like(self.initial_count), need & ~plausible
        )
        append = need & plausible
        ix = torch.arange(self.n, device=self.device)
        old_a = self.initial_accel[ix, self.initial_cursor]
        old_t = self.initial_times[ix, self.initial_cursor]
        self.initial_accel[ix, self.initial_cursor] = torch.where(
            append[:, None], accel, old_a
        )
        self.initial_times[ix, self.initial_cursor] = torch.where(append, t, old_t)
        self._put(self.initial_cursor, (self.initial_cursor + 1) % self.window, append)
        self._put(
            self.initial_count, (self.initial_count + 1).clamp_max(self.window), append
        )
        mean = self.initial_accel.mean(1)
        std = torch.sqrt(((self.initial_accel - mean[:, None]) ** 2).mean(1))
        deviation = torch.rad2deg(
            torch.acos(
                (norm(self.initial_accel) * norm(mean)[:, None]).sum(-1).clamp(-1, 1)
            )
        ).amax(1)
        initialize = (
            append
            & (self.initial_count >= self.window)
            & ((self.initial_times.amax(1) - self.initial_times.amin(1)) >= 0.18 - 1e-9)
            & (std.amax(1) <= 0.20)
            & (deviation <= 3.0)
        )
        self._put(self.orientation, between(mean, self.up), initialize)
        self.initialized |= initialize
        dt = torch.where(
            torch.isfinite(self.last_sample), t - self.last_sample, torch.zeros_like(t)
        )
        mu = torch.where((an > 1e-6)[:, None], norm(accel), self.up)
        pu = rotate(inverse(self.orientation), self.up)
        gain = torch.where(
            (an >= 8.3) & (an <= 11.3),
            torch.full_like(t, 1 / p.orientation_correction_tau_s),
            torch.zeros_like(t),
        )
        omega = gyro + gain[:, None] * cross(mu, pu)
        speed = torch.linalg.vector_norm(omega, dim=-1)
        angle = speed * dt
        dq = torch.cat(
            (
                omega / speed.clamp_min(1e-30)[:, None] * torch.sin(angle / 2)[:, None],
                torch.cos(angle / 2)[:, None],
            ),
            -1,
        )
        self._put(
            self.orientation,
            norm(qmul(self.orientation, dq)),
            acquire & old_initialized & torch.isfinite(self.last_sample),
        )
        self._put(self.last_sample, t, acquire)
        self._put(self.latest_gyro, gyro, acquire)
        self._put(self.latest_accel, accel, acquire)
        self._put(self.latest_sample_t, t, acquire)
        self._put(self.latest_gyro_clipped, gc, acquire)
        self._put(self.latest_accel_clipped, ac, acquire)
        self.acquired_seq += acquire.long()
        self.has_pending |= acquire
        flush_index = torch.floor(t * 50 + 1e-9).long()
        flush = (flush_index >= self.next_read) & enabled
        self._put(self.next_read, flush_index + 1, flush)
        read = flush & self.has_pending
        self.has_pending &= ~flush
        counts = (
            torch.round(self.latest_gyro / GYRO_LSB)
            .clamp(-32768, 32767)
            .to(torch.int16)
        )
        sign = torch.where(self.orientation[:, 3:4] < 0, -1.0, 1.0)
        xyz = torch.where(
            self.initialized[:, None],
            self.orientation[:, :3] * sign,
            torch.zeros_like(self.orientation[:, :3]),
        )
        half = xyz.to(torch.float16)
        bits = half.view(torch.int16)
        xyz = half.to(self.dtype)
        sq = (xyz * xyz).sum(-1)
        update = (bits != 0).any(-1) & torch.isfinite(xyz).all(-1) & (sq <= 1.02)
        qd = torch.cat((xyz, torch.sqrt((1 - sq).clamp_min(0))[:, None]), -1)
        update &= (qd * qd).sum(-1) > 0.25
        self._put(self.decoder_quat, norm(qd), read & update)
        self.live_reads += (read & update).long()
        gravity = norm(rotate(inverse(self.decoder_quat), self.down))
        raw_gyro = counts.to(self.dtype) * GYRO_LSB
        filtered_g = torch.median(
            torch.cat((self.gyro_history, raw_gyro[:, None]), 1), dim=1
        ).values
        filtered_v = torch.median(
            torch.cat((self.gravity_history, gravity[:, None]), 1), dim=1
        ).values
        self._put(
            self.gyro_history, torch.stack((self.gyro_history[:, 1], raw_gyro), 1), read
        )
        self._put(
            self.gravity_history,
            torch.stack((self.gravity_history[:, 1], gravity), 1),
            read,
        )
        self._put(self.delivered_gyro, filtered_g, read)
        self._put(self.delivered_gravity, filtered_v, read)
        self._put(self.delivered_quat, self.decoder_quat, read)
        self._put(
            self.delivered_valid, self.initialized & (self.live_reads >= 25), read
        )
        self.delivered_present |= read
        self._put(self.delivered_sample_t, self.latest_sample_t, read)
        self._put(self.delivered_read_t, t, read)
        self._put(self.delivered_seq, self.read_seq, read)
        self._put(self.delivered_sample_seq, self.acquired_seq - 1, read)
        self._put(self.gyro_counts, counts, read)
        self._put(self.quat_half_bits, bits, read)
        self._put(self.quaternion_updated, update, read)
        self._put(self.delivered_gyro_clipped, self.latest_gyro_clipped, read)
        self._put(self.delivered_accel_clipped, self.latest_accel_clipped, read)
        self.read_seq += read.long()
        return self.observe(t, delivered_mask=read)

    def observe(self, timestamp_s, delivered_mask=None):
        t = torch.as_tensor(timestamp_s, device=self.device, dtype=self.dtype).expand(
            self.n
        )
        age = t - self.delivered_sample_t
        valid = (
            self.delivered_present
            & self.delivered_valid
            & (age >= -1e-9)
            & (age <= 0.08)
        )
        return {
            "delivered_mask": (
                torch.zeros_like(valid) if delivered_mask is None else delivered_mask
            ),
            "orientation_valid": valid,
            "gyro_body_rad_s": self.delivered_gyro.clone(),
            "gravity_body": self.delivered_gravity.clone(),
            "quat_wxyz": self.delivered_quat[:, [3, 0, 1, 2]].clone(),
            "timestamp_s": self.delivered_sample_t.clone(),
            "read_timestamp_s": self.delivered_read_t.clone(),
            "measurement_age_s": age,
            "seq": self.delivered_seq.clone(),
            "sample_seq": self.delivered_sample_seq.clone(),
            "live_quaternion_reads": self.live_reads.clone(),
            "gyro_counts": self.gyro_counts.clone(),
            "quaternion_xyz_half_bits": self.quat_half_bits.clone(),
            "gyro_clipped": self.delivered_gyro_clipped.clone(),
            "accel_clipped": self.delivered_accel_clipped.clone(),
        }
