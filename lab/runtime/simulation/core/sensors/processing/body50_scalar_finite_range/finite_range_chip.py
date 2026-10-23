# Frozen scalar sensor parameters; only initialized clipping invalidation changed.
# Proprietary SFLP firmware is not reproduced.
"""Local head IMU observations, with arbitrary yaw and explicit modeled noise.

The official Beta decoder publishes head FLU, WXYZ orientation, gyro rad/s,
and accelerometer specific force m/s². Chip fusion firmware is not reproduced.
Do not feed the simulator's absolute orientation into the student controller.
"""
from __future__ import annotations
import math, random
from collections import deque
from dataclasses import dataclass, asdict
from beta_sensor_profile import HEAD_IMU_PROFILE


def normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError("Finite nonzero vector required")
    return tuple(x / n for x in v)


def cross(a, b):
    x, y, z = a
    X, Y, Z = b
    return y * Z - z * Y, z * X - x * Z, x * Y - y * X


def qmul(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return (
        w * X + x * W + y * Z - z * Y,
        w * Y - x * Z + y * W + z * X,
        w * Z + x * Y - y * X + z * W,
        w * W - x * X - y * Y - z * Z,
    )


def inverse(q):
    x, y, z, w = normalize(q)
    return -x, -y, -z, w


def rotate(q, v):
    q = normalize(q)
    t = tuple(2 * x for x in cross(q[:3], v))
    u = cross(q[:3], t)
    return tuple(v[i] + q[3] * t[i] + u[i] for i in range(3))


def quaternion_between(a, b):
    a, b = normalize(a), normalize(b)
    d = sum(x * y for x, y in zip(a, b))
    if d < -0.999999:
        axis = cross(a, (1.0, 0.0, 0.0))
        if sum(x * x for x in axis) < 1e-12:
            axis = cross(a, (0.0, 1.0, 0.0))
        return (*normalize(axis), 0.0)
    return normalize((*cross(a, b), 1 + d))


@dataclass(frozen=True)
class ImuInitializationCfg:
    """Explicit simulation fusion assumptions, not a model of ST firmware."""

    minimum_samples: int = 12
    minimum_duration_s: float = 0.18
    maximum_sample_gap_s: float = 0.04
    gravity_norm_tolerance_m_s2: float = 1.18
    maximum_gyro_norm_rad_s: float = 0.25
    maximum_accel_component_std_m_s2: float = 0.20
    maximum_direction_deviation_deg: float = 3.0


class BetaHeadImu:
    """60 Hz acquisitions and 15 Hz delivery, timestamps belong to acquisition.

    Call poll once per physics tick. Late polls acquire only one real sample;
    missing samples are counted, not manufactured. World truth is an input to
    the virtual sensor only; the fused output starts with arbitrary yaw and
    integrates noisy local gyro. Consumers receive no world pose or position.
    """

    def __init__(
        self,
        *,
        profile=HEAD_IMU_PROFILE,
        seed=38111,
        initialization=ImuInitializationCfg(),
    ):
        self.profile = profile
        self.initialization = initialization
        self.rng = random.Random(seed)
        self.reset()

    def reset(self):
        self.orientation = None
        self.last_time = None
        self.next_sample = 0
        self.next_flush = 0
        self.queue = []
        self.latest = None
        self.seq = 0
        self.missed = 0
        self._initial_samples = deque(maxlen=self.initialization.minimum_samples)
        self.initialized_at = None
        self.initialization_status = "waiting_for_stationary_gravity"

    def _initialize_from_measurements(
        self, timestamp_s, gyro, accel, acc_norm, clipped
    ):
        """Use only measured gyro/specific force; never the true head pose.

        A free-falling accelerometer has almost zero specific force, so its
        direction is noise. A single apparently plausible sample is also
        insufficient after a landing impulse: require a continuous quiet window.
        """
        cfg = self.initialization
        gyro_norm = math.sqrt(sum(x * x for x in gyro))
        if clipped:
            reason = "waiting_sensor_saturated"
        elif abs(acc_norm - 9.80665) > cfg.gravity_norm_tolerance_m_s2:
            reason = "waiting_gravity_magnitude"
        elif gyro_norm > cfg.maximum_gyro_norm_rad_s:
            reason = "waiting_low_angular_speed"
        else:
            reason = None
        if reason:
            self._initial_samples.clear()
            self.initialization_status = reason
            return
        self._initial_samples.append((timestamp_s, accel))
        window = list(self._initial_samples)
        count = len(window)
        self.initialization_status = "collecting_stationary_gravity"
        if (
            count < cfg.minimum_samples
            or window[-1][0] - window[0][0] < cfg.minimum_duration_s - 1e-9
        ):
            return
        mean = tuple(sum(sample[1][i] for sample in window) / count for i in range(3))
        std = tuple(
            math.sqrt(sum((sample[1][i] - mean[i]) ** 2 for sample in window) / count)
            for i in range(3)
        )
        up = normalize(mean)
        deviation = max(
            math.degrees(
                math.acos(
                    max(
                        -1.0,
                        min(1.0, sum(a * b for a, b in zip(normalize(sample[1]), up))),
                    )
                )
            )
            for sample in window
        )
        if (
            max(std) > cfg.maximum_accel_component_std_m_s2
            or deviation > cfg.maximum_direction_deviation_deg
        ):
            self.initialization_status = "waiting_stable_gravity_direction"
            return
        self.orientation = quaternion_between(mean, (0.0, 0.0, 1.0))
        self.initialized_at = timestamp_s
        self.initialization_status = "initialized_from_stationary_accelerometer"

    def _measure(
        self,
        timestamp_s,
        q_world_from_head_xyzw,
        angular_velocity_world_rad_s,
        linear_acceleration_world_m_s2,
    ):
        p = self.profile
        q_inv = inverse(q_world_from_head_xyzw)
        gyro = rotate(q_inv, angular_velocity_world_rad_s)
        specific_force = tuple(
            a - g for a, g in zip(linear_acceleration_world_m_s2, (0.0, 0.0, -9.80665))
        )
        accel = rotate(q_inv, specific_force)
        gyro_lsb = math.radians(0.0175)
        accel_lsb = 0.000122 * 9.80665
        noisy_g = [
            gyro[i] + p.gyro_bias_rad_s[i] + self.rng.gauss(0.0, p.gyro_noise_std_rad_s)
            for i in range(3)
        ]
        noisy_a = [a + self.rng.gauss(0.0, p.accel_noise_std_m_s2) for a in accel]
        gs = [bool(abs(x) > math.radians(500)) for x in noisy_g]
        a_s = [bool(abs(x) > 4 * 9.80665) for x in noisy_a]
        gyro = tuple(
            round(max(-math.radians(500), min(math.radians(500), v)) / gyro_lsb)
            * gyro_lsb
            for v in noisy_g
        )
        accel = tuple(
            round(max(-4 * 9.80665, min(4 * 9.80665, v)) / accel_lsb) * accel_lsb
            for v in noisy_a
        )
        acc_norm = math.sqrt(sum(x * x for x in accel))
        gap = (
            self.last_time is not None
            and timestamp_s - self.last_time > self.initialization.maximum_sample_gap_s
        )
        if gap:
            # Lost gyro chronology cannot be repaired with a world-pose lookup.
            self.orientation = None
            self.initialized_at = None
            self._initial_samples.clear()
            self.initialization_status = (
                "waiting_after_sample_gap" if gap else "waiting_sensor_saturated"
            )
        if self.orientation is None:
            self._initialize_from_measurements(
                timestamp_s, gyro, accel, acc_norm, any(gs) or any(a_s)
            )
        elif self.last_time is not None:
            dt = timestamp_s - self.last_time
            measured_up = normalize(accel) if acc_norm > 1e-6 else (0.0, 0.0, 1.0)
            predicted_up = rotate(inverse(self.orientation), (0.0, 0.0, 1.0))
            error = cross(measured_up, predicted_up)
            # Dynamic acceleration is not gravity: suppress tilt correction if
            # the norm is outside a conservative experiment-selected interval.
            gain = (
                1 / p.orientation_correction_tau_s if 8.3 <= acc_norm <= 11.3 else 0.0
            )
            omega = tuple(g + gain * e for g, e in zip(gyro, error))
            speed = math.sqrt(sum(x * x for x in omega))
            angle = speed * dt
            dq = (
                tuple(x / speed * math.sin(angle / 2) for x in omega)
                + (math.cos(angle / 2),)
                if speed > 1e-12
                else (0.0, 0.0, 0.0, 1.0)
            )
            self.orientation = normalize(qmul(self.orientation, dq))
        self.last_time = timestamp_s
        q = self.orientation
        frame = {
            "seq": self.seq,
            "timestamp_s": timestamp_s,
            "gyro": list(gyro),
            "accel": list(accel),
            "quat": [q[3], q[0], q[1], q[2]] if q is not None else None,
            "orientation_valid": q is not None,
            "orientation_initialized_at_s": self.initialized_at,
            "initialization_status": self.initialization_status,
            "initialization_stable_samples": len(self._initial_samples),
            "quaternion_order": "wxyz",
            "axes": "head FLU",
            "gyro_unit": "rad/s",
            "accel_unit": "m/s2 specific force (includes gravity)",
            "orientation_reference": "gravity-aligned local frame with arbitrary initial yaw",
            "absolute_world_yaw": False,
            "gyro_clipped": gs,
            "accel_clipped": a_s,
            "noise_is_measured": False,
            "sflp_firmware_reproduced": False,
        }
        self.seq += 1
        self.latest = frame
        return frame

    def poll(
        self,
        timestamp_s,
        *,
        q_world_from_head_xyzw,
        angular_velocity_world_rad_s,
        linear_acceleration_world_m_s2,
    ):
        if (
            not math.isfinite(timestamp_s)
            or timestamp_s < 0
            or (self.last_time is not None and timestamp_s < self.last_time)
        ):
            raise ValueError(
                "Monotonic nonnegative simulation time required; reset for a new episode"
            )
        for v, n in (
            (q_world_from_head_xyzw, 4),
            (angular_velocity_world_rad_s, 3),
            (linear_acceleration_world_m_s2, 3),
        ):
            if len(v) != n or not all(math.isfinite(x) for x in v):
                raise ValueError("Finite IMU state required")
        latest = int(math.floor(timestamp_s * self.profile.simulated_sample_hz + 1e-9))
        if latest >= self.next_sample:
            self.missed += latest - self.next_sample
            self.next_sample = latest + 1
            self.queue.append(
                self._measure(
                    timestamp_s,
                    q_world_from_head_xyzw,
                    angular_velocity_world_rad_s,
                    linear_acceleration_world_m_s2,
                )
            )
        flush = int(math.floor(timestamp_s * self.profile.fifo_read_hz + 1e-9))
        if flush < self.next_flush:
            return []
        self.next_flush = flush + 1
        batch = self.queue
        self.queue = []
        return [
            {
                **frame,
                "delivered_timestamp_s": timestamp_s,
                "fifo_latency_s": timestamp_s - frame["timestamp_s"],
            }
            for frame in batch
        ]

    def metadata(self):
        return {
            **self.profile.to_dict(),
            "successful_samples": self.seq,
            "missed_samples": self.missed,
            "initialization": {
                **asdict(self.initialization),
                "thresholds_are_measured": False,
                "invalid_output": "orientation_valid=false and quat=null until a continuous stationary gravity window",
                "assumption": "near-gravity magnitude, low gyro and stable acceleration are evidence for approximate rest; sustained constant linear acceleration remains unobservable without another sensor",
            },
            "fusion_limitations": [
                "not ST SFLP firmware",
                "no measured temperature bias",
                "no real device timestamp jitter",
            ],
            "truth_boundary": "World kinematics are used only inside sensor synthesis, never exported as task localization",
        }
