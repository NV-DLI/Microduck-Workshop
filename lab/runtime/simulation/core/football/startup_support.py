"""Explicit, temporary physical support for measured-IMU initialization.

Privileged pose enters this initialization fixture only. It is never exported
to navigation. No root/joint state is written and no sensor threshold changes.
"""

from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass(frozen=True)
class SupportConfig:
    calibration_timeout_s: float = 5.0
    supported_stand_s: float = 0.5
    release_duration_s: float = 1.0
    position_kp_n_m: float = 150.0
    position_kd_ns_m: float = 12.0
    orientation_kp_nm_rad: float = 2.0
    orientation_kd_nms_rad: float = 0.16
    maximum_force_n: float = 12.0
    maximum_torque_nm: float = 0.6


def rotation_error(target, current):
    from beta_head_imu import qmul, inverse

    q = np.asarray(qmul(target, inverse(current)), float)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    sine = np.linalg.norm(q[:3])
    return q[:3] * 2 if sine < 1e-9 else q[:3] / sine * (2 * math.atan2(sine, q[3]))


def clip_norm(v, maximum):
    return v * min(1.0, maximum / max(float(np.linalg.norm(v)), 1e-12))


class StartupSupport:
    def __init__(self, reference_pose_xyzw, total_mass_kg, config=SupportConfig()):
        if (
            len(reference_pose_xyzw) != 7
            or not all(math.isfinite(v) for v in reference_pose_xyzw)
            or total_mass_kg <= 0
        ):
            raise ValueError(
                "Finite initialization fixture pose and positive actual mass required"
            )
        self.reference = np.asarray(reference_pose_xyzw, float)
        self.mass = total_mass_kg
        self.cfg = config
        self.valid_at = None
        self.release_at = None
        self.task_started_at = None
        self.last_time = -1.0
        self.failure = None
        if (
            config.calibration_timeout_s <= 0
            or config.supported_stand_s < 0
            or config.release_duration_s <= 0
        ):
            raise ValueError("Finite positive startup timings required")

    def update(
        self,
        timestamp_s,
        *,
        position_world,
        quaternion_xyzw,
        velocity_world,
        angular_velocity_world,
        imu_orientation_valid
    ):
        if timestamp_s <= self.last_time or not math.isfinite(timestamp_s):
            raise ValueError("Strictly advancing physical support clock required")
        self.last_time = timestamp_s
        if self.task_started_at is not None:
            phase = "TASK"
            gain = 0.0
        elif self.failure is not None:
            phase = "FAILED"
            gain = 0.0
        elif self.valid_at is None:
            if imu_orientation_valid:
                self.valid_at = timestamp_s
                phase = "SUPPORTED_STAND"
                gain = 1.0
            elif timestamp_s >= self.cfg.calibration_timeout_s:
                self.failure = "measured_imu_calibration_timeout"
                phase = "FAILED"
                gain = 0.0
            else:
                phase = "CALIBRATE"
                gain = 1.0
        elif not imu_orientation_valid:
            self.failure = "measured_imu_invalidated_during_startup"
            phase = "FAILED"
            gain = 0.0
        elif timestamp_s < self.valid_at + self.cfg.supported_stand_s:
            phase = "SUPPORTED_STAND"
            gain = 1.0
        else:
            self.release_at = self.valid_at + self.cfg.supported_stand_s
            u = (timestamp_s - self.release_at) / self.cfg.release_duration_s
            if u >= 1 - 1e-10:
                self.task_started_at = timestamp_s
                phase = "TASK"
                gain = 0.0
            else:
                phase = "RELEASE"
                gain = 0.5 * (1 + math.cos(math.pi * max(0.0, u)))
        if gain:
            values = np.asarray(
                [
                    *position_world,
                    *quaternion_xyzw,
                    *velocity_world,
                    *angular_velocity_world,
                ],
                float,
            )
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite physical fixture feedback")
            force = (
                self.cfg.position_kp_n_m
                * (self.reference[:3] - np.asarray(position_world))
                - self.cfg.position_kd_ns_m * np.asarray(velocity_world)
                + [0, 0, self.mass * 9.80665]
            )
            torque = self.cfg.orientation_kp_nm_rad * rotation_error(
                self.reference[3:], quaternion_xyzw
            ) - self.cfg.orientation_kd_nms_rad * np.asarray(angular_velocity_world)
            force = clip_norm(force, self.cfg.maximum_force_n) * gain
            torque = clip_norm(torque, self.cfg.maximum_torque_nm) * gain
        else:
            force = np.zeros(3)
            torque = np.zeros(3)
        if phase == "TASK" and (np.any(force) or np.any(torque)):
            raise RuntimeError("Support must be exactly zero throughout task scoring")
        return {
            "timestamp_s": timestamp_s,
            "phase": phase,
            "support_gain": gain,
            "force_world_n": force.tolist(),
            "torque_world_nm": torque.tolist(),
            "imu_orientation_valid": bool(imu_orientation_valid),
            "policy_enabled": bool(imu_orientation_valid)
            and phase not in ("FAILED", "CALIBRATE"),
            "navigation_enabled": phase == "TASK",
            "task_started_at_s": self.task_started_at,
            "task_time_s": (
                None
                if self.task_started_at is None
                else timestamp_s - self.task_started_at
            ),
            "failure": self.failure,
            "score_enabled": phase == "TASK",
            "physical_state_written": False,
        }

    def metadata(self):
        return {
            "kind": "temporary external-wrench initialization fixture",
            "config": asdict(self.cfg),
            "total_robot_mass_kg": self.mass,
            "reference_pose_xyzw_initialization_only": self.reference.tolist(),
            "measured_imu_valid_at_s": self.valid_at,
            "release_started_at_s": self.release_at,
            "task_started_at_s": self.task_started_at,
            "failure": self.failure,
            "world_feedback_scope": "initialization support only; not navigator input",
            "no_state_projection": True,
            "hardware_equivalent": "robot supported while powering up and calibrating; not demonstrated autonomous get-up",
        }


class NewtonSupportAdapter:
    """Submit only external wrench, then read back the buffer before stepping."""

    def __init__(self, robot):
        self.robot = robot
        ids, names = robot.find_bodies(["trunk_base"], preserve_order=True)
        if len(ids) != 1:
            raise ValueError("Expected one named trunk_base for startup fixture")
        self.ids = ids

    def submit(self, row):
        import torch

        f = torch.tensor(
            [[row["force_world_n"]]], device=self.robot.device, dtype=torch.float32
        )
        t = torch.tensor(
            [[row["torque_world_nm"]]], device=self.robot.device, dtype=torch.float32
        )
        # Set even zero values: merely disabling a composer can leave a previous
        # simulation wrench active. This writes all-env-zero then this one body.
        self.robot.permanent_wrench_composer.set_forces_and_torques_index(
            forces=f, torques=t, body_ids=self.ids, is_global=True
        )

    def check_after_write(self, row):
        buffer = self.robot.data._sim_bind_body_external_wrench.numpy()
        if not np.isfinite(buffer).all():
            raise RuntimeError("Nonfinite actual external wrench buffer")
        actual_peak = float(np.abs(buffer).max())
        if row["score_enabled"] and actual_peak > 1e-12:
            raise RuntimeError("Nonzero actual external wrench during task scoring")
        return actual_peak
