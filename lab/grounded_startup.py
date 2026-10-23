"""Ground-supported initialization with a temporary lateral/orientation fixture.

Start from the official standing pose. The existing executor holds home until
the measured body IMU is ready, then runs its walking actor at zero command.
The fixture never cancels weight. It releases before task scoring.
Only initialization inspects world motion, never navigation or kick authority.
"""

import math
import numpy as np
from startup_support import rotation_error, clip_norm
from types import SimpleNamespace


class GroundedStartup:
    def __init__(self, reference_pose_xyzw, total_mass_kg):
        self.cfg = SimpleNamespace(
            calibration_timeout_s=8.0, supported_stand_s=0.0, release_duration_s=0.0
        )
        self.reference = list(reference_pose_xyzw)
        self.mass = float(total_mass_kg)
        self.valid_at = None
        self.task_started_at = None
        self.release_at = None
        self.failure = None
        self.stable_since = None
        self.last_time = -1.0

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
        t = float(timestamp_s)
        if not math.isfinite(t) or t <= self.last_time:
            raise ValueError("Startup clock must advance")
        self.last_time = t
        if imu_orientation_valid and self.valid_at is None:
            self.valid_at = t
        if self.task_started_at is None:
            qx, qy, _, _ = quaternion_xyzw
            upright = 1 - 2 * (qx * qx + qy * qy) > math.cos(math.radians(12))
            quiet = (
                sum(v * v for v in velocity_world) < 0.025**2
                and sum(w * w for w in angular_velocity_world) < 0.3**2
            )
            stable = (
                imu_orientation_valid
                and upright
                and quiet
                and position_world[2] > 0.075
            )
            self.stable_since = (
                (self.stable_since if self.stable_since is not None else t)
                if stable
                else None
            )
            if (
                self.valid_at is not None
                and t >= self.valid_at + 1.0
                and self.stable_since is not None
                and t - self.stable_since >= 0.5
            ):
                self.task_started_at = t
            elif t >= 8.0:
                self.failure = "grounded_initialization_not_stable"
        phase = (
            "FAILED"
            if self.failure
            else ("TASK" if self.task_started_at is not None else "GROUNDED_STAND")
        )
        gain = (
            0.0
            if phase in ("TASK", "FAILED")
            else (1.0 if self.valid_at is None else max(0.0, 1.0 - (t - self.valid_at)))
        )
        force = [
            gain
            * (150 * (self.reference[i] - position_world[i]) - 12 * velocity_world[i])
            for i in range(2)
        ] + [0.0]
        torque = gain * clip_norm(
            2 * rotation_error(self.reference[3:], quaternion_xyzw)
            - 0.16 * np.asarray(angular_velocity_world),
            0.6,
        )
        return dict(
            timestamp_s=t,
            phase=phase,
            support_gain=gain,
            force_world_n=clip_norm(np.asarray(force), 12).tolist(),
            torque_world_nm=torque.tolist(),
            imu_orientation_valid=bool(imu_orientation_valid),
            policy_enabled=bool(imu_orientation_valid),
            navigation_enabled=phase == "TASK",
            score_enabled=phase == "TASK",
            task_started_at_s=self.task_started_at,
            task_time_s=(
                None if self.task_started_at is None else t - self.task_started_at
            ),
            failure=self.failure,
            physical_state_written=False,
        )

    def metadata(self):
        return dict(
            kind="ground-supported measured-IMU initialization",
            total_robot_mass_kg=self.mass,
            reference_pose_xyzw_initialization_only=self.reference,
            measured_imu_valid_at_s=self.valid_at,
            task_started_at_s=self.task_started_at,
            failure=self.failure,
            external_force_and_torque="lateral/orientation stabilization during calibration only; zero vertical force; zero all axes before task",
            stable_window_s=0.5,
            max_speed_m_s=0.025,
            max_angular_speed_rad_s=0.3,
            max_tilt_deg=12,
            no_state_projection=True,
            world_feedback_scope="startup gate only; never navigator input",
        )
