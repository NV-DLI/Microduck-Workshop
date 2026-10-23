"""Short stationary-ball memory from delivered encoders/IMU, without motor authority.

No simulator pose/contact, world localization, or future observations are inputs.
The no-slip foot hypothesis and approximately constant unseen ball are explicit
assumptions. Expiry and public diagnostics are guards, not calibrated confidence.
"""

from dataclasses import dataclass
import math
import numpy as np


def finite_vector(value, size):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Finite public vector with declared dimensions required")
    return result.copy()


def level_rotation(gravity):
    up = -finite_vector(gravity, 3)
    length = np.linalg.norm(up)
    if not 0.5 < length < 1.5:
        raise ValueError("Valid public projected gravity required")
    up /= length
    forward = np.array([1.0, 0.0, 0.0])
    forward -= up * np.dot(up, forward)
    if np.linalg.norm(forward) < 0.3:
        raise ValueError("Body pitch is too steep for the heading frame")
    forward /= np.linalg.norm(forward)
    return np.stack([forward, np.cross(up, forward), up])


def rotate_z(angle, vector):
    c, s = math.cos(angle), math.sin(angle)
    x, y, z = vector
    return np.array([c * x - s * y, s * x + c * y, z])


@dataclass(frozen=True)
class PublicRead:
    """All channels must belong to one delivered hardware read."""

    timestamp_s: float
    encoder_timestamp_s: float
    orientation_valid: bool
    q_rad: tuple
    qd_rad_s: tuple
    gyro_body_rad_s: tuple
    gravity_body: tuple
    joint_names: tuple


class BallMemory:
    def __init__(self, leg_fk, joint_names, *, maximum_gap_s=0.085, maximum_age_s=4.5):
        if not 0 < maximum_gap_s <= 0.1 or not 0 < maximum_age_s <= 6:
            raise ValueError("Explicit bounded engineering limits required")
        self.leg = leg_fk
        self.joint_names = tuple(joint_names)
        if len(self.joint_names) != 14 or len(set(self.joint_names)) != 14:
            raise ValueError(
                "Fourteen unique effective output encoder channels required"
            )
        self.maximum_gap_s, self.maximum_age_s = maximum_gap_s, maximum_age_s
        self.reset()

    def reset(self):
        self.previous = None
        self.pose = np.zeros(4)  # xyz and yaw in the first valid read's level frame
        self.last_R = None
        self.anchor = None
        self.capture_time = None
        self.range_uncertainty = None
        self.samples_since_capture = 0
        self.low_confidence_since_capture = 0
        self.capture_read_age = None
        self.reason = "no_complete_ball_observation"

    def invalidate(self, reason):
        self.reset()
        self.reason = reason

    def update(self, read):
        if type(read) is not PublicRead:
            self.invalidate("undeclared_public_read_schema")
            raise ValueError("PublicRead schema required; no hidden truth fields")
        try:
            t = float(read.timestamp_s)
            if (
                not math.isfinite(t)
                or t < 0
                or not read.orientation_valid
                or not math.isfinite(read.encoder_timestamp_s)
                or abs(read.encoder_timestamp_s - t) > 1e-7
                or tuple(read.joint_names) != self.joint_names
            ):
                raise ValueError("Valid same-read public sensor fields required")
            q, qd = finite_vector(read.q_rad, 14), finite_vector(read.qd_rad_s, 14)
            gyro, gravity = finite_vector(read.gyro_body_rad_s, 3), finite_vector(
                read.gravity_body, 3
            )
            R = level_rotation(gravity)
            kin = self.leg.estimate(q, qd, gyro, gravity, self.joint_names)
            velocity = R @ finite_vector(kin["velocity_body_m_s"], 3)
            omega = float((R @ gyro)[2])
        except (ValueError, TypeError, KeyError):
            self.invalidate("invalid_public_sensor_read")
            raise
        if self.previous is not None:
            pt, pv, pw = self.previous
            dt = t - pt
            if dt <= 0:
                self.invalidate("duplicate_or_nonmonotonic_read")
                raise ValueError("Only fresh delivered reads may integrate memory")
            if dt > self.maximum_gap_s + 1e-8:
                self.invalidate("sensor_gap_requires_new_ball_observation")
            else:
                omega_mid = 0.5 * (pw + omega)
                self.pose[:3] += rotate_z(
                    self.pose[3] + 0.5 * omega_mid * dt, 0.5 * (pv + velocity) * dt
                )
                angle = self.pose[3] + omega_mid * dt
                self.pose[3] = math.atan2(math.sin(angle), math.cos(angle))
        self.previous = (t, velocity, omega)
        self.last_R = R
        if self.anchor is not None:
            self.samples_since_capture += 1
            self.low_confidence_since_capture += int(kin["diagnostic_low_confidence"])
        return self.status(t)

    def predicted_pose(self, timestamp_s):
        """Bounded causal hold for a camera exposure between public reads."""
        if self.previous is None:
            raise ValueError("A public read is required")
        age = timestamp_s - self.previous[0]
        if not 0 <= age <= 0.02000001:
            raise ValueError(
                "At most20ms explicitly declared image/read offset supported"
            )
        _, velocity, omega = self.previous
        pose = self.pose.copy()
        pose[:3] += rotate_z(pose[3] + 0.5 * omega * age, velocity * age)
        pose[3] += omega * age
        return pose

    def observe_complete_ball(
        self,
        *,
        timestamp_s,
        point_body_m,
        estimated_range_uncertainty_m,
        complete,
        geometry_accepted,
        capture_read_age_s=0.0
    ):
        """Use body-frame geometry, never a ToF-level or map-frame point.

        Caller must establish the sphere is complete in the captured RGB image.
        This module reports any innovation before refreshing the anchor.
        """
        if (
            self.previous is None
            or not math.isfinite(timestamp_s)
            or not math.isfinite(capture_read_age_s)
            or abs(timestamp_s - self.previous[0] - capture_read_age_s) > 1e-7
        ):
            raise ValueError("Declare the actual image/read time difference")
        capture_pose = self.predicted_pose(timestamp_s)
        point = finite_vector(point_body_m, 3)
        uncertainty = float(estimated_range_uncertainty_m)
        if (
            not complete
            or not geometry_accepted
            or not math.isfinite(uncertainty)
            or not 0 <= uncertainty <= 0.1
            or point[0] <= 0
        ):
            raise ValueError("Accepted complete fresh sphere geometry required")
        before = self.status(timestamp_s)
        level = self.last_R @ point
        innovation = None
        if before["memory_available"]:
            innovation = (level - np.asarray(before["point_level_m"])).tolist()
        self.anchor = capture_pose[:3] + rotate_z(capture_pose[3], level)
        self.capture_time = timestamp_s
        self.capture_read_age = capture_read_age_s
        self.range_uncertainty = uncertainty
        self.samples_since_capture = 0
        self.low_confidence_since_capture = 0
        self.reason = "unseen_ball_assumed_stationary"
        return {
            **self.status(timestamp_s),
            "reobservation_innovation_level_m": innovation,
        }

    def status(self, now_s):
        t = float(now_s)
        if (
            not math.isfinite(t)
            or self.previous is not None
            and t < self.previous[0] - 1e-8
        ):
            raise ValueError("Evaluation time cannot precede the latest public read")
        age = None if self.capture_time is None else t - self.capture_time
        gap = None if self.previous is None else t - self.previous[0]
        available = bool(
            self.anchor is not None
            and 0 <= age <= self.maximum_age_s
            and 0 <= gap <= 0.02000001
        )
        reason = self.reason
        if self.anchor is not None and not available:
            reason = (
                "ball_memory_expired"
                if age > self.maximum_age_s
                else "sensor_gap_requires_new_ball_observation"
            )
        pose = self.predicted_pose(t) if available else self.pose
        level = rotate_z(-pose[3], self.anchor - pose[:3]) if available else None
        return dict(
            memory_available=available,
            reason=reason,
            point_level_m=None if level is None else level.tolist(),
            point_body_m=None if level is None else (self.last_R.T @ level).tolist(),
            frame="BODY50_gravity_level_with_projected_body_forward",
            ball_capture_timestamp_s=self.capture_time,
            ball_age_s=age,
            capture_sensor_read_age_s=self.capture_read_age,
            sensor_read_timestamp_s=None if self.previous is None else self.previous[0],
            sensor_read_age_s=gap,
            samples_since_capture=self.samples_since_capture,
            low_confidence_samples_since_capture=self.low_confidence_since_capture,
            original_range_uncertainty_m=self.range_uncertainty,
            prediction_horizon_s=gap if available else None,
            public_motion_hold_is_measurement=False,
            propagated_uncertainty_calibrated=False,
            stationary_ball_verified=False,
            no_slip_verified=False,
            kick_authorized=False,
        )
