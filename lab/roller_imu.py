"""Ideal trunk IMU and a separate inertial velocity estimate for short roller stops.

Simulation synthesizes specific force, gyro and gravity; the controller receives
only those packets. Integration starts from the declared zero-velocity reset.
Bias, vibration and real-hardware drift are not calibrated in this experiment.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class IdealTrunkImu:
    def __init__(self, initial_world_velocity):
        self.previous_velocity = np.asarray(initial_world_velocity, dtype=float)

    def sample(self, velocity_world, quaternion_xyzw, gyro_body, timestamp_s, dt=0.005):
        velocity = np.asarray(velocity_world, dtype=float)
        rotation = Rotation.from_quat(quaternion_xyzw).as_matrix()
        acceleration = (velocity - self.previous_velocity) / dt
        self.previous_velocity = velocity.copy()
        gravity = rotation.T @ np.asarray([0.0, 0.0, -1.0])
        specific = rotation.T @ (acceleration - np.asarray([0.0, 0.0, -9.81]))
        return dict(
            timestamp_s=timestamp_s,
            dt_s=dt,
            specific_force_body_m_s2=specific.tolist(),
            gyro_body_rad_s=list(gyro_body),
            gravity_body=gravity.tolist(),
        )


class InertialVelocity:
    def __init__(self):
        self.velocity = np.zeros(3)

    def update(self, packet):
        acceleration = np.asarray(
            packet["specific_force_body_m_s2"]
        ) + 9.81 * np.asarray(packet["gravity_body"])
        omega = np.asarray(packet["gyro_body_rad_s"])
        self.velocity += packet["dt_s"] * (
            acceleration - np.cross(omega, self.velocity)
        )
        return self.velocity.copy()


def braking_command(velocity_x):
    """Damp estimated travel in either direction; zero command otherwise coasts."""
    return (
        float(np.clip(-3.0 * velocity_x, -0.5, 0.35))
        if abs(velocity_x) > 0.008
        else 0.0
    )


class RollerBrake:
    """Slow velocity regulation with anti-windup; zero-command actor can coast."""

    def __init__(self):
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def update(self, velocity_x, dt=0.02):
        candidate = float(np.clip(self.integral + velocity_x * dt, -0.15, 0.15))
        requested = -3.0 * velocity_x - 4.0 * candidate
        if -0.5 <= requested <= 0.35:
            self.integral = candidate
        return float(np.clip(-3.0 * velocity_x - 4.0 * self.integral, -0.5, 0.35))


def turn_command(error, gyro_z):
    if abs(error) < 0.025:
        return float(np.clip(-0.15 * gyro_z, -0.1, 0.1))
    request = np.sign(error) * max(0.15, min(0.3, 1.8 * abs(error))) - 0.15 * gyro_z
    return float(np.clip(request, -0.3, 0.3))
