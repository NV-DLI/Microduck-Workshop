"""Experimental transport optimization; kinematics stay inside sensor producers.

This does not replace the production wrappers automatically. Call only after a
physics step and scene.update, at the same 200 Hz cadence as the legacy calls.
Do not also call the legacy head/body update methods for that tick.
"""

from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch

from beta_head_imu import rotate as body_rotate


def _tensor(value):
    return getattr(value, "torch", value)


def _head_rotate(q, vector):
    # Match the existing rig's NumPy operations, including normalization.
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    return np.asarray(vector) + 2 * np.cross(
        q[:3], np.cross(q[:3], vector) + q[3] * np.asarray(vector)
    )


def _head_qmul(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return np.asarray(
        [
            w * X + x * W + y * Z - z * Y,
            w * Y - x * Z + y * W + z * X,
            w * Z + x * Y - y * X + z * W,
            w * W - x * X - y * Y - z * Z,
        ]
    )


class SharedIsaacImuProducer:
    """Owns a private two-body snapshot and exposes only delivered sensor data.

    body is an existing IsaacBodyImu; rig is an existing IsaacBetaSensors. Their
    measurement/fusion/decoder objects and random states are retained unchanged.
    The actor/planner should never receive this object's private snapshot.
    """

    def __init__(self, body, rig):
        if body.robot is not rig.robot:
            raise ValueError("Head and body must belong to the same articulation")
        if body.env_index != 0:
            raise ValueError("The existing head rig is explicitly env_0")
        names = list(body.robot.body_names)
        head_id = names.index(rig.mount.body_name)
        if rig.head_id is not None and rig.head_id != head_id:
            raise ValueError("Head mount/body selection changed")
        rig.head_id = head_id
        self._body, self._rig = body, rig
        self._body_ids = (body.body_id, head_id)
        self._indices = None
        self._last_timestamp = None
        self.cpu_snapshot_transfers = 0

    def _snapshot(self):
        data = self._body.robot.data
        position = _tensor(data.body_pos_w)
        quaternion = _tensor(data.body_quat_w)
        angular = _tensor(data.body_ang_vel_w)
        if not all(
            isinstance(x, torch.Tensor) for x in (position, quaternion, angular)
        ):
            raise TypeError(
                "This adapter requires the native Torch-backed articulation arrays"
            )
        if self._indices is None:
            self._indices = torch.tensor(
                self._body_ids, dtype=torch.long, device=position.device
            )
        # Select on GPU before copying. Do not copy every body of every world.
        packed = torch.cat(
            (
                position[0].index_select(0, self._indices),
                quaternion[0].index_select(0, self._indices),
                angular[0].index_select(0, self._indices),
            ),
            dim=-1,
        )
        snapshot = packed.detach().cpu().numpy().astype(np.float64, copy=False)
        self.cpu_snapshot_transfers += 1
        if snapshot.shape != (2, 10) or not np.isfinite(snapshot).all():
            raise ValueError("Invalid articulation kinematics at sensor boundary")
        return snapshot

    def update(self, *, timestamp_s):
        if self._last_timestamp is not None and timestamp_s <= self._last_timestamp:
            raise ValueError("Shared sensor update requires advancing simulation time")
        if any(
            previous is not None and timestamp_s <= previous
            for previous in (self._body._time, self._rig._imu_last_time)
        ):
            raise ValueError("A sensor was already updated; do not run both adapters")
        snapshot = self._snapshot()
        body_packets = self._update_body(timestamp_s, snapshot[0])
        head_packets = self._update_head(timestamp_s, snapshot[1])
        self._last_timestamp = timestamp_s
        # Kinematics deliberately do not appear in this return value.
        return {"body": body_packets, "head": head_packets}

    def _update_body(self, timestamp_s, row):
        body = self._body
        quaternion = row[3:7]
        if body.body_quaternion_order == "wxyz":
            quaternion = quaternion[[1, 2, 3, 0]]
        position = row[:3] + body_rotate(
            quaternion, body.profile.sensor_position_trunk_m
        )
        delivered = []
        if body._time is not None:
            dt = timestamp_s - body._time
            if dt <= 0:
                raise ValueError(
                    "Body IMU was already updated; do not run both adapters"
                )
            velocity = (position - body._position) / dt
            if body._velocity is not None:
                acceleration = (velocity - body._velocity) / dt
                delivered = body.sensor.poll(
                    timestamp_s,
                    q_world_from_trunk_xyzw=quaternion,
                    angular_velocity_world_rad_s=row[7:10],
                    linear_acceleration_world_m_s2=acceleration,
                )
            body._velocity = velocity
        body._position, body._time = position, timestamp_s
        return delivered

    def _update_head(self, timestamp_s, row):
        rig = self._rig
        quaternion = row[3:7]
        if rig.body_quaternion_order == "wxyz":
            quaternion = quaternion[[1, 2, 3, 0]]
        quaternion = quaternion / np.linalg.norm(quaternion)
        position = row[:3] + _head_rotate(quaternion, rig.mount.imu_position_body_m)
        rig._imu_batch = []
        if rig._imu_last_time is not None:
            dt = timestamp_s - rig._imu_last_time
            if dt <= 0:
                raise ValueError(
                    "Head IMU was already updated; do not run both adapters"
                )
            velocity = (position - rig._imu_last_position) / dt
            if rig._imu_last_velocity is not None:
                acceleration = (velocity - rig._imu_last_velocity) / dt
                q_flu = _head_qmul(quaternion, rig.mount.site_quaternion_xyzw)
                rig._imu_batch = rig.imu.poll(
                    timestamp_s,
                    q_world_from_head_xyzw=q_flu,
                    angular_velocity_world_rad_s=row[7:10],
                    linear_acceleration_world_m_s2=acceleration,
                )
            rig._imu_last_velocity = velocity
        rig._imu_last_position, rig._imu_last_time = position, timestamp_s
        return list(rig._imu_batch)

    def reset_after_sensor_resets(self):
        """Call after the existing body.reset and rig.reset; do not reset RNG twice."""
        if self._body._time is not None or self._rig._imu_last_time is not None:
            raise ValueError("Reset both original sensor producers first")
        self._last_timestamp = None

    def metadata(self):
        return {
            "schema": "microduck.shared-imu-transport-experiment.v1",
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "world_kinematics_exposed_in_packet": False,
            "cpu_snapshot_transfers": self.cpu_snapshot_transfers,
            "sensors_share_one_physics_tick": True,
            "measurement_and_delivery_rates_changed": False,
            "source_profiles": {
                "body": self._body.metadata(),
                "head": self._rig.imu.metadata(),
            },
            "gpu_performance_verified": False,
            "limitation": "Two sensor bodies in env_0; not a vectorized1024-env training implementation. No rendering, contact, encoder or BAM synchronization is combined here.",
        }
