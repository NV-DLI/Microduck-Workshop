"""Causal body-IMU prediction from delivered Beta FIFO and encoder history.

Hardware rates stay60Hz sampling/15Hz FIFO delivery. Prediction does not claim
an additional measurement. No renderer/root pose/privileged velocity accepted.
"""

from collections import deque
import math
import numpy as np


def rotation_vector_apply(v, r):
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.asarray(v, float).copy()
    axis = np.asarray(r) / angle
    v = np.asarray(v, float)
    return (
        v * math.cos(angle)
        + np.cross(axis, v) * math.sin(angle)
        + axis * np.dot(axis, v) * (1 - math.cos(angle))
    )


class DeliveredBodyImu:
    def __init__(self, head_adapter, joint_names, *, maximum_prediction_age_s=0.08):
        self.head = head_adapter
        self.names = list(joint_names)
        self.max_age = maximum_prediction_age_s
        self.encoders = deque()
        self.latest = None
        self.latest_head_sample = None
        self.last_seq = -1
        self.last_delivery = -math.inf

    def record_encoders(self, t, q, qd):
        if self.encoders and t <= self.encoders[-1][0]:
            raise ValueError("Encoder chronology must advance")
        q = np.asarray(q, float)
        qd = np.asarray(qd, float)
        if (
            q.shape != (14,)
            or qd.shape != (14,)
            or not np.isfinite(q).all()
            or not np.isfinite(qd).all()
        ):
            raise ValueError("14finite effective encoder values required")
        self.encoders.append((float(t), q.copy(), qd.copy()))
        while self.encoders and self.encoders[0][0] < t - 0.5:
            self.encoders.popleft()

    def ingest(self, batch, *, now_s):
        delivered = []
        for sample in batch:
            acquisition = float(sample["timestamp_s"])
            arrival = float(sample["delivered_timestamp_s"])
            seq = int(sample["seq"])
            if (
                arrival > now_s + 1e-8
                or acquisition > arrival + 1e-8
                or arrival < self.last_delivery - 1e-8
                or seq <= self.last_seq
            ):
                raise ValueError("Only newly delivered causal FIFO samples allowed")
            self.last_seq = seq
            self.last_delivery = arrival
            self.latest_head_sample = dict(sample)
            if (
                sample.get("orientation_valid") is not True
                or sample.get("quat") is None
                or any(sample.get("gyro_clipped", []))
            ):
                self.latest = None
                delivered.append(
                    {
                        "seq": seq,
                        "timestamp_s": acquisition,
                        "delivered_timestamp_s": arrival,
                        "valid": False,
                    }
                )
                continue
            matched = [r for r in self.encoders if abs(r[0] - acquisition) < 1e-8]
            if len(matched) != 1:
                raise ValueError(
                    "Acquisition-time encoder sample missing; do not substitute current or truth FK"
                )
            _, q, qd = matched[0]
            body = self.head.body_imu(sample, self.names, q, qd, capture_s=acquisition)
            self.latest = {
                **body,
                "seq": seq,
                "timestamp_s": acquisition,
                "delivered_timestamp_s": arrival,
            }
            delivered.append({"valid": True, **self.latest})
        return delivered

    def predict(self, now_s):
        if self.latest is None:
            return None
        dt = float(now_s - self.latest["timestamp_s"])
        if dt < -1e-9:
            raise ValueError("Prediction may not travel backward")
        if dt > self.max_age:
            return None
        omega = np.asarray(self.latest["gyro_body_rad_s"])
        gravity = rotation_vector_apply(self.latest["gravity_body"], -omega * dt)
        gravity /= np.linalg.norm(gravity)
        return {
            **self.latest,
            "gyro_body_rad_s": omega.tolist(),
            "gravity_body": gravity.tolist(),
            "control_timestamp_s": float(now_s),
            "measurement_age_s": dt,
            "prediction": "constant acquired body-angular-velocity, gravity Exp(-omega*age); gyro zero-order hold",
            "new_measurement": False,
            "maximum_prediction_age_s": self.max_age,
            "scope": "delivered FIFO and same-time encoder FK only; no simulator root state",
        }
