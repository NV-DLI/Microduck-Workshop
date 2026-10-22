"""Consume the official body control-IMU stream without accessing simulator state.

The producer already models the 50 Hz bus pipeline and median-of-three decoder.
Do not add another delay, normalize the median gravity, or use head telemetry as
a silent substitute. Missing or invalid delivered data means no policy input.
"""
import math
from collections import deque


class BodyImuInput:
    def __init__(self, maximum_age_s=.08):
        self.maximum_age_s = float(maximum_age_s)
        self.latest = None
        self.last_sequence = -1
        self.last_delivery = -math.inf
        self.encoders = deque()

    def record_encoders(self, timestamp_s, q_rad, qd_rad_s):
        if self.encoders and timestamp_s <= self.encoders[-1][0]:
            raise ValueError('Encoder measurements must advance')
        if len(q_rad) != 14 or len(qd_rad_s) != 14 or not all(math.isfinite(x) for x in (*q_rad, *qd_rad_s)):
            raise ValueError('Exactly14 finite encoder values required')
        # Actual XL330 register resolutions used in the official combined read.
        position_lsb = 2 * math.pi / 4096
        velocity_lsb = .229 * 2 * math.pi / 60
        q = [round(x / position_lsb) * position_lsb for x in q_rad]
        qd = [round(x / velocity_lsb) * velocity_lsb for x in qd_rad_s]
        self.encoders.append((float(timestamp_s), q, qd))
        while self.encoders and self.encoders[0][0] < timestamp_s - .5:
            self.encoders.popleft()

    def ingest(self, packets, *, now_s):
        for packet in packets:
            acquisition = float(packet['timestamp_s'])
            delivery = float(packet['delivered_timestamp_s'])
            sequence = int(packet['seq'])
            if packet.get('axes') != 'trunk FLU' or packet.get('source') != 'body_powerboard_control_imu':
                raise ValueError('A delivered body power-board packet is required')
            if not all(math.isfinite(v) for v in (acquisition, delivery, now_s)):
                raise ValueError('Finite sensor timestamps required')
            if acquisition > delivery + 1e-9 or delivery > now_s + 1e-9 or delivery < self.last_delivery - 1e-9 or sequence <= self.last_sequence:
                raise ValueError('Body packet must be newly delivered and causal')
            self.last_sequence, self.last_delivery = sequence, delivery
            self.latest = dict(packet)
            read_time = float(packet['read_timestamp_s'])
            rows = [r for r in self.encoders if abs(r[0] - read_time) < 1e-8]
            if len(rows) != 1:
                raise ValueError('Same-read encoder sample required; no current-state substitution')
            self.latest.update(encoder_read_timestamp_s=read_time,
                               encoder_q_rad=rows[0][1], encoder_qd_rad_s=rows[0][2])

    def observe(self, now_s):
        packet = self.latest
        if packet is None or packet.get('orientation_valid') is not True:
            return None
        age = float(now_s - packet['timestamp_s'])
        if age < -1e-9:
            raise ValueError('Body input cannot read a future sample')
        if age > self.maximum_age_s:
            return None
        gyro, gravity = packet['gyro'], packet['gravity']
        if len(gyro) != 3 or len(gravity) != 3 or not all(math.isfinite(x) for x in (*gyro, *gravity)):
            raise ValueError('Finite body gyro and gravity required')
        return {**packet, 'gyro_body_rad_s': list(gyro), 'gravity_body': list(gravity),
                'control_timestamp_s': float(now_s), 'measurement_age_s': age,
                'prediction': 'none; official decoded median3 values held between delivered packets',
                'new_measurement': False, 'maximum_prediction_age_s': self.maximum_age_s,
                'scope': 'body power-board measurements only; no head FIFO substitution or root state'}
