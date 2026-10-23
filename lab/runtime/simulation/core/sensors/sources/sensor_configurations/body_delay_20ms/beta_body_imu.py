"""Beta power-board control IMU; head FIFO remains a separate visual sensor.

The virtual chip integrates noisy local gyro and acceleration. Only that fused
estimate is encoded/decoded: no simulator orientation is copied into a policy
packet. Register transport, Beta axes and decoder match the pinned robotd code;
the proprietary SFLP algorithm, internal clock and noise remain assumptions.
"""

from collections import deque
from dataclasses import asdict, dataclass
import math
import struct

from beta_head_imu import BetaHeadImu, ImuInitializationCfg, normalize, rotate, inverse
from beta_sensor_profile import OFFICIAL_BASE, PROFILE_ID


@dataclass(frozen=True)
class BetaBodyImuProfile:
    model: str = "LSM6DSV16X"
    simulated_sample_hz: float = 120.0
    fifo_read_hz: float = 50.0  # Compatibility name used only by virtual chip sampler.
    pipeline_delay_s: float = 0.020
    additional_transport_delay_s: float = 0.0
    gyro_bias_rad_s: tuple = (0.0, 0.0, 0.002)
    gyro_noise_std_rad_s: float = 0.003
    accel_noise_std_m_s2: float = 0.03
    orientation_correction_tau_s: float = 0.5
    sensor_position_trunk_m: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.fifo_read_hz != 50.0:
            raise ValueError(
                "Pinned body control readout is 50 Hz; head FIFO is separate"
            )
        numeric = (
            self.simulated_sample_hz,
            self.pipeline_delay_s,
            self.additional_transport_delay_s,
            self.gyro_noise_std_rad_s,
            self.accel_noise_std_m_s2,
            self.orientation_correction_tau_s,
            *self.gyro_bias_rad_s,
            *self.sensor_position_trunk_m,
        )
        if not all(math.isfinite(x) for x in numeric) or self.simulated_sample_hz < 50:
            raise ValueError("Finite body IMU assumptions required")
        if (
            min(
                self.pipeline_delay_s,
                self.additional_transport_delay_s,
                self.gyro_noise_std_rad_s,
                self.accel_noise_std_m_s2,
            )
            < 0
            or self.orientation_correction_tau_s <= 0
        ):
            raise ValueError("Negative delay/noise or invalid fusion time constant")
        if len(self.gyro_bias_rad_s) != 3 or len(self.sensor_position_trunk_m) != 3:
            raise ValueError("Three-axis bias and mounting position required")

    def to_dict(self):
        return {
            **asdict(self),
            "profile_id": PROFILE_ID,
            "placement": "body power board",
            "role": "50 Hz locomotion/control IMU; separate from head_imu.stream",
            "transport": "Dynamixel combined sync_read with servo feedback",
            "control_read_hz": 50.0,
            "bus_register_address": 124,
            "bus_block_bytes": 12,
            "sensor_to_trunk_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "published_axes": "trunk FLU",
            "gyro_range_deg_s": 500.0,
            "gyro_lsb_deg_s": 0.0175,
            "quaternion_encoding": "xyz binary16; positive w reconstructed",
            "decoder": "component-wise median3 gyro/gravity, unfiltered quaternion; ready after25 live quaternion reads",
            "minimum_ready_reads": 25,
            "minimum_ready_read_time_s": 0.5,
            "pipeline_delay_origin": "Experiment assumption only: robotd reads, infers and applies in the same50Hz loop; no fixed post-read20ms queue is established by upstream source",
            "pipeline_delay_is_measured": False,
            "pipeline_delay_default_scope": "Historical conservative20ms stress case; use explicit0/5/10/20ms sensitivity cases, not an official measured latency",
            "median_phase_note": "Component-wise median3 already contributes approximately one50Hz tick of phase lag on monotonic signals; its delay must not be counted again as transport",
            "internal_sample_rate_is_measured": False,
            "noise_is_measured": False,
            "sensor_position_is_measured": False,
            "sflp_firmware_reproduced": False,
            "additional_transport_delay_is_measured": False,
            "assumptions": [
                "120 Hz virtual internal fusion rate is not confirmed body firmware configuration",
                "gyro bias/noise and acceleration noise are experiment inputs",
                "power-board translation is unknown; trunk origin is a parameterized approximation",
                "Additional post-read pipeline delay is unknown; the historical20ms default is an processing stress setting, not a measured bus/inference duration",
            ],
            "source_urls": [
                OFFICIAL_BASE + "/duck-control/src/imu.rs",
                OFFICIAL_BASE + "/duck-control/src/bus.rs",
                OFFICIAL_BASE + "/duck-control/src/obs.rs",
                OFFICIAL_BASE + "/robotd/src/main.rs",
            ],
        }


class BodySflpDecoder:
    """Literal Beta subset of duck-control/src/imu.rs, including its ready gate."""

    GYRO_LSB = math.radians(0.0175)

    def __init__(self):
        self.last_quat = (1.0, 0.0, 0.0, 0.0)
        self.quat_samples = 0
        self.gyro_history = deque([(0.0, 0.0, 0.0)] * 2, maxlen=2)
        self.gravity_history = deque([(0.0, 0.0, -1.0)] * 2, maxlen=2)

    @property
    def ready(self):
        return self.quat_samples >= 25

    def decode(self, raw):
        if len(raw) != 12:
            raise ValueError("Body IMU block must be12 bytes")
        gyro = tuple(x * self.GYRO_LSB for x in struct.unpack("<3h", raw[:6]))
        packed = struct.unpack("<3H", raw[6:])
        updated = False
        if packed != (0, 0, 0):
            xyz = struct.unpack("<3e", raw[6:])
            norm_sq = sum(x * x for x in xyz)
            if all(math.isfinite(x) for x in xyz) and norm_sq <= 1.02:
                q = (math.sqrt(max(0.0, 1.0 - norm_sq)), *xyz)
                if sum(x * x for x in q) > 0.25:
                    self.last_quat = normalize(q)
                    self.quat_samples += 1
                    updated = True
        w, x, y, z = self.last_quat
        gravity = normalize(rotate(inverse((x, y, z, w)), (0.0, 0.0, -1.0)))

        def median(history, now):
            return [sorted((history[0][i], history[1][i], now[i]))[1] for i in range(3)]

        result = {
            "gyro": median(self.gyro_history, gyro),
            "gravity": median(self.gravity_history, gravity),
            "quat": list(self.last_quat),
            "decoder_ready": self.ready,
            "live_quaternion_reads": self.quat_samples,
            "quaternion_updated": updated,
        }
        self.gyro_history.append(gyro)
        self.gravity_history.append(gravity)
        return result


class BetaBodyImu:
    def __init__(self, *, profile=None, seed=38112, initialization=None):
        self.profile = profile or BetaBodyImuProfile()
        if initialization is None:
            # Keep a 0.2s observation window when the assumed internal rate
            # changes; a fixed12-entry60Hz window would never fill0.18s at120Hz.
            initialization = ImuInitializationCfg(
                minimum_samples=math.ceil(0.20 * self.profile.simulated_sample_hz) + 1,
                minimum_duration_s=0.18,
                maximum_sample_gap_s=0.04,
            )
        self.chip = BetaHeadImu(
            profile=self.profile, seed=seed, initialization=initialization
        )
        self.reset()

    def reset(self):
        self.chip.reset()
        self.decoder = BodySflpDecoder()
        self.pending = deque()
        self.read_timestamps = deque([None, None], maxlen=3)
        self.latest = None  # Only delivered packets; never the unreceived chip sample.
        self.read_count = 0
        self.delivered_count = 0

    def poll(
        self,
        timestamp_s,
        *,
        q_world_from_trunk_xyzw,
        angular_velocity_world_rad_s,
        linear_acceleration_world_m_s2
    ):
        # The inherited acquisition/fusion math is frame-generic. Its raw local
        # measurements now belong to the body, with the official identity mount.
        acquired = self.chip.poll(
            timestamp_s,
            q_world_from_head_xyzw=q_world_from_trunk_xyzw,
            angular_velocity_world_rad_s=angular_velocity_world_rad_s,
            linear_acceleration_world_m_s2=linear_acceleration_world_m_s2,
        )
        if acquired:
            sample = acquired[
                -1
            ]  # Body register exposes latest, not a FIFO to the host.
            gyro_counts = [
                max(-32768, min(32767, round(x / BodySflpDecoder.GYRO_LSB)))
                for x in sample["gyro"]
            ]
            quaternion = sample["quat"]
            if quaternion is None:
                xyz = (0.0, 0.0, 0.0)
            else:
                sign = -1.0 if quaternion[0] < 0 else 1.0
                xyz = tuple(sign * x for x in quaternion[1:])
            raw = struct.pack("<3h3e", *gyro_counts, *xyz)
            decoded = self.decoder.decode(raw)
            self.read_timestamps.append(sample["timestamp_s"])
            valid = bool(sample["orientation_valid"] and decoded["decoder_ready"])
            due = (
                timestamp_s
                + self.profile.pipeline_delay_s
                + self.profile.additional_transport_delay_s
            )
            packet = {
                **decoded,
                "seq": self.read_count,
                "timestamp_s": sample["timestamp_s"],
                "read_timestamp_s": timestamp_s,
                "scheduled_delivery_timestamp_s": due,
                "orientation_valid": valid,
                "quat": decoded["quat"] if valid else None,
                "gravity": decoded["gravity"] if valid else None,
                "axes": "trunk FLU",
                "quaternion_order": "wxyz",
                "gyro_unit": "rad/s",
                "raw_block_hex": raw.hex(),
                "body_sensor_sample_seq": sample["seq"],
                "median_window_acquisition_timestamps_s": list(self.read_timestamps),
                "gyro_clipped": sample["gyro_clipped"],
                "accel_clipped": sample["accel_clipped"],
                "initialization_status": sample["initialization_status"],
                "absolute_world_yaw": False,
                "sflp_firmware_reproduced": False,
                "noise_is_measured": False,
                "source": "body_powerboard_control_imu",
                "truth_boundary": "decoded noisy gyro and sensor-fused orientation; no root pose copy",
            }
            self.read_count += 1
            self.pending.append((due, packet))
        delivered = []
        while self.pending and self.pending[0][0] <= timestamp_s + 1e-9:
            due, packet = self.pending.popleft()
            packet = {
                **packet,
                "delivered_timestamp_s": timestamp_s,
                "sample_age_at_delivery_s": timestamp_s - packet["timestamp_s"],
                "delivery_lateness_s": max(0.0, timestamp_s - due),
            }
            self.latest = packet
            self.delivered_count += 1
            delivered.append(packet)
        return delivered

    def metadata(self):
        return {
            **self.profile.to_dict(),
            "acquired_samples": self.chip.seq,
            "missed_internal_samples": self.chip.missed,
            "body_register_reads": self.read_count,
            "delivered_packets": self.delivered_count,
            "fusion_initialization": self.chip.metadata()["initialization"],
            "head_stream_is_used_for_locomotion": False,
        }
