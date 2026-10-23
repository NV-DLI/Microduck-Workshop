"""Beta sensor contracts, separating verified hardware from virtual optics.

GC2093 lens calibration is unpublished in the pinned official runtime. All K/D
below are explicit experiment inputs, never the alpha IMX219 calibration.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math

PROFILE_ID = "beta-gc2093-l8cx-v1"

OFFICIAL_COMMIT = "53230463927095ea890f231d29089bb0a538c06f"
OFFICIAL_BASE = f"https://github.com/pollen-robotics/microduck/blob/{OFFICIAL_COMMIT}"


@dataclass(frozen=True)
class BetaCameraProfile:
    width_px: int = 1920
    height_px: int = 1080
    fps: float = 30.0
    assumed_horizontal_fov_deg: float = 70.0
    distortion_k1_k2_p1_p2_k3: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    principal_offset_fraction: tuple[float, float] = (0.0, 0.0)
    mount_rotation_cw_deg: int = 0
    calibration_family: str = "beta-uncalibrated; explicit virtual pinhole experiment"
    source_commit: str = OFFICIAL_COMMIT

    def __post_init__(self):
        if (
            type(self.width_px) is not int
            or type(self.height_px) is not int
            or self.width_px < 2
            or self.height_px < 2
            or self.mount_rotation_cw_deg != 0
        ):
            raise ValueError(
                "Beta requires positive raster dimensions and native 0-degree rotation"
            )
        if self.width_px * 1080 != self.height_px * 1920:
            raise ValueError("Preserve the pinned 1920x1080 raster aspect ratio")
        if not math.isfinite(self.fps) or not 0 < self.fps <= 30:
            raise ValueError(
                "GC2093 pinned driver rate is 30 Hz; use an explicit lower acquisition rate"
            )
        if (
            not math.isfinite(self.assumed_horizontal_fov_deg)
            or not 30 <= self.assumed_horizontal_fov_deg <= 120
        ):
            raise ValueError(
                "Virtual horizontal FoV must be within the tested 30..120 degree envelope"
            )
        if len(self.distortion_k1_k2_p1_p2_k3) != 5 or not all(
            math.isfinite(x) for x in self.distortion_k1_k2_p1_p2_k3
        ):
            raise ValueError("Five finite Brown-Conrady coefficients required")
        if len(self.principal_offset_fraction) != 2 or not all(
            math.isfinite(x) and abs(x) <= 0.1 for x in self.principal_offset_fraction
        ):
            raise ValueError(
                "Virtual principal offset must be finite within 10 percent of raster size"
            )

    @property
    def fx(self):
        return self.width_px / (
            2 * math.tan(math.radians(self.assumed_horizontal_fov_deg) / 2)
        )

    @property
    def fy(self):
        return self.fx

    @property
    def cx(self):
        return (self.width_px - 1) / 2 + self.principal_offset_fraction[
            0
        ] * self.width_px

    @property
    def cy(self):
        return (self.height_px - 1) / 2 + self.principal_offset_fraction[
            1
        ] * self.height_px

    @property
    def K(self):
        return [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]

    @property
    def vertical_fov_deg(self):
        return math.degrees(2 * math.atan(self.height_px / (2 * self.fy)))

    def intrinsics_at(self, width_px, height_px):
        if (
            type(width_px) is not int
            or type(height_px) is not int
            or min(width_px, height_px) < 2
        ):
            raise ValueError("Integer image dimensions >= 2 required")
        sx, sy = width_px / self.width_px, height_px / self.height_px
        # Pixel-centre preserving resize; resize raw pixels the same way.
        return [
            [self.fx * sx, 0.0, (self.cx + 0.5) * sx - 0.5],
            [0.0, self.fy * sy, (self.cy + 0.5) * sy - 0.5],
            [0.0, 0.0, 1.0],
        ]

    def display_intrinsics(self):
        return {
            "width_px": self.width_px,
            "height_px": self.height_px,
            "K": self.K,
            "distortion_k1_k2_p1_p2_k3": list(self.distortion_k1_k2_p1_p2_k3),
            "pixel_operation": "identity",
            "optical_axis_operation": "identity",
            "rotation_clockwise_deg": 0,
        }

    def to_dict(self):
        return {
            **asdict(self),
            "profile_id": PROFILE_ID,
            "sensor": "GalaxyCore GC2093",
            "board": "beta / hw_rev 2",
            "hardware_driver_raster": [1920, 1080],
            "hardware_driver_rate_hz": 30,
            "simulated_output_raster": [self.width_px, self.height_px],
            "raw_format_hardware": "SRGGB10_1X10",
            "simulated_output_format": "RGB8 after virtual ISP",
            "K": self.K,
            "assumed_vertical_fov_deg": self.vertical_fov_deg,
            "physical_lens_model": None,
            "physical_lens_hfov_deg": None,
            "physical_K": None,
            "physical_D": None,
            "unit_specific_calibration": False,
            "calibrated": False,
            "optics_origin": "course sensitivity experiment; not alpha calibration and not measured Beta optics",
            "raw_stream_rotated": False,
            "rotation_applied_by": "none: Beta camera is mounted at 0 degrees",
            "source_urls": [
                OFFICIAL_BASE + "/robotd-params/src/board.rs",
                OFFICIAL_BASE + "/mediad/src/sensor.rs",
            ],
        }


@dataclass(frozen=True)
class BetaHeadImuProfile:
    model: str = "LSM6DSV16X"
    enabled: bool = True
    reader: str = "robotd"
    simulated_sample_hz: float = 60.0
    fifo_read_hz: float = 15.0
    sensor_internal_odr_hz: float = 120.0
    gyro_bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.002)
    gyro_noise_std_rad_s: float = 0.003
    accel_noise_std_m_s2: float = 0.03
    orientation_correction_tau_s: float = 0.5

    def __post_init__(self):
        values = (
            self.simulated_sample_hz,
            self.gyro_noise_std_rad_s,
            self.accel_noise_std_m_s2,
            self.orientation_correction_tau_s,
            *self.gyro_bias_rad_s,
        )
        if (
            not all(math.isfinite(v) for v in values)
            or self.simulated_sample_hz <= 0
            or min(self.gyro_noise_std_rad_s, self.accel_noise_std_m_s2) < 0
            or self.orientation_correction_tau_s <= 0
            or len(self.gyro_bias_rad_s) != 3
        ):
            raise ValueError("Invalid virtual IMU assumptions")

    def to_dict(self):
        return {
            **asdict(self),
            "hardware_features": [
                "SFLP on-chip orientation fusion",
                "FIFO gyro/accel/quaternion batching",
            ],
            "stream_rate_is_official": True,
            "noise_is_measured": False,
            "gyro_range_deg_s": 500.0,
            "accel_range_g": 4.0,
            "sensor_to_head_quaternion_wxyz": [0.5, 0.5, -0.5, -0.5],
            "sensor_to_head_axes": "head = [-sensor.z, -sensor.x, +sensor.y]",
            "published_axes": "head FLU after official decode/to_head",
            "simulation_model": "biased/noisy local gyro and specific force; lightweight complementary tilt fusion",
            "sflp_firmware_reproduced": False,
            "absolute_yaw_available": False,
            "orientation_reference": "gravity-aligned local frame, yaw initialized at zero and integrated from gyro",
            "source_url": OFFICIAL_BASE + "/robotd/src/head_imu.rs",
        }


CAMERA_PROFILE = BetaCameraProfile()
HEAD_IMU_PROFILE = BetaHeadImuProfile()
