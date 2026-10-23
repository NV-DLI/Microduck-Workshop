"""Independent 8x8 ToF geometric approximation, pinned to Beta SPI L8CX.

Copied from the course's reviewed angular sampler; optics/no-return semantics
remain unchanged. SPI is metadata; no electrical bus or ST photon firmware is
simulated. Source provenance is in sources/course-source-manifest.json.
"""

from __future__ import annotations
from dataclasses import asdict, dataclass
import math, random
from typing import Sequence
from beta_sensor_profile import OFFICIAL_COMMIT, OFFICIAL_BASE


@dataclass(frozen=True)
class ToFProfile:
    rows: int = 8
    cols: int = 8
    fps: float = 15.0
    horizontal_fov_deg: float = 45.0
    vertical_fov_deg: float = 45.0
    minimum_reliable_range_m: float = 0.1
    maximum_range_m: float = 4.0
    sensor_generations: tuple[str, ...] = ("VL53L8CX",)
    source_commit: str = OFFICIAL_COMMIT

    def __post_init__(self):
        if self.rows != 8 or self.cols != 8:
            raise ValueError("This hardware profile is the official 8x8 mode")
        if not 0 < self.horizontal_fov_deg < 90 or not 0 < self.vertical_fov_deg < 90:
            raise ValueError(
                "ToF angular field must be positive and less than 90 degrees"
            )
        if (
            not 0 < self.minimum_reliable_range_m < self.maximum_range_m
            or self.fps <= 0
        ):
            raise ValueError("Invalid ToF range or update rate")

    def to_dict(self):
        return {
            **asdict(self),
            "distance": "slant range along a normalized spherical-angle ray",
            "sensor_axes": "FLU: +x forward, +y left, +z up",
            "zone_order": "row-major; row 0 top, column 0 left",
            "maximum_range_is_guaranteed": False,
            "close_range_semantics": "Below 0.1 m is TooClose/unreliable, never a valid free-space observation",
            "minimum_range_is_chip_physical_limit": False,
            "close_range_model": "Conservative course slant-range filter; official kinematics also filters near points. This does not reproduce the chip's target-status firmware",
            "bus": "SPI3",
            "device": "/dev/spidev3.0",
            "spi_clock_hz": 8000000,
            "spi_bus_is_simulated": False,
            "board": "beta / hw_rev 2",
            "source_url": OFFICIAL_BASE + "/kinematics/src/tof.rs#L108",
        }


TOF_PROFILE = ToFProfile()


def _ray(azimuth, elevation):
    return (
        math.cos(elevation) * math.cos(azimuth),
        math.cos(elevation) * math.sin(azimuth),
        math.sin(elevation),
    )


def tof_zone_rays(profile: ToFProfile = TOF_PROFILE):
    """64 official unit FLU rays, spherical angular centres, row-major."""
    return tuple(
        _ray(
            math.radians(
                profile.horizontal_fov_deg * (0.5 - (col + 0.5) / profile.cols)
            ),
            math.radians(profile.vertical_fov_deg * (0.5 - (row + 0.5) / profile.rows)),
        )
        for row in range(profile.rows)
        for col in range(profile.cols)
    )


@dataclass(frozen=True)
class ToFCell:
    row: int
    col: int
    status: str  # valid, no_return, invalid
    reason: str
    range_m: float | None
    range_mm: int | None
    ray_sensor_flu: tuple[float, float, float]
    mixed_edge: bool
    valid_samples: int
    no_return_samples: int
    invalid_samples: int
    samples_total: int
    sample_range_spread_m: float | None
    contains_too_close: bool


@dataclass(frozen=True)
class ToFFrame:
    timestamp_s: float
    cells: tuple[tuple[ToFCell, ...], ...]
    profile: ToFProfile
    model: dict
    frame_dropped: bool = False

    def to_dict(self):
        return {
            "timestamp_s": self.timestamp_s,
            "cells": [[asdict(cell) for cell in row] for row in self.cells],
            "profile": self.profile.to_dict(),
            "model": self.model,
            "frame_dropped": self.frame_dropped,
        }


def sample_tof_from_depth(
    depth: Sequence[Sequence[float]],
    *,
    profile: ToFProfile = TOF_PROFILE,
    depth_kind: str = "camera_z",
    source_horizontal_fov_deg: float = 50.0,
    source_vertical_fov_deg: float = 50.0,
    samples_per_zone: int = 4,
    quantize_mm: bool = True,
    noise_std_m: float = 0.0,
    dropout_probability: float = 0.0,
    frame_dropout_probability: float = 0.0,
    seed: int = 0,
    timestamp_s: float = 0.0,
    mixed_edge_threshold_m: float = 0.1,
) -> ToFFrame:
    """Project independent square pinhole RTX depth into 8x8 angular zones.

    Depth optical coordinates are +x right, +y down, +z forward and aligned with
    ToF FLU through [-left,-up,forward]. ``camera_z`` is axial depth in metres;
    ``ray_range`` is RTX distance-to-camera in metres. Inputs are not RGB-raw
    rotated images. 50-degree source FoV covers the corners of the 45x45-degree
    spherical-angle sensor region. Unrendered rays remain invalid.

    Select the nearest valid sample in each zone, annotate depth discontinuity
    or mixed valid/unknown samples. A TooClose sample conservatively invalidates
    the zone rather than hiding a nearby object behind a farther return.
    Gaussian range noise and dropout are optional user-chosen approximations,
    not measured IMX219/ST noise models. Quantization is nearest millimetre.
    """
    if depth_kind not in ("camera_z", "ray_range"):
        raise ValueError("depth_kind must explicitly be camera_z or ray_range")
    if samples_per_zone not in (3, 4):
        raise ValueError("Use a 3x3 or 4x4 sample grid in every angular zone")
    values = (
        source_horizontal_fov_deg,
        source_vertical_fov_deg,
        noise_std_m,
        dropout_probability,
        frame_dropout_probability,
        timestamp_s,
        mixed_edge_threshold_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Sampling parameters must be finite")
    if not 0 < source_horizontal_fov_deg < 180 or not 0 < source_vertical_fov_deg < 180:
        raise ValueError("Source pinhole FoV must be between 0 and 180 degrees")
    if (
        noise_std_m < 0
        or mixed_edge_threshold_m < 0
        or not 0 <= dropout_probability <= 1
        or not 0 <= frame_dropout_probability <= 1
    ):
        raise ValueError("Invalid noise/dropout/edge parameters")
    height = len(depth)
    if height < 1 or any(len(row) != height for row in depth):
        raise ValueError(
            "The independent ToF depth render must be a nonempty square 2D array"
        )
    width = height
    fx = width / (2 * math.tan(math.radians(source_horizontal_fov_deg) / 2))
    fy = height / (2 * math.tan(math.radians(source_vertical_fov_deg) / 2))
    cx, cy = (width - 1) / 2, (height - 1) / 2
    rng = random.Random(seed)
    frame_dropped = (
        frame_dropout_probability > 0 and rng.random() < frame_dropout_probability
    )
    rays = tof_zone_rays(profile)
    result = []
    count = samples_per_zone**2
    for row in range(profile.rows):
        result_row = []
        for col in range(profile.cols):
            valid, missing, invalid, too_close = [], 0, 0, False
            reasons = set()
            for subrow in range(samples_per_zone):
                for subcol in range(samples_per_zone):
                    az = math.radians(
                        profile.horizontal_fov_deg
                        * (
                            0.5
                            - (col + (subcol + 0.5) / samples_per_zone) / profile.cols
                        )
                    )
                    el = math.radians(
                        profile.vertical_fov_deg
                        * (
                            0.5
                            - (row + (subrow + 0.5) / samples_per_zone) / profile.rows
                        )
                    )
                    forward, left, up = _ray(az, el)
                    u, v = cx - fx * left / forward, cy - fy * up / forward
                    column_index, row_index = math.floor(u + 0.5), math.floor(v + 0.5)
                    if not 0 <= column_index < width or not 0 <= row_index < height:
                        invalid += 1
                        reasons.add("outside_render")
                        continue
                    raw_value = depth[row_index][column_index]
                    try:
                        distance = float(raw_value)
                    except (TypeError, ValueError, OverflowError):
                        invalid += 1
                        reasons.add("invalid_depth")
                        continue
                    if (
                        isinstance(raw_value, bool)
                        or math.isnan(distance)
                        or distance < 0
                    ):
                        invalid += 1
                        reasons.add("invalid_depth")
                        continue
                    if math.isinf(distance):
                        missing += 1
                        reasons.add("renderer_no_hit")
                        continue
                    slant = distance / forward if depth_kind == "camera_z" else distance
                    if slant < profile.minimum_reliable_range_m:
                        invalid += 1
                        too_close = True
                        reasons.add("too_close")
                    elif slant > profile.maximum_range_m:
                        missing += 1
                        reasons.add("beyond_range")
                    else:
                        valid.append(slant)
            spread = max(valid) - min(valid) if valid else None
            mixed = bool(
                valid and (missing or invalid or spread > mixed_edge_threshold_m)
            ) or bool(missing and invalid)
            range_m, range_mm = None, None
            if frame_dropped:
                status, reason = "invalid", "modeled_frame_dropout"
            elif dropout_probability > 0 and rng.random() < dropout_probability:
                status, reason = "invalid", "modeled_zone_dropout"
            elif too_close:
                status, reason = "invalid", "too_close"
            elif valid:
                selected = min(valid)
                if noise_std_m:
                    selected += rng.gauss(0.0, noise_std_m)
                if selected < profile.minimum_reliable_range_m:
                    status, reason = "invalid", "modeled_noise_below_minimum"
                elif selected > profile.maximum_range_m:
                    status, reason = "no_return", "modeled_noise_beyond_range"
                else:
                    status, reason = "valid", "nearest_valid_geometric_sample"
                    range_mm = (
                        math.floor(selected * 1000 + 0.5) if quantize_mm else None
                    )
                    range_m = range_mm / 1000 if quantize_mm else selected
            elif invalid:
                status, reason = "invalid", "+".join(sorted(reasons))
            else:
                status, reason = "no_return", "+".join(sorted(reasons))
            result_row.append(
                ToFCell(
                    row,
                    col,
                    status,
                    reason,
                    range_m,
                    range_mm,
                    rays[row * profile.cols + col],
                    mixed,
                    len(valid),
                    missing,
                    invalid,
                    count,
                    spread,
                    too_close,
                )
            )
        result.append(tuple(result_row))
    return ToFFrame(
        timestamp_s,
        tuple(result),
        profile,
        {
            "implementation": "angular-zone nearest-valid geometric approximation",
            "photon_model": False,
            "st_firmware_target_selection_model": False,
            "multi_target_output": False,
            "depth_kind": depth_kind,
            "source_width_px": width,
            "source_height_px": height,
            "source_horizontal_fov_deg": source_horizontal_fov_deg,
            "source_vertical_fov_deg": source_vertical_fov_deg,
            "source_must_be_independent_tof_render": True,
            "source_optical_axes": "+x right,+y down,+z forward; aligned to ToF, not raw RGB roll",
            "samples_per_zone_axis": samples_per_zone,
            "quantize_mm": quantize_mm,
            "noise_std_m": noise_std_m,
            "zone_dropout_probability": dropout_probability,
            "frame_dropout_probability": frame_dropout_probability,
            "noise_seed": seed,
            "noise_is_measured_hardware_model": False,
            "mixed_edge_threshold_m": mixed_edge_threshold_m,
            "seed_scope": "One frame; caller varies seed by frame index when modeling temporally changing noise",
            "too_close_policy": "Any sampled TooClose invalidates its zone; do not infer free space",
            "sampling_cadence": "Caller schedules this at profile.fps; this function does not advance a clock",
        },
        frame_dropped,
    )


def tof_to_forward_scan(frame: ToFFrame):
    """Conservative obstacle-marking projection of the eight observed columns.

    Taking the minimum horizontal projection across vertical zones is a sector
    summary, not a coplanar spinning lidar. No-return/invalid become null; they
    never become infinity, maximum-range free space or invented 360-degree rays.
    """
    profile = frame.profile
    angles, ranges, statuses, coverage, mixed = [], [], [], [], []
    for col in reversed(range(profile.cols)):
        azimuth = math.radians(
            profile.horizontal_fov_deg * (0.5 - (col + 0.5) / profile.cols)
        )
        cells = [frame.cells[row][col] for row in range(profile.rows)]
        observed = [cell for cell in cells if cell.status == "valid"]
        too_close = any(cell.contains_too_close for cell in cells)
        if observed and not too_close:
            horizontal_range = min(
                cell.range_m
                * math.hypot(cell.ray_sensor_flu[0], cell.ray_sensor_flu[1])
                for cell in observed
            )
            status = "valid"
        else:
            horizontal_range = None
            status = (
                "invalid"
                if too_close or any(cell.status == "invalid" for cell in cells)
                else "no_return"
            )
        angles.append(azimuth)
        ranges.append(horizontal_range)
        statuses.append(status)
        coverage.append(len(observed))
        mixed.append(
            any(cell.mixed_edge for cell in cells) or len(observed) != profile.rows
        )
    return {
        "timestamp_s": frame.timestamp_s,
        "angles_rad": angles,
        "ranges_m": ranges,
        "status": statuses,
        "valid_vertical_zones_per_bin": coverage,
        "mixed_or_partial_column": mixed,
        "angle_min_rad": angles[0],
        "angle_max_rad": angles[-1],
        "angle_increment_rad": math.radians(profile.horizontal_fov_deg / profile.cols),
        "range_min_m": profile.minimum_reliable_range_m,
        "range_max_m": profile.maximum_range_m,
        "observed_sector_edges_rad": [
            -math.radians(profile.horizontal_fov_deg) / 2,
            math.radians(profile.horizontal_fov_deg) / 2,
        ],
        "outside_sector": "unobserved_unknown",
        "no_return_semantics": "unknown, never free space",
        "obstacle_marking_only": True,
        "free_space_clearing_allowed": False,
        "projection": "nearest horizontal range of vertical-zone returns; sector obstacle approximation, not coplanar lidar",
    }
