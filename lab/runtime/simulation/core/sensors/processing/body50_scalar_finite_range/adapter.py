"""Explicit parameters for a fresh scalar BetaBodyImu / IsaacBodyImu only."""

from beta_body_imu import BetaBodyImu
from finite_range_chip import BetaHeadImu as FiniteChip


def install_body_before_first_tick(body):
    if body.read_count or body.chip.seq or body.chip.last_time is not None:
        raise ValueError(
            "Select parameters before any sensor acquisition or register read"
        )
    old = body.chip
    parameters = FiniteChip(
        profile=old.profile, seed=0, initialization=old.initialization
    )
    parameters.rng.setstate(old.rng.getstate())
    body.chip = parameters
    return body


def install_before_first_update(isaac_imu):
    if isaac_imu._time is not None:
        raise ValueError("Select parameters before any IsaacBodyImu derivative tick")
    return install_body_before_first_tick(isaac_imu.sensor)


class FiniteRangeBodyImu(BetaBodyImu):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        install_body_before_first_tick(self)

    def metadata(self):
        return {**super().metadata(), **parameters_metadata()}


def parameters_metadata():
    return {
        "fusion_hypothesis": "body50-scalar-finite-range-v1",
        "gyro_clipping_resets_initialized_fusion": False,
        "initial_saturation_blocks_initialization": True,
        "sample_gap_invalidates_orientation": True,
        "measured_gyro_range_deg_s": 500.0,
        "clipped_measurement_integrated": True,
        "sflp_firmware_reproduced": False,
        "hardware_fit_verified": False,
        "interpretation": "Clipping flags and finite-range integration error are preserved; no truth orientation correction.",
    }
