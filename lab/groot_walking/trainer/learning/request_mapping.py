"""Explicit task-specific conversion from learned requests to the roller bridge."""

import math

DEMONSTRATED_FORWARD = (
    "forward_gt_0.05_to_demonstrated_0.6_v1; all other components unchanged"
)


def map_request(action, mode="continuous"):
    if len(action) != 4 or not all(math.isfinite(float(v)) for v in action):
        raise ValueError("Four finite action values required")
    if mode not in ("continuous", "demonstrated-forward"):
        raise ValueError("Unknown request mapping")
    bounded = [
        max(lo, min(hi, float(v)))
        for v, lo, hi in zip(
            action, [-0.5, -0.3, -0.2, 0.0], [0.6, 0.3, 1.0, 1.0], strict=True
        )
    ]
    mapped = bounded.copy()
    if mode == "demonstrated-forward" and mapped[0] > 0.05:
        mapped[0] = 0.6
    return mapped, bounded


def mapping_identity(mode):
    if mode == "continuous":
        return {}
    if mode == "demonstrated-forward":
        return {"inference_output_mapping": DEMONSTRATED_FORWARD}
    raise ValueError("Unknown request mapping")
