"""Display model of the head sensors, built from frames the sensor rig already delivered.

numpy (cv2 if present), no Kit imports: used by the live Isaac Sim panel (sensor_panel.py), by the
offline media tools and by the CPU tests. Inputs are the dictionary returned by
`IsaacBetaSensors.capture()`:

- `rgb_raw`: uint8 HxWx3, the distortion-remapped head-camera image. The same array is
  passed to the detector (onboard) or to the GR00T client, so the panel shows exactly what
  the controller received, only downscaled.
- `tof`: `ToFFrame.to_dict()`, 8x8 zones in row-major order (row 0 top, column 0 left as
  seen from the robot), each with status valid / no_return / invalid and range_m.
- `timestamp_s`: simulation time of the capture (both images share it).

`tof_source_ray_range` (the 32x32 renderer depth that the ToF model samples) is not a
sensor output and is never displayed. Inputs are never modified.
"""
from dataclasses import dataclass
import math

import numpy as np

# Defaults of beta_tof_geometry.ToFProfile; a frame's own profile takes precedence.
TOF_MIN_M = 0.1
TOF_MAX_M = 4.0
CAPTURE_PERIOD_S = 1 / 15
# A frame is stale once more than this many capture periods passed without a new one.
STALE_AFTER_PERIODS = 2.0

# Perceptual sequential scale (viridis samples): near = yellow, far = dark violet.
_STOPS = np.array(
    [
        (253, 231, 37),
        (148, 216, 64),
        (68, 191, 112),
        (31, 158, 137),
        (33, 145, 140),
        (49, 104, 142),
        (62, 74, 137),
        (72, 40, 120),
        (68, 1, 84),
    ],
    dtype=np.float64,
)
NO_RETURN_FILL = (24, 26, 30)
INVALID_FILL = (128, 128, 128)
CELL_KINDS = ("valid", "no_return", "too_close", "invalid")


def range_color(range_m, min_m=TOF_MIN_M, max_m=TOF_MAX_M):
    """Fill colour of a valid range on the fixed linear scale min_m..max_m (clamped)."""
    if not (math.isfinite(range_m) and max_m > min_m):
        raise ValueError("range and scale must be finite with max > min")
    x = min(max((range_m - min_m) / (max_m - min_m), 0.0), 1.0) * (len(_STOPS) - 1)
    i = min(int(x), len(_STOPS) - 2)
    c = _STOPS[i] + (x - i) * (_STOPS[i + 1] - _STOPS[i])
    return tuple(int(round(v)) for v in c)


def text_color(fill):
    """Black or white text, whichever contrasts with the fill (relative luminance)."""
    r, g, b = (v / 255 for v in fill)
    return (0, 0, 0) if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.45 else (255, 255, 255)


@dataclass(frozen=True)
class TofCellView:
    row: int
    col: int
    kind: str  # one of CELL_KINDS
    text: str
    fill: tuple
    text_fill: tuple
    range_m: float | None
    reason: str


@dataclass(frozen=True)
class TofGridView:
    timestamp_s: float
    cells: tuple  # 8 rows x 8 TofCellView
    min_m: float
    max_m: float
    frame_dropped: bool
    counts: dict


def tof_view(tof):
    """Classify the 64 zones of one delivered ToF frame for display (no inference added)."""
    profile = tof.get("profile") or {}
    min_m = float(profile.get("minimum_reliable_range_m", TOF_MIN_M))
    max_m = float(profile.get("maximum_range_m", TOF_MAX_M))
    rows = tof["cells"]
    if len(rows) != 8 or any(len(r) != 8 for r in rows):
        raise ValueError("Expected the 8x8 ToF mode")
    counts = dict.fromkeys(CELL_KINDS, 0)
    grid = []
    for r, row in enumerate(rows):
        out = []
        for c, cell in enumerate(row):
            if (cell["row"], cell["col"]) != (r, c):
                raise ValueError("ToF zones are not in row-major order")
            status, reason, value = cell["status"], cell.get("reason", ""), cell["range_m"]
            if status == "valid":
                if value is None or not math.isfinite(value):
                    raise ValueError("Valid ToF zone without a finite range")
                fill = range_color(value, min_m, max_m)
                kind, text = "valid", f"{value:.2f}"
            elif status == "no_return":
                kind, text, fill = "no_return", "NR", NO_RETURN_FILL
            elif status == "invalid":
                close = cell.get("contains_too_close") or "too_close" in reason
                kind = "too_close" if close else "invalid"
                text = f"<{min_m:g}" if close else "X"
                fill = INVALID_FILL
            else:
                raise ValueError("Unknown ToF zone status: " + str(status))
            counts[kind] += 1
            out.append(
                TofCellView(r, c, kind, text, fill, text_color(fill),
                            float(value) if kind == "valid" else None, reason)
            )
        grid.append(tuple(out))
    return TofGridView(float(tof["timestamp_s"]), tuple(grid), min_m, max_m,
                       bool(tof.get("frame_dropped", False)), counts)


def rgb_preview(rgb, factor=3):
    """Area-averaged RGBA copy of a delivered uint8 HxWx3 frame, 1/factor per side."""
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected uint8 HxWx3 RGB, got {image.dtype} {image.shape}")
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be >= 1")
    h, w = image.shape[0] // factor, image.shape[1] // factor
    if not (h and w):
        raise ValueError("Image smaller than the downscale factor")
    crop = image[: h * factor, : w * factor]
    try:  # same rounded block mean, about 12x faster than numpy
        import cv2

        small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    except ImportError:
        block = crop.reshape(h, factor, w, factor, 3)
        small = (block.sum(axis=(1, 3), dtype=np.uint32) + factor * factor // 2) // (factor * factor)
    out = np.empty((h, w, 4), np.uint8)
    out[..., :3] = small
    out[..., 3] = 255
    return out


def legend_image(width=256, height=10, min_m=TOF_MIN_M, max_m=TOF_MAX_M):
    """RGBA strip of the range scale, min_m at the left edge and max_m at the right edge."""
    out = np.empty((height, width, 4), np.uint8)
    for x in range(width):
        out[:, x, :3] = range_color(min_m + (max_m - min_m) * x / max(width - 1, 1), min_m, max_m)
    out[..., 3] = 255
    return out


def tof_grid_image(view, cell_px=24, gap_px=1):
    """RGBA image of the grid fills (no text) for offline media."""
    size = 8 * cell_px + 9 * gap_px
    out = np.zeros((size, size, 4), np.uint8)
    out[..., 3] = 255
    for row in view.cells:
        for cell in row:
            y = gap_px + cell.row * (cell_px + gap_px)
            x = gap_px + cell.col * (cell_px + gap_px)
            out[y:y + cell_px, x:x + cell_px, :3] = cell.fill
    return out


@dataclass(frozen=True)
class Freshness:
    state: str  # waiting, live, stale
    age_s: float | None
    text: str


def freshness(sim_now_s, frame_t_s, period_s=CAPTURE_PERIOD_S, stale_after=STALE_AFTER_PERIODS):
    """Age of the shown frame against the simulation clock (never against wall time)."""
    if frame_t_s is None:
        return Freshness("waiting", None, "waiting for first frame")
    if sim_now_s is None:
        sim_now_s = frame_t_s
    age = float(sim_now_s) - float(frame_t_s)
    if age < -1e-9:
        raise ValueError("Simulation clock is behind the frame timestamp")
    age = max(age, 0.0)
    state = "stale" if age > stale_after * period_s + 1e-9 else "live"
    return Freshness(state, age, f"{state.upper()} t={frame_t_s:.3f} s, age {age * 1000:.0f} ms")
