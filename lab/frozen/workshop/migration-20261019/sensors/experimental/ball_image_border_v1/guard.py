"""Additional original-RGB border veto for the declared plain-white ball.

This checks an observed component only. It does not replace the existing
21-sector geometry gate, measure a sphere, reconstruct missing arcs, or grant
movement. Appearance parameters are copied from the existing contour candidate;
none are fitted against the held historical regression.
"""
import math
import cv2
import numpy as np

PARAMETERS = {
    "roi_pad_fraction": .15, "roi_pad_min_px": 4,
    "white_value_min": 90, "white_chroma_over_value_max": .035,
    "green_G_over_R_min": 1.12, "green_G_over_B_min": 1.07,
    "green_delta_min": 10, "component_min_area_px": 30,
    "secondary_component_area_fraction": .25,
}


def check_observed_ball_border(rgb, box):
    """Return an additional required gate; unknown identity is not acceptance."""
    out = {
        "schema": "microduck.observed-ball-image-border.v1",
        "accepted": False, "reason": None, "component": None,
        "original_sector_gate_replaced": False, "missing_boundary_reconstructed": False,
        "truth_used": False, "kick_authorized": False, "body_authorized": False,
        "hardware_validated": False, "appearance_scope": "declared plain neutral-white simulation ball",
    }

    def reject(reason):
        out["reason"] = reason
        return out

    a = np.asarray(rgb)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 3:
        return reject("invalid_original_RGB")
    try:
        b = np.asarray(box, float)
    except (TypeError, ValueError):
        return reject("invalid_detection_box")
    height, width = a.shape[:2]
    if b.shape != (4,) or not np.isfinite(b).all():
        return reject("invalid_detection_box")
    x0, y0, x1, y1 = b
    if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
        return reject("invalid_detection_extent")
    pad = max(PARAMETERS["roi_pad_min_px"], PARAMETERS["roi_pad_fraction"] * max(x1 - x0, y1 - y0))
    left, top = max(0, math.floor(x0 - pad)), max(0, math.floor(y0 - pad))
    right, bottom = min(width, math.ceil(x1 + pad)), min(height, math.ceil(y1 + pad))
    crop = a[top:bottom, left:right].astype(np.float32)
    red, green, blue = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    value, low = crop.max(axis=2), crop.min(axis=2)
    field = ((green > red * PARAMETERS["green_G_over_R_min"]) &
             (green > blue * PARAMETERS["green_G_over_B_min"]) &
             (green - np.minimum(red, blue) >= PARAMETERS["green_delta_min"]))
    white = ((value >= PARAMETERS["white_value_min"]) &
             (value - low <= PARAMETERS["white_chroma_over_value_max"] * value) & ~field)
    # No morphology, closing, circle/ellipse fit, temporal fill or hull repair.
    n, labels, stats, centers = cv2.connectedComponentsWithStats(white.astype(np.uint8), 8)
    candidates = []
    for i in range(1, n):
        cx, cy = centers[i] + [left, top]
        if stats[i, cv2.CC_STAT_AREA] >= PARAMETERS["component_min_area_px"] and x0 <= cx <= x1 and y0 <= cy <= y1:
            candidates.append(i)
    candidates.sort(key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))
    out["roi_xyxy_px"] = [left, top, right, bottom]
    out["candidate_count"] = len(candidates)
    if not candidates:
        return reject("unknown_ball_component")
    chosen = candidates[0]
    if len(candidates) > 1 and stats[candidates[1], cv2.CC_STAT_AREA] >= stats[chosen, cv2.CC_STAT_AREA] * PARAMETERS["secondary_component_area_fraction"]:
        return reject("ambiguous_ball_component")
    x, y, w, h, area = map(int, stats[chosen])
    observed = labels == chosen
    borders = {
        "left": int(np.count_nonzero(observed[:, 0])) if left == 0 else 0,
        "right": int(np.count_nonzero(observed[:, -1])) if right == width else 0,
        "top": int(np.count_nonzero(observed[0])) if top == 0 else 0,
        "bottom": int(np.count_nonzero(observed[-1])) if bottom == height else 0,
    }
    out["component"] = {
        "bbox_xywh": [x + left, y + top, w, h], "area_px": area,
        "centroid_xy_px": (centers[chosen] + [left, top]).tolist(),
        "actual_image_border_pixels": borders,
    }
    if any(borders.values()):
        return reject("observed_ball_component_touches_image_edge")
    if x == 0 or y == 0 or x + w == right - left or y + h == bottom - top:
        return reject("unknown_component_truncated_by_ROI")
    out.update(accepted=True, reason="associated_observed_component_inside_image")
    return out
