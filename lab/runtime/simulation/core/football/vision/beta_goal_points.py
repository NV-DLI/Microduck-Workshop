"""Observed post bases for a YOLO goal box over the declared green course floor.

Uses lower post/grass contrast rather than a connected white crossbar: the
opening can legitimately reveal a neutral distant background. Never reads map
coordinates, goal poses, masks, depth, or labels.
"""

import cv2, numpy as np, math


def extract_beta_goal_bases(rgb, box, gravity_raw, camera):
    r = {
        "kind": "RGB_beta_goal_post_base_parameters_v1",
        "accepted": False,
        "reason": None,
        "footpoints_upright_px": None,
        "truth_used": False,
        "task_qualified": False,
        "parameters_only": True,
        "missing_boundary_reconstructed": False,
        "assumption": "YOLO goal identity; complete white lower posts against locally green course floor",
    }

    def reject(reason):
        r["reason"] = reason
        return r

    g = np.asarray(gravity_raw, float)
    b = np.asarray(box, float)
    x0, y0, x1, y1 = b
    if np.linalg.norm(g[:2]) < 0.1:
        return reject("gravity_projection_undefined")
    if (
        min(x1 - x0, y1 - y0) < 18
        or x0 < 3
        or y0 < 3
        or x1 > camera.width_px - 4
        or y1 > camera.height_px - 4
    ):
        return reject("small_or_cropped_goal")
    pad = 8
    left = max(0, int(x0) - pad)
    right = min(camera.width_px, int(math.ceil(x1)) + pad + 1)
    top = max(0, int(y0) - pad)
    bottom = min(camera.height_px, int(math.ceil(y1)) + pad + 1)
    crop = rgb[top:bottom, left:right].astype(float)
    R, G, B = crop[:, :, 0], crop[:, :, 1], crop[:, :, 2]
    green = (G > 1.12 * R) & (G > 1.07 * B) & (G - np.minimum(R, B) > 10)
    white = (
        (crop.min(axis=2) >= 160) & (crop.max(axis=2) - crop.min(axis=2) <= 35) & ~green
    )
    yy, xx = np.nonzero(white)
    pixels = np.column_stack((xx + left, yy + top)).astype(float)
    if len(pixels) < 30:
        return reject("no_observed_posts")
    down = g[:2] / np.linalg.norm(g[:2])
    across = np.array([down[1], -down[0]])
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    rect = np.column_stack(((pixels - centre) @ across, (pixels - centre) @ down))
    width = x1 - x0
    feet = []
    details = []
    # Left/right strips are evidence selection only. Base y is always taken from
    # observed pixels; the lower detector-box edge is never emitted as a keypoint.
    for sign in (-1, 1):
        parameterss = np.flatnonzero(
            (sign * rect[:, 0] > 0.27 * width)
            & (sign * rect[:, 0] < 0.57 * width)
            & (rect[:, 1] > -0.15 * (y1 - y0))
        )
        if len(parameterss) < 12:
            return reject("one_post_missing")
        low = rect[parameterss, 1].max()
        base = parameterss[rect[parameterss, 1] >= low - 1.5]
        if len(base) < 3:
            return reject("insufficient_base_pixels")
        observed = pixels[base]
        foot = np.mean(observed, axis=0)
        if (
            min(
                foot[0] - left, right - 1 - foot[0], foot[1] - top, bottom - 1 - foot[1]
            )
            <= 3
        ):
            return reject("base_at_crop_boundary")
        probe = np.round(observed + down * 4).astype(int)
        valid = (
            (probe[:, 0] >= 0)
            & (probe[:, 0] < camera.width_px)
            & (probe[:, 1] >= 0)
            & (probe[:, 1] < camera.height_px)
        )
        if not valid.all():
            return reject("base_below_image")
        values = rgb[probe[:, 1], probe[:, 0]].astype(float)
        floor = (values[:, 1] > 1.12 * values[:, 0]) & (
            values[:, 1] > 1.07 * values[:, 2]
        )
        fraction = float(floor.mean())
        if fraction < 0.7:
            return reject("post_base_not_supported_by_observed_floor")
        feet.append(foot.tolist())
        details.append(
            {"observed_base_pixels": len(base), "floor_fraction_below_base": fraction}
        )
    # The segment between lower posts must reveal actual green floor. This rejects
    # a solid white rectangle without demanding a green background behind the bar.
    a, b = np.asarray(feet)
    line = np.array([a * (1 - u) + b * u for u in np.linspace(0.30, 0.70, 12)])
    line = np.round(line + down * 1).astype(int)
    v = rgb[line[:, 1], line[:, 0]].astype(float)
    floor = (v[:, 1] > 1.12 * v[:, 0]) & (v[:, 1] > 1.07 * v[:, 2])
    r["green_fraction_between_bases"] = float(floor.mean())
    if floor.mean() < 0.7:
        return reject("lower_goal_opening_not_visible")
    return {
        **r,
        "accepted": True,
        "reason": "two_observed_floor_supported_bases",
        "footpoints_upright_px": feet,
        "posts": details,
        "full_visibility_verified": False,
        "semantic_correctness_verified": False,
        "subpixel_accuracy_calibrated": False,
    }
