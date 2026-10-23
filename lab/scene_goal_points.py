"""Observed front-post bases: neutral vertical support above green-floor pixels.
Uses only RGB, detector boxes and measured gravity. No pose, depth or labels.
"""

import numpy as np


def extract_scene_goal_bases(rgb, box, gravity_raw, camera):
    result = dict(
        kind="RGB_beta_goal_post_base_parameters_v1",
        extractor="vertical_support_v5",
        accepted=False,
        reason=None,
        footpoints_upright_px=None,
        truth_used=False,
        task_qualified=False,
        parameters_only=True,
        missing_boundary_reconstructed=False,
    )

    def reject(reason):
        return dict(result, reason=reason)

    b = np.asarray(box, float)
    g = np.asarray(gravity_raw, float)
    if (
        b.shape != (4,)
        or g.shape != (3,)
        or not np.isfinite(b).all()
        or not np.isfinite(g).all()
    ):
        return reject("invalid_input")
    if np.asarray(rgb).shape != (camera.height_px, camera.width_px, 3):
        return reject("invalid_image")
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    if (
        min(w, h) < 24
        or min(x0, y0) < 3
        or x1 > camera.width_px - 4
        or y1 > camera.height_px - 4
    ):
        return reject("small_or_cropped_goal")
    if np.linalg.norm(g[:2]) < 0.1:
        return reject("gravity_projection_undefined")
    down = g[:2] / np.linalg.norm(g[:2])
    across = np.array([down[1], -down[0]])
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    a = np.asarray(rgb, float)
    green = (a[:, :, 1] > 1.12 * a[:, :, 0]) & (a[:, :, 1] > 1.07 * a[:, :, 2])
    neutral = (a.min(2) >= 100) & (a.max(2) - a.min(2) <= 35) & ~green
    left = max(1, int(x0) - 6)
    right = min(camera.width_px - 1, int(np.ceil(x1)) + 7)
    top = max(1, int(y0))
    bottom = min(camera.height_px - 1, int(np.ceil(y1)) + 7)
    yy, xx = np.nonzero(neutral[top:bottom, left:right])
    pixels = np.column_stack([xx + left, yy + top]).astype(float)
    if len(pixels) == 0:
        return reject("no_neutral_pixels")
    u = (pixels - centre) @ across
    v = (pixels - centre) @ down
    # A few pixels of observed upright support reject thin transverse markings.
    length = max(8, min(20, int(h * 0.1)))

    def sample(mask, points):
        q = np.rint(points).astype(int)
        inside = (
            (q[:, 0] >= 0)
            & (q[:, 0] < camera.width_px)
            & (q[:, 1] >= 0)
            & (q[:, 1] < camera.height_px)
        )
        values = np.zeros(len(q), bool)
        values[inside] = mask[q[inside, 1], q[inside, 0]]
        return values

    feet = []
    details = []
    for sign in [-1, 1]:
        candidates = pixels[
            (sign * u > 0.40 * w) & (sign * u < 0.57 * w) & (v > -0.05 * h)
        ]
        if not len(candidates):
            return reject("one_post_missing")
        supported = np.ones(len(candidates), bool)
        for distance in np.linspace(2, length, 7):
            point = candidates - down * distance
            # One-pixel cross-section allowance for anti-aliasing/tilt.
            supported &= (
                sample(neutral, point)
                | sample(neutral, point + across)
                | sample(neutral, point - across)
            )
        below = sample(green, candidates + down * 4) & sample(
            green, candidates + down * 6
        )
        good = candidates[supported & below]
        if len(good) < 2:
            return reject("no_supported_post_base")
        height = (good - centre) @ down
        low = height.max()
        base = good[height >= low - 1.2]
        # Do not average separate endpoints (post plus field-line clutter).
        if np.ptp(base @ across) > 12:
            return reject("ambiguous_base_components")
        foot = base.mean(0)
        if (
            foot[0] < 3
            or foot[0] > camera.width_px - 4
            or foot[1] > camera.height_px - 7
        ):
            return reject("base_at_image_boundary")
        feet.append(foot.tolist())
        details.append(dict(vertical_support_px=length, observed_base_pixels=len(base)))
    return dict(
        result,
        accepted=True,
        reason="two_observed_vertical_supported_bases",
        footpoints_upright_px=feet,
        posts=details,
        full_visibility_verified=False,
        semantic_correctness_verified=False,
        subpixel_accuracy_calibrated=False,
    )
