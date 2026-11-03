"""Private front-goal-post image extraction candidate.

Derived from the bounded-ROI course extractor. Intentionally changes recognition:
perspective-projected gravity gives each candidate its local image vertical;
a neutral-metal chromaticity test tolerates modest green reflection and shading.
Uses only RGB, detector boxes, declared camera intrinsics, measured gravity.
Known-spacing, height, three-frame consistency and freshness gates are unchanged.
This file documents the tested function; the byte-exact executed version is kept
in profiler-continuity/candidates_src/scene_goal_points_roi.py for provenance.
"""

import numpy as np

ORIGINAL_SHA256 = "fe6b1bbb17c30d4509245678f81f997922781c35126d97e76ad081fe90f28da0"
HALO = 22
STATS = dict(calls=0, region=0, full_image=0, outside_region=0)


def _green(a):
    return (a[..., 1] > 1.12 * a[..., 0]) & (a[..., 1] > 1.07 * a[..., 2])


def _neutral(a, green):
    low = np.minimum(np.minimum(a[..., 0], a[..., 1]), a[..., 2])
    high = np.maximum(np.maximum(a[..., 0], a[..., 1]), a[..., 2])
    return (low >= 60) & (high - low <= 45) & (a[..., 1] <= 1.18 * np.maximum(a[..., 0], a[..., 2]))


def extract_scene_goal_bases(rgb, box, gravity_raw, camera):
    result = dict(
        kind="RGB_beta_goal_post_base_parameters_v1",
        extractor="perspective_vertical_support_v2_chromatic",
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

    STATS["calls"] += 1
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
    left = max(1, int(x0) - 6)
    right = min(camera.width_px - 1, int(np.ceil(x1)) + 7)
    top = max(1, int(y0))
    bottom = min(camera.height_px - 1, int(np.ceil(y1)) + 7)
    source = np.asarray(rgb)
    if source.dtype.kind in "biuf" and 0 <= top < bottom and 0 <= left < right:
        STATS["region"] += 1
        r0, r1 = max(0, top - HALO), min(camera.height_px, bottom + HALO)
        c0, c1 = max(0, left - HALO), min(camera.width_px, right + HALO)
        a = np.asarray(source[r0:r1, c0:c1], float)
    else:  # original whole-image computation
        STATS["full_image"] += 1
        r0, c0 = 0, 0
        a = np.asarray(rgb, float)
        r1, c1 = a.shape[0], a.shape[1]
    green = _green(a)
    neutral = _neutral(a, green)
    yy, xx = np.nonzero(neutral[top - r0:bottom - r0, left - c0:right - c0])
    pixels = np.column_stack([xx + left, yy + top]).astype(float)
    if len(pixels) == 0:
        return reject("no_neutral_pixels")
    u = (pixels - centre) @ across
    v = (pixels - centre) @ down
    # A few pixels of observed upright support reject thin transverse markings.
    length = max(8, min(20, int(h * 0.1)))

    def exact_outside(kind, rows, cols):
        full = np.asarray(source[rows, cols], float)
        gm = _green(full)
        return gm if kind == "green" else _neutral(full, gm)

    def sample(kind, mask, points):
        q = np.rint(points).astype(int)
        inside = (
            (q[:, 0] >= 0)
            & (q[:, 0] < camera.width_px)
            & (q[:, 1] >= 0)
            & (q[:, 1] < camera.height_px)
        )
        values = np.zeros(len(q), bool)
        region = inside & (q[:, 0] >= c0) & (q[:, 0] < c1) & (q[:, 1] >= r0) & (q[:, 1] < r1)
        values[region] = mask[q[region, 1] - r0, q[region, 0] - c0]
        outside = inside & ~region
        if outside.any():
            STATS["outside_region"] += int(outside.sum())
            values[outside] = exact_outside(kind, q[outside, 1], q[outside, 0])
        return values

    feet = []
    details = []
    for sign in [-1, 1]:
        candidates = pixels[
            (sign * u > 0.40 * w) & (sign * u < 0.57 * w) & (v > -0.05 * h)
        ]
        if not len(candidates):
            return reject("one_post_missing")
        # A 3D vertical has a perspective vanishing point: its pixel direction
        # depends on the pixel, especially with a downward-looking head.
        local_down = np.column_stack([
            camera.fx * g[0] - (candidates[:, 0] - camera.cx) * g[2],
            camera.fy * g[1] - (candidates[:, 1] - camera.cy) * g[2]])
        norms = np.linalg.norm(local_down, axis=1)
        if np.any(norms < 1e-6):
            return reject("local_vertical_undefined")
        local_down /= norms[:, None]
        local_across = np.column_stack([local_down[:, 1], -local_down[:, 0]])
        supported = np.ones(len(candidates), bool)
        for distance in np.linspace(2, length, 7):
            point = candidates - local_down * distance
            # One-pixel cross-section allowance for anti-aliasing/tilt.
            supported &= (
                sample("neutral", neutral, point)
                | sample("neutral", neutral, point + local_across)
                | sample("neutral", neutral, point - local_across)
            )
        below = sample("green", green, candidates + local_down * 4) & sample(
            "green", green, candidates + local_down * 6
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
