"""Candidate `floor-cache-cross`: floor-cache plus scalar 3-component cross product.

Separate from `floor-cache` (candidates_src/ground_plane_cached.py, sha 96255ce3...), which it
copies unchanged except one statement in the RANSAC loop: np.cross(b - a, c - a) becomes the
same three expressions numpy evaluates for 3-vectors (numpy/_core/numeric.py cross: cp0 = a1*b2
- a2*b1, cp1 = a2*b0 - a0*b2, cp2 = a0*b1 - a1*b0; each product rounded, then the subtraction),
on Python floats read exactly from the same points (tolist), giving a float64 array with
bit-identical components. b - a and c - a are the same elementwise IEEE subtractions. The norm,
dot products, thresholds, RNG stream, triplet order, a/b/c indexing and every other statement
are unchanged. No vectorization or change of reduction order.

Base description of floor-cache follows.

Candidate `floor-cache`: football/ground_plane.fit_virtual_floor with exact immutable caches.

Same function body, order of operations, thresholds, seed and return values as the original
(source sha256 f118fe0ae51b806e29d0508e9fc78bcc892159679bd20299f5cd18d27fa0d566). Only two
values that never depend on the frame are memoised:

1. Sub-ray table per zone. In the original, `rays` for a zone is a pure function of
   (row, col, samples_per_zone_axis, profile FoVs, source raster/FoVs). It is built here by the
   identical Python/numpy statements and stored as the same float64 array; every caller gets
   the same values the original computes.
2. RANSAC index triplets per point count. The original draws, on a fresh
   default_rng(979031), `rng.choice(n, 3, replace=False)` exactly 256 times and every draw
   happens before any `continue`, so the sequence depends only on n. It is drawn once per n
   with the same generator calls and replayed.

Inputs are validated exactly as before (same checks, same order, same exceptions). Cached
arrays are never handed out mutable (copies/indexing only). Not a threshold or model change.
"""

import math
import numpy as np

ORIGINAL_SHA256 = "f118fe0ae51b806e29d0508e9fc78bcc892159679bd20299f5cd18d27fa0d566"
_RAYS = {}
_TRIPLETS = {}
STATS = dict(ray_hits=0, ray_misses=0, triplet_hits=0, triplet_misses=0)


def _zone_rays(row, col, model, profile):
    key = (
        row,
        col,
        model["samples_per_zone_axis"],
        profile["horizontal_fov_deg"],
        profile["vertical_fov_deg"],
        model["source_width_px"],
        model["source_height_px"],
        model["source_horizontal_fov_deg"],
        model["source_vertical_fov_deg"],
    )
    cached = _RAYS.get(key)
    if cached is not None:
        STATS["ray_hits"] += 1
        return cached
    STATS["ray_misses"] += 1
    rays = []
    count = model["samples_per_zone_axis"]
    for subrow in range(count):
        for subcol in range(count):
            az = math.radians(
                profile["horizontal_fov_deg"] * (0.5 - (col + (subcol + 0.5) / count) / 8)
            )
            el = math.radians(
                profile["vertical_fov_deg"] * (0.5 - (row + (subrow + 0.5) / count) / 8)
            )
            f = math.cos(el) * math.cos(az)
            left = math.cos(el) * math.sin(az)
            up = math.sin(el)
            width = model["source_width_px"]
            height = model["source_height_px"]
            fx = width / (2 * math.tan(math.radians(model["source_horizontal_fov_deg"]) / 2))
            fy = height / (2 * math.tan(math.radians(model["source_vertical_fov_deg"]) / 2))
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            c = math.floor(cx - fx * left / f + 0.5)
            rr = math.floor(cy - fy * up / f + 0.5)
            if not 0 <= c < width or not 0 <= rr < height:
                continue
            ray = np.array([1.0, (cx - c) / fx, (cy - rr) / fy])
            ray /= np.linalg.norm(ray)
            rays.append(ray.tolist())
    rays = np.asarray(rays)
    rays.setflags(write=False)
    _RAYS[key] = rays
    return rays


def _triplets(n):
    cached = _TRIPLETS.get(n)
    if cached is not None:
        STATS["triplet_hits"] += 1
        return cached
    STATS["triplet_misses"] += 1
    rng = np.random.default_rng(979031)
    out = tuple(rng.choice(n, 3, replace=False) for _ in range(256))
    for idx in out:
        idx.setflags(write=False)
    _TRIPLETS[n] = out
    return out


def fit_virtual_floor(tof_frame, proprio, head_fk):
    p = proprio
    if set(p) != {
        "timestamp_s",
        "joint_names",
        "q_rad",
        "qd_rad_s",
        "gyro_body_rad_s",
        "gravity_body",
    }:
        raise ValueError("Encoder/IMU input only; no extra privileged fields")
    if abs(tof_frame["timestamp_s"] - p["timestamp_s"]) > 1e-6:
        raise ValueError("Same-capture ToF/encoder input required")
    if len(tof_frame["cells"]) != 8 or any(len(r) != 8 for r in tof_frame["cells"]):
        raise ValueError("Exactly8x8 actual zones required")
    fk = head_fk.forward_policy14(p["q_rad"], joint_names=p["joint_names"])["tof_flu"]
    R = np.asarray(fk.rotation)
    origin = np.asarray(fk.translation_m)
    points = []
    normal = -np.asarray(p["gravity_body"], dtype=float)
    if (
        normal.shape != (3,)
        or not np.isfinite(normal).all()
        or not 0.5 < np.linalg.norm(normal) < 1.5
    ):
        raise ValueError("Measured finite gravity direction required")
    normal /= np.linalg.norm(normal)
    frame = tof_frame
    model = frame["model"]
    profile = frame["profile"]
    if (
        model.get("samples_per_zone_axis") != 4
        or model.get("source_width_px") != 32
        or model.get("source_height_px") != 32
        or model.get("source_horizontal_fov_deg") != 50.0
        or model.get("source_vertical_fov_deg") != 50.0
        or profile.get("horizontal_fov_deg") != 45.0
        or profile.get("vertical_fov_deg") != 45.0
    ):
        raise ValueError(
            "Declared virtual32raster/16subray/45degree zone profile required"
        )
    if model["implementation"] != "angular-zone nearest-valid geometric approximation":
        raise ValueError(
            "Explicit virtual sensor model required; this is not a generic ST firmware correction"
        )
    for cell in sum(tof_frame["cells"], []):
        if (
            cell["status"] == "valid"
            and not cell.get("contains_too_close", False)
            and cell["range_m"] is not None
            and 0.1 <= cell["range_m"] <= 2
        ):
            if cell.get("no_return_samples") or cell.get("invalid_samples"):
                continue
            rays = _zone_rays(cell["row"], cell["col"], model, profile)
            world = rays @ R.T
            selected = world[np.argmin(world @ normal)]
            points.append(origin + selected * cell["range_m"])
    points = np.asarray(points)
    normal = -np.asarray(p["gravity_body"])
    normal /= np.linalg.norm(normal)
    if len(points) < 12:
        return {
            "accepted": False,
            "reason": "insufficient_returns",
            "valid_returns": len(points),
        }
    best = None
    rows = points.tolist()
    for idx in _triplets(len(points)):
        a, b, c = points[idx]
        i, j, k = idx.tolist()
        ax, ay, az = rows[i]
        bx, by, bz = rows[j]
        cx, cy, cz = rows[k]
        u0, u1, u2 = bx - ax, by - ay, bz - az
        v0, v1, v2 = cx - ax, cy - ay, cz - az
        n = np.array([u1 * v2 - u2 * v1, u2 * v0 - u0 * v2, u0 * v1 - u1 * v0])
        size = np.linalg.norm(n)
        if size < 1e-5:
            continue
        n /= size
        if n @ normal < 0:
            n = -n
        d = n @ a
        if n @ normal < np.cos(np.deg2rad(25)) or not -0.22 < d < -0.05:
            continue
        residual = np.abs(points @ n - d)
        inliers = residual < 0.005
        score = (
            int(inliers.sum()),
            -float(np.median(residual[inliers])) if inliers.any() else -1,
        )
        if best is None or score > best[0]:
            best = (score, inliers)
    if best is None or best[0][0] < 12:
        return {
            "accepted": False,
            "reason": "no_consistent_floor",
            "valid_returns": len(points),
        }
    q = points[best[1]]
    center = q.mean(axis=0)
    _, sv, v = np.linalg.svd(q - center)
    n = v[-1]
    if n @ normal < 0:
        n = -n
    d = float(n @ center)
    residual = np.abs(q @ n - d)
    return {
        "accepted": True,
        "normal_body": n.tolist(),
        "offset_m": d,
        "valid_returns": len(points),
        "inliers": len(q),
        "residual_rms_m": float(np.sqrt(np.mean(residual**2))),
        "second_spread_m": float(sv[1] / np.sqrt(len(q))),
        "angle_from_imu_deg": float(np.rad2deg(np.arccos(np.clip(n @ normal, -1, 1)))),
        "truth_used": False,
        "tof_aggregation_bias_calibrated": False,
        "virtual_forward_model": "nearest-zone rasterized subray from declared virtual sensor projection; no raw depth array consumed",
    }
