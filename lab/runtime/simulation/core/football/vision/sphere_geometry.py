"""Camera-relative known-radius sphere fit from observed silhouette rays.

No floor height, robot pose, object coordinates or temporal truth is accepted.
The input must be the complete, visible silhouette, not a fitted ellipse or a
detector box. These diagnostic rejection thresholds are design assumptions;
passing them does not certify a physical ball, calibration or metric accuracy.
"""

import math
import numpy as np

VERSION = "known_radius_silhouette_cone_v1"
PARAMETERS = {
    "min_unique_rays": 24,
    "unit_norm_tolerance": 1e-5,
    "min_angular_radius_rad": 0.003,
    "max_angular_radius_rad": 0.7,
    "max_azimuth_gap_rad": math.pi / 6,
    "min_tangent_variance_ratio": 0.2,
    "max_rms_over_angular_radius": 0.03,
    "max_residual_over_angular_radius": 0.12,
}


def fit_sphere_rays(rays, radius_m=0.035):
    """Fit optical x-right/y-down/z-forward unit rays to a tangent cone.

    For a sphere at C, tangent directions r satisfy n.r=cos(alpha), where
    n=C/|C| and sin(alpha)=radius/|C|. Total least squares estimates that
    plane on the unit sphere. No rejection repairs, guessed missing arcs or
    outlier trimming are performed. Radius error scales estimated distance
    linearly; pixel sampling/calibration bias is not removed by this fit.
    """
    result = {
        "kind": VERSION,
        "accepted": False,
        "reason": None,
        "center_camera_m": None,
        "diagnostics": {},
        "parameters": PARAMETERS.copy(),
        "coordinate_convention": "optical_x_right_y_down_z_forward",
        "world_or_ground_inputs": False,
        "validated_metric_accuracy": False,
    }

    def reject(reason):
        result["reason"] = reason
        return result

    try:
        radius = float(radius_m)
        points = np.asarray(rays, dtype=np.float64)
    except (ValueError, TypeError):
        return reject("invalid_input")
    if not math.isfinite(radius) or radius <= 0:
        return reject("invalid_radius")
    result["radius_m"] = radius
    if points.ndim != 2 or points.shape[1] != 3:
        return reject("requires_N_by_3_unit_rays")
    if not np.isfinite(points).all():
        return reject("nonfinite_rays")
    count = len(points)
    result["diagnostics"]["ray_count"] = count
    if count < PARAMETERS["min_unique_rays"]:
        return reject("insufficient_rays")
    norms = np.linalg.norm(points, axis=1)
    if np.max(np.abs(norms - 1)) > PARAMETERS["unit_norm_tolerance"]:
        return reject("nonunit_rays")
    if np.any(points[:, 2] <= 0):
        return reject("rays_behind_optical_camera")
    unique = np.unique(np.round(points, decimals=12), axis=0)
    if len(unique) < PARAMETERS["min_unique_rays"]:
        return reject("insufficient_unique_rays")
    # Roundoff alone may perturb nominal unit directions; only now normalize.
    points = points / norms[:, None]
    mean = points.mean(axis=0)
    centered = points - mean
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / count)
    diag = result["diagnostics"]
    diag["covariance_eigenvalues"] = eigenvalues.tolist()
    if (
        eigenvalues[2] <= 1e-14
        or eigenvalues[1] / eigenvalues[2] < PARAMETERS["min_tangent_variance_ratio"]
    ):
        return reject("degenerate_or_narrow_boundary")
    normal = eigenvectors[:, 0]
    if np.dot(normal, mean) < 0:
        normal = -normal
    cosine = float(np.dot(normal, mean))
    if not 0 < cosine < 1 or normal[2] <= 0:
        return reject("invalid_forward_tangent_cone")
    alpha = math.acos(cosine)
    diag["angular_radius_rad"] = alpha
    if (
        not PARAMETERS["min_angular_radius_rad"]
        <= alpha
        <= PARAMETERS["max_angular_radius_rad"]
    ):
        return reject("unsupported_angular_radius")
    deviations = np.arccos(np.clip(points @ normal, -1, 1)) - alpha
    rms = float(np.sqrt(np.mean(deviations**2)))
    maximum = float(np.max(np.abs(deviations)))
    diag.update(
        angular_residual_rms_rad=rms,
        angular_residual_max_rad=maximum,
        angular_residual_P95_rad=float(np.quantile(np.abs(deviations), 0.95)),
    )
    if (
        rms / alpha > PARAMETERS["max_rms_over_angular_radius"]
        or maximum / alpha > PARAMETERS["max_residual_over_angular_radius"]
    ):
        return reject("boundary_not_consistent_with_sphere")
    tangent_a = eigenvectors[:, 2]
    tangent_b = np.cross(normal, tangent_a)
    azimuths = np.sort(np.arctan2(points @ tangent_b, points @ tangent_a))
    gaps = np.diff(np.r_[azimuths, azimuths[0] + 2 * math.pi])
    diag["max_azimuth_gap_rad"] = float(gaps.max())
    if gaps.max() > PARAMETERS["max_azimuth_gap_rad"]:
        return reject("incomplete_silhouette_coverage")
    distance = radius / math.sin(alpha)
    result.update(
        accepted=True,
        reason="complete_cone_geometry_consistent",
        center_camera_m=(normal * distance).tolist(),
        center_distance_m=distance,
    )
    diag["radius_relative_error_distance_gain"] = 1.0
    diag["angular_radius_distance_derivative_m_per_rad"] = -distance / math.tan(alpha)
    return result
