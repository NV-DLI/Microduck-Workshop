"""Conditional RGB landmark SE2 update. No simulator pose inputs."""

import math

POSTS = ((0.93, 0.1325), (0.93, -0.1325))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def rotate(a, p):
    c, s = math.cos(a), math.sin(a)
    return [c * p[0] - s * p[1], s * p[0] + c * p[1]]


def compose(a, b):
    p = rotate(a[2], b[:2])
    return [a[0] + p[0], a[1] + p[1], wrap(a[2] + b[2])]


def inverse(a):
    p = rotate(-a[2], [-a[0], -a[1]])
    return [*p, wrap(-a[2])]


def mean_pose(poses):
    return [sum(p[i] for p in poses) / len(poses) for i in range(2)] + [
        math.atan2(
            sum(math.sin(p[2]) for p in poses), sum(math.cos(p[2]) for p in poses)
        )
    ]


def fit_posts(points_heading, expected_yaw=0.0):
    """Rigid fit, no scale adjustment. Initial facing/inside assumption resolves pi."""
    if len(points_heading) != 2 or any(
        len(p) != 3 or not all(math.isfinite(x) for x in p) for p in points_heading
    ):
        raise ValueError("Exactly two finite observed heading points required")
    a, b = points_heading
    distance = math.dist(a[:2], b[:2])
    if abs(distance - 0.265) > 0.030:
        raise ValueError("Original observed spacing bound failed")
    center = [(a[i] + b[i]) / 2 for i in range(2)]
    parameterss = []
    for swapped in (False, True):
        x, y = (b, a) if swapped else (a, b)
        yaw = wrap(
            math.atan2(POSTS[1][1] - POSTS[0][1], POSTS[1][0] - POSTS[0][0])
            - math.atan2(y[1] - x[1], y[0] - x[0])
        )
        v = rotate(yaw, center)
        pose = [0.93 - v[0], -v[1], yaw]
        parameterss.append(
            {
                "pose_map_xyyaw": pose,
                "swapped_correspondence": swapped,
                "distance_to_expected_yaw_rad": abs(wrap(yaw - expected_yaw)),
            }
        )
    chosen = min(parameterss, key=lambda c: c["distance_to_expected_yaw_rad"])
    if chosen["distance_to_expected_yaw_rad"] >= math.pi / 2:
        raise ValueError("Initial goal-side/yaw ambiguity")
    return {
        **chosen,
        "both_pose_parameterss": parameterss,
        "observed_width_m": distance,
        "rigid_fit_per_post_residual_m": abs(distance - 0.265) / 2,
        "scale_fitted": False,
        "uncertainty_calibrated": False,
        "physical_hardware_calibration": False,
    }


def correction_from_window(frames, previous_correction=None):
    """Each real image uses its own capture odom, avoiding cross-frame motion smear."""
    if len(frames) != 3:
        raise ValueError("Require original accepted three-image window")
    corrections = []
    reports = []
    for f in frames:
        predicted = (
            0.0
            if previous_correction is None
            else compose(previous_correction, f["odom_pose_xyyaw"])[2]
        )
        fit = fit_posts(f["post_points_heading3"], predicted)
        c = compose(fit["pose_map_xyyaw"], inverse(f["odom_pose_xyyaw"]))
        corrections.append(c)
        reports.append({**f, **fit, "map_to_odom_xyyaw": c})
    result = mean_pose(corrections)
    return {
        "map_to_odom_xyyaw": result,
        "frames": reports,
        "within_window_translation_spread_m": max(
            math.dist(p[:2], result[:2]) for p in corrections
        ),
        "within_window_yaw_spread_rad": max(
            abs(wrap(p[2] - result[2])) for p in corrections
        ),
        "fusion_method": "three equal-weight map-to-odom landmark corrections; no Kalman covariance claim",
    }
