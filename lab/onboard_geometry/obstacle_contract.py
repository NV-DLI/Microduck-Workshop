"""Explicit surveyed map prior; never obtains simulation state."""

import math

DEFAULT_PROFILE = "known-flat-floor-20cm-v2"
VISUAL_MAT_BOUNDS_XY_M = (-0.7, 0.9, -0.5, 0.5)
KNOWN_FLAT_FLOOR_BOUNDS_XY_M = (-0.9, 1.1, -0.7, 0.7)

OBSTACLE = {
    "id": "blocking_box01",
    "center_xy_m": [-0.31, 0.025],
    "size_xy_m": [0.04, 0.05],
    "yaw_rad": 0.0,
    "height_m": 0.08,
}


def prior(layout="empty", *, profile=DEFAULT_PROFILE):
    if layout not in ("empty", "one-obstacle"):
        raise ValueError("Unknown layout")
    if profile != DEFAULT_PROFILE:
        raise ValueError("Unknown declared map prior profile")
    result = {
        "kind": "beta_surveyed_course_field",
        "layout": layout,
        "bounds_xy_m": [-0.7, 0.9, -0.5, 0.5],
        "goal_posts_map_xy_m": [[0.68, 0.197], [0.68, -0.197]],
        "empty_field_assumed": layout == "empty",
        "obstacles": [] if layout == "empty" else [dict(OBSTACLE)],
        "information_source": "explicit surveyed static prior; not a sensor observation",
    }


    if profile == DEFAULT_PROFILE:
        result.update(
            profile=DEFAULT_PROFILE,
            bounds_xy_m=list(KNOWN_FLAT_FLOOR_BOUNDS_XY_M),
            visual_mat_bounds_xy_m=list(VISUAL_MAT_BOUNDS_XY_M),
            information_source="explicit known flat-floor prior including 20 cm beyond the visual mat; not a sensor observation",
        )
    return result


def bounds(obstacle=OBSTACLE):
    c = obstacle["center_xy_m"]
    s = obstacle["size_xy_m"]
    return [c[0] - s[0] / 2, c[0] + s[0] / 2, c[1] - s[1] / 2, c[1] + s[1] / 2]
