"""Conditional goal geometry from two observed post-base parameterss.

Requires independently estimated image-time camera pose and flat ground. A
known post spacing is a rejection check, never a scale fit or repaired pixel.
No simulator state, map pose, teacher mask, or hidden goal position is read.
"""

from dataclasses import asdict
import math
import sys
from pathlib import Path

# Standalone local module bundle; no external import-path mutation.
from beta_perception_geometry import (
    CaptureCameraPose,
    PlaneEstimate,
    project_pixel_to_plane,
    GeometryError,
)
from beta_sensor_profile import CAMERA_PROFILE

POST_CENTER_SEPARATION_M = 0.265
SPACING_TOLERANCE_M = (
    0.030  # Development rejection bound, not measured sensor uncertainty.
)


def project_goal_parameterss(
    parameterss,
    *,
    pose,
    ground,
    capture_timestamp_s,
    now_s,
    max_age_s=0.2,
    camera=CAMERA_PROFILE
):
    result = {
        "kind": "conditional_observed_goal_post_geometry_v2",
        "accepted": False,
        "reason": None,
        "goal_center_reference_m": None,
        "ground_contact_and_full_post_visibility_assumed": True,
        "semantic_correctness_verified": False,
        "task_qualified": False,
        "scale_fitted": False,
        "missing_pixels_reconstructed": False,
        "known_center_separation_m": POST_CENTER_SEPARATION_M,
        "spacing_rejection_tolerance_m": SPACING_TOLERANCE_M,
        "uncertainty_calibrated": False,
        "point_retimed_to_current_robot": False,
    }

    def reject(reason):
        result["reason"] = reason
        return result

    if not isinstance(parameterss, dict) or parameterss.get("accepted") is not True:
        return reject("no_two_observed_parameterss")
    if parameterss.get("kind") not in (
        "RGB_goal_post_base_parameters_v3",
        "RGB_beta_goal_post_base_parameters_v1",
    ):
        return reject("requires_crop_checked_v3_parameterss")
    points = parameterss.get("footpoints_upright_px")
    if not isinstance(points, (list, tuple)) or len(points) != 2:
        return reject("requires_exactly_two_points")
    if not isinstance(pose, CaptureCameraPose) or not isinstance(ground, PlaneEstimate):
        return reject("requires_explicit_pose_and_ground_estimates")
    if (
        any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            for v in (capture_timestamp_s, now_s, max_age_s)
        )
        or max_age_s < 0
    ):
        return reject("invalid_time")
    age = now_s - capture_timestamp_s
    result["age_s"] = age
    if age < -1e-9 or age > max_age_s:
        return reject("future_or_stale_parameterss")
    if abs(pose.timestamp_s - capture_timestamp_s) > 1e-6:
        return reject("capture_pose_timestamp_mismatch")
    try:
        projected = [
            project_pixel_to_plane(
                p,
                pixel_space="upright",
                image_size=(camera.width_px, camera.height_px),
                capture_timestamp_s=capture_timestamp_s,
                pose=pose,
                plane=ground,
                camera=camera,
                min_abs_incidence=0.01,
                max_range_m=3.0,
            )
            for p in points
        ]
    except (GeometryError, ValueError, TypeError) as error:
        return reject("projection_rejected:" + str(error))
    a, b = [p.point_reference_m for p in projected]
    separation = math.dist(a, b)
    result["observed_separation_m"] = separation
    result["post_points_reference_m"] = [list(a), list(b)]
    if abs(separation - POST_CENTER_SEPARATION_M) > SPACING_TOLERANCE_M:
        return reject("post_spacing_inconsistent_with_known_goal")
    result.update(
        accepted=True,
        reason="conditional_grounded_post_geometry_and_spacing_consistent",
        goal_center_reference_m=[(x + y) / 2 for x, y in zip(a, b)],
        reference_frame=pose.reference_frame,
        capture_timestamp_s=capture_timestamp_s,
        pose_source=pose.source,
        ground_source=ground.source,
        projections=[asdict(p) for p in projected],
        precision_limit="Width consistency rejects large errors; it does not prove unoccluded bases or metric accuracy.",
    )
    return result
