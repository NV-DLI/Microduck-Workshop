"""Conditional grounded sphere centre from an already accepted public image.

This parameters does not reconstruct cropped silhouettes or authorize motion.
Its parent detector boundary checks and ToF floor evaluation remain necessary.
"""

import copy
import math
import numpy as np


def refine_grounded_complete_ball(original, pose, ground, floor, *, now_s):
    result = {
        "kind": "grounded_complete_YOLO_sphere_v1",
        "accepted": False,
        "reason": "invalid_public_geometry",
        "truth_used": False,
        "kick_authorized": False,
        "task_or_hardware_validated": False,
    }
    try:
        if (
            original.get("kind") != "beta_YOLO_box_known_radius_v1"
            or original.get("accepted") is not True
        ):
            return {**result, "reason": "original_complete_YOLO_geometry_required"}
        if (
            original.get("observed_boundary_required") is not True
            or original.get("missing_boundary_reconstructed") is not False
        ):
            return {**result, "reason": "observed_complete_boundary_required"}
        if original.get("supported_boundary_sectors", 0) < original.get(
            "required_boundary_sectors", 21
        ):
            return {**result, "reason": "original_boundary_support_insufficient"}
        if not floor or floor.get("accepted") is not True:
            return {**result, "reason": "accepted_actual_ToF_floor_required"}
        stamp = float(original["capture_timestamp_s"])
        if (
            not math.isfinite(now_s)
            or not 0 <= now_s - stamp <= 0.2
            or abs(float(pose["timestamp_s"]) - stamp) > 1e-6
        ):
            return {**result, "reason": "stale_or_mixed_capture"}
        if (
            pose["reference_frame"] != ground["reference_frame"]
            or original["reference_frame"] != pose["reference_frame"]
        ):
            return {**result, "reason": "mixed_reference_frame"}
        radius = float(original["radius_assumption_m"])
        if abs(radius - 0.035) > 1e-9:
            return {**result, "reason": "declared35mm_sphere_required"}
        point = np.asarray(original["point_reference_m"], float)
        origin = np.asarray(pose["translation_m"], float)
        normal = np.asarray(ground["normal"], float)
        if point.shape != (3,) or origin.shape != (3,) or normal.shape != (3,):
            return result
        offset = float(ground["offset_m"])
        if not all(
            np.isfinite(v).all() for v in (point, origin, normal)
        ) or not math.isfinite(offset):
            return result
        if abs(np.linalg.norm(normal) - 1.0) > 1e-6:
            return {**result, "reason": "unit_floor_normal_required"}
        ray = point - origin
        norm = float(np.linalg.norm(ray))
        if norm < 1e-8:
            return result
        ray /= norm
        denominator = float(normal @ ray)
        if abs(denominator) < 0.15:
            return {**result, "reason": "floor_ray_nearly_parallel"}
        distance = float((offset + radius - normal @ origin) / denominator)
        if not 0.06 < distance < 3.0:
            return {
                **result,
                "reason": "forward_grounded_range_outside_existing_domain",
            }
        refined = origin + distance * ray
        result = copy.deepcopy(original)
        result.update(
            kind="grounded_complete_YOLO_sphere_v1",
            accepted=True,
            reason="complete_ball_and_actual_floor_assuming_supported_sphere",
            original_box_geometry=copy.deepcopy(original),
            point_reference_m=refined.tolist(),
            slant_range_m=distance,
            original_to_refined_distance_m=float(np.linalg.norm(refined - point)),
            ray_dot_floor_normal=denominator,
            estimated_center_height_above_plane_m=radius,
            ground_contact_verified=False,
            supported_sphere_assumption=True,
            missing_boundary_reconstructed=False,
            truth_used=False,
            kick_authorized=False,
            task_or_hardware_validated=False,
            uncertainty_calibrated=False,
            uncertainty_scope="Preserve parent box allowance without reduction; floor/FK/radius systematics are additional. This parameters has no calibrated uncertainty bound.",
        )
        return result
    except (KeyError, TypeError, ValueError):
        return result
