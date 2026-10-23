"""CPU-only geometry prerequisite for the hardware-reference camera.

This module consumes caller-supplied pose/plane estimates. It neither obtains
robot/object state nor controls Isaac, FK, a head joint, ToF or a detector.
Pixels are original-image pixel centres, after undoing detector letterboxing.
The default K/D is an explicit uncalibrated Beta pinhole experiment, not the
alpha calibration or a measured calibration of a Beta unit. A bounding-box centre is not automatically the
projection of a ball centre; silhouette/keypoint estimation remains upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from beta_sensor_profile import CAMERA_PROFILE, BetaCameraProfile as CameraProfile

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


class GeometryError(ValueError):
    """An explicit rejected input or geometric estimate; never a free-space hit."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


def _finite(value, name):
    if isinstance(value, bool):
        raise GeometryError(name + "_not_finite_number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise GeometryError(name + "_not_finite_number") from None
    if not math.isfinite(result):
        raise GeometryError(name + "_not_finite_number")
    return result


def _vector(values, length, name):
    try:
        if len(values) != length:
            raise GeometryError(name + "_wrong_shape")
        return tuple(_finite(value, name) for value in values)
    except TypeError:
        raise GeometryError(name + "_wrong_shape") from None


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _matvec(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise GeometryError(name + "_required")


@dataclass(frozen=True)
class CaptureCameraPose:
    """T_reference_from_raw_optical at the image's acquisition time.

    Column-vector convention: p_reference = R @ p_raw + translation_m.
    Raw optical axes are +x right, +y down, +z forward; this is NOT the upright
    optical frame, OpenGL camera frame, or ToF frame. The reference frame may
    be estimated base-at-capture or an odometry frame. `source` must describe
    how encoder FK/IMU/odometry supplied the estimate; a string does not verify
    its accuracy. This API provides no simulation-truth adapter.
    """

    timestamp_s: float
    reference_frame: str
    rotation_reference_from_raw: Matrix3
    translation_m: Vector3
    source: str

    def __post_init__(self):
        _nonempty(self.reference_frame, "reference_frame")
        _nonempty(self.source, "pose_source")
        object.__setattr__(
            self, "timestamp_s", _finite(self.timestamp_s, "pose_timestamp")
        )
        object.__setattr__(
            self, "translation_m", _vector(self.translation_m, 3, "translation")
        )
        try:
            if len(self.rotation_reference_from_raw) != 3:
                raise GeometryError("rotation_wrong_shape")
            rotation = tuple(
                _vector(row, 3, "rotation") for row in self.rotation_reference_from_raw
            )
        except TypeError:
            raise GeometryError("rotation_wrong_shape") from None
        for i in range(3):
            for j in range(3):
                if (
                    abs(_dot(rotation[i], rotation[j]) - (1.0 if i == j else 0.0))
                    > 1e-6
                ):
                    raise GeometryError("rotation_not_orthonormal")
        a, b, c = rotation
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if abs(determinant - 1.0) > 1e-6:
            raise GeometryError("rotation_not_proper")
        object.__setattr__(self, "rotation_reference_from_raw", rotation)


@dataclass(frozen=True)
class PlaneEstimate:
    """n dot p = offset_m in the named reference frame, with unit normal n.

    For a ground plane the normal points out of the floor toward the robot.
    Offset/normal must come from an explicitly declared estimate or teaching
    assumption. This module cannot infer or verify ground height. If the frame
    moves, the caller must express the plane in that same capture-time frame.
    """

    reference_frame: str
    normal: Vector3
    offset_m: float
    source: str

    def __post_init__(self):
        _nonempty(self.reference_frame, "reference_frame")
        _nonempty(self.source, "plane_source")
        normal = _vector(self.normal, 3, "plane_normal")
        if abs(_dot(normal, normal) - 1.0) > 1e-6:
            raise GeometryError("plane_normal_not_unit")
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "offset_m", _finite(self.offset_m, "plane_offset"))


@dataclass(frozen=True)
class PlaneProjection:
    point_reference_m: Vector3
    slant_range_m: float
    raw_pixel: tuple[float, float]
    ray_raw_optical: Vector3
    ray_reference: Vector3
    capture_timestamp_s: float
    reference_frame: str
    pose_source: str
    plane_source: str
    plane_offset_m: float


def _camera(camera):
    if (
        type(camera.width_px) is not int
        or type(camera.height_px) is not int
        or camera.width_px <= 0
        or camera.height_px <= 0
        or camera.mount_rotation_cw_deg != 0
    ):
        raise GeometryError(
            "beta_camera_requires_positive_raw_dimensions_and_native_rotation"
        )
    for name in ("fx", "fy", "cx", "cy"):
        _finite(getattr(camera, name), "camera_" + name)
    if camera.fx <= 0 or camera.fy <= 0:
        raise GeometryError("camera_focal_length_not_positive")
    _vector(camera.distortion_k1_k2_p1_p2_k3, 5, "distortion")


def _pixel(pixel, width, height):
    u, v = _vector(pixel, 2, "pixel")
    if not 0 <= u <= width - 1 or not 0 <= v <= height - 1:
        raise GeometryError("pixel_outside_original_image_centres")
    return u, v


def raw_to_upright(pixel: Sequence[float], camera: CameraProfile = CAMERA_PROFILE):
    """Beta native orientation: identity, no resize/letterbox."""
    _camera(camera)
    u, v = _pixel(pixel, camera.width_px, camera.height_px)
    return u, v


def upright_to_raw(pixel: Sequence[float], camera: CameraProfile = CAMERA_PROFILE):
    """Beta display and raw raster use the same pixel coordinates."""
    _camera(camera)
    return _pixel(pixel, camera.width_px, camera.height_px)


def _brown_with_jacobian(x, y, coefficients):
    k1, k2, p1, p2, k3 = coefficients
    r2 = x * x + y * y
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    derivative = k1 + 2 * k2 * r2 + 3 * k3 * r2 * r2
    dx, dy = 2 * x * derivative, 2 * y * derivative
    distorted = (
        x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
        y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y,
    )
    jacobian = (
        radial + x * dx + 2 * p1 * y + 6 * p2 * x,
        x * dy + 2 * p1 * x + 2 * p2 * y,
        y * dx + 2 * p1 * x + 2 * p2 * y,
        radial + y * dy + 6 * p1 * y + 2 * p2 * x,
    )
    return distorted, jacobian


def raw_pixel_to_unit_ray(
    pixel: Sequence[float], camera: CameraProfile = CAMERA_PROFILE
):
    """Invert five-coefficient Brown–Conrady with checked damped Newton steps.

    Returns a forward unit ray in RAW optical axes. The small central inverse
    branch is used; non-convergence, a folded/singular Jacobian, and solutions
    outside radius 2 in normalized coordinates are rejected. Extreme replacement calibrations may be rejected.
    Residual tolerance is 1e-8 original pixels; this is numerical precision,
    not physical calibration accuracy.
    """
    _camera(camera)
    u, v = _pixel(pixel, camera.width_px, camera.height_px)
    target = ((u - camera.cx) / camera.fx, (v - camera.cy) / camera.fy)
    x, y = target
    for _ in range(32):
        if math.hypot(x, y) > 2.0:
            raise GeometryError("distortion_inverse_outside_supported_branch")
        distorted, jacobian = _brown_with_jacobian(
            x, y, camera.distortion_k1_k2_p1_p2_k3
        )
        ex, ey = distorted[0] - target[0], distorted[1] - target[1]
        a, b, c, d = jacobian
        determinant = a * d - b * c
        if not math.isfinite(determinant) or determinant <= 1e-12:
            raise GeometryError("distortion_inverse_singular_or_folded")
        residual = math.hypot(ex * camera.fx, ey * camera.fy)
        if residual <= 1e-8:
            norm = math.sqrt(x * x + y * y + 1.0)
            return x / norm, y / norm, 1.0 / norm
        dx, dy = (d * ex - b * ey) / determinant, (-c * ex + a * ey) / determinant
        for backtrack in range(16):
            scale = 2.0**-backtrack
            nx, ny = x - scale * dx, y - scale * dy
            if not all(math.isfinite(t) for t in (nx, ny)) or math.hypot(nx, ny) > 2.0:
                continue
            nd, _ = _brown_with_jacobian(nx, ny, camera.distortion_k1_k2_p1_p2_k3)
            next_residual = math.hypot(
                (nd[0] - target[0]) * camera.fx, (nd[1] - target[1]) * camera.fy
            )
            if next_residual < residual:
                x, y = nx, ny
                break
        else:
            raise GeometryError("distortion_inverse_no_convergence")
    raise GeometryError("distortion_inverse_no_convergence")


def project_pixel_to_plane(
    pixel: Sequence[float],
    *,
    pixel_space: str,
    image_size: tuple[int, int],
    capture_timestamp_s: float,
    pose: CaptureCameraPose,
    plane: PlaneEstimate,
    camera: CameraProfile = CAMERA_PROFILE,
    max_range_m: float = 4.0,
    min_abs_incidence: float = 1e-6,
    timestamp_tolerance_s: float = 1e-6,
) -> PlaneProjection:
    """Project a pixel using the explicit image-time SE(3) and declared plane.

    image_size is (width, height), after detector preprocessing is reversed.
    The timestamp comparison detects mismatched input, not stale reception:
    sensor scheduling, receive latency, FK interpolation and timeouts remain
    caller responsibilities. Generic planes are two-sided; behind-camera,
    grazing/horizon and out-of-range intersections are rejected.
    """
    _camera(camera)
    if pixel_space not in ("raw", "upright"):
        raise GeometryError("pixel_space_must_be_raw_or_upright")
    expected = (
        (camera.width_px, camera.height_px)
        if pixel_space == "raw"
        else (camera.width_px, camera.height_px)
    )
    if (
        not isinstance(image_size, (tuple, list))
        or len(image_size) != 2
        or any(type(value) is not int for value in image_size)
        or tuple(image_size) != expected
    ):
        raise GeometryError("original_image_size_mismatch_undo_letterbox_first")
    timestamp = _finite(capture_timestamp_s, "capture_timestamp")
    tolerance = _finite(timestamp_tolerance_s, "timestamp_tolerance")
    maximum = _finite(max_range_m, "max_range")
    minimum = _finite(min_abs_incidence, "min_abs_incidence")
    if tolerance < 0 or maximum <= 0 or not 0 < minimum < 1:
        raise GeometryError("invalid_projection_limits")
    if abs(timestamp - pose.timestamp_s) > tolerance:
        raise GeometryError("capture_pose_timestamp_mismatch")
    if pose.reference_frame != plane.reference_frame:
        raise GeometryError("plane_pose_reference_frame_mismatch")
    raw = (
        upright_to_raw(pixel, camera)
        if pixel_space == "upright"
        else _pixel(pixel, camera.width_px, camera.height_px)
    )
    ray_raw = raw_pixel_to_unit_ray(raw, camera)
    ray_reference = _matvec(pose.rotation_reference_from_raw, ray_raw)
    incidence = _dot(plane.normal, ray_reference)
    if abs(incidence) < minimum:
        raise GeometryError("ray_near_plane_horizon")
    distance = (plane.offset_m - _dot(plane.normal, pose.translation_m)) / incidence
    if not math.isfinite(distance) or distance <= 1e-9:
        raise GeometryError("plane_intersection_not_in_front_of_camera")
    if distance > maximum:
        raise GeometryError("plane_intersection_beyond_max_range")
    point = tuple(
        origin + distance * direction
        for origin, direction in zip(pose.translation_m, ray_reference)
    )
    if not all(math.isfinite(value) for value in point):
        raise GeometryError("projection_not_finite")
    return PlaneProjection(
        point,
        distance,
        raw,
        ray_raw,
        ray_reference,
        timestamp,
        pose.reference_frame,
        pose.source,
        plane.source,
        plane.offset_m,
    )


def project_grounded_ball_center(
    pixel: Sequence[float],
    *,
    radius_m: float,
    ground: PlaneEstimate,
    pose: CaptureCameraPose,
    **projection_options,
) -> PlaneProjection:
    """Intersect a supplied BALL-CENTRE ray with ground offset by known radius.

    Assumes a spherical ball resting on the declared ground. Does not infer a
    centre pixel from a distorted silhouette/bounding box, verify ball contact,
    or estimate airborne balls. Incorrect positive ground height remains an
    estimation risk, but a camera at/below this centre plane is rejected.
    """
    radius = _finite(radius_m, "ball_radius")
    if radius <= 0:
        raise GeometryError("ball_radius_not_positive")
    if pose.reference_frame != ground.reference_frame:
        raise GeometryError("plane_pose_reference_frame_mismatch")
    offset = ground.offset_m + radius
    if _dot(ground.normal, pose.translation_m) - offset <= 1e-9:
        raise GeometryError("camera_not_above_grounded_ball_centre_plane")
    centre_plane = PlaneEstimate(
        ground.reference_frame,
        ground.normal,
        offset,
        ground.source + "; grounded sphere radius offset",
    )
    return project_pixel_to_plane(
        pixel, pose=pose, plane=centre_plane, **projection_options
    )
