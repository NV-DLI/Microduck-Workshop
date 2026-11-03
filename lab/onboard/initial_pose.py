"""CPU-only grounded-pose composition for independent football resets.

Yaw is an additional world-Z rotation of the measured standing61 pose, not a
replacement by a pure-yaw quaternion. Isaac Lab's pinned backend uses XYZW.
"""
import math


def yaw_xyzw(quaternion):
    x, y, z, w = quaternion
    norm2 = x*x + y*y + z*z + w*w
    return math.atan2(2*(w*z + x*y), norm2 - 2*(y*y + z*z))


def rotate_world_z(quaternion, yaw_rad):
    q = tuple(float(v) for v in quaternion)
    if len(q) != 4 or not all(math.isfinite(v) for v in (*q, yaw_rad)):
        raise ValueError("Finite XYZW quaternion and yaw required")
    if abs(sum(v*v for v in q) - 1.0) > 1e-5:
        raise ValueError("Grounded quaternion must be unit length")
    if yaw_rad == 0:
        return q  # Preserve the nominal recorded pose exactly.
    x, y, z, w = q
    s, c = math.sin(yaw_rad/2), math.cos(yaw_rad/2)
    # Hamilton product q_world_z * q_grounded, in XYZW order.
    return (c*x - s*y, c*y + s*x, c*z + s*w, c*w - s*z)


def configure_grounded_pose(cfg, fixture, reference, yaw_rad):
    """Finalize scene/fixture together after outer scene wrappers, before creation."""
    if not math.isfinite(yaw_rad) or abs(yaw_rad) > math.pi/6 + 1e-12:
        raise ValueError("Reset yaw offset must lie within +/-30 degrees")
    xy = fixture["root_pose_xyzw"][:2]
    z = float(reference["root"][2])
    if len(xy) != 2 or not all(math.isfinite(v) for v in (*xy, z)):
        raise ValueError("Finite grounded root position required")
    q = rotate_world_z(reference["quat"], yaw_rad)
    pose = [*xy, z, *q]
    fixture["root_pose_xyzw"] = pose
    fixture["root_yaw_offset_rad"] = yaw_rad
    fixture["root_yaw_rad"] = yaw_xyzw(q)
    cfg.scene.robot.init_state.pos = tuple(pose[:3])
    cfg.scene.robot.init_state.rot = q
    return pose
