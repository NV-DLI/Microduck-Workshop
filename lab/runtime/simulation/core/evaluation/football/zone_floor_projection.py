"""Correct only supported floor rays for the declared virtual nearest-zone model.

No raw depth, object labels, root pose or contact truth. The reported distance
is the minimum of raster subrays, not necessarily the central zone ray. An
accepted public floor plus that same-capture sensor pose predicts the nearest
floor subray. Only a measured return within 5 mm of that plane is reprojected.
Unexplained/partial/TooClose returns and the immediate guard remain unchanged.
This is specific to the existing simulator approximation, not ST calibration.
"""

from functools import lru_cache
import copy, math
from nav2_service.map_contract import rotate


@lru_cache(maxsize=64)
def raster_rays(row, col):
    rays = []
    f = 32 / (2 * math.tan(math.radians(50) / 2))
    c = 15.5
    for sr in range(4):
        for sc in range(4):
            az = math.radians(45 * (0.5 - (col + (sc + 0.5) / 4) / 8))
            el = math.radians(45 * (0.5 - (row + (sr + 0.5) / 4) / 8))
            x = math.cos(el) * math.cos(az)
            y = math.cos(el) * math.sin(az)
            z = math.sin(el)
            ix = math.floor(c - f * y / x + 0.5)
            iy = math.floor(c - f * z / x + 0.5)
            if not (0 <= ix < 32 and 0 <= iy < 32):
                continue
            r = [1.0, (c - ix) / f, (c - iy) / f]
            length = math.sqrt(sum(v * v for v in r))
            rr = tuple(v / length for v in r)
            if rr not in rays:
                rays.append(rr)
    return tuple(rays)


def project_floor_zones(sample, frame, *, floor_accepted):
    result = copy.deepcopy(sample)
    report = {
        "schema": "microduck.virtual-floor-zone-projection.v1",
        "capture_s": sample["capture_s"],
        "corrected_zones": [],
        "floor_residual_limit_m": 0.005,
        "truth_used": False,
        "distance_values_changed": False,
        "occupancy_cleared": False,
        "source": "same-capture accepted public floor and declared virtual nearest-raster projection",
    }
    model = frame.get("model", {})
    profile = frame.get("profile", {})
    expected = {
        "samples_per_zone_axis": 4,
        "source_width_px": 32,
        "source_height_px": 32,
        "source_horizontal_fov_deg": 50.0,
        "source_vertical_fov_deg": 50.0,
        "implementation": "angular-zone nearest-valid geometric approximation",
    }
    if (
        not floor_accepted
        or any(model.get(k) != v for k, v in expected.items())
        or profile.get("horizontal_fov_deg") != 45.0
        or profile.get("vertical_fov_deg") != 45.0
    ):
        report["hold_reason"] = "accepted_floor_and_exact_virtual_model_required"
        return result, report
    if (
        frame["timestamp_s"] != sample["capture_s"]
        or sample["fk_capture_s"] != sample["capture_s"]
    ):
        raise ValueError("Floor correction requires exact same capture and FK")
    cells = [c for row in frame["cells"] for c in row]
    if len(cells) != 64:
        raise ValueError("Exactly 64 zones required")
    for zone, cell in enumerate(cells):
        distance = sample["ranges_m"][zone]
        if (
            not sample["isaac_valid_mask"][zone]
            or cell["status"] != "valid"
            or cell.get("contains_too_close")
            or cell.get("no_return_samples")
            or cell.get("invalid_samples")
            or distance is None
        ):
            continue
        if cell["row"] != zone // 8 or cell["col"] != zone % 8:
            raise ValueError("Zone index mismatch")
        if distance != cell["range_m"]:
            raise ValueError("Measured distance changed before projection")
        parameterss = raster_rays(zone // 8, zone % 8)
        ray = min(
            parameterss,
            key=lambda r: rotate(sample["sensor_quaternion_odom_xyzw"], r)[2],
        )
        rz = rotate(sample["sensor_quaternion_odom_xyzw"], ray)[2]
        residual = sample["sensor_translation_odom"][2] + distance * rz
        if rz >= 0 or abs(residual) > 0.005:
            continue
        old = sample["unit_rays_sensor"][zone]
        central_height = (
            sample["sensor_translation_odom"][2]
            + distance * rotate(sample["sensor_quaternion_odom_xyzw"], old)[2]
        )
        result["unit_rays_sensor"][zone] = list(ray)
        report["corrected_zones"].append(
            {
                "zone": zone,
                "range_m": distance,
                "central_height_m": central_height,
                "floor_residual_m": residual,
                "ray_sensor_flu": list(ray),
            }
        )
    return result, report
