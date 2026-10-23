"""Conservative, offline sensor-snapshot adapter; never an actuator bridge."""

import math

if __package__:
    from .obstacle_contract import prior as obstacle_prior, bounds as obstacle_bounds
else:
    from obstacle_contract import prior as obstacle_prior, bounds as obstacle_bounds

WIDTH, HEIGHT, RESOLUTION = 160, 120, 0.025
UNKNOWN, FREE, OCCUPIED = -1, 0, 100


def finite(seq, n):
    if len(seq) != n or any(
        isinstance(x, bool) or not math.isfinite(float(x)) for x in seq
    ):
        raise ValueError("Invalid finite vector")
    return [float(x) for x in seq]


def rotate(q, v):
    x, y, z, w = finite(q, 4)
    if abs(sum(a * a for a in q) - 1) > 1e-5:
        raise ValueError("Quaternion must be unit xyzw")
    vx, vy, vz = finite(v, 3)
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return [
        vx + w * tx + y * tz - z * ty,
        vy + w * ty + z * tx - x * tz,
        vz + w * tz + x * ty - y * tx,
    ]


def cell(x, y, origin):
    i, j = math.floor((x - origin[0]) / RESOLUTION), math.floor(
        (y - origin[1]) / RESOLUTION
    )
    return (i, j) if 0 <= i < WIDTH and 0 <= j < HEIGHT else None


def index(ij):
    return ij[1] * WIDTH + ij[0]


def raster_line(a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = max(abs(dx), abs(dy), 1)
    return list(
        dict.fromkeys(
            (round(a[0] + dx * k / n), round(a[1] + dy * k / n)) for k in range(n + 1)
        )
    )


def snapshot(case, provenance):
    """Only project measured valid rays. No return/invalid/out-of-FOV stays unknown.

    2-D ray clearing is deliberately an unqualified planar assumption: this
    resource/interface probe does not establish volumetric collision coverage.
    """
    if provenance["kind"] not in (
        "synthetic_test_only",
        "recorded_sensor_snapshot",
        "isaac_recorded_tof_approximation",
        "isaac_recorded_tof_approximation_with_manual_field_prior",
        "isaac_recorded_tof_approximation_with_RGB_field_localization",
    ):
        raise ValueError("Unsupported provenance")
    if provenance.get("odometry_origin") != "sensor_only":
        raise ValueError("Truth/command-integrated odometry is not permitted")
    if provenance["kind"] != "synthetic_test_only" and not provenance.get(
        "source_file_sha256"
    ):
        raise ValueError("Recorded snapshots require source hashes")
    query = float(case["query_s"])
    pose = finite(case["odom"]["pose_xyyaw"], 3)
    finite(case["odom"]["twist_body_xyw"], 3)
    odom_stamp = float(case["odom"]["capture_s"])
    if not (0 <= query - odom_stamp <= 0.10):
        raise ValueError("Odometry stale/future")
    # This profile has a surveyed map frame. Keep its raster origin fixed:
    # Jazzy StaticLayer suppresses origin changes smaller than 10 micrometres.
    # The full-content barrier retains its stricter tolerance, without treating
    # unknown as free or changing the supplied robot/sensor poses.
    origin = (
        [-WIDTH * RESOLUTION / 2, -HEIGHT * RESOLUTION / 2]
        if provenance["kind"]
        == "isaac_recorded_tof_approximation_with_RGB_field_localization"
        else [pose[0] - WIDTH * RESOLUTION / 2, pose[1] - HEIGHT * RESOLUTION / 2]
    )
    grid = [UNKNOWN] * (WIDTH * HEIGHT)
    manual = provenance["kind"] in (
        "isaac_recorded_tof_approximation_with_manual_field_prior",
        "isaac_recorded_tof_approximation_with_RGB_field_localization",
    )
    if manual:
        prior = provenance.get("static_prior")
        expected = (
            obstacle_prior(provenance.get("layout", "empty"))
            if provenance["kind"].endswith("RGB_field_localization")
            else {
                "kind": "manual_calibration_fixture",
                "bounds_xy_m": [-1.0, 1.0, -0.5, 0.5],
                "map_to_odom_xyyaw": [0.0, 0.0, 0.0],
                "empty_field_assumed": True,
            }
        )
        if prior != expected:
            raise ValueError(
                "Explicit surveyed-field and initial-goal-side assumptions required"
            )
        # Only whole cells strictly inside the measured boundary are prior free.
        # Outside is unknown; the one-cell boundary is a planning keepout.
        xmin, xmax, ymin, ymax = prior["bounds_xy_m"]
        for j in range(HEIGHT):
            for i in range(WIDTH):
                x0 = origin[0] + i * RESOLUTION
                y0 = origin[1] + j * RESOLUTION
                x1 = x0 + RESOLUTION
                y1 = y0 + RESOLUTION
                if x0 >= xmin and x1 <= xmax and y0 >= ymin and y1 <= ymax:
                    grid[j * WIDTH + i] = FREE
                elif x1 >= xmin and x0 <= xmax and y1 >= ymin and y0 <= ymax:
                    grid[j * WIDTH + i] = OCCUPIED
    if (
        provenance["kind"].endswith("RGB_field_localization")
        and provenance.get("layout") == "one-obstacle"
    ):
        # A surveyed occupied cell can never be cleared by a high/partial ToF ray.
        lo_x, hi_x, lo_y, hi_y = obstacle_bounds()
        for j in range(HEIGHT):
            for i in range(WIDTH):
                x0 = origin[0] + i * RESOLUTION
                y0 = origin[1] + j * RESOLUTION
                if (
                    x0 <= hi_x
                    and x0 + RESOLUTION >= lo_x
                    and y0 <= hi_y
                    and y0 + RESOLUTION >= lo_y
                ):
                    grid[j * WIDTH + i] = OCCUPIED
    samples = case.get("tof_samples", [])
    counts = {
        "valid_ranges": 0,
        "unknown_or_invalid_ranges": 0,
        "samples": len(samples),
    }
    for sample in samples:
        stamp = float(sample["capture_s"])
        if not (0 <= query - stamp <= 0.20):
            raise ValueError("ToF stale/future")
        if abs(float(sample["fk_capture_s"]) - stamp) > 1e-6:
            raise ValueError("FK is not matched to exposure time")
        distances, statuses, rays = (
            sample["ranges_m"],
            sample["status"],
            sample["unit_rays_sensor"],
        )
        if len(distances) != 64 or len(statuses) != 64 or len(rays) != 64:
            raise ValueError("Stock sensor requires exactly 8x8 observations")
        sensor = finite(sample["sensor_translation_odom"], 3)
        q = finite(sample["sensor_quaternion_odom_xyzw"], 4)
        rotate(q, [1, 0, 0])
        start = cell(*sensor[:2], origin)
        if start is None:
            raise ValueError("Sensor outside bounded local grid")
        isaac = provenance["kind"].startswith("isaac_recorded_tof_approximation")
        if isaac:
            if sample.get("provenance") != "simulation_approximation" or any(
                v is not None for v in statuses
            ):
                raise ValueError(
                    "Isaac packet must retain unsimulated chip status None"
                )
            valid = sample.get("isaac_valid_mask")
            if (
                not isinstance(valid, list)
                or len(valid) != 64
                or any(type(v) is not bool for v in valid)
            ):
                raise ValueError("Explicit Isaac 8x8 valid mask required")
            if sample.get("near_m") != 0.1 or sample.get("far_m") != 4.0:
                raise ValueError("Retain recorded Isaac near/far profile")
        endpoints = []
        for zone, (distance, status, ray) in enumerate(zip(distances, statuses, rays)):
            r = finite(ray, 3)
            if abs(sum(x * x for x in r) - 1) > 1e-5:
                raise ValueError("Calibrated ray must be unit length")
            status_valid = valid[zone] if isaac else status in (5, 9)
            minimum = 0.1 if isaac else 0.02
            if (
                not status_valid
                or distance is None
                or isinstance(distance, bool)
                or not math.isfinite(float(distance))
                or not minimum <= float(distance) <= 4
            ):
                counts["unknown_or_invalid_ranges"] += 1
                continue
            counts["valid_ranges"] += 1
            direction = rotate(q, r)
            hit = [sensor[k] + float(distance) * direction[k] for k in range(3)]
            stop = cell(*hit[:2], origin)
            # Reject endpoints outside map; do not extend observations to a border.
            if stop is None:
                continue
            # No claim that a ray over the duck's collision envelope cleared it.
            if not 0.025 <= hit[2] <= 0.50:
                continue
            for ij in raster_line(start, stop)[:-1]:
                at = index(ij)
                if grid[at] != OCCUPIED:
                    grid[at] = FREE
            endpoints.append(stop)
        for ij in endpoints:
            grid[index(ij)] = OCCUPIED
    if "synthetic_grid" in case:
        if provenance["kind"] != "synthetic_test_only":
            raise ValueError("Explicit grids are fixture-only")
        data = case["synthetic_grid"]
        if len(data) != len(grid) or any(
            type(v) is not int or v not in (-1, 0, 100) for v in data
        ):
            raise ValueError("Invalid fixture grid")
        grid = list(data)
    if not samples and "synthetic_grid" not in case:
        raise ValueError("No observed input")
    local = finite(case["goal_body_xy"], 2)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    goal = [
        pose[0] + c * local[0] - s * local[1],
        pose[1] + s * local[0] + c * local[1],
    ]
    counts.update({str(v): grid.count(v) for v in (-1, 0, 100)})
    return {
        "data": grid,
        "origin": origin,
        "pose": pose,
        "goal": goal,
        "counts": counts,
        "source_query_s": query,
        "source_odom_capture_s": odom_stamp,
        "source_ToF_capture_s": [s["capture_s"] for s in samples],
        "map_scope": (
            "declared surveyed single-obstacle field plus observed RGB pose and current ToF; not unknown-obstacle detection qualification"
            if manual
            else "frozen sensor snapshot; projected rays are not validated volumetric free space"
        ),
    }


def dense_path(points):
    dense = []
    for a, b in zip(points, points[1:]):
        n = max(1, math.ceil(math.dist(a, b) / (RESOLUTION / 2)))
        dense.extend(
            [
                [a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n]
                for k in range(n)
            ]
        )
    return dense + [points[-1]] if points else []


def centerline_is_known_free(points, snap):
    for x, y in dense_path(points):
        at = cell(x, y, snap["origin"])
        if at is None or snap["data"][index(at)] != FREE:
            return False
    return True


def check_path(points, snap, radius=0.10):
    """Continuous circular envelope test, conservatively including touched cells.

    .10m is an explicit test envelope, not measured Microduck swept geometry.
    """
    if not points:
        return {"accepted": False, "reason": "no_path"}
    pose, goal = snap["pose"], snap["goal"]
    if math.dist(points[0], pose[:2]) > 0.06 or math.dist(points[-1], goal) > 0.06:
        return {"accepted": False, "reason": "endpoint_mismatch"}
    dense = dense_path(points)
    expanded = radius + RESOLUTION * math.sqrt(2) / 2
    for x, y in dense:
        ij = cell(x, y, snap["origin"])
        if ij is None:
            return {"accepted": False, "reason": "outside_map"}
        n = math.ceil(expanded / RESOLUTION)
        for j in range(ij[1] - n, ij[1] + n + 1):
            for i in range(ij[0] - n, ij[0] + n + 1):
                cx = snap["origin"][0] + (i + 0.5) * RESOLUTION
                cy = snap["origin"][1] + (j + 0.5) * RESOLUTION
                if math.hypot(cx - x, cy - y) <= expanded:
                    if not (0 <= i < WIDTH and 0 <= j < HEIGHT):
                        return {"accepted": False, "reason": "envelope_outside_map"}
                    v = snap["data"][j * WIDTH + i]
                    if v != FREE:
                        return {
                            "accepted": False,
                            "reason": (
                                "unknown_in_envelope"
                                if v == UNKNOWN
                                else "occupied_in_envelope"
                            ),
                        }
    return {
        "accepted": True,
        "reason": "known_fixture_or_projected_free_envelope",
        "test_radius_m": radius,
    }
