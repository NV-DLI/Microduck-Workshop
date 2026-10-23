"""Pure request/response contract; simulation and ROS transaction clocks stay distinct."""

import hashlib, json, math, os, time, uuid
from pathlib import Path

if __package__:
    from .map_contract import snapshot, finite, WIDTH, HEIGHT, RESOLUTION
else:
    from map_contract import snapshot, finite, WIDTH, HEIGHT, RESOLUTION
SCHEMA = "microduck-beta-nav2-request-v1"
RESPONSE = "microduck-beta-nav2-response-v1"


def canonical(x):
    return json.dumps(
        x, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def atomic(p, x):
    p = Path(p)
    tmp = p.with_name(p.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_bytes(canonical(x) + b"\n")
    os.replace(tmp, p)


def need(v, why):
    if not v:
        raise ValueError(why)


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return [a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], a[2] + b[2]]


def validate_request(x, nonce, last_sequence=-1, last_sim=-1.0):
    need(
        x.get("schema") == SCHEMA and x.get("run_nonce") == nonce,
        "Wrong protocol/run nonce",
    )
    seq = x.get("sequence")
    need(
        type(seq) is int and 0 <= seq < 10000 and seq > last_sequence,
        "Duplicate/out-of-order sequence",
    )
    t = x.get("sim_query_s")
    need(
        type(t) in (int, float) and math.isfinite(t) and t >= 0 and t > last_sim,
        "Duplicate/out-of-order simulation query",
    )
    need(
        0 <= time.monotonic_ns() - x["sent_monotonic_ns"] <= 5_000_000_000,
        "Stale/future wall request transport",
    )
    case = x["case"]
    provenance = x["provenance"]
    need(case["query_s"] == t, "Source query time changed")
    need(
        provenance["kind"]
        == "isaac_recorded_tof_approximation_with_RGB_field_localization",
        "Explicit RGB localized Isaac approximation required",
    )
    need(provenance["odometry_origin"] == "sensor_only", "Odometry must be sensor-only")
    need(
        provenance.get("hardware_profile_id") == "beta-gc2093-l8cx-v1",
        "Beta profile required",
    )
    need(provenance.get("runtime_id") == "newton-bam-20261019", "New runtime required")
    need("synthetic_grid" not in case, "No explicit/synthetic grid in live requests")
    e = case["recorded_evidence"]
    need(e.get("truth_read") is False, "Runtime truth prohibited")
    raw = finite(e["raw_sensor_odom_xyyaw"], 3)
    correction = finite(e["map_to_odom_xyyaw"], 3)
    fused = finite(case["odom"]["pose_xyyaw"], 3)
    expected = compose(correction, raw)
    need(
        max(abs(a - b) for a, b in zip(fused[:2], expected[:2])) < 1e-8
        and abs(
            math.atan2(
                math.sin(fused[2] - expected[2]), math.cos(fused[2] - expected[2])
            )
        )
        < 1e-8,
        "Fused map pose is not declared RGB correction plus raw sensor odometry",
    )
    need(
        0
        <= e["RGB_localization_capture_end_s"]
        <= e["RGB_localization_available_s"]
        <= t,
        "Noncausal RGB localization",
    )
    need(
        len(case.get("tof_samples", [])) == 1,
        "Exactly one current 64-zone ToF packet required",
    )
    snap = snapshot(case, provenance)
    return snap


def validate_response(r, request, now_sim_s):
    need(
        r.get("schema") == RESPONSE and r.get("run_nonce") == request["run_nonce"],
        "Wrong response protocol/nonce",
    )
    need(
        r.get("sequence") == request["sequence"]
        and r.get("request_sha256") == digest(request),
        "Old/unrelated response",
    )
    need(
        r.get("source_sim_query_s") == request["sim_query_s"],
        "Response retimestamped sensor query",
    )
    need(
        0 <= now_sim_s - request["sim_query_s"] <= 0.20 + 1e-9,
        "Path expired in actual simulation time",
    )
    need(
        r.get("status") == "complete", "Nav2 transaction failed: " + str(r.get("error"))
    )
    need(
        r.get("map_barrier", {}).get("matched") is True
        and r.get("planner_outcome_valid") is True,
        "No map-consumption/planner proof",
    )
    return r


def map_compatible(data, snap):
    """Validate input occupancy topology after Nav2 inflation, not plugin receipt time.

    Inflation may turn unknown near an obstacle into INSCRIBED (253), never free;
    unknown still fails the independent path-envelope gate. Free input may carry
    inflation 0..253. All lethal (254) locations must equal actual input obstacles.
    """
    if len(data) != len(snap["data"]):
        return False
    return all(
        (v == 254 if x == 100 else v in (253, 255) if x == -1 else 0 <= v <= 253)
        for x, v in zip(snap["data"], data)
    )
