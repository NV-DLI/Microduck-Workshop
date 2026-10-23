"""Recovery config: Microduck simulation and control."""

import hashlib
import json
import math
from pathlib import Path

from retreat_recovery import (
    RecoveryParams,
    HELPER_PATH,
    STAGE_A_VX,
    STAGE_A_DURATION_S,
    STAGE_A_OFFSETS_RAD,
    POLICY_DT_S,
)

SCHEMA = "microduck.football-retreat-recovery-approval.v2"
REFUSED_SCHEMAS = {
    "microduck.football-retreat-recovery-approval.v1": "v120_constant_motion_approval_refused_openloop_stage_a_failed"
}
BOUND_FILES = (
    "retreat_recovery.py",
    "heading_feedback.py",
    "recovery_handoff.py",
    "recovery_config.py",
    "executor.py",
    "execute.py",
    "execute_media.py",
    "paths.py",
)
STAGE_A_SPEC = {
    "seed": 981131,
    "worlds": 8,
    "repetitions_per_offset": 4,
    "vx": STAGE_A_VX,
    "vy": 0.0,
    "duration_s": STAGE_A_DURATION_S,
    "target_heading_offsets_rad": list(STAGE_A_OFFSETS_RAD),
    "gains": {
        "k_heading": 2.5,
        "k_rate": 0.25,
        "tau_heading_s": 0.12,
        "tau_rate_s": 0.10,
        "wz_clip": 0.5,
    },
    "start": "heading and heading LP start at +offset",
    "post_stop_authority": False,
}
STAGE_A_CRITERIA_KEYS = (
    "no_fall",
    "mechanical_original",
    "onset_le_1s",
    "backward_0p12_to_0p24",
    "yaw_stop_plus_1s_le_0p25",
    "lateral_le_0p05",
    "post_stop_drift_le_0p04",
    "public_odometry_error_le_0p03",
    "feedback_no_fault",
    "audits_passed",
)
STAGE_B_MIN_TESTS = 2
STAGE_B_LIMITS = {"max_foot_ball_force_n": 0.001, "ball_displacement_m": 0.005}


class ApprovalError(ValueError):
    pass


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters_hashes(parameters_dir):
    d = Path(parameters_dir)
    return {name: sha(d / name) for name in BOUND_FILES}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _evidence(stage, items):
    if not isinstance(items, list) or not items:
        raise ApprovalError(stage + "_evidence_missing")
    for item in items:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ApprovalError(stage + "_evidence_schema")
        p = Path(item["path"])
        if not p.is_absolute() or not p.is_file():
            raise ApprovalError(stage + "_evidence_file_missing:" + str(p))
        if sha(p) != item["sha256"]:
            raise ApprovalError(stage + "_evidence_hash_mismatch:" + str(p))


def _motion(record, where):
    """Exact fixed motion named by a stage record."""
    keys = ("helper_sha256", "duration_s", "heading_offset_rad", "vx")
    if not isinstance(record, dict) or any(k not in record for k in keys):
        raise ApprovalError(where + "_motion_binding_missing")
    return tuple(record[k] for k in keys)


def verify_approval(path, *, parameters_dir, allow_test_only=False):
    """Return {'parameters': RecoveryParams, ...} or raise ApprovalError. Production: allow_test_only=False."""
    if path is None:
        raise ApprovalError(
            "recovery_approval_manifest_required_physical_gates_pending"
        )
    path = Path(path)
    if not path.is_file():
        raise ApprovalError("recovery_approval_manifest_missing:" + str(path))
    try:
        m = json.loads(path.read_text())
    except (ValueError, OSError) as error:
        raise ApprovalError("recovery_approval_manifest_unreadable") from error
    if not isinstance(m, dict):
        raise ApprovalError("recovery_approval_schema")
    if m.get("schema") in REFUSED_SCHEMAS:
        raise ApprovalError(REFUSED_SCHEMAS[m["schema"]])
    if m.get("schema") != SCHEMA:
        raise ApprovalError("recovery_approval_schema")
    test_only = m.get("test_only")
    if type(test_only) is not bool:
        raise ApprovalError("test_only_flag_required")
    if test_only and not allow_test_only:
        raise ApprovalError("test_only_manifest_rejected_in_production")
    if not test_only and (
        m.get("approved_by") != "root"
        or m.get("physical_calibration_complete") is not True
    ):
        raise ApprovalError("root_physical_approval_required")
    if Path(m.get("parameters_dir", "")).resolve() != Path(parameters_dir).resolve():
        raise ApprovalError("approval_bound_to_different_parameters_dir")
    bound = m.get("parameters_source_sha256")
    if not isinstance(bound, dict) or set(bound) != set(BOUND_FILES):
        raise ApprovalError("parameters_source_binding_incomplete")
    actual = parameters_hashes(parameters_dir)
    changed = sorted(k for k in BOUND_FILES if actual[k] != bound[k])
    if changed:
        raise ApprovalError(
            "parameters_source_changed_since_approval:" + ",".join(changed)
        )
    helper_sha = actual["heading_feedback.py"]
    if (
        Path(parameters_dir).resolve() / "heading_feedback.py" != HELPER_PATH.resolve()
        and sha(HELPER_PATH) != helper_sha
    ):
        raise ApprovalError("loaded_helper_differs_from_parameters_helper")
    a, b = m.get("stage_a"), m.get("stage_b")
    if not isinstance(a, dict) or a.get("passed") is not True:
        raise ApprovalError("stage_a_heading_feedback_calibration_not_passed")
    if not isinstance(b, dict) or b.get("passed") is not True:
        raise ApprovalError("stage_b_nearball_no_contact_not_passed")
    for stage, value in (("stage_a", a), ("stage_b", b)):
        protocol = value.get("protocol")
        if not isinstance(protocol, dict):
            raise ApprovalError(stage + "_protocol_missing")
        _evidence(stage + "_protocol", [protocol])
        _evidence(stage, value.get("evidence"))
        if not test_only and value.get("test_only") is not False:
            raise ApprovalError(
                "unqualified_or_test_only_" + stage + "_evidence_rejected_in_production"
            )
    # Stage A: exact protocol, per-offset results, smaller eligible offset.
    if a.get("spec") != STAGE_A_SPEC:
        raise ApprovalError(
            "stage_a_spec_is_not_seed981131_8world_1p6s_offset_protocol"
        )
    results = a.get("offset_results")
    if not isinstance(results, dict) or set(results) != {
        "%.2f" % o for o in STAGE_A_OFFSETS_RAD
    }:
        raise ApprovalError("stage_a_offset_results_incomplete")
    eligible = []
    for name, r in results.items():
        worlds = r.get("worlds") if isinstance(r, dict) else None
        if (
            not isinstance(worlds, list)
            or len(worlds) != STAGE_A_SPEC["repetitions_per_offset"]
        ):
            raise ApprovalError("stage_a_offset_world_count:" + name)
        all_pass = all(
            isinstance(w, dict) and all(w.get(k) is True for k in STAGE_A_CRITERIA_KEYS)
            for w in worlds
        )
        if r.get("eligible") is not all_pass:
            raise ApprovalError("stage_a_eligibility_inconsistent:" + name)
        if all_pass:
            eligible.append(float(name))
    if not eligible:
        raise ApprovalError("stage_a_no_eligible_offset")
    selected = min(eligible)
    if a.get("selected_heading_offset_rad") != selected:
        raise ApprovalError("stage_a_selection_is_not_smaller_eligible_offset")
    motion_a = _motion(a, "stage_a")
    expected_motion = (helper_sha, STAGE_A_DURATION_S, selected, STAGE_A_VX)
    if motion_a != expected_motion:
        raise ApprovalError(
            "stage_a_motion_binding_differs_from_parameters_helper_duration_offset_vx"
        )
    # Stage B: exact selected motion on >=2 new near-ball fixtures.
    if _motion(b, "stage_b") != expected_motion:
        raise ApprovalError("stage_b_motion_binding_differs_from_stage_a_selection")
    tests = b.get("tests")
    if not isinstance(tests, list) or len(tests) < STAGE_B_MIN_TESTS:
        raise ApprovalError("stage_b_requires_two_nearball_tests")
    seeds, fixtures = set(), set()
    for t in tests:
        if (
            not isinstance(t, dict)
            or t.get("pre_kick_ball_contact_observed") is not False
            or t.get("fell") is not False
        ):
            raise ApprovalError("stage_b_contact_or_fall")
        if (
            _motion(t, "stage_b_test") != expected_motion
            or t.get("audit_passed") is not True
        ):
            raise ApprovalError("stage_b_test_not_exact_selected_motion_or_audit")
        for key, limit in STAGE_B_LIMITS.items():
            if (
                not _num(t.get(key))
                or not 0 <= t[key] <= limit
                or (key == "ball_displacement_m" and t[key] >= limit)
            ):
                raise ApprovalError("stage_b_limit:" + key)
        if (
            not isinstance(t.get("seed"), int)
            or t["seed"] in seeds
            or t.get("fixture") in fixtures
            or t.get("fixture") is None
        ):
            raise ApprovalError("stage_b_distinct_new_seeds_and_fixtures_required")
        seeds.add(t["seed"])
        fixtures.add(t["fixture"])
    params = m.get("parameters")
    expected_params = {
        "retreat_vx": STAGE_A_VX,
        "duration_s": STAGE_A_DURATION_S,
        "move_steps": round(STAGE_A_DURATION_S / POLICY_DT_S),
        "heading_offset_rad": selected,
        "helper_sha256": helper_sha,
    }
    if params != expected_params:
        raise ApprovalError("parameters_must_equal_stage_bindings")
    rp = RecoveryParams(
        retreat_vx=STAGE_A_VX,
        duration_s=STAGE_A_DURATION_S,
        move_steps=expected_params["move_steps"],
        heading_offset_rad=selected,
        helper_sha256=helper_sha,
        calibrated=not test_only,
        source=("TEST_ONLY_synthetic_manifest:" if test_only else "root_approval:")
        + str(path),
    )
    return {
        "parameters": rp,
        "manifest_path": str(path),
        "manifest_sha256": sha(path),
        "test_only": test_only,
        "parameters_source_sha256": actual,
        "stage_a_binding": {
            "selected_heading_offset_rad": selected,
            "eligible_offsets_rad": sorted(eligible),
            "motion": list(expected_motion),
        },
        "stage_b_binding": {"tests": len(tests), "motion": list(expected_motion)},
    }
