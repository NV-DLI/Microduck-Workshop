"""Independent trace replay for the 0.5 s settled-ball episode contract.

No ShotCompletion or executor code is imported. Official ONNX networks run on
CPU, and their composition is recomputed here. A legal miss/timeout passes this
integrity audit; recovery is a separate diagnostic, never an audit prerequisite.
"""
from pathlib import Path
import argparse
import bisect
import hashlib
import json
import math
import traceback

import numpy as np

MODE = "official_teacher_only_no_residual"
MODEL_SHA = {
    "velstand.onnx": "2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492",
    "ball_kick_right.onnx": "147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af",
}
PHYSICS_DT = .01
CONTROL_DT = .02
HOLD_S = .14
BLEND_S = .12
REST_S = .5
SPEED_M_S = .02
CONTACT_N = .001


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def vector(value, size):
    out = np.asarray(value, dtype=np.float32).reshape(-1)
    if out.shape != (size,) or not np.isfinite(out).all():
        raise ValueError("Missing, malformed or nonfinite vector")
    return out


def composition_weight(phase):
    phase = np.float32(phase)
    u = np.clip((phase - np.float32(HOLD_S)) / np.float32(BLEND_S),
                np.float32(0), np.float32(1))
    return np.float32(1) - u * u * (np.float32(3) - np.float32(2) * u)


class OfficialNetworks:
    """Frozen network identity, independent composition, CPU execution only."""
    def __init__(self, materialized_root):
        import onnxruntime as ort
        root = Path(materialized_root)
        paths = [root / "models/reference/velstand.onnx",
                 root / "football/sources/ball_kick_right.onnx"]
        self.source_sha256 = {str(p): sha(p) for p in paths}
        for p in paths:
            if self.source_sha256[str(p)] != MODEL_SHA[p.name]:
                raise ValueError("Official model identity mismatch: " + str(p))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.sessions = [ort.InferenceSession(str(p), options,
                         providers=["CPUExecutionProvider"]) for p in paths]

    def __call__(self, observation, phase):
        values = [s.run(None, {s.get_inputs()[0].name: observation[None]})[0][0]
                  for s in self.sessions]
        weight = composition_weight(phase)
        return weight * values[1] + (np.float32(1) - weight) * values[0]


def recovery_diagnostic(physics, completed_at):
    if not physics or completed_at is None:
        return dict(recovered=False, applicable=False, reason="No executed completed kick")
    end = float(physics[-1]["timestamp_s"])
    window = [r for r in physics if end - .5 - 1e-8 <= r["timestamp_s"] <= end
              and r.get("score_enabled") and r["timestamp_s"] >= completed_at - 1e-8]
    if len(window) < 51:
        return dict(recovered=False, applicable=True, samples=len(window),
                    reason="Less than 0.5 s post-kick observation at episode end")
    gravity = np.asarray([r["projected_gravity_body"] for r in window], float)
    velocity = np.asarray([r["root_linear_velocity_body_m_s"] for r in window], float)
    omega = np.asarray([r["root_angular_velocity_body_rad_s"] for r in window], float)
    feet = np.asarray([r["actual_ground_paired_forces_n"] for r in window], float)
    heights = np.asarray([r["root_position_world_m"][2] for r in window], float)
    if not all(np.isfinite(v).all() for v in (gravity, velocity, omega, feet, heights)):
        return dict(recovered=False, applicable=True, reason="Nonfinite recovery evidence")
    tilt = np.arctan2(np.linalg.norm(gravity[:, :2], axis=1), -gravity[:, 2])
    support = float((np.linalg.norm(feet, axis=-1) > .2).all(axis=1).mean())
    linear = float(np.sqrt(np.mean(np.sum(velocity[:, :2] ** 2, axis=1))))
    angular = float(np.sqrt(np.mean(np.sum(omega ** 2, axis=1))))
    checks = dict(full_window=window[-1]["timestamp_s"] - window[0]["timestamp_s"] >= .5 - 1e-8,
                  upright_height=float(heights.min()) >= .09, tilt=float(tilt.max()) <= .35,
                  planar_speed=linear <= .03, angular_speed=angular <= .2,
                  both_feet_supported=support >= .9)
    return dict(recovered=all(checks.values()), applicable=True, checks=checks,
                samples=len(window), start_s=window[0]["timestamp_s"], end_s=end,
                minimum_height_m=float(heights.min()), maximum_tilt_rad=float(tilt.max()),
                planar_speed_rms_m_s=linear, angular_speed_rms_rad_s=angular,
                both_feet_supported_fraction=support,
                scope="Last 0.5 s standing diagnostic; not old 4 s recovery qualification or termination authority")


def audit_data(manifest, policy, physics, infer, *, task_limit_s=60.0):
    """Pure replay; infer independently evaluates the frozen official networks."""
    if not math.isfinite(task_limit_s) or not 0 <= task_limit_s <= 180:
        raise ValueError("Task limit must be 0 (manual reset) or at most 180 s")
    checks = dict(physics_present=bool(physics), policy_present=bool(policy))
    if not physics or not policy:
        return dict(passed=False, checks=checks, recovered=False,
                    reason="No complete trace; interruption before first step is not a completed trial")
    pt = [float(r["timestamp_s"]) for r in policy]
    ft = [float(r["timestamp_s"]) for r in physics]
    checks["physics_100hz"] = all(abs(b - a - PHYSICS_DT) < 1e-7 for a, b in zip(ft, ft[1:]))
    checks["policy_50hz"] = all(abs(b - a - CONTROL_DT) < 1e-7 for a, b in zip(pt, pt[1:]))
    checks["finite_clocks"] = all(math.isfinite(x) and x >= 0 for x in pt + ft)
    checks["policy_covers_physics"] = pt[0] < ft[0] and 0 < ft[-1] - pt[-1] <= CONTROL_DT + 1e-7
    histories, formulas, overrides, weights, phases = [], [], [], [], []
    valid_kick = [False] * len(policy)
    completed_tick = [False] * len(policy)
    onset = None
    previous = None
    zero_residual = neutral = mode_ok = authority_ok = True
    active_count = 0
    for i, row in enumerate(policy):
        obs = vector(row["observation"], 61)
        raw = vector(row["raw_action"], 14)
        actual = vector(row["action"], 14)
        if previous is not None:
            histories.append(float(np.max(abs(obs[34:48] - previous))))
        previous = raw.copy()
        overrides.append(float(np.max(abs(raw - actual))))
        k = row.get("native_k10")
        requested = row.get("kick_requested") is True and row.get("policy") == "kick_right"
        if not k or not k.get("active"):
            authority_ok &= not requested
            continue
        active_count += 1
        phase = float(k["phase_s"])
        if not math.isfinite(phase) or not 0 <= phase <= 4.000001:
            raise ValueError("Invalid actual kick phase")
        onset = pt[i] - phase if onset is None else onset
        phases.append(abs((pt[i] - onset) - phase))
        expected = vector(infer(obs, phase), 14)
        formulas.append(float(np.max(abs(expected - raw))))
        weights.append(abs(float(k["kick_weight"]) - float(composition_weight(phase))))
        zero_residual &= bool(not np.any(vector(k["residual14"], 14))
                              and not np.any(vector(k["delta14"], 14))
                              and np.array_equal(vector(k["raw14"], 14), vector(k["teacher14"], 14))
                              and np.array_equal(vector(k["raw14"], 14), raw))
        neutral &= bool(np.all(obs[48:61] == 0))
        mode_ok &= k.get("mode") == MODE and k.get("course_residual_network_loaded") is False
        authority_ok &= requested and k.get("authority_owner") == "external public-sensor supervisor"
        valid_kick[i] = requested
        completed_tick[i] = requested and phase >= HOLD_S + BLEND_S - 1e-7 and composition_weight(phase) == 0
    checks.update(actual_previous_raw_history=max(histories, default=0.) < 1e-7,
                  official_only_formula_matches_sent_raw=max(formulas, default=0.) < 1e-6,
                  smoothstep_weight_matches=max(weights, default=0.) < 1e-7,
                  phase_matches_executed_clock=max(phases, default=0.) < 1e-6,
                  zero_residual_trace=zero_residual, official_only_mode=mode_ok,
                  no_hidden_target_override=max(overrides, default=0.) == 0,
                  zero_base_head_body_during_kick=neutral, actual_kick_authority=authority_ok)
    names = manifest.get("paired_contact_order", [])
    checks["named_paired_contact_channels"] = len(names) == 2 and set(names) == {"ankle_right", "ankle_left"}
    if not checks["named_paired_contact_channels"]:
        raise ValueError("Cannot identify physical kicking-foot contact channel")
    right = names.index("ankle_right")
    epoch = None
    completion = contact_at = rest_start = first_end = None
    reason = None
    previous_position = None
    last_speed = None
    clock_ok = sampled_state = no_task_assistance = True
    for j, row in enumerate(physics):
        t = ft[j]
        # Policy rows are logged before physics advances. Equal-time policy rows
        # belong to the NEXT step, so strictly preceding rows own this sample.
        i = bisect.bisect_left(pt, t - 1e-8) - 1
        sampled_state &= i >= 0 and 0 < t - pt[max(0, i)] <= CONTROL_DT + 1e-7
        i = max(i, 0)
        task = row.get("task_time_s")
        if task is not None:
            candidate = t - float(task)
            epoch = candidate if epoch is None else epoch
            clock_ok &= abs(candidate - epoch) < 1e-7 and (row.get("score_enabled") is True
                or abs(t - epoch - PHYSICS_DT) < 1e-7)
        else:
            clock_ok &= epoch is None and row.get("score_enabled") is False
        forces = np.asarray(row["raw_paired_foot_ball_forces_n"], float)
        if forces.shape != (2, 3) or not np.isfinite(forces).all():
            raise ValueError("Invalid physical foot-ball force sample")
        if row.get("score_enabled") and valid_kick[i] and np.linalg.norm(forces[right]) > CONTACT_N:
            contact_at = t if contact_at is None else contact_at
        if completed_tick[i] and completion is None:
            completion = t
        xyz = np.asarray(row["ball_position_world_m"], float)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("Invalid physical ball position")
        if previous_position is not None:
            last_speed = float(np.linalg.norm(xyz - previous_position) / (t - ft[j-1]))
        previous_position = xyz
        low = (completion is not None and contact_at is not None and last_speed is not None
               and last_speed <= SPEED_M_S and j > 0 and t - ft[j-1] <= .025)
        rest_start = (t if rest_start is None else rest_start) if low else None
        elapsed = t if task is None else float(task)
        detected = ("task_duration_limit" if task_limit_s > 0 and elapsed >= task_limit_s - 1e-9 else
                    "shot_complete_ball_settled" if rest_start is not None and t-rest_start >= REST_S-1e-9 else None)
        if detected is not None and first_end is None:
            first_end, reason = t, detected
        no_task_assistance &= not np.any(vector(row["support_force_world_n"], 3)) and not np.any(vector(row["support_torque_world_nm"], 3))
    interrupted = manifest.get("terminal_reason") in ("user_reset", "window_closed")
    checks["physics_policy_alignment"] = sampled_state
    checks["task_clock_never_resets"] = clock_ok
    checks["no_external_body_assistance"] = bool(no_task_assistance)
    checks["release_epoch_matches_trace"] = (epoch is None or abs(epoch - float(manifest["initialization_brake"]["release_timestamp_s"])) < 1e-7)
    checks["manifest_end_matches_trace"] = abs(float(manifest["simulated_seconds"]) - ft[-1]) < 1e-7
    if interrupted:
        checks["terminal_contract"] = task_limit_s == 0 or first_end is None
    else:
        checks["terminal_contract"] = (first_end is not None and abs(first_end-ft[-1]) < 1e-7
                                       and reason == manifest.get("terminal_reason"))
    declared = manifest.get("shot_completion") or {}
    checks["declared_thresholds_match_contract"] = (declared.get("rest_duration_s") == REST_S
        and declared.get("speed_limit_m_s") == SPEED_M_S and declared.get("task_limit_s") == task_limit_s)
    checks["recorded_completion_flags_match_replay"] = (declared.get("kick_completed") is (completion is not None)
        and declared.get("physical_kick_contact") is (contact_at is not None))
    checks["recorded_rest_start_matches_replay"] = (declared.get("rest_since_s") is None if rest_start is None else
        declared.get("rest_since_s") is not None and abs(float(declared["rest_since_s"])-rest_start) < 1e-7)
    recovery = recovery_diagnostic(physics, completion)
    return dict(passed=all(checks.values()), checks={k: bool(v) for k, v in checks.items()},
                interrupted=interrupted, completed_attempt=not interrupted,
                recovered=recovery["recovered"], recovery=recovery,
                kick_onset_s=onset, kick_complete_applied_at_s=completion,
                physical_kick_contact_at_s=contact_at, first_legal_end_s=first_end,
                independently_replayed_terminal_reason=reason, task_epoch_s=epoch,
                ball_speed_m_s=last_speed, rest_since_s=rest_start,
                physics_samples=len(physics), policy_samples=len(policy), active_kick_ticks=active_count,
                raw_inference_max_error=max(formulas, default=0.), history_max_error=max(histories, default=0.),
                thresholds=dict(rest_duration_s=REST_S, speed_limit_m_s=SPEED_M_S, task_limit_s=task_limit_s,
                                kick_hold_s=HOLD_S, kick_blend_s=BLEND_S, contact_force_n=CONTACT_N))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--materialized-root", type=Path, default=Path("/workspace/state/student-course/simulation/core"))
    parser.add_argument("--task-limit", type=float, default=60.0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new audit output; preserve earlier evidence")
    paths = [Path(__file__), args.runtime, args.run / "manifest.json",
             args.run / "physics-trace.jsonl", args.run / "policy-trace.jsonl"]
    out = dict(schema="microduck.shot-actual-audit.v1", passed=False, checks={})
    try:
        manifest = json.loads(paths[2].read_text())
        networks = OfficialNetworks(args.materialized_root)
        out.update(audit_data(manifest, rows(paths[4]), rows(paths[3]), networks,
                              task_limit_s=args.task_limit))
        expected = manifest.get("official_teacher_kick_runtime", {}).get("source_sha256", {}).get(str(args.runtime.resolve()))
        out["checks"]["runtime_source_identity"] = expected == sha(args.runtime)
        out["passed"] = all(out["checks"].values())
        out["model_source_sha256"] = networks.source_sha256
    except Exception as error:
        out.update(passed=False, error=repr(error), traceback=traceback.format_exc())
    out["source_sha256"] = {str(p): sha(p) for p in paths if p.is_file()}
    out["scope"] = ("Independent replay of recorded 100 Hz ball/contact physics and 50 Hz actual policy actions; "
                    "official CPU ONNX formula checked without importing the executor or ShotCompletion. "
                    "Goal/miss and recovery do not determine audit pass. Recovery is only the last 0.5 s diagnostic. "
                    "Interrupted attempts are labelled separately; user input itself is not proven by physics traces. "
                    "Other source/actuator/contact-placement audits remain required; these records alone cannot prove absence of unlogged overrides.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2, allow_nan=False))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
