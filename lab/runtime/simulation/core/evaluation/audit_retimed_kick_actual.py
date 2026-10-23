from pathlib import Path
import argparse, importlib.util, json, hashlib, math
import numpy as np

M = Path(__file__).resolve().parents[1]
O = M.parents[1] / "configuration/core"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


a = argparse.ArgumentParser()
a.add_argument("run", type=Path)
a.add_argument("--runtime", type=Path, required=True)
a.add_argument("--output", type=Path, required=True)
a = a.parse_args()
assert not a.output.exists()
spec = importlib.util.spec_from_file_location("audited_runtime", a.runtime)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
runtime = mod.NativeKickRuntime(O / "kick_policy/export/deployment.json")
rows = [json.loads(l) for l in (a.run / "policy-trace.jsonl").read_text().splitlines()]
physics = [
    json.loads(l) for l in (a.run / "physics-trace.jsonl").read_text().splitlines()
]
maxerr = 0.0
hist = 0.0
actualerr = 0.0
onset = None
count = 0
previous = None
headzero = True
for r in rows:
    obs = np.array(r["observation"], np.float32)
    sent = np.array(r["raw_action"], np.float32).reshape(-1)
    if previous is not None:
        hist = max(hist, float(abs(obs[34:48] - previous).max()))
    previous = sent.copy()
    k = r.get("native_k10")
    if not k or not k.get("active"):
        continue
    headzero &= bool(np.all(obs[48:61] == 0))
    phase = k["phase_s"]
    onset = r["timestamp_s"] - phase if onset is None else onset
    expected = runtime.step(
        obs,
        previous_raw14=obs[34:48].copy(),
        phase_s=phase,
        ready=True,
        authorized=True,
        head_return_complete=True,
    )["raw14"]
    maxerr = max(maxerr, float(abs(expected - sent).max()))
    actualerr = max(
        actualerr,
        float(abs(np.array(r["action"], np.float32).reshape(-1) - sent).max()),
    )
    count += 1
assert onset is not None
win = [r for r in physics if 3.5 - 1e-6 <= r["timestamp_s"] - onset <= 4.0 + 1e-6]
v = np.array([r["root_linear_velocity_body_m_s"] for r in win])
w = np.array([r["root_angular_velocity_body_rad_s"] for r in win])
g = np.array([r["projected_gravity_body"] for r in win])
feet = np.array([r["actual_ground_paired_forces_n"] for r in win])
height = min(r["root_position_world_m"][2] for r in win)
tilt = np.arctan2(np.linalg.norm(g[:, :2], axis=1), -g[:, 2])
support = (np.linalg.norm(feet, axis=-1) > 0.2).all(axis=1).mean()
recovery = dict(
    samples=len(win),
    start_phase_s=win[0]["timestamp_s"] - onset,
    end_phase_s=win[-1]["timestamp_s"] - onset,
    minimum_height_m=height,
    maximum_tilt_rad=float(tilt.max()),
    planar_speed_rms_m_s=float(np.sqrt((v[:, :2] ** 2).sum(axis=1).mean())),
    angular_speed_rms_rad_s=float(np.sqrt((w * w).sum(axis=1).mean())),
    both_feet_supported_fraction=float(support),
)
checks = dict(
    full200kick_ticks=count == 200,
    actual_previous_raw_history=hist < 1e-7,
    retimed_formula_matches_sent_raw=maxerr < 1e-6,
    no_hidden_target_override=actualerr == 0,
    zero_base_head_body_duringkick=headzero,
    fullrecoverywindow=len(win) >= 100,
    recovery=height >= 0.09
    and tilt.max() <= 0.35
    and recovery["planar_speed_rms_m_s"] <= 0.03
    and recovery["angular_speed_rms_rad_s"] <= 0.2
    and support >= 0.9,
    no_task_assistance=all(
        not any(r["support_force_world_n"]) and not any(r["support_torque_world_nm"])
        for r in physics
        if r["score_enabled"]
    ),
)
out = dict(
    schema="microduck.retimed-kick-actual-audit.v1",
    passed=all(checks.values()),
    checks={k: bool(v) for k, v in checks.items()},
    kick_onset_s=onset,
    raw_inference_max_error=maxerr,
    history_max_error=hist,
    recovery=recovery,
    scope="Actual same-read public policy inputs and sent actions; private physics only scores recovery. One development episode, no success-rate qualification.",
    source_sha256={
        str(p): sha(p)
        for p in [
            Path(__file__),
            a.runtime,
            a.run / "policy-trace.jsonl",
            a.run / "physics-trace.jsonl",
        ]
    },
)
a.output.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
assert out["passed"]
