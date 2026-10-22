"""Supplement the original media audit with exact finite-episode tick checks."""
import argparse
import hashlib
import json
from pathlib import Path
from audit_media_archive_identity import audit, rows


def cadence(physics, policy, manifest):
    hz = manifest["physics_hz"]
    phz = manifest["policy_hz"]
    ratio = hz / phz
    if ratio != int(ratio):
        return False
    ratio = int(ratio)
    n = len(physics)
    expected_policy = (n + ratio - 1) // ratio
    return (
        n > 0
        and n == manifest["physics_samples"]
        and len(policy) == manifest["policy_updates"] == expected_policy
        and abs(n / hz - manifest["simulated_seconds"]) < 1e-9
        and all(abs(row["timestamp_s"] - (i + 1) / hz) < 1e-9 for i, row in enumerate(physics))
        and all(abs(row["timestamp_s"] - i / phz) < 1e-9 for i, row in enumerate(policy))
        and 1 <= n - (len(policy) - 1) * ratio <= ratio
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    original = audit(a.run)
    manifest = json.loads((a.run / "manifest.json").read_text())
    physics = rows(a.run / "physics-trace.jsonl")
    policy = rows(a.run / "policy-trace.jsonl")
    good = cadence(physics, policy, manifest)
    # Actual trace negative checks reject missing or displaced ticks.
    negative_checks = {
        "missing_physics_rejected": not cadence(physics[:-1], policy, manifest),
        "missing_policy_rejected": not cadence(physics, policy[:-1], manifest),
        "displaced_physics_rejected": not cadence(
            [dict(physics[0], timestamp_s=physics[0]["timestamp_s"] + .005), *physics[1:]], policy, manifest),
        "displaced_policy_rejected": not cadence(
            physics, [dict(policy[0], timestamp_s=.02), *policy[1:]], manifest),
    }
    checks = {k: v for k, v in original["checks"].items() if k != "cadence_counts_match"}
    checks["all_actual_tick_times_and_terminal_partial_interval"] = good
    checks.update(negative_checks)
    result = {
        "schema": "microduck.media-terminal-cadence-supplement.v1",
        "passed": all(checks.values()),
        "original_audit_retained": original,
        "checks": checks,
        "final_policy_interval_physics_ticks": len(physics) - (len(policy) - 1) * 4,
        "explanation": "Policy runs at t=0 before physics. Natural early termination can leave the final 20ms policy interval with 1–4 actual 5ms physics ticks. The old equality physics=4*policy rejected valid partial intervals. This supplement checks every original timestamp, both manifest counts, and exact terminal duration.",
        "source_sha256": {str(Path(__file__)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "qualification_claimed": False,
    }
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "original_cadence_check": original["checks"]["cadence_counts_match"],
                      "final_policy_interval_physics_ticks": result["final_policy_interval_physics_ticks"]}))
    assert result["passed"]


if __name__ == "__main__":
    main()
