from pathlib import Path
import argparse
import json
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--runtime", type=Path, required=True)
p.add_argument("--run", type=Path, required=True)
p.add_argument("--evaluation-output", type=Path)
a = p.parse_args()
A = a.runtime / "evaluation"


def finish(job, config, code):
    output = job
    controller = Path(config["controller"])
    evaluation = a.evaluation_output or job
    if a.evaluation_output:
        evaluation.mkdir(parents=True, exist_ok=False)
    if code:
        raise RuntimeError(
            "Physics execution failed; original logs retained: " + str(code)
        )
    manifest = json.loads((output / "manifest.json").read_text())
    score = manifest["scorer"]
    audits = [
        ["audit_media_terminal_cadence.py", "independent-media-audit.json"],
        ["audit_football.py", "independent-root-audit.json"],
    ]
    if score["real_foot_ball_contact"]:
        audits.append(
            [
                "audit_retimed_kick_actual.py",
                "independent-retimed-recovery-audit.json",
                "--runtime",
                str(controller / "kick_runtime.py"),
            ]
        )
    if any(
        json.loads(line).get("public_neutral_fine_step", {}).get("state")
        in ("TRACK", "MACRO")
        for line in (output / "policy-trace.jsonl").open()
    ):
        audits.append(
            [
                "audit_motion_heading_reference.py",
                "independent-motion-heading-audit.json",
            ]
        )
    if any(
        json.loads(line)
        .get("public_handoff", {})
        .get("memory", {})
        .get("memory_available")
        for line in (output / "policy-trace.jsonl").open()
    ):
        audits.append(
            [
                "audit_sole_memory_actual.py",
                "independent-sole-memory-audit.json",
                "--controller",
                str(controller),
            ]
        )
    traces = [json.loads(line) for line in (output / "policy-trace.jsonl").open()]
    if any(
        r.get("public_neutral_fine_step", {}).get("state") == "TRACK" for r in traces
    ):
        extra = [
            "--lateral-target=-0.035",
            "--coarse-lateral-target=-0.028",
            "--brake-duration=.16",
            "--brake-start-x=.125",
        ]
        if any(
            r.get("public_neutral_fine_step", {}).get("track_brake_started_s")
            is not None
            for r in traces
        ):
            extra.append("--require-brake")
        audits.append(
            [
                "audit_track_commands_actual.py",
                "independent-track-commands-audit.json",
                *extra,
            ]
        )
    if any((output / "nav2/service/result").glob("request-*.json")) or any(
        r.get("coarse_track_path", {}).get("map_request")
        for r in map(json.loads, (output / "control.jsonl").open())
    ):
        audits.append(
            [
                "audit_floor_projection_actual.py",
                "independent-floor-projection-audit.json",
            ]
        )
    if any(
        r.get("public_neutral_fine_step", {}).get("coarse_track")
        and r.get("public_neutral_fine_step", {}).get("state") == "TRACK"
        for r in traces
    ):
        audits.append(
            ["audit_coarse_track_actual.py", "independent-coarse-track-audit.json"]
        )
    audits.append(
        ["audit_guard_stop_actual.py", "independent-guard-stop-final-audit-v1.json"]
    )
    if any(r.get("public_neutral_fine_step", {}).get("turn_goal_only") for r in traces):
        audits.append(
            [
                "audit_goal_turn_progress_actual.py",
                "independent-goal-turn-progress-audit.json",
                "--require-turn",
            ]
        )
    if (output / "recovery-trace.jsonl").exists():
        audits.append(
            [
                "audit_retreat_recovery.py",
                "independent-retreat-recovery-audit-v2.json",
                "--controller",
                str(controller),
            ]
        )
    codes = []
    for i, (script, name, *extra) in enumerate(audits):
        with (evaluation / f"audit-{i+1}.log").open("x") as log:
            codes.append(
                subprocess.run(
                    [
                        sys.executable,
                        str(A / script),
                        str(output),
                        "--output",
                        str(evaluation / name),
                        *extra,
                    ],
                    stdout=log,
                    stderr=log,
                ).returncode
            )
    recovery = evaluation / "independent-retimed-recovery-audit.json"
    recovered = recovery.exists() and json.loads(recovery.read_text())["passed"]
    passed = bool(
        score["physical_episode_success"]
        and not score["contact"]["pre_kick_ball_contact_observed"]
        and recovered
        and not any(codes)
    )
    report = dict(
        status="completed",
        seed=manifest["fixture"]["seed"],
        task_seconds=manifest["task_seconds"],
        scorer=score,
        course_evaluation=dict(
            passed=passed, recovered=recovered, audit_exit_codes=codes
        ),
    )
    (evaluation / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if any(codes):
        raise RuntimeError("Independent checks failed; all results retained")


finish(a.run, {"controller": str(A / "football")}, 0)
