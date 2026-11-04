from pathlib import Path
import argparse
import json
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--runtime", type=Path, required=True)
p.add_argument("--run", type=Path, required=True)
p.add_argument("--evaluation-output", type=Path)
p.add_argument("--scene-contract", action="store_true")
a = p.parse_args()
if not a.scene_contract:
    p.error("The onboard100 evaluator requires --scene-contract")
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
    settled_shot = manifest.get("shot_completion", {}).get("protocol") == "settled-shot-v2"
    if a.scene_contract:
        scene = json.loads((output / "scene-contract.json").read_text())
        if scene.get("source_changed") or scene["profile"] not in (
            "indoor-steel-goal-v1",
            "indoor-trapezoid-official-ball-v2",
        ):
            raise ValueError("Changed or unknown scene contract")
        if scene["goal_plane_x_m"] != 0.68 or scene["goal_opening_width_m"] != 0.388:
            raise ValueError("Audit geometry does not match the recorded scene")
    if not settled_shot:
        raise ValueError("The course requires the settled-shot-v2 completion contract")
    traces = [json.loads(line) for line in (output / "policy-trace.jsonl").open()]
    controls = [json.loads(line) for line in (output / "control.jsonl").open()]
    audits = [
        ["audit_media_terminal_cadence.py", "independent-media-audit.json"],
        ["audit_football.py", "independent-root-audit.json"],
        ["audit_shot_actual.py", "independent-shot-audit.json", "--runtime",
         str(Path(__file__).parent / "official_kick_runtime.py"), "--materialized-root", str(a.runtime),
         "--task-limit", str(manifest["shot_completion"]["task_limit_s"])],
        ["audit_direct_navigation.py", "independent-direct-navigation-audit.json"],
    ]
    if any((row.get("public_handoff") or {}).get("memory", {}).get("memory_available") for row in traces):
        audits.append(["audit_sole_memory_actual.py", "independent-sole-memory-audit.json",
                       "--controller", str(controller)])
    if ((output / "direct-path/queries.jsonl").stat().st_size or
            any((row.get("coarse_track_path") or {}).get("map_request") or
                (row.get("coarse_active_path") or {}).get("map_request") for row in controls)):
        audits.append(["audit_floor_projection_actual.py", "independent-floor-projection-audit.json"])
    if (output / "recovery-trace.jsonl").exists():
        audits.append(["audit_retreat_recovery.py", "independent-retreat-recovery-audit.json",
                       "--controller", str(controller)])
    codes = []
    for i, (script, name, *extra) in enumerate(audits):
        with (evaluation / f"audit-{i+1}.log").open("x") as log:
            codes.append(
                subprocess.run(
                    [
                        sys.executable,
                        *(
                            [
                                str(Path(__file__).parent / "scene_audit_frequency.py"),
                                "--scene-runtime",
                                str(a.runtime),
                                "--script",
                                script,
                            ]
                            if a.scene_contract
                            else [str(A / script)]
                        ),
                        str(output),
                        "--output",
                        str(evaluation / name),
                        *extra,
                    ],
                    stdout=log,
                    stderr=log,
                ).returncode
            )
    recovered = False
    shot_audit = evaluation / "independent-shot-audit.json"
    if settled_shot and shot_audit.exists():
        recovered = json.loads(shot_audit.read_text()).get("recovered", False)
    completion = (manifest.get("terminal_reason") == "shot_complete_ball_settled"
                  if settled_shot else recovered)
    passed = bool(
        score["physical_episode_success"]
        and not score["contact"]["pre_kick_ball_contact_observed"]
        and completion
        and not any(codes)
    )
    report = dict(
        status="completed",
        seed=manifest["fixture"]["seed"],
        task_seconds=manifest["task_seconds"],
        scorer=score,
        course_evaluation=dict(
            passed=passed, recovered=recovered, audit_exit_codes=codes,
            completion_rule="contacted completed kick with ball settled",
            recovery_is_separate_diagnostic=settled_shot
        ),
    )
    (evaluation / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if any(codes):
        raise RuntimeError("Independent checks failed; all results retained")


finish(a.run, {"controller": str(A / "football")}, 0)
