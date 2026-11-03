"""Evaluate all R-reset rounds without dropping unsuccessful physical episodes."""
from pathlib import Path
import json
import subprocess
import sys


def summarize(rows, state):
    completed = [r for r in rows if r["status"] == "completed"]
    successes = sum(r["success"] for r in completed)
    return dict(episodes=rows, completed=len(completed), successes=successes,
        failures=len(completed)-successes,
        technical_errors=sum(r["status"] == "technical_error" or bool(r.get("technical_error")) for r in rows),
        audit_errors=sum(bool(r.get("audit_error")) for r in rows),
        interrupted=sum(r["status"] == "interrupted" for r in rows),
        success_rate=successes/len(completed) if completed else None,
        denominator="All completed physical episodes, including failed or unverified recovery; only interrupted/incomplete technical runs excluded",
        session_state=state)


def evaluate(runtime, folder):
    state = json.loads((folder / "session.json").read_text())
    if state["state"] != "closed":
        raise RuntimeError("Close the Isaac window before evaluating this session")
    rows = []
    # Include a round that failed in configuration before its output folder existed.
    names = {p.name for p in folder.glob("episode-*") if p.is_dir()}
    names.update(f"episode-{r['index']:03d}" for r in state["episodes"])
    session_rows = {f"episode-{r['index']:03d}": r for r in state["episodes"]}
    for name in sorted(names):
        episode = folder / name
        path = episode / "manifest.json"
        row = dict(episode=name)
        manifest = None
        try:
            if not path.exists():
                row.update(status="technical_error", reason="missing manifest")
            else:
                manifest = json.loads(path.read_text())
                reason = manifest.get("terminal_reason")
                if reason in ("user_reset", "window_closed") and manifest.get("status") == "failed":
                    row.update(status="technical_error", interrupted=True, reason=manifest.get("error", manifest.get("media_error", "cleanup failed")))
                elif reason in ("user_reset", "window_closed"):
                    row.update(status="interrupted", reason=reason)
                elif manifest.get("status") != "development_episode_finished":
                    row.update(status="technical_error", reason=manifest.get("error", "incomplete episode"))
                else:
                    # A completed physical round always stays in the denominator.
                    # Failed recovery or an audit error cannot inflate the success rate.
                    result_path = episode / "result.json"
                    if result_path.exists():
                        result = json.loads(result_path.read_text())
                        code = int(any(result["course_evaluation"]["audit_exit_codes"]))
                    else:
                        code = subprocess.call([sys.executable,
                            str(Path(__file__).parent / "onboard/evaluate.py"),
                            "--runtime", str(runtime), "--run", str(episode), "--scene-contract"])
                        result = json.loads(result_path.read_text()) if result_path.exists() else {}
                    row.update(status="completed", success=bool(not code and result.get("course_evaluation", {}).get("passed")),
                        audit_error=bool(code), reason=reason, seed=manifest["fixture"]["seed"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            completed = isinstance(manifest, dict) and manifest.get("status") == "development_episode_finished" and manifest.get("terminal_reason") not in ("user_reset", "window_closed")
            row.update(status="completed" if completed else "technical_error",
                       reason="Unreadable evaluation evidence: " + str(error))
            if completed:
                row.update(success=False, audit_error=True)
        session_row = session_rows.get(name, {})
        if session_row.get("technical_error"):
            row.update(technical_error=True,
                technical_error_detail=session_row.get("technical_error_detail"),
                technical_error_phase=session_row.get("technical_error_phase"))
            if row["status"] == "interrupted":
                row.update(status="technical_error", interrupted=True)
        rows.append(row)
    report = summarize(rows, state["state"])
    (folder / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return int(bool(report["technical_errors"] or report["audit_errors"]))
