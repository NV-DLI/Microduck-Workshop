"""Evaluate the course direct-path shot or an interactive shooting session."""
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
if (a.run / "session.json").is_file():
    from evaluate_onboard_session import evaluate
    raise SystemExit(evaluate(a.runtime, a.run))
manifest = json.loads((a.run / "manifest.json").read_text())
if manifest.get("physics_hz") != 100 or manifest.get("onboard_route", {}).get("kick_composition") != "official_teacher_only_no_residual":
    p.error("Use a recorded course onboard run")
raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).parent / "onboard/evaluate.py"), *sys.argv[1:]]))
