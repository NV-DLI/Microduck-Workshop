"""Run preserved offline checks with this scene's explicit map contract."""

import argparse
from pathlib import Path
import runpy
import sys

p = argparse.ArgumentParser()
p.add_argument("--scene-runtime", type=Path, required=True)
p.add_argument("--script", required=True)
a, remaining = p.parse_known_args()
allowed = {
    "audit_media_terminal_cadence.py",
    "audit_football.py",
    "audit_retimed_kick_actual.py",
    "audit_motion_heading_reference.py",
    "audit_sole_memory_actual.py",
    "audit_track_commands_actual.py",
    "audit_floor_projection_actual.py",
    "audit_coarse_track_actual.py",
    "audit_guard_stop_actual.py",
    "audit_goal_turn_progress_actual.py",
    "audit_retreat_recovery.py",
}
if a.script not in allowed:
    raise ValueError("Unknown scene audit")
M = a.scene_runtime.resolve()
sys.path[:0] = [
    str(M / "evaluation/football"),
    str(M / "evaluation"),
    str(M / "football"),
    str(M / "sensors"),
]
from scene_nav2 import map_contract, obstacle_contract

sys.modules["nav2_service.map_contract"] = map_contract
sys.modules["nav2_service.obstacle_contract"] = obstacle_contract
file = M / "evaluation" / a.script
sys.argv = [str(file), *remaining]
if a.script == "audit_coarse_track_actual.py":
    # Keep every threshold/assertion. Only the declared goal-centre coordinate differs.
    source = file.read_text()
    old = "dx, dy = 0.93 - ball[0], -ball[1]"
    if source.count(old) != 1:
        raise RuntimeError("Unexpected preserved audit version")
    source = source.replace(old, "dx, dy = 0.68 - ball[0], -ball[1]")
    exec(
        compile(source, str(file), "exec"),
        {"__name__": "__main__", "__file__": str(file)},
    )
else:
    runpy.run_path(str(file), run_name="__main__")
