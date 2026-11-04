"""Onboard football with the grounded HOME start-up and official kick/standing policies.

The onboard executor runs Newton at 100 Hz, low-level policies at 50 Hz and
RGB/ToF at 15 Hz. The shared scene supplies the pitch and sensor-only navigation.
Temporary joint limits are restored before applying the first real walking output;
the root is always free and receives no external support. Interactive sessions
rebuild each episode in the same Isaac Sim application when R is pressed.

Only the onboard controller is exposed through this entry.
"""

from pathlib import Path
import hashlib, json, sys

HERE = Path(__file__).resolve().parent
STARTUP_DIR = HERE / "course_labs/startup"
if str(HERE) not in sys.path:  # already first when run as a script
    sys.path.insert(0, str(HERE))
import football_scene  # noqa: E402

EXECUTOR = HERE / "onboard/executor.py"
SOURCES = [
    Path(__file__).resolve(),
    EXECUTOR,
    STARTUP_DIR / "joint_brake.py",
    STARTUP_DIR / "grounded-pose.json",
    HERE / "course_labs/onboard_startup.py",
    HERE / "course_labs/kit_settings.py",
    HERE / "course_labs/realtime_view.py",
    *sorted((HERE / "onboard").glob("*.py")),
]
REFUSED = ("--development-contour", "--calibration-command")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def onboard_argv(argv):
    """Check a football_scene command line and select the onboard route of the executor."""
    for flag in REFUSED:
        if any(x == flag or x.startswith(flag + "=") for x in argv):
            raise ValueError(flag + " is not part of the onboard route")
    if not any(x == "--detector-deployment" or x.startswith("--detector-deployment=") for x in argv):
        raise ValueError("The onboard route requires --detector-deployment")
    if "--output" not in argv:
        raise ValueError("--output is required")
    return argv if "--course-onboard" in argv else [*argv, "--course-onboard"]


def install_executor():
    """Load the student executor wherever football_scene loads the installed one."""
    original_load = football_scene.load

    def load(name, path):
        module = original_load(name, EXECUTOR if name == "isaac_execute" else path)
        if name == "scene_preserved_controller":
            original_build = module.build
            def build(a):
                cfg, fixture = original_build(a)
                module.contract["controller"] = module.contract["controller"].replace(
                    "one native K10 kick", "official kick and standing teachers; no residual")
                module.contract["official_only"] = dict(models=["ball_kick_right.onnx", "velstand.onnx"],
                    residual=False, hold_s=.14, blend_s=.12)
                return cfg, fixture
            module.build = build
        return module

    football_scene.load = load
    return original_load


def main():
    sys.argv = [sys.argv[0], *onboard_argv(sys.argv[1:])]
    if "kick_runtime" in sys.modules:
        raise RuntimeError("Fresh process required: no alternate kick runtime may be preloaded")
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    hashes = {str(p): sha(p) for p in SOURCES}
    pose = json.loads((STARTUP_DIR / "grounded-pose.json").read_text())
    with output.with_name(output.name + "-onboard-start.json").open("x") as f:
        json.dump(
            dict(
                source_sha256=hashes,
                argv=sys.argv[1:],
                start_up=(
                    "standing61: HOME joints locked by temporary joint limits on the measured "
                    "grounded root, free root, zero external wrench; released after the first "
                    "walking-actor output and before it is applied"
                ),
                grounded_pose=dict(case=pose["case"], root=pose["root"], quat=pose["quat"]),
                ground="flat Newton plane at z=0 shared by football styles A-D",
                controller="detector, straight public-observation path, continuous approach and official kick/standing policies",
            ),
            f,
            indent=2,
        )
    sys.path.insert(0, str(HERE / "onboard"))
    install_executor()
    football_scene.main()
    if any(sha(p) != h for p, h in hashes.items()):
        raise RuntimeError("Onboard start-up sources changed during physics")


if __name__ == "__main__":
    main()
