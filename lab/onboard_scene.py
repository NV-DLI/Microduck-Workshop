"""Onboard football on the indoor pitch with the course joint-brake start-up.

Same scene, detector, walking actor, navigator, Nav2 session and kick skill as
football_scene.py; only the start-up differs. Instead of the temporary external
support fixture, the robot starts at HOME on the measured grounded root of this
pitch with every joint locked by a temporary joint-limit brake (root free, zero
external wrench). The brake is released after the walking actor's first real output
is computed and before it is applied; the task clock and scoring start there.

The executor is the course student executor (also used by the GR00T route) with
--course-onboard: no GR00T policy, WalkingPolicy or kick-yaw patches are loaded.
"""

from pathlib import Path
import hashlib, json, sys

HERE = Path(__file__).resolve().parent
STUDENT = HERE / "groot_walking/student"
if str(HERE) not in sys.path:  # already first when run as a script
    sys.path.insert(0, str(HERE))
import football_scene  # noqa: E402

EXECUTOR = STUDENT / "walking_student_executor.py"
SOURCES = [
    Path(__file__).resolve(),
    EXECUTOR,
    STUDENT / "joint_brake.py",
    STUDENT / "grounded-pose.json",
    HERE / "course_labs/onboard_startup.py",
    HERE / "course_labs/kit_settings.py",
    HERE / "course_labs/realtime_view.py",
]
REFUSED = ("--walking-policy-server", "--development-contour", "--calibration-command")


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
        return original_load(name, EXECUTOR if name == "isaac_execute" else path)

    football_scene.load = load
    return original_load


def main():
    sys.argv = [sys.argv[0], *onboard_argv(sys.argv[1:])]
    if "kick_runtime" in sys.modules or "learning.walking_runtime" in sys.modules:
        raise RuntimeError("Fresh process required: no GR00T skill modules may be preloaded")
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    hashes = {str(p): sha(p) for p in SOURCES}
    pose = json.loads((STUDENT / "grounded-pose.json").read_text())
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
                controller="installed detector, BallReacquisitionNavigator, Nav2 session and K10",
            ),
            f,
            indent=2,
        )
    install_executor()
    football_scene.main()
    if any(sha(p) != h for p, h in hashes.items()):
        raise RuntimeError("Onboard start-up sources changed during physics")


if __name__ == "__main__":
    main()
