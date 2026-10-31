#!/usr/bin/env python3
"""Bind the course modules to the environments installed in the course image.

Runs from docker/entrypoint.sh at every container start. It performs the same
binding as `run.py setup` on a host install (runtime.json, relocated course
modules, Isaac Lab and detector links), but writes only to $MICRODUCK_STATE.
Nothing is downloaded and no model or course source is changed.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_runtime import LOCAL, prepare  # noqa: E402
from setup_runtime import COMMIT, IMAGE as PINNED_NAV2_IMAGE  # noqa: E402

NAV2_RECORD = Path("/usr/local/share/microduck-nav2/install-record.json")


def main():
    lab = Path(os.environ.get("ISAACLAB_PATH", "/workspace/IsaacLab"))
    python = lab / ".venv/bin/python"
    detector = Path("/workspace/detector/bin/python")
    ffmpeg = shutil.which("ffmpeg")
    for path in (python, detector, Path(ffmpeg or "")):
        if not path.is_file():
            raise RuntimeError("Missing environment in the image: " + str(path))
    for name in ("pyproject.toml", "uv.lock"):
        if (lab / name).read_bytes() != (ROOT / "lab/environment" / name).read_bytes():
            raise RuntimeError("Image Isaac Lab lock differs from lab/environment/" + name)
    LOCAL.mkdir(parents=True, exist_ok=True)
    # The relocation keeps the pinned Nav2 image digest in the course modules. The
    # container never starts that separate image: the onboard football route runs
    # the same Nav2 service as a CPU child process (lab/course_labs/nav2_local.py)
    # with the ROS 2 Jazzy packages installed by docker/nav2/install.sh.
    config = dict(python=str(python), isaaclab=str(lab), detector_python=str(detector),
                  ffmpeg=ffmpeg, nav2_image=PINNED_NAV2_IMAGE, container=True)
    (LOCAL / "runtime.json").write_text(json.dumps(config, indent=2) + "\n")
    materialized = prepare()
    config["nav2_image"] = None
    config["nav2_backend"] = "local"
    config["nav2_install_record"] = str(NAV2_RECORD) if NAV2_RECORD.is_file() else None
    link = materialized / "runtime" / ("IsaacLab-" + COMMIT)
    link.parent.mkdir(parents=True, exist_ok=True)
    if not link.exists():
        link.symlink_to(lab, target_is_directory=True)
    overlay = materialized / "perception/.venv"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    if not overlay.exists():
        overlay.symlink_to(detector.parent.parent, target_is_directory=True)
    config["materialized"] = str(materialized)
    # Scene detector deployment with absolute artifact paths, checked by digest.
    manifest = ROOT / "models/scene-detector/deployment.json"
    deployment = json.loads(manifest.read_text())
    for key in ("weights", "onnx"):
        artifact = manifest.parent / deployment[key]
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != deployment[key + "_sha256"]:
            raise RuntimeError("Indoor detector artifact changed")
        deployment[key] = str(artifact)
    (LOCAL / "scene-detector-deployment.json").write_text(json.dumps(deployment, indent=2) + "\n")
    (LOCAL / "runtime.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
