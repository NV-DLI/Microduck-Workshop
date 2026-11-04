"""Install or bind a pinned Isaac Lab runtime without modifying existing installs."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import platform
from prepare_runtime import prepare

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "e527032b25939a8fc061609d3502c99e0c57f9c3"


def setup_runtime(a):
    if platform.system() != "Linux" or platform.machine() not in ("aarch64", "x86_64"):
        raise RuntimeError(
            "This pinned course installer supports Linux aarch64 (DGX Spark) and x86_64 (Intel/AMD). Read the platform requirements before continuing."
        )
    os.environ.setdefault("UV_HTTP_TIMEOUT", "1800")
    local = ROOT / ".local"
    local.mkdir(exist_ok=True)
    ffmpeg = str(a.ffmpeg.resolve()) if a.ffmpeg else shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required. Install it with: sudo apt-get install ffmpeg"
        )
    os.environ["PATH"] = (
        str(Path(ffmpeg).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    lab = a.isaaclab.resolve() if a.isaaclab else local / ("IsaacLab-" + COMMIT)
    if not a.isaaclab:
        uv = shutil.which("uv")
        if not uv:
            bootstrap = local / "bootstrap"
            if not bootstrap.exists():
                subprocess.run(
                    [sys.executable, "-m", "venv", str(bootstrap)], check=True
                )
            subprocess.run(
                [str(bootstrap / "bin/pip"), "install", "uv==0.9.5"], check=True
            )
            uv = str(bootstrap / "bin/uv")
        if not lab.exists():
            archive = local / "isaaclab.tar.gz"
            urllib.request.urlretrieve(
                f"https://codeload.github.com/isaac-sim/IsaacLab/tar.gz/{COMMIT}",
                archive,
            )
            if (
                hashlib.sha256(archive.read_bytes()).hexdigest()
                != "6a2e2eab3f6f324673e91cc51ade1df7bdef26d8a7a0feaf4cf94fb18a4e39ce"
            ):
                raise RuntimeError("Isaac Lab source archive differs")
            with tarfile.open(archive) as tf:
                tf.extractall(local, filter="data")
            for name in ["pyproject.toml", "uv.lock"]:
                shutil.copy2(ROOT / "lab/environment" / name, lab / name)
        subprocess.run(
            [uv, "sync", "--frozen", "--extra", "isaacsim", "--extra", "rsl-rl"],
            cwd=lab,
            check=True,
        )
        subprocess.run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(lab / ".venv/bin/python"),
                "onnx==1.22.0",
                "onnxruntime==1.26.0",
                "tensorboard",
            ],
            check=True,
        )
    py = lab / ".venv/bin/python"
    if not py.is_file():
        raise RuntimeError("Expected Isaac Lab .venv/bin/python at " + str(lab))
    # Require the course source lock even when binding an existing environment.
    for name in ["pyproject.toml", "uv.lock"]:
        if (
            hashlib.sha256((lab / name).read_bytes()).digest()
            != hashlib.sha256((ROOT / "lab/environment" / name).read_bytes()).digest()
        ):
            raise RuntimeError(
                "The supplied Isaac Lab environment does not match the course " + name
            )
    detector = (
        a.detector_python.absolute()
        if a.detector_python
        else local / "detector/bin/python"
    )
    if not detector.is_file():
        prepare_script = (
            ROOT / "lab/runtime/simulation/core/perception/prepare_environment.py"
        )
        subprocess.run(
            [
                str(py),
                str(prepare_script),
                "--source-python",
                str(py),
                "--venv",
                str(local / "detector"),
                "--install",
            ],
            check=True,
        )
    data = dict(
        python=str(py),
        isaaclab=str(lab),
        detector_python=str(detector),
        ffmpeg=ffmpeg,
    )
    (local / "runtime.json").write_text(json.dumps(data, indent=2) + "\n")
    m = prepare()
    runtime_link = m / "runtime" / ("IsaacLab-" + COMMIT)
    runtime_link.parent.mkdir(parents=True, exist_ok=True)
    if not runtime_link.exists():
        runtime_link.symlink_to(lab, target_is_directory=True)
    overlay = m / "perception/.venv"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    if not overlay.exists():
        overlay.symlink_to(detector.parent.parent, target_is_directory=True)
    data["materialized"] = str(m)
    (local / "runtime.json").write_text(json.dumps(data, indent=2) + "\n")
    subprocess.run(
        [str(py), str(m / "evaluation/football/execute_media.py"), "--verify-only"],
        check=True,
    )
    print("Runtime ready. Run: python3 scripts/run.py doctor")
