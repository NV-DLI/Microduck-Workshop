"""Rebuild the CPU navigation image from pinned public package archives."""

from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def build_nav2():
    source = ROOT / "lab/nav2"
    context = ROOT / ".local/nav2-build"
    (context / "debs").mkdir(parents=True, exist_ok=True)
    for name in ["Dockerfile", "deb-sha256.txt"]:
        shutil.copy2(source / name, context / name)
    urls = {
        filename: url
        for url, filename in re.findall(
            r"'([^']+)'\s+(\S+\.deb)\s+", (source / "apt-download-plan.txt").read_text()
        )
    }
    manifest = json.loads((source / "package-closure.json").read_text())
    for item in manifest["debs"]:
        path = context / "debs" / item["filename"]
        if not path.exists():
            url = urls.get(item["filename"])
            if not url:
                raise RuntimeError("No pinned package URL: " + item["filename"])
            if "ports.ubuntu.com/ubuntu-ports/" in url:
                url = url.replace(
                    "http://ports.ubuntu.com/ubuntu-ports/",
                    "https://snapshot.ubuntu.com/ubuntu/20261019T000000Z/",
                )
            if "packages.ros.org/ros2/ubuntu/" in url:
                url = url.replace(
                    "packages.ros.org/ros2/ubuntu/",
                    "snapshots.ros.org/jazzy/2026-09-11/ubuntu/",
                )
            if item["filename"].startswith("libxml2-dev_"):
                url = url.replace("20261019T000000Z", "20260921T000000Z")
            # ROS publishes these pinned archives through its official HTTP endpoint.
            # Verify every byte against the committed SHA-256 manifest below.
            temporary = path.with_suffix(".deb.part")
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open(
                "wb"
            ) as target:
                shutil.copyfileobj(response, target)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
                raise RuntimeError(
                    "Downloaded package identity mismatch: " + item["filename"]
                )
            temporary.replace(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise RuntimeError("Package identity mismatch: " + item["filename"])
    subprocess.run(
        [
            "docker",
            "build",
            "--network=none",
            "--tag",
            "microduck-workshop-nav2:20261022",
            str(context),
        ],
        check=True,
    )
    return subprocess.check_output(
        [
            "docker",
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            "microduck-workshop-nav2:20261022",
        ],
        text=True,
    ).strip()


if __name__ == "__main__":
    print(build_nav2())
