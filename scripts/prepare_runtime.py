#!/usr/bin/env python3
"""Prepare verified course modules in an isolated local workspace.

Only paths and their dependent digest bindings are changed. Published weights and
historical results stay immutable. A materialized digest is not new physics evidence.
"""
from pathlib import Path
import hashlib
import json
import re
import shutil
import os

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare():
    manifest = json.loads((ROOT / "lab/bundle-manifest.json").read_text())
    dest = ROOT / ".local/student-course"
    dest.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install ffmpeg before preparing the runtime.")
    runtime_config = (
        json.loads((ROOT / ".local/runtime.json").read_text())
        if (ROOT / ".local/runtime.json").exists()
        else {}
    )
    replacements = {
        "@COURSE@": str(dest),
        "@REPO@": str(ROOT),
        "@FFMPEG@": ffmpeg,
        "sha256:c60e955f781c31c00b46b415c9bad7dc8e5faa922f0e3ea0dbb44f604685bfbd": runtime_config.get(
            "nav2_image",
            "sha256:c60e955f781c31c00b46b415c9bad7dc8e5faa922f0e3ea0dbb44f604685bfbd",
        ),
    }
    baseline = {}
    digests = {}
    records = manifest["files"]
    for item in records:
        src = ROOT / "lab/runtime" / item["path"]
        if sha(src) != item["template_sha256"]:
            raise RuntimeError("Course source changed: " + item["path"])
        raw = src.read_bytes()
        p = dest / item["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix in (".py", ".json", ".yaml", ".yml", ".usda", ".txt"):
            text = raw.decode()
            for old, new in replacements.items():
                text = text.replace(old, new)
            baseline[p] = text
            p.write_text(text)
        else:
            p.write_bytes(raw)
        digests[item["original_sha256"]] = sha(p)
    for entry in manifest.get("external_files", []):
        digests[entry["original_sha256"]] = sha(ffmpeg)
    # Identity maps in JSON and source form a dependency graph. Recompute to a fixed point.
    for round_index in range(40):
        changed = False
        for item in records:
            p = dest / item["path"]
            if p not in baseline:
                continue
            value = re.sub(
                r"\b[0-9a-f]{64}\b",
                lambda m: digests.get(m.group(), m.group()),
                baseline[p],
            )
            if p.read_text() != value:
                p.write_text(value)
                changed = True
            digests[item["original_sha256"]] = sha(p)
        if not changed:
            break
    else:
        raise RuntimeError("Path/digest relocation graph did not converge")
    for old, new in manifest["model_aliases"].items():
        rel = old
        link = dest / rel
        link.parent.mkdir(parents=True, exist_ok=True)
        target = ROOT / new.split("@REPO@/", 1)[1]
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise RuntimeError("Model alias changed: " + str(link))
        elif link.exists():
            raise RuntimeError("Refusing to replace existing file: " + str(link))
        else:
            link.symlink_to(target)
    fflink = dest / "simulation/core/runtime/tools/bin/ffmpeg"
    fflink.parent.mkdir(parents=True, exist_ok=True)
    if not fflink.exists():
        fflink.symlink_to(ffmpeg)
    record = {
        "schema": "microduck.path-relocation.v1",
        "source_bundle_sha256": sha(ROOT / "lab/bundle-manifest.json"),
        "rounds": round_index + 1,
        "files": len(records),
        "weights_changed": False,
        "new_physics_evidence": False,
        "resolved_root": str(dest),
    }
    (ROOT / ".local/relocation.json").write_text(json.dumps(record, indent=2) + "\n")
    return dest / "simulation/core"


if __name__ == "__main__":
    print(prepare())
