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
LOCAL = Path(os.environ.get("MICRODUCK_STATE", ROOT / ".local"))


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare():
    manifest = json.loads((ROOT / "lab/bundle-manifest.json").read_text())
    dest = LOCAL / "student-course"
    dest.mkdir(parents=True, exist_ok=True)
    # Remove only explicitly retired bundled sources, including installations
    # prepared before the direct-path-only course. Experiment outputs are outside
    # this managed source tree and are never inspected or removed here.
    for relative in manifest.get("retired_files", []):
        candidate = dest / relative
        if candidate.is_symlink() or candidate.is_file():
            candidate.unlink()
        if candidate.suffix == ".py":
            for cache in (candidate.parent / "__pycache__").glob(candidate.stem + ".*.pyc"):
                cache.unlink()
        for directory in (candidate.parent / "__pycache__", *candidate.parents):
            if directory == dest:
                break
            if directory.is_dir() and not directory.is_symlink():
                try:
                    directory.rmdir()
                except OSError:
                    pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("Install ffmpeg before preparing the runtime.")
    replacements = {
        "@COURSE@": str(dest),
        "@REPO@": str(ROOT),
        "@FFMPEG@": ffmpeg,
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
    (LOCAL / "relocation.json").write_text(json.dumps(record, indent=2) + "\n")
    return dest / "simulation/core"


if __name__ == "__main__":
    print(prepare())
