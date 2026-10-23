#!/usr/bin/env python3
"""Execute: Microduck simulation and control."""
import hashlib, importlib.util, json, sys
from pathlib import Path

D = Path(__file__).resolve().parent
M = D.parents[1]


def verify():
    release = json.loads((D / "release.json").read_text())
    for item in release["sourceFiles"]:
        path = Path(item["path"])
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise RuntimeError("Frozen media parameters source mismatch: " + str(path))
    return release


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    release = verify()
    if sys.argv[1:] == ["--verify-only"]:
        print(
            json.dumps(
                {
                    "verified": True,
                    "files": len(release["sourceFiles"]),
                    "scope": release["scope"],
                }
            )
        )
        return
    if "isaac_execute" in sys.modules:
        raise RuntimeError("A new process is required; no cached executor substitution")
    sys.path.insert(0, str(D))
    entry = load("isaac_execute", D / "executor.py")
    if entry.ROOT != M / "football" or entry.M != M:
        raise RuntimeError("Copied executor path anchor mismatch")
    controller = load("media_v1_frozen_controller", Path(release["controllerEntry"]))
    if controller.entry is not entry or controller.base_run is not entry.run:
        raise RuntimeError("Controller did not bind the private media executor")
    try:
        controller.main()
    finally:
        verify()


if __name__ == "__main__":
    main()
