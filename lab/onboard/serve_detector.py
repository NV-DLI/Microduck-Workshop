#!/usr/bin/env python3
"""Detector server shim: phase-4 raw server (unchanged) + detector-cuda-graph after the warm-ups.

  <detector python> serve_detector_raw_graph.py [--raw-server <phase4/serve_detector_raw.py>] <server args>

Same arguments as serve_detector_raw.py (the phase-4 client can launch it directly with server=this
file; the raw server then defaults to ../phase4/serve_detector_raw.py). The perception/detector.py module is imported first (same
module object the raw server then imports); BetaDetector.detect is wrapped only to count calls: after
the second completed call on an instance (the server's two normal warm-ups, serve_detector_raw.py
main) the graph is installed on that instance, before the HTTP server starts. Detection itself is
the original method. GET /transport reports server_sha256 = this shim (what the client launched),
inner_server_sha256 = the phase-4 server, and the live graph record. MICRODUCK_DETECTOR_GRAPH_SHADOW=N
enables shadow comparison for the first N calls.
"""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PHASE4_SERVER_SHA256 = "10e9eff8ff7f34fe52f4ddc4e48f4ccbda80aecad7b1ca58fa158b408a4afd7c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--raw-server":
        raw_path, args = Path(sys.argv[2]).resolve(), sys.argv[3:]
    else:  # launched by the phase-4 client with the raw server's own arguments
        raw_path, args = (HERE / "serve_detector_raw.py").resolve(), sys.argv[1:]
    if sha(raw_path) != PHASE4_SERVER_SHA256:
        raise SystemExit("phase-4 raw server differs from the reviewed version")
    folder = Path(args[args.index("--perception-dir") + 1]).resolve()
    sys.path.insert(0, str(folder))
    import detector  # the module the raw server imports next (sys.modules)
    import ultralytics

    graph = load("microduck_detector_graph", HERE / "detector_graph.py")
    record = {"candidate": "detector-cuda-graph", "active": False, "reason": "not yet installed"}
    original_detect = detector.BetaDetector.detect
    counts = {}

    def detect(self, rgb):
        nonlocal record
        rows = original_detect(self, rgb)
        n = counts[id(self)] = counts.get(id(self), 0) + 1
        if n == 2:  # after the server's two normal warm-ups, before serving
            record = graph.install(self, detector_file=detector.__file__, ultralytics_module=ultralytics,
                                   shadow_calls=int(os.environ.get("MICRODUCK_DETECTOR_GRAPH_SHADOW", "0")))
        return rows

    detector.BetaDetector.detect = detect
    raw = load("serve_detector_raw", raw_path)
    make_handler = raw.make_handler

    def handler(model, identity):
        identity = dict(identity, server_sha256=sha(__file__), inner_server_sha256=identity["server_sha256"],
                        detector_graph=record, detector_graph_sha256=sha(HERE / "detector_graph.py"))
        return make_handler(model, identity)

    raw.make_handler = handler
    sys.argv = [str(raw_path), *args]
    raw.main()


if __name__ == "__main__":
    main()
