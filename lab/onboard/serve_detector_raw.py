#!/usr/bin/env python3
"""Private detector server for candidate `detector-raw-rgb` (opt-in; the course server is unchanged).

Same process role, model construction, warm-up, /health dictionary and response dictionary as
perception/serve_detector.py (sha256 d6085cbb..., mirrored below). The only difference is the
request transport: POST /detect-raw-rgb carries the exact 960x540x3 uint8 RGB bytes
(application/octet-stream, 1,555,200 bytes) with their SHA-256 in a header, instead of a PNG in
base64 JSON. The server rebuilds the array with np.frombuffer + copy (C-contiguous, writable, like
cv2.cvtColor's output), refuses any length/shape/dtype/hash mismatch with HTTP 400 in the original
error format, and reports image_sha256 of the array actually passed to the model.
GET /transport describes the protocol and the sources this server was checked against.

Started by detector_raw_client.RawDetectorSession with the detector Python, exactly as the course
client starts the original server (same arguments plus --perception-dir).
"""
import argparse
import hashlib
import json
import signal
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

PROTOCOL = "microduck.detector.raw-rgb.v1"
SHAPE = (540, 960, 3)
NBYTES = SHAPE[0] * SHAPE[1] * SHAPE[2]
ORIGINAL_SERVER_SHA256 = "d6085cbb073e135c1434b808cb01d2d1b951b64a8bf3617edd8e9c4fad8dce5f"
DETECTOR_SHA256 = "a5d2149143960aac7935786c4c2017e0e259ddecf8598ab989bfbc58a2583d9e"
HEADER_SHA = "X-Microduck-RGB-SHA256"
HEADER_SHAPE = "X-Microduck-RGB-Shape"
HEADER_DTYPE = "X-Microduck-RGB-Dtype"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode_raw(headers, body):
    """Validate one raw request; returns the RGB array or raises ValueError (HTTP 400)."""
    if headers.get("Content-Type") != "application/octet-stream":
        raise ValueError("Raw RGB transport requires application/octet-stream")
    if headers.get(HEADER_SHAPE) != "540,960,3" or headers.get(HEADER_DTYPE) != "uint8":
        raise ValueError("Actual Beta RGB960x540 uint8 required")
    if len(body) != NBYTES:
        raise ValueError("Bounded image request required")
    digest = headers.get(HEADER_SHA, "")
    if len(digest) != 64 or sha(body) != digest:
        raise ValueError("Raw RGB transport hash mismatch")
    return np.frombuffer(body, dtype=np.uint8).reshape(SHAPE).copy()


def make_handler(model, identity):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, obj):
            raw = json.dumps(obj, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/health":  # same dictionary as the original server
                return self.send(
                    200,
                    {
                        "ready": True,
                        "hardware_profile_id": "beta-gc2093-l8cx-v1",
                        "weights_sha256": model.manifest["weights_sha256"],
                        "device": model.device,
                        "onnx_sha256": model.manifest["onnx_sha256"],
                        "backend": model.backend,
                    },
                )
            if self.path == "/transport":
                return self.send(200, identity)
            return self.send(404, {"error": "unknown route"})

        def do_POST(self):
            try:
                if self.path != "/detect-raw-rgb":
                    return self.send(404, {"error": "unknown route"})
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 8_000_000:
                    raise ValueError("Bounded image request required")
                body = self.rfile.read(n)
                rgb = decode_raw(self.headers, body)
                start = time.monotonic()
                rows = model.detect(rgb)
                return self.send(
                    200,
                    {
                        "detections": rows,
                        "weights_sha256": model.manifest["weights_sha256"],
                        "teacher_inputs": False,
                        "onnx_sha256": model.manifest["onnx_sha256"],
                        "backend": model.backend,
                        "image_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                        "inference_wall_ms": (time.monotonic() - start) * 1000,
                    },
                )
            except (ValueError, KeyError, TypeError) as error:
                return self.send(400, {"error": str(error), "detections": []})

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deployment", required=True)
    p.add_argument("--port", type=int, default=8039)
    p.add_argument("--allow-smoke", action="store_true")
    p.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    p.add_argument("--perception-dir", type=Path, required=True)
    p.add_argument("--require-detector-sha256", default=DETECTOR_SHA256,
                   help="tests only substitute a stub detector; the client always passes the default")
    a = p.parse_args()
    folder = a.perception_dir.resolve()
    sources = {"serve_detector.py": sha((folder / "serve_detector.py").read_bytes()),
               "detector.py": sha((folder / "detector.py").read_bytes())}
    if sources["serve_detector.py"] != ORIGINAL_SERVER_SHA256 or sources["detector.py"] != a.require_detector_sha256:
        raise SystemExit("Detector sources differ from the reviewed versions: " + json.dumps(sources))
    sys.path.insert(0, str(folder))
    from detector import BetaDetector

    model = BetaDetector(a.deployment, allow_smoke=a.allow_smoke, backend=a.backend, device=a.device)
    for _ in range(2):
        model.detect(np.zeros(SHAPE, dtype=np.uint8))  # Startup warmup, not a scored observation.
    identity = dict(protocol=PROTOCOL, shape=list(SHAPE), dtype="uint8", bytes=NBYTES,
                    server_sha256=sha(Path(__file__).read_bytes()), mirrored_sources_sha256=sources,
                    lossless=True, route="/detect-raw-rgb")

    def stop(*args):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    server = HTTPServer(("127.0.0.1", a.port), make_handler(model, identity))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
