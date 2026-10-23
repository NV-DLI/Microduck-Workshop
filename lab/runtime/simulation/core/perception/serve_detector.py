#!/usr/bin/env python3
"""Optional independent CPU detector process; images only, no teacher labels."""
import argparse, base64, hashlib, json, signal, time
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2, numpy as np
from detector import BetaDetector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deployment", required=True)
    p.add_argument("--port", type=int, default=8039)
    p.add_argument("--allow-smoke", action="store_true")
    p.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    p.add_argument("--device", choices=["cpu", "cuda:0"], default="cpu")
    a = p.parse_args()
    model = BetaDetector(
        a.deployment, allow_smoke=a.allow_smoke, backend=a.backend, device=a.device
    )
    for _ in range(2):
        model.detect(
            np.zeros((540, 960, 3), dtype=np.uint8)
        )  # Startup warmup, not a scored observation.

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
            return (
                self.send(
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
                if self.path == "/health"
                else self.send(404, {"error": "unknown route"})
            )

        def do_POST(self):
            try:
                if self.path != "/detect":
                    return self.send(404, {"error": "unknown route"})
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 8_000_000:
                    raise ValueError("Bounded image request required")
                request = json.loads(self.rfile.read(n))
                if set(request) != {"rgb_png_base64"}:
                    raise ValueError(
                        "Only rendered RGB accepted; labels/world state forbidden"
                    )
                bgr = cv2.imdecode(
                    np.frombuffer(
                        base64.b64decode(request["rgb_png_base64"], validate=True),
                        np.uint8,
                    ),
                    cv2.IMREAD_COLOR,
                )
                if bgr is None:
                    raise ValueError("Invalid image")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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

    def stop(*args):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    server = HTTPServer(("127.0.0.1", a.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
