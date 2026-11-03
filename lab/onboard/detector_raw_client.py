"""Client of candidate `detector-raw-rgb`: football/detector_client.DetectorSession with raw RGB transport.

`make_session_class(detector_client)` returns a subclass of the course DetectorSession class (the
module object the executor imports; sha256 b7cd0ddc... is required). Its constructor repeats the
course checks in the same order (backend/device pairing, runtime/profile/smoke, weights and ONNX
hashes, owned child process, health check of weights/ONNX/backend/device, start-up timeout, close on
failure) and starts serve_detector_raw.py with the same detector Python and arguments.

detect(rgb) keeps the course input contract (uint8, (540, 960, 3)), sends the array's exact bytes
with their SHA-256, and applies the same response checks: weights SHA, ONNX SHA,
teacher_inputs False, and image_sha256 equal to sha256(a.tobytes()). The returned dictionary has the
same keys as the course client's (client_wall_ms included), so detections.jsonl rows keep their form.
Errors keep their classes: bad input ValueError before any request, HTTP 400/404 urllib HTTPError
(an OSError), provenance mismatch ValueError. close() is the course method.
"""
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COURSE_CLIENT_SHA256 = "b7cd0ddc358f18e4446cfb644dfcab6f44c6b8fd4eb8ace91eb435eb419b732f"
PROTOCOL = "microduck.detector.raw-rgb.v1"
SHAPE = (540, 960, 3)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_session_class(detector_client, *, server=HERE / "serve_detector_raw.py",
                       detector_sha256=None, python=None):
    """Subclass of detector_client.DetectorSession using the raw transport.

    detector_sha256/python are for CPU tests with a stub model only; production leaves them None.
    """
    actual = sha_file(detector_client.__file__)
    if actual != COURSE_CLIENT_SHA256:
        raise RuntimeError("Course detector_client.py differs from the reviewed version: " + actual)
    base = detector_client.DetectorSession

    class RawDetectorSession(base):
        transport = PROTOCOL

        def __init__(self, deployment, output, *, startup_timeout_s=90.0, backend="onnx", device="cpu",
                     perception_dir=None):
            self.manifest = json.loads(Path(deployment).read_text())
            self.output = Path(output)
            self.output.mkdir(parents=True, exist_ok=False)
            self.proc = None
            self.log = (self.output / "server.log").open("x")
            self.calls = 0
            self.backend = backend
            self.device = device
            self.transport_identity = None
            if (
                backend not in ("onnx", "torch")
                or device not in ("cpu", "cuda:0")
                or (backend == "onnx" and device != "cpu")
            ):
                raise ValueError("Use verified ONNX CPU or Torch CUDA/CPU pairing")
            if (
                self.manifest["runtime_id"] != "newton-bam-20261019"
                or self.manifest["hardware_profile_id"] != "beta-gc2093-l8cx-v1"
                or self.manifest["smoke_only"]
            ):
                raise ValueError("Full matching Beta detector required")
            for key in ("weights", "onnx"):
                if (
                    hashlib.sha256(Path(self.manifest[key]).read_bytes()).hexdigest()
                    != self.manifest[key + "_sha256"]
                ):
                    raise ValueError("Detector artifact hash mismatch")
            folder = Path(perception_dir) if perception_dir else Path(detector_client.__file__).resolve().parent.parent / "perception"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            self.url = f"http://127.0.0.1:{port}"
            env = dict(os.environ)
            env["OMP_NUM_THREADS"] = "4"
            if device == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            env["OPENBLAS_NUM_THREADS"] = "4"
            command = [
                str(python or folder / ".venv/bin/python"), str(server),
                "--deployment", str(Path(deployment).resolve()), "--port", str(port),
                "--backend", backend, "--device", device, "--perception-dir", str(folder),
            ]
            if detector_sha256:
                command += ["--require-detector-sha256", detector_sha256]
            self.proc = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT, env=env)
            try:
                until = time.monotonic() + startup_timeout_s
                while time.monotonic() < until:
                    if self.proc.poll() is not None:
                        raise RuntimeError("Owned CPU detector exited during startup")
                    try:
                        with urllib.request.urlopen(self.url + "/health", timeout=1) as response:
                            self.health = json.load(response)
                        if (
                            self.health["weights_sha256"] != self.manifest["weights_sha256"]
                            or self.health["onnx_sha256"] != self.manifest["onnx_sha256"]
                            or self.health["backend"] != backend
                            or self.health["device"] != device
                        ):
                            raise ValueError("Wrong running detector")
                        break
                    except OSError:
                        time.sleep(0.1)
                else:
                    raise TimeoutError("Owned CPU detector startup timeout")
                with urllib.request.urlopen(self.url + "/transport", timeout=5) as response:
                    self.transport_identity = json.load(response)
                if (
                    self.transport_identity.get("protocol") != PROTOCOL
                    or self.transport_identity.get("server_sha256") != sha_file(server)
                ):
                    raise ValueError("Wrong raw detector transport")
            except BaseException:
                self.close()
                raise

        def detect(self, rgb):
            a = np.asarray(rgb)
            if a.dtype != np.uint8 or a.shape != SHAPE:
                raise ValueError("Real RGB960x540 required")
            raw = a.tobytes()  # C order, exactly the bytes the course client hashes
            digest = hashlib.sha256(raw).hexdigest()
            request = urllib.request.Request(
                self.url + "/detect-raw-rgb",
                data=raw,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Microduck-RGB-SHA256": digest,
                    "X-Microduck-RGB-Shape": "540,960,3",
                    "X-Microduck-RGB-Dtype": "uint8",
                },
            )
            start = time.monotonic()
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
            if (
                result["weights_sha256"] != self.manifest["weights_sha256"]
                or result["onnx_sha256"] != self.manifest["onnx_sha256"]
                or result["image_sha256"] != digest
                or result["teacher_inputs"] is not False
            ):
                raise ValueError("Detector image/model provenance mismatch")
            result["client_wall_ms"] = (time.monotonic() - start) * 1000
            self.calls += 1
            return result

    RawDetectorSession.__qualname__ = "RawDetectorSession"
    return RawDetectorSession
