"""Owned verified ONNX/Torch detector child: rendered RGB is the only observation input."""

from pathlib import Path
import base64, hashlib, json, os, socket, subprocess, time, urllib.request
import cv2, numpy as np


class DetectorSession:
    def __init__(
        self,
        deployment,
        output,
        *,
        startup_timeout_s=90.0,
        backend="onnx",
        device="cpu",
    ):
        self.manifest = json.loads(Path(deployment).read_text())
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.proc = None
        self.log = (self.output / "server.log").open("x")
        self.calls = 0
        self.backend = backend
        self.device = device
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
        folder = Path(__file__).resolve().parent.parent / "perception"
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = "4"
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        env["OPENBLAS_NUM_THREADS"] = "4"
        self.proc = subprocess.Popen(
            [
                str(folder / ".venv/bin/python"),
                str(folder / "serve_detector.py"),
                "--deployment",
                str(Path(deployment).resolve()),
                "--port",
                str(port),
                "--backend",
                backend,
                "--device",
                device,
            ],
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            until = time.monotonic() + startup_timeout_s
            while time.monotonic() < until:
                if self.proc.poll() is not None:
                    raise RuntimeError("Owned CPU detector exited during startup")
                try:
                    with urllib.request.urlopen(
                        self.url + "/health", timeout=1
                    ) as response:
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
        except BaseException:
            self.close()
            raise

    def detect(self, rgb):
        a = np.asarray(rgb)
        if a.dtype != np.uint8 or a.shape != (540, 960, 3):
            raise ValueError("Real RGB960x540 required")
        ok, png = cv2.imencode(".png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("RGB encode failed")
        request = urllib.request.Request(
            self.url + "/detect",
            data=json.dumps(
                {"rgb_png_base64": base64.b64encode(png).decode()}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.monotonic()
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
        if (
            result["weights_sha256"] != self.manifest["weights_sha256"]
            or result["onnx_sha256"] != self.manifest["onnx_sha256"]
            or result["image_sha256"] != hashlib.sha256(a.tobytes()).hexdigest()
            or result["teacher_inputs"] is not False
        ):
            raise ValueError("Detector image/model provenance mismatch")
        result["client_wall_ms"] = (time.monotonic() - start) * 1000
        self.calls += 1
        return result

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        self.log.close()
        return {
            "owned_detector_pid": None if self.proc is None else self.proc.pid,
            "terminated": self.proc is None or self.proc.poll() is not None,
            "requests": self.calls,
        }
