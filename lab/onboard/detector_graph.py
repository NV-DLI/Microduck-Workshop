"""Candidate `detector-cuda-graph`: CUDA-graph replay of the Ultralytics AutoBackend.forward (opt-in).

Installed on a constructed, warmed-up perception/detector.py BetaDetector (sha a5d21491...):
BetaDetector.detect (preprocessing, imgsz 640, rect False, conf 0.45, BGR flip, clamping, class
names, row order, image hash in the server) is unchanged. Only the instance method `forward` of the
predictor's AutoBackend (ultralytics 8.4.182; nn/autobackend.py:302-352) is replaced:

  eligible call  = forward(im) with augment False, embed None, no other kwargs, im a contiguous CUDA
                   float32 tensor of shape (1, 3, 640, 640), inference mode on, no capture running,
                   and precision flags equal to those at capture;
                   -> copy im into the static input, replay the graph, return a CLONE of the static
                   output tree (tuple/list/dict of tensors), so retained results are never
                   overwritten by later replays.
  anything else  -> the original forward, counted with its reason.

Capture (install time, after the server's two normal warm-ups, before any request): 3 more original
calls on a side stream, then torch.cuda.graph(capture_error_mode="thread_local") of the same
original forward on a zero input, then a replay compared bit-exactly with the eager output. Any
failure leaves the original forward in place and records the reason. No half precision, TF32, compile,
fusion or operator change: the captured kernels are those of the original call.

Why capture is expected to work (source): the end-to-end Detect head caches anchors once the input
shape is fixed (nn/modules/head.py:219-224, set during the warm-ups), top-k uses a static k =
min(max_det, anchors) (head.py:266-287), and the pt backend calls the module directly
(nn/backends/pytorch.py:124-134). Host-side Python in AutoBackend.forward (metadata, batch/pad
checks, from_numpy) is evaluated once at capture; the eligibility test guarantees the same branch.
If the installed graph contains a host sync or dynamic shape, capture raises and the original stays.

Bit-exact detection rows on real frames must be shown by root on the GPU (gpu_acceptance.py) before
integration. shadow_calls=N compares replay and original for the first N calls; the first mismatch
disables the graph for the rest of the run and returns the original output.
"""
import hashlib
import time
from pathlib import Path

import torch

DETECTOR_SHA256 = "a5d2149143960aac7935786c4c2017e0e259ddecf8598ab989bfbc58a2583d9e"
ULTRALYTICS_VERSION = "8.4.182"
ULTRALYTICS_SHA256 = {
    "nn/autobackend.py": "62e7d83beea56911691d58dacca3c9f20319920fbb2e2b07309c963946b707d6",
    "engine/predictor.py": "9b29a5322293167a705a6d6054ef653a62d27f6e4bff589b242946df89b5b62f",
    "nn/modules/head.py": "b471bb8a39cc1e38672c20feaa8c28bdf43739adb8eafd14226807b34b31f3ee",
    "nn/backends/pytorch.py": "aabae215c5474913be5887ce64c1d832f5f60b0e3c49c89a058fd3dfc91e03b3",
}
SHAPE = (1, 3, 640, 640)


def precision_flags():
    return dict(
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )


def clone_tree(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(clone_tree(v) for v in value)
    if isinstance(value, list):
        return [clone_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: clone_tree(v) for k, v in value.items()}
    return value


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    return a == b


class TorchCudaGraph:
    def __init__(self, fn, static_input, warmup=3):
        side = torch.cuda.Stream(device=static_input.device)
        side.wait_stream(torch.cuda.current_stream(static_input.device))
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn(static_input)
        torch.cuda.current_stream(static_input.device).wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            self.static_output = fn(static_input)
        torch.cuda.synchronize(static_input.device)

    def replay(self):
        self.graph.replay()
        return self.static_output


def identity_checks(beta_detector, detector_file, ultralytics_module):
    root = Path(ultralytics_module.__file__).resolve().parent
    files = {rel: hashlib.sha256((root / rel).read_bytes()).hexdigest() for rel in ULTRALYTICS_SHA256}
    return dict(
        detector_sha256_ok=hashlib.sha256(Path(detector_file).read_bytes()).hexdigest() == DETECTOR_SHA256,
        ultralytics_version_ok=getattr(ultralytics_module, "__version__", None) == ULTRALYTICS_VERSION,
        ultralytics_files_ok={rel: files[rel] == ULTRALYTICS_SHA256[rel] for rel in files},
        backend_torch=beta_detector.backend == "torch",
        weights_sha256=beta_detector.manifest.get("weights_sha256"),
    )


def install(beta_detector, *, detector_file, ultralytics_module, shadow_calls=0, graph_backend=TorchCudaGraph,
            require_cuda=True, checks=None):
    """Replace the warmed predictor's AutoBackend.forward; returns the JSON-safe record (always)."""
    record = dict(candidate="detector-cuda-graph", active=False, reason=None, capture_wall_s=None, calls=0,
                  graph_calls=0, eager_calls=0, eager_reasons={}, shadow_calls=shadow_calls, shadow_compared=0,
                  shadow_mismatches=0, flags_at_capture=None)
    try:
        checks = checks if checks is not None else identity_checks(beta_detector, detector_file, ultralytics_module)
        record["checks"] = checks
        if not (checks["detector_sha256_ok"] and checks["ultralytics_version_ok"]
                and all(checks["ultralytics_files_ok"].values()) and checks["backend_torch"]):
            raise RuntimeError("source/backend identity checks failed")
        predictor = getattr(beta_detector.model, "predictor", None)
        if predictor is None or not getattr(predictor, "done_warmup", False):
            raise RuntimeError("install only after the detector's normal warm-up calls")
        backend = predictor.model
        if type(backend).__name__ != "AutoBackend":
            raise RuntimeError("predictor model is not a plain AutoBackend (compiled or wrapped)")
        props = dict(format=getattr(backend, "format", None), fp16=bool(getattr(backend, "fp16", True)),
                     nhwc=bool(getattr(backend, "nhwc", True)), batch=getattr(backend, "batch", None),
                     device=str(getattr(backend, "device", None)))
        record["backend"] = props
        if props["format"] != "pt" or props["fp16"] or props["nhwc"] or props["batch"] != 1:
            raise RuntimeError("AutoBackend outside the captured contract: " + str(props))
        device = torch.device(props["device"])
        if require_cuda and device.type != "cuda":
            raise RuntimeError("detector device is not CUDA")
        original = backend.forward  # bound method of the instance's class
        start = time.perf_counter()
        with torch.inference_mode():
            static_input = torch.zeros(SHAPE, dtype=torch.float32, device=device)
            flags = precision_flags()
            graph = graph_backend(lambda x: original(x), static_input)
            static_input.zero_()
            expected = original(static_input)
            replayed = clone_tree(graph.replay())
            if not equal_tree(expected, replayed):
                raise RuntimeError("graph replay differs from the original forward on the zero input")
        record["capture_wall_s"] = time.perf_counter() - start
    except Exception as error:
        record["reason"] = "not installed: " + repr(error)
        return record

    state = dict(graph=graph)

    def ineligible(im, augment, embed, kwargs):
        if state["graph"] is None:
            return "graph inactive"
        if kwargs or augment or embed is not None:
            return "augment/embed/kwargs outside capture"
        if not isinstance(im, torch.Tensor) or tuple(im.shape) != SHAPE or im.dtype != torch.float32:
            return "input shape/dtype outside capture"
        if im.device != static_input.device or not im.is_contiguous():
            return "input device/layout outside capture"
        if not torch.is_inference_mode_enabled():
            return "not in inference mode"
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return "stream capture in progress"
        if precision_flags() != flags:
            return "precision flags changed since capture"
        return None

    def forward(im, augment=False, embed=None, **kwargs):
        record["calls"] += 1
        reason = ineligible(im, augment, embed, kwargs)
        if reason is not None:
            record["eager_calls"] += 1
            record["eager_reasons"][reason] = record["eager_reasons"].get(reason, 0) + 1
            return original(im, augment=augment, embed=embed, **kwargs)
        static_input.copy_(im)
        result = clone_tree(state["graph"].replay())
        if record["shadow_compared"] < record["shadow_calls"]:
            expected = original(im)
            record["shadow_compared"] += 1
            if not equal_tree(result, expected):
                record["shadow_mismatches"] += 1
                record.update(active=False, reason="shadow mismatch: graph disabled")
                state["graph"] = None
                record["eager_calls"] += 1
                return expected
        record["graph_calls"] += 1
        return result

    forward.__wrapped__ = original
    backend.forward = forward  # instance attribute; nn.Module.__call__ dispatches to it
    record.update(active=True, flags_at_capture=flags)
    return record
