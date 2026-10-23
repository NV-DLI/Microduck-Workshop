"""Frozen K10 inference only; public preparation/authorization belongs to caller.

One 50 Hz call consumes the supplied same-read BODY50 observation. This adapter
has no simulator, vision, ball state, contact state, or authority to start a kick.
The caller must feed the raw action actually sent back as next-step history.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import onnxruntime as ort


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NativeKickRuntime:
    def __init__(self, deployment_json):
        path = Path(deployment_json)
        self.deployment = json.loads(path.read_text())
        d = self.deployment
        if (
            d.get("schema") != "microduck.native-residual-kick-export.v10"
            or d.get("adapter")
            != "kick_training_v10_goal/native-raw-mixture-with-original-history"
            or d.get("input_dimension") != 62
            or d.get("output_dimension") != 14
        ):
            raise ValueError("Exact native K10 deployment contract required")
        if d["phase_contract"] != {
            "kick_s": 0.5,
            "recovery_blend_s": 0.4,
            "episode_s": 4.0,
            "phase_input": "clip(phase_seconds,-1,4)/4",
        }:
            raise ValueError("K10 clock contract changed")
        ledger = path.parent / "training-ledger.snapshot.json"
        if sha(ledger) != d["training_ledger_snapshot_sha256"]:
            raise ValueError("Frozen training ledger changed")
        teachers = d["teacher_sha256"]
        standing = [p for p in teachers if Path(p).name == "velstand.onnx"]
        kicking = [p for p in teachers if Path(p).name == "ball_kick_right.onnx"]
        if len(standing) != 1 or len(kicking) != 1 or len(teachers) != 2:
            raise ValueError("Exactly the two frozen teachers required")
        files = {
            **teachers,
            d["onnx"]: d["onnx_sha256"],
            d["checkpoint"]: d["checkpoint_sha256"],
        }
        if any(sha(p) != h for p, h in files.items()):
            raise ValueError("A bound policy/checkpoint changed")
        self.source_sha256 = {str(path): sha(path), str(ledger): sha(ledger), **files}
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.sessions = [
            ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"])
            for p in (standing[0], kicking[0], d["onnx"])
        ]
        for session, width in zip(self.sessions, (61, 61, 62)):
            if (
                len(session.get_inputs()) != 1
                or len(session.get_outputs()) != 1
                or session.get_inputs()[0].shape[-1] != width
                or session.get_outputs()[0].shape[-1] != 14
            ):
                raise ValueError("Unexpected network shape")
        self.network_calls = [0, 0, 0]

    def _infer(self, index, observation):
        session = self.sessions[index]
        self.network_calls[index] += 1
        result = session.run(None, {session.get_inputs()[0].name: observation[None]})[
            0
        ][0]
        if result.shape != (14,) or not np.isfinite(result).all():
            raise ValueError("Nonfinite or malformed network output")
        return result

    def step(
        self,
        body61,
        *,
        previous_raw14,
        phase_s,
        ready,
        authorized,
        head_return_complete
    ):
        """Return native raw action and diagnostics, without changing any input.

        phase_s=-1 is preparation; phase_s in [0,4] is the caller's elapsed
        authorized window. No timestamp means no hidden automatic phase advance.
        Boolean flags must be actual booleans. A withdrawn authorization returns
        standing, so the supervisor owns safe abort handling and terminal logic.
        """
        for flag in (ready, authorized, head_return_complete):
            if not isinstance(flag, (bool, np.bool_)):
                raise ValueError("Explicit boolean authority/readiness flags required")
        if isinstance(phase_s, (bool, np.bool_)) or not np.isfinite(phase_s):
            raise ValueError("Finite public elapsed time required")
        phase = np.float32(phase_s)
        if not (phase_s == -1 or 0 <= phase_s <= 4):
            raise ValueError("Only preparation -1 or declared 0..4 s task window")
        obs = np.asarray(body61)
        history = np.asarray(previous_raw14)
        if obs.dtype != np.float32 or obs.shape != (61,):
            raise ValueError("FP32 original61D observation required")
        if (
            history.dtype != np.float32
            or history.shape != (14,)
            or not np.isfinite(history).all()
        ):
            raise ValueError("Actual finite FP32 previous raw14 required")
        active = bool(ready and authorized and head_return_complete and phase_s >= 0)
        result = {
            "raw14": np.zeros(14, np.float32),
            "teacher14": np.zeros(14, np.float32),
            "residual14": np.zeros(14, np.float32),
            "delta14": np.zeros(14, np.float32),
            "actor62": None,
            "kick_weight": 0.0,
            "active": active,
            "kick_authorized": False,
            "authority_owner": "external public-sensor supervisor",
            "phase_s": float(phase),
            "mode": "sensor_unready_raw_home",
        }
        # Invalid sensor rows never enter a network, matching the training mask.
        if not ready:
            return result
        if not np.isfinite(obs).all():
            raise ValueError("Ready observation must be entirely finite")
        if not np.array_equal(obs[34:48], history):
            raise ValueError("History must equal the actual preceding sent raw action")
        if active and np.any(obs[48:61] != 0):
            raise ValueError("Active K10 requires zero base3/head4/body6 commands")
        actor = np.concatenate(
            (obs, np.array([np.clip(phase, -1, 4) / np.float32(4)], np.float32))
        )
        standing = self._infer(0, obs)
        if active:
            kicking = self._infer(1, obs)
            residual = self._infer(2, actor)
            u = np.clip(
                (phase - np.float32(0.5)) / np.float32(0.4),
                np.float32(0),
                np.float32(1),
            )
            weight = np.float32(1) - u * u * (np.float32(3) - np.float32(2) * u)
            teacher = weight * kicking + (np.float32(1) - weight) * standing
            delta = np.float32(0.2) * np.tanh(residual)
            result.update(
                raw14=teacher + delta,
                teacher14=teacher,
                residual14=residual,
                delta14=delta,
                kick_weight=float(weight),
                mode="native_k10",
            )
        else:
            result.update(
                raw14=standing.copy(), teacher14=standing, mode="standing_only"
            )
        result["actor62"] = actor
        return result
