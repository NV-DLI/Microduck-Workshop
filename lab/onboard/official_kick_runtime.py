"""Official kick and standing policies for the onboard football task.

Same call contract and validation as the installed course composition
(evaluation/football/kick_runtime.py, "k10_early_return_parameters"): one 50 Hz
call consumes the supplied same-read BODY50 61D observation, the caller owns
authorization, the elapsed phase and termination. The kick teacher holds for
0.14 s, then a smoothstep over 0.12 s hands over to the standing teacher, which
continues to the end of the caller's 4 s window.

Differences from the installed composition, and nothing else:
  * the course-trained residual network, its checkpoint, the training ledger and
    the K10 deployment file are never opened or inferred;
  * raw14 == teacher14 exactly; residual14 and delta14 are zero vectors;
  * mode/identity say "official_teacher_only_no_residual".

This is not the official 3 s demo timing and not the course K10 controller.
"""

import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

MODE = "official_teacher_only_no_residual"
# Frozen Pollen community v8 networks used by the course (velstand doubles as the
# onboard walking actor; ball_kick_right is the kick teacher).
FROZEN_SHA256 = {
    "velstand.onnx": "2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492",
    "ball_kick_right.onnx": "147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af",
}
HOLD_S = 0.14
BLEND_S = 0.12


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class OfficialTeacherKickRuntime:
    MODE = MODE

    def __init__(self, standing_onnx, kick_onnx):
        standing_onnx, kick_onnx = Path(standing_onnx), Path(kick_onnx)
        for path, name in ((standing_onnx, "velstand.onnx"), (kick_onnx, "ball_kick_right.onnx")):
            if sha(path) != FROZEN_SHA256[name]:
                raise ValueError("Frozen official teacher changed: " + str(path))
        self.source_sha256 = {
            str(Path(__file__).resolve()): sha(__file__),
            str(standing_onnx): FROZEN_SHA256["velstand.onnx"],
            str(kick_onnx): FROZEN_SHA256["ball_kick_right.onnx"],
        }
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.sessions = [
            ort.InferenceSession(str(p), options, providers=["CPUExecutionProvider"])
            for p in (standing_onnx, kick_onnx)
        ]
        for session in self.sessions:
            if (
                len(session.get_inputs()) != 1
                or len(session.get_outputs()) != 1
                or session.get_inputs()[0].shape[-1] != 61
                or session.get_outputs()[0].shape[-1] != 14
            ):
                raise ValueError("Unexpected network shape")
        self.network_calls = [0, 0]
        self.identity = dict(
            mode=MODE,
            networks=["velstand.onnx (standing)", "ball_kick_right.onnx (kick)"],
            composition_hold_s=HOLD_S,
            composition_blend_s=BLEND_S,
            blend="smoothstep weight on kick teacher, float32, as installed course composition",
            course_residual_network_loaded=False,
            course_residual_checkpoint_loaded=False,
            course_training_ledger_loaded=False,
            k10_deployment_file_loaded=False,
            model_weights_changed=False,
            official_demo_timing=False,
            qualification_claimed=False,
        )

    @classmethod
    def from_installed(cls, materialized_root):
        m = Path(materialized_root)
        return cls(
            m / "models/reference/velstand.onnx",
            m / "football/sources/ball_kick_right.onnx",
        )

    def _infer(self, index, observation):
        session = self.sessions[index]
        self.network_calls[index] += 1
        result = session.run(None, {session.get_inputs()[0].name: observation[None]})[0][0]
        if result.shape != (14,) or not np.isfinite(result).all():
            raise ValueError("Nonfinite or malformed network output")
        return result

    def step(self, body61, *, previous_raw14, phase_s, ready, authorized, head_return_complete):
        """Same arguments, validation and result keys as the installed composition."""
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
        if history.dtype != np.float32 or history.shape != (14,) or not np.isfinite(history).all():
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
            "course_residual_network_loaded": False,
        }
        if not ready:
            return result
        if not np.isfinite(obs).all():
            raise ValueError("Ready observation must be entirely finite")
        if not np.array_equal(obs[34:48], history):
            raise ValueError("History must equal the actual preceding sent raw action")
        if active and np.any(obs[48:61] != 0):
            raise ValueError("Active kick requires zero base3/head4/body6 commands")
        # Diagnostic only, kept for log compatibility; no network consumes it.
        actor = np.concatenate((obs, np.array([np.clip(phase, -1, 4) / np.float32(4)], np.float32)))
        standing = self._infer(0, obs)
        if active:
            kicking = self._infer(1, obs)
            u = np.clip((phase - np.float32(HOLD_S)) / np.float32(BLEND_S), np.float32(0), np.float32(1))
            weight = np.float32(1) - u * u * (np.float32(3) - np.float32(2) * u)
            teacher = weight * kicking + (np.float32(1) - weight) * standing
            result.update(
                raw14=teacher.copy(),
                teacher14=teacher,
                kick_weight=float(weight),
                mode=MODE,
                composition_hold_s=HOLD_S,
                composition_blend_s=BLEND_S,
                original_composition_changed=True,
                model_weights_changed=False,
            )
        else:
            result.update(raw14=standing.copy(), teacher14=standing, mode="standing_only")
        result["actor62"] = actor
        result["actor62_consumed_by_network"] = False
        return result
