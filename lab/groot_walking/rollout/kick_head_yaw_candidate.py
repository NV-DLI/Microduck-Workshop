"""CPU-prepared candidate: frozen K10 composition with active standing head-yaw clamp.

Only raw index7 changes, to already-computed official standing yaw clipped at
+/-0.15. Current validated execution mapping is scale1/defaultyaw0/bias0;
this bounds the requested target, not the measured joint. New physical results
are required. Original network/deployment identities and authority gates remain.

Untrained early-return composition; public preparation/authorization belongs to caller.

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
    @staticmethod
    def verify_action_mapping(term):
        """Read-only mapping check before using the +/-0.15 raw-yaw candidate.

        Only the known affine mapping is supported; no state or target is reset.
        Scalar scale/offset or per-environment Nx14 terms are accepted.
        """
        names = list(term._joint_names)
        if len(names) != 14 or names[7] != "head_yaw":
            raise ValueError("Exact14 action order with head_yaw at index7 required")

        def checked_yaw(value, label, expected):
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            array = np.asarray(value)
            if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
                raise ValueError(label + " must be finite numeric values")
            if array.ndim == 0:
                yaw = array.reshape(1)
            elif array.ndim == 2 and array.shape[0] >= 1 and array.shape[1] == 14:
                yaw = array[:, 7]
            else:
                raise ValueError(label + " must be scalar or nonempty Nx14")
            if not np.all(yaw == expected):
                raise ValueError(label + " head_yaw mapping differs from candidate")
            return {"shape": list(array.shape), "yaw_values": yaw.astype(float).tolist()}

        scale = checked_yaw(term._scale, "action_scale", 1.0)
        offset = checked_yaw(term._offset, "default_offset", 0.0)
        bias = checked_yaw(term.encoder_bias, "encoder_bias", 0.0)
        return {
            "verified": True,
            "joint_name": "head_yaw",
            "policy_joint_index": 7,
            "action_joint_names": names,
            "scale": scale,
            "offset": offset,
            "encoder_bias": bias,
            "physical_target_formula": "raw*scale+offset-encoder_bias",
            "raw_yaw_bound_rad": 0.15,
            "requested_target_yaw_bound_rad": 0.15,
            "actual_encoder_bound_guaranteed": False,
            "candidate_source": str(Path(__file__).resolve()),
            "candidate_source_sha256": sha(Path(__file__)),
        }

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
        self.source_sha256[str(Path(__file__).resolve())] = sha(Path(__file__))
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
                (phase - np.float32(0.14)) / np.float32(0.12),
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
                mode="k10_early_return_parameters",
                composition_hold_s=0.14,
                composition_blend_s=0.12,
                original_composition_changed=True,
                model_weights_changed=False,
            )
        else:
            result.update(
                raw14=standing.copy(), teacher14=standing, mode="standing_only"
            )
        # Explicit candidate execution adapter; never mutate teacher/residual.
        # The caller must feed this final raw14 as the next actual history.
        result["raw14_before_head_yaw_adapter"] = result["raw14"].copy()
        result["head_yaw_execution_adapter"] = {
            "kind": "active_standing_head_yaw_clamp_v1",
            "applied": active,
            "policy_joint_index": 7,
            "joint_name": "head_yaw",
            "raw_bound_rad": 0.15,
            "expected_action_scale": 1.0,
            "expected_default_yaw_rad": 0.0,
            "expected_encoder_bias_rad": 0.0,
            "physical_mapping_must_be_checked_by_caller": True,
            "actual_joint_bound_guaranteed": False,
            "qualification_claimed": False,
            "model_weights_changed": False,
        }
        if active:
            candidate = result["raw14"].copy()
            candidate[7] = np.clip(standing[7], np.float32(-0.15), np.float32(0.15))
            result["raw14"] = candidate
            result["head_yaw_execution_adapter"].update(
                standing_raw_yaw=float(standing[7]),
                original_raw_yaw=float(result["raw14_before_head_yaw_adapter"][7]),
                sent_raw_yaw=float(candidate[7]),
            )
        result["actor62"] = actor
        return result
