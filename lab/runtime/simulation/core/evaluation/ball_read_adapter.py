"""CPU-only same-read adapter for the frozen public ball memory.

This is not installed in the current football controller. A future entry may
call it after BodyImuInput.ingest with the actual delivered packet, and must
record the returned readback. No simulator or camera timestamp is substituted.
"""

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / "football/ball_memory/memory.py"
spec = importlib.util.spec_from_file_location(
    "frozen_public_ball_memory_for_read_adapter_v1", SOURCE
)
memory_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = memory_module
spec.loader.exec_module(memory_module)


class SameReadMemoryAdapter:
    def __init__(self, leg_fk, joint_names):
        self.joint_names = tuple(joint_names)
        self.memory = memory_module.BallMemory(leg_fk, self.joint_names)
        self.last_sequence = None
        self.last_fingerprint = None
        self.last_read_s = None
        self.delivered_reads = 0

    def consume(self, packet, *, now_s):
        """Consume BodyImuInput.latest or observe() without re-quantization."""
        try:
            if (
                packet["source"] != "body_powerboard_control_imu"
                or packet["axes"] != "trunk FLU"
            ):
                raise ValueError("Delivered body sensor source required")
            sequence = packet["seq"]
            if type(sequence) is not int or sequence < 0:
                raise ValueError("Nonnegative integral sequence required")
            sample_s = float(packet["timestamp_s"])
            read_s = float(packet["read_timestamp_s"])
            delivery_s = float(packet["delivered_timestamp_s"])
            encoder_s = float(packet["encoder_read_timestamp_s"])
            if not all(
                math.isfinite(t) and t >= 0
                for t in (sample_s, read_s, delivery_s, encoder_s, now_s)
            ):
                raise ValueError("Finite actual timestamps required")
            if (
                not sample_s <= read_s + 1e-9
                or not read_s <= delivery_s + 1e-9
                or not delivery_s <= now_s + 1e-9
            ):
                raise ValueError("Causal acquisition/read/delivery required")
            if abs(read_s - encoder_s) > 1e-7:
                raise ValueError("Same-read encoder fields required")
            read = memory_module.PublicRead(
                read_s,
                encoder_s,
                packet["orientation_valid"],
                tuple(packet["encoder_q_rad"]),
                tuple(packet["encoder_qd_rad_s"]),
                tuple(packet["gyro"]),
                tuple(packet["gravity"]),
                self.joint_names,
            )
            payload = dict(
                seq=sequence,
                sample_timestamp_s=sample_s,
                read_timestamp_s=read_s,
                delivered_timestamp_s=delivery_s,
                encoder_timestamp_s=encoder_s,
                orientation_valid=read.orientation_valid,
                q_rad=read.q_rad,
                qd_rad_s=read.qd_rad_s,
                gyro_body_rad_s=read.gyro_body_rad_s,
                gravity_body=read.gravity_body,
                joint_names=read.joint_names,
            )
            digest = hashlib.sha256(
                json.dumps(payload, sort_keys=True, allow_nan=False).encode()
            ).hexdigest()
            if self.last_sequence == sequence:
                if self.last_fingerprint != digest:
                    raise ValueError("Repeated sequence changed public payload")
                return {
                    "integrated": False,
                    "reason": "same_delivered_read",
                    "seq": sequence,
                    "read_timestamp_s": read_s,
                    "kick_authorized": False,
                }
            if self.last_sequence is not None and (
                sequence < self.last_sequence or read_s <= self.last_read_s
            ):
                raise ValueError("Delivered read must advance")
        except (KeyError, TypeError, ValueError):
            self.memory.invalidate("invalid_public_packet_adapter_input")
            raise
        sequence_gap = (
            self.last_sequence is not None and sequence != self.last_sequence + 1
        )
        if sequence_gap:
            self.memory.invalidate("delivered_sequence_gap_requires_new_ball")
        self.last_sequence, self.last_read_s, self.last_fingerprint = (
            sequence,
            read_s,
            digest,
        )
        self.delivered_reads += 1
        if read.orientation_valid is not True:
            self.memory.invalidate("delivered_orientation_invalid")
            status = self.memory.status(now_s)
            integrated = False
        else:
            status = self.memory.update(read)
            integrated = True
        return {
            "integrated": integrated,
            "seq": sequence,
            "sequence_gap": sequence_gap,
            "actual_packet": payload,
            "input_sha256": digest,
            "source_gyro_clipped": packet.get("gyro_clipped"),
            "source_accel_clipped": packet.get("accel_clipped"),
            "memory_readback": status,
            "kick_authorized": False,
            "timestamp_semantics": "PublicRead uses the hardware read time; original sample/delivery times remain separately recorded.",
        }
