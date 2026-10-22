"""Current-time status for the frozen public ball memory; no motor authority.

Call consume(packet, now_s=...) for a delivered packet and poll(now_s=...) on
every caller tick with no new packet. consume(None, ...) is identical to poll.
Only memory_readback is the current-time answer. read_time_memory_readback is
a retained diagnostic, never current permission. The caller clock must be the
same monotonic simulation/hardware clock as the packet, not wall-clock time.

This standalone candidate is not installed in a football executor. It keeps the
v1 packet validation and numerical integration unchanged, without modifying v1
or memory.py. It provides no kick authority, calibrated error bound, or proof
that an unseen ball is stationary.
"""
import copy
import hashlib
import importlib.util
import math
from pathlib import Path
import sys

DIRECTORY = Path(__file__).resolve().parent
PINS = {
    DIRECTORY / 'public_ball_read_adapter_v1.py': '6a4667dbe20daed4276da590d286a5d21aa8606b7d0e240f960b8d547f3d8a9b',
    DIRECTORY.parent / 'football/public_ball_memory_v1/memory.py': '6b268da8af9bd0351cb5204809eba247fab2409f53b03fc859b953f087cc0bb6',
}
for path, expected in PINS.items():
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise RuntimeError('Frozen public memory dependency changed: ' + str(path))
spec = importlib.util.spec_from_file_location('frozen_public_read_adapter_parent_v1', DIRECTORY / 'public_ball_read_adapter_v1.py')
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)


class SameReadMemoryAdapter(parent.SameReadMemoryAdapter):
    """Preserve read integration, explicitly evaluate every return at now_s."""
    def __init__(self, leg_fk, joint_names):
        super().__init__(leg_fk, joint_names)
        self.last_caller_timestamp_s = None
        self._last_packet_readback = None
        self._read_time_diagnostic = None

    def _clock(self, now_s):
        try:
            if isinstance(now_s, bool):
                raise ValueError('Boolean is not a caller timestamp')
            now = float(now_s)
            if not math.isfinite(now) or now < 0:
                raise ValueError('Finite nonnegative caller timestamp required')
            if self.last_caller_timestamp_s is not None and now < self.last_caller_timestamp_s:
                raise ValueError('Caller clock must not go backwards')
        except (TypeError, ValueError, OverflowError):
            self.memory.invalidate('invalid_or_reversed_caller_clock')
            raise ValueError('Finite nonnegative monotonic caller timestamp required') from None
        return now

    def consume(self, packet, *, now_s):
        now = self._clock(now_s)
        if packet is None:
            result = {'integrated': False, 'reason': 'no_new_delivered_read',
                      'seq': self.last_sequence, 'sequence_gap': False}
        else:
            previous_sequence = self.last_sequence
            # v1 rejects future acquisition/read/delivery and changes to a
            # repeated sequence before performing any memory integration.
            result = super().consume(packet, now_s=now)
            if self.last_sequence != previous_sequence:
                self._last_packet_readback = {
                    k: copy.deepcopy(result[k])
                    for k in ('actual_packet', 'input_sha256', 'source_gyro_clipped', 'source_accel_clipped')
                }
                self._read_time_diagnostic = self.memory.status(self.last_read_s)

        # An expired diagnostic must never be presented as current availability,
        # including duplicate and no-packet calls. status does not integrate.
        current = self.memory.status(now)
        self.last_caller_timestamp_s = now
        if self._last_packet_readback is not None:
            result.update(copy.deepcopy(self._last_packet_readback))
            original = result['actual_packet']
            ages = {
                'sample_age_s': now - original['sample_timestamp_s'],
                'read_age_s': now - original['read_timestamp_s'],
                'delivery_age_s': now - original['delivered_timestamp_s'],
            }
        else:
            ages = None
        result.update(
            schema='microduck.public-ball-read-adapter.v2',
            memory_readback=current,
            status_evaluated_at_s=now,
            source_packet_ages_at_call=ages,
            read_time_memory_readback=copy.deepcopy(self._read_time_diagnostic),
            read_time_diagnostic_timestamp_s=self.last_read_s,
            read_time_diagnostic_is_current_permission=False,
            kick_authorized=False,
            timestamp_semantics='Sample/read/delivery remain original. memory_readback is evaluated at caller now_s; the retained read-time diagnostic is not current authority.',
        )
        return result

    def poll(self, *, now_s):
        """Mandatory current-time check on a caller tick without new delivery."""
        return self.consume(None, now_s=now_s)
