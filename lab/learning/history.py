"""Causal proprioception/request history; no time, object pose or phase input."""

from collections import deque
import math

MODE = "proprio_0_12_24_previous_request_v1"
LAGS = (0, 12, 24)


def names(base, actions):
    if len(base) != 37 or len(actions) != 4:
        raise ValueError("Expected base37/request4 contract")
    return [f"lag{lag}/{n}" for lag in LAGS for n in base] + [
        "previous_request/" + n for n in actions
    ]


def specification():
    return dict(
        mode=MODE,
        state_lags_samples=list(LAGS),
        sample_hz=12.5,
        padding="repeat first measured state within the same physical episode",
        previous_request="last actually executed high-level request; zero only at reset",
        reset="physical episode only; keep on authority changes",
        no_historical_images=True,
    )


class CausalHistory:
    def __init__(self):
        self.states = deque(maxlen=25)
        self.frame = 0

    def observe(self, state, previous_request, frame):
        if frame != self.frame:
            raise ValueError("Non-contiguous physical control frames")
        if len(state) != 37 or len(previous_request) != 4:
            raise ValueError("History feature shape mismatch")
        if not all(math.isfinite(float(v)) for v in [*state, *previous_request]):
            raise ValueError("Non-finite history features")
        if frame == 0 and any(float(v) != 0 for v in previous_request):
            raise ValueError("Reset previous request must be zero")
        self.states.append([float(v) for v in state])
        output = []
        for lag in LAGS:
            output.extend(self.states[max(0, len(self.states) - 1 - lag)])
        output.extend(float(v) for v in previous_request)
        self.frame += 1
        return output


def transform_rows(rows):
    memory = CausalHistory()
    previous = [0.0] * 4
    result = []
    for index, row in enumerate(rows):
        if row["frame"] != index or not math.isclose(
            row["timestamp_s"], index * 0.08, abs_tol=1e-6
        ):
            raise ValueError("Transform the complete unfiltered physical episode")
        state = memory.observe(row["state"], previous, index)
        result.append(dict(row, state=state))
        previous = row["action"]
    return result
