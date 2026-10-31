"""Local GR00T inference bridge; simulation time waits for each response.

This measures model execution latency, not a real-time hardware guarantee.
Bind only to loopback for the simulator. A physical robot deployment additionally
needs authenticated transport, command deadlines and an onboard stop watchdog.
"""

import argparse
import hashlib
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
from pathlib import Path
import time
import sys
sys.path[:0]=[str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[2])]
sys.path.append(str(Path(__file__).resolve().parents[1]))
from walking_history_packet import decode_observation
from weight_page_cache import release_after_hash

import numpy as np
from PIL import Image
import torch
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.factory import make_pre_post_processors

try:
    from .horizon import configure_prediction_horizon
    from .request_mapping import map_request, mapping_identity
except ImportError:
    from learning.horizon import configure_prediction_horizon
    from learning.request_mapping import map_request, mapping_identity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument(
        "--prediction-horizon", choices=["training", "checkpoint"], default="training"
    )
    p.add_argument(
        "--action-execution-steps", type=int,
        help="Walking only: execute this many actions before replanning; defaults to checkpoint",
    )
    p.add_argument(
        "--request-mapping",
        choices=["continuous", "demonstrated-forward"],
        default="continuous",
        help="Optional task-specific forward intent mapping; never changes gaze or yaw",
    )
    a = p.parse_args()
    mapping_metadata = mapping_identity(a.request_mapping)
    log = a.log.open("x")
    identity = hashlib.sha256()
    for artifact in sorted(a.checkpoint.iterdir()):
        if artifact.suffix in (".json", ".safetensors"):
            with artifact.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                release_after_hash(stream)
            identity.update((artifact.name + ":" + digest + "\n").encode())
    checkpoint_sha256 = identity.hexdigest()
    contract = json.loads((a.checkpoint / "control-contract.json").read_text())
    walking = contract.get("locomotion") == "walking"
    if not walking or contract.get("version") != 3 or len(contract["state_names"]) != 127:
        raise ValueError("Explicit version3 previous-decision walking checkpoint required")
    from walking_visual_history import specification
    joint_names = [name.removeprefix("q_relative/") for name in contract["state_names"][6:20]]
    if contract != specification(joint_names):
        raise ValueError("Previous-decision contract differs")
    policy_config = GrootConfig.from_pretrained(a.checkpoint)
    from walking_visual_history import IMAGE_KEYS
    image_keys=sorted(k for k in policy_config.input_features if k.startswith("observation.images."))
    if image_keys != list(IMAGE_KEYS) or list(policy_config.input_features["observation.state"].shape) != [127]:
        raise ValueError("Exact history model observation keys required")
    saved_execution_steps = int(policy_config.n_action_steps)
    if a.action_execution_steps is not None:
        if not walking:
            raise ValueError("Execution override is supported only for walking")
        if not 1 <= a.action_execution_steps <= int(policy_config.chunk_size):
            raise ValueError("Execution steps must fit the checkpoint training chunk")
        # The queue capacity is constructed in __init__, not in reset().
        policy_config.n_action_steps = a.action_execution_steps
    policy = GrootPolicy.from_pretrained(a.checkpoint, config=policy_config).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy.config, pretrained_path=str(a.checkpoint)
    )
    horizon_contract = configure_prediction_horizon(policy, pre, a.prediction_horizon)
    policy.reset()
    if a.action_execution_steps is not None and (
            policy._action_queue_steps != a.action_execution_steps
            or policy._action_queue.maxlen != a.action_execution_steps):
        raise ValueError("Actual action queue differs from the requested execution steps")
    horizon_contract.update(
        saved_execution_horizon=saved_execution_steps,
        execution_override=a.action_execution_steps,
        actual_queue_steps=policy._action_queue_steps,
        actual_queue_capacity=policy._action_queue.maxlen,
    )
    if walking and a.request_mapping != "continuous":
        raise ValueError("Roller request mapping cannot be used for walking")
    action_dimension = len(contract["action_names"])
    expected_state = policy.config.input_features["observation.state"].shape[0]
    previous = -1

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/health":
                self.send_error(404)
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    dict(
                        **mapping_metadata,
                        ready=True,
                        checkpoint=str(a.checkpoint),
                        state_dim=expected_state,
                        control_contract=contract,
                        checkpoint_sha256=checkpoint_sha256,
                        native_horizon_contract=horizon_contract,
                    )
                ).encode()
            )

        def do_POST(self):
            nonlocal previous
            if self.path != "/act":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length < 8_000_000:
                    raise ValueError("Invalid payload size")
                packet = json.loads(self.rfile.read(length))
                if packet.get("control_contract") != contract:
                    raise ValueError("Model and actuator bridge contracts differ")
                seq = int(packet["sequence"])
                if packet.get("reset") is True:
                    if seq == 0:
                        # Paired assisted/unassisted episodes begin with the same
                        # model noise; the seed never enters model observations.
                        torch.manual_seed(int(packet.get("episode_seed", 42)))
                    policy.reset()
                    pre.reset()
                    post.reset()
                    previous = seq - 1
                if seq != previous + 1:
                    raise ValueError("Out-of-order observation")
                state = np.asarray(packet["state"], dtype=np.float32)
                if state.shape != (expected_state,) or not np.isfinite(state).all():
                    raise ValueError("Invalid robot state")
                batch = decode_observation(packet, contract, expected_state)
                started = time.monotonic()
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    result = (
                        post(policy.select_action(pre(batch))).float().cpu().numpy()[0]
                    )
                if result.shape != (action_dimension,) or not np.isfinite(result).all():
                    raise ValueError("Invalid GR00T output")
                # Explicit physical command limits also cover floating point
                # unnormalization round-off; retain the raw output in the log.
                if walking:
                    try:
                        from .walking import bound_request
                    except ImportError:
                        from learning.walking import bound_request
                    mapped = bound_request(result)
                    bounded = bool(np.any(np.asarray(mapped) != result))
                else:
                    mapped, bounded = map_request(result, a.request_mapping)
                clipped = np.asarray(mapped)
                row = dict(
                    **mapping_metadata,
                    episode_id=str(packet.get("episode_id", "unspecified")),
                    reset=packet.get("reset") is True,
                    sequence=seq,
                    action=clipped.tolist(),
                    raw_action=result.tolist(),
                    native_horizon_contract=horizon_contract,
                    latency_s=time.monotonic() - started,
                )
                if mapping_metadata:
                    row["pre_projection_action"] = bounded
                log.write(json.dumps(row) + "\n")
                log.flush()
                previous = seq
                body = json.dumps(row).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                self.send_error(422, str(exc))

    print(f"GR00T ready at http://127.0.0.1:{a.port}", flush=True)
    HTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
