#!/usr/bin/env python3
"""Local CPU controller service; own Nav2 child, no GPU or simulator imports.

POST /step JSON: rgb_png_base64, proprio, tof_frame, now_s, optional detections.
GET /health. POST /reset {layout, gait_id, kick_ball_body_xy_m} reinitializes a
controller. No subprocess/paths/commands are accepted in requests.
"""
import argparse, base64, json, math, signal, time
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2, numpy as np
from beta_sensor_profile import BetaCameraProfile
from controller import BetaNavigator, ControllerConfig
from nav2_bridge import Nav2Session


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8038)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--hfov", type=float, default=70.0)
    p.add_argument("--layout", choices=["empty", "one-obstacle"], default="empty")
    p.add_argument("--max-wall-s", type=float, default=1800)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    camera = BetaCameraProfile(
        width_px=a.width, height_px=a.height, assumed_horizontal_fov_deg=a.hfov
    )
    nav = Nav2Session(a.output, max_wall_s=a.max_wall_s)
    controller = [
        BetaNavigator(nav, camera=camera, config=ControllerConfig(layout=a.layout))
    ]
    log = (a.output / "control.jsonl").open("x")
    start = time.monotonic()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, obj):
            raw = json.dumps(obj, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/health":
                return self.send(404, {"error": "unknown route"})
            return self.send(
                200,
                {
                    "status": "ready",
                    "runtime_id": "newton-bam-20261019",
                    "hardware_profile_id": "beta-gc2093-l8cx-v1",
                    "GPU": False,
                    "camera": camera.to_dict(),
                    "controller_state": controller[0].state,
                    "Nav2_planner_pid": nav.ready["planner_pid"],
                },
            )

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 8_000_000:
                    raise ValueError("Bounded JSON request required")
                x = json.loads(self.rfile.read(n))
                if self.path == "/reset":
                    if set(x) - {"layout", "gait_id", "kick_ball_body_xy_m"}:
                        raise ValueError("Unknown reset fields")
                    if x.get("layout", "empty") not in ("empty", "one-obstacle"):
                        raise ValueError("Unknown layout")
                    if "kick_ball_body_xy_m" in x and (
                        len(x["kick_ball_body_xy_m"]) != 2
                        or not all(math.isfinite(v) for v in x["kick_ball_body_xy_m"])
                    ):
                        raise ValueError("Invalid kick offset")
                    controller[0] = BetaNavigator(
                        nav, camera=camera, config=ControllerConfig(**x)
                    )
                    # Nav2's monotone transaction clock spans episodes: restart its owned
                    # service if the simulator clock will reset to0. Caller uses new service.
                    return self.send(
                        200,
                        {
                            "status": "reset",
                            "clock_requirement": "Continue globally monotone timestamps for this service; restart service for clock reset",
                        },
                    )
                if self.path != "/step":
                    return self.send(404, {"error": "unknown route"})
                if set(x) - {
                    "rgb_png_base64",
                    "proprio",
                    "tof_frame",
                    "now_s",
                    "detections",
                }:
                    raise ValueError(
                        "Unknown fields; hidden simulator metadata forbidden"
                    )
                raw = base64.b64decode(x["rgb_png_base64"], validate=True)
                bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("PNG decode failed")
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                out = controller[0].step(
                    rgb,
                    x["proprio"],
                    x["tof_frame"],
                    now_s=x["now_s"],
                    detections=x.get("detections"),
                )
                log.write(json.dumps(out, allow_nan=False) + "\n")
                log.flush()
                self.send(200, out)
            except (ValueError, KeyError, TypeError, TimeoutError, OSError) as e:
                self.send(
                    400,
                    {
                        "error": str(e),
                        "velocity_body_xyw": [0.0, 0.0, 0.0],
                        "kick_request": False,
                    },
                )

    def stop_owned_service(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_owned_service)
    signal.signal(signal.SIGINT, stop_owned_service)
    server = None
    try:
        server = HTTPServer(("127.0.0.1", a.port), Handler)
        server.timeout = 0.5
        while time.monotonic() - start < a.max_wall_s:
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        log.close()
        cleanup = nav.close()
        (a.output / "cleanup.json").write_text(json.dumps(cleanup, indent=2) + "\n")


if __name__ == "__main__":
    main()
