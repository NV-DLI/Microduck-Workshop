#!/usr/bin/env python3
"""Execute navigation: Microduck simulation and control."""
import numpy as np
import execute_velocity_policy as entry
import controller

Original = controller.BetaNavigator


class BoundedGazeNavigator(Original):
    def _output(self, t, command=(0.0, 0.0, 0.0), reason="", gaze=None):
        requested = np.asarray([0.0, 0.0] if gaze is None else gaze, dtype=float)
        bounded = np.clip(requested, [-0.4, 0.0], [0.4, 1.4])
        out = super()._output(t, command, reason, bounded.tolist())
        out["trained_gaze_domain"] = {
            "yaw_rad": [-0.4, 0.4],
            "pitch_rad": [0.0, 1.4],
            "requested": requested.tolist(),
            "limited": bool(np.any(requested != bounded)),
            "extra_body_turn_implemented": False,
        }
        return out


def navigation_build(a):
    result = entry.build(a)
    entry.contract["source_sha256"][str(entry.Path(__file__).resolve())] = entry.sha(
        __file__
    )
    return result


if __name__ == "__main__":
    controller.BetaNavigator = BoundedGazeNavigator
    entry.contract["navigation_head_domain"] = {
        "yaw_rad": [-0.4, 0.4],
        "pitch_rad": [0.0, 1.4],
        "source": str(__file__),
        "source_sha256": entry.sha(__file__),
        "scope": "Explicit new controller bundle; source outside original frozen v14",
    }
    entry.isaac_execute.build_cfg = navigation_build
    entry.isaac_execute.run = entry.run
    entry.nav2_bridge.Nav2Session = entry.BoundedLifetimeSession
    entry.isaac_execute.main()
