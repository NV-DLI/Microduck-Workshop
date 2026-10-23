"""Constrained image servo: do not compensate an unavailable yaw motion.

The two-variable convex objective is solved over the actual remaining command
domain and slew box. Enumerating each edge and the interior is exact in2D;
post-hoc clipping of an unconstrained coupled solution is not equivalent.
"""

import numpy as np
from image_head_servo import attention_correction as geometry


def bounded_step(J, error, lower, upper, damping=0.025):
    J = np.asarray(J, float)
    error = np.asarray(error, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    if J.shape != (2, 2) or error.shape != (2,) or np.any(lower > upper):
        raise ValueError("Finite2D box required")
    H = J.T @ J + np.eye(2) * damping
    g = J.T @ error
    parameterss = []
    unconstrained = -np.linalg.solve(H, g)
    if np.all(unconstrained >= lower) and np.all(unconstrained <= upper):
        parameterss.append(unconstrained)
    for fixed in range(2):
        free = 1 - fixed
        for bound in (lower[fixed], upper[fixed]):
            x = np.zeros(2)
            x[fixed] = bound
            x[free] = np.clip(
                -(g[free] + H[free, fixed] * bound) / H[free, free],
                lower[free],
                upper[free],
            )
            parameterss.append(x)
    cost = lambda x: float(np.linalg.norm(J @ x + error) ** 2 + damping * np.dot(x, x))
    best = min(parameterss, key=cost)
    return best, {
        "objective_before": cost(np.zeros(2)),
        "objective_after": cost(best),
        "lower_increment_rad": lower.tolist(),
        "upper_increment_rad": upper.tolist(),
        "constraint_active": bool(
            np.any(np.isclose(best, lower)) or np.any(np.isclose(best, upper))
        ),
    }


def attention_correction(
    attention,
    *,
    command,
    rotation_body_from_raw,
    camera,
    now_s,
    dt_s,
    gain=2.0,
    maximum_rate=0.45
):
    out = geometry(
        attention,
        command=command,
        rotation_body_from_raw=rotation_body_from_raw,
        camera=camera,
        now_s=now_s,
        dt_s=dt_s,
        gain=gain,
        maximum_rate=maximum_rate,
    )
    if out is None:
        return None
    old = np.asarray(command, float)
    lower = np.maximum(-maximum_rate * dt_s, np.array([-0.4, 0.0]) - old)
    upper = np.minimum(maximum_rate * dt_s, np.array([0.4, 1.4]) - old)
    delta, diagnostic = bounded_step(
        out["local_image_jacobian"],
        np.array(out["image_error_normalized"]) * gain * dt_s,
        lower,
        upper,
    )
    out["gaze_body_yaw_pitch_rad"] = (old + delta).tolist()
    out.pop("unclipped_increment_rad")
    out.update(
        constrained_increment_rad=delta.tolist(),
        constrained_solution=diagnostic,
        control="incremental visible-pixel error; jointly bounded2D damped least-squares step",
    )
    return out
