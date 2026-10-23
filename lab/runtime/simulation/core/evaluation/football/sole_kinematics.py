"""Public-only foot kinematics on the frozen LegFK chain plus the actual USD sole collision hull.

CPU parameters preparation, not installed in any executor. Inputs are only the 14
delivered encoder angles/velocities, the body gyro and the projected gravity. The
frozen LegFK object (football/leg_odometry.py) is read, never modified: its parsed
chain (`paths`) gives the hinge anchors/axes, so any material point of an ankle
body can be given a velocity, not only the fixed CAD site.

Sole geometry: football/sources/sole-collision-geometry-backlash.json (USD
convexHull collision vertices of left/right_foot_collision, expressed in the
ankle rigid-body frame). Only the convex-hull vertices are kept (the lowest point
of a convex body along any direction is a hull vertex); they are stored in
sole-hull-vertices.json and pinned by SHA-256.
"""

import hashlib, json, math
from pathlib import Path
import numpy as np

DIRECTORY = Path(__file__).resolve().parent
HULL_PATH = DIRECTORY / "sole-hull-vertices.json"
HULL_SHA256 = "86d0bfe13521d52939d9782199fceebe1ff2ce08a916995de346153517c384ba"
FOOT_NAMES = ("left_foot", "right_foot")


def _rot(axis, angle):
    x, y, z = axis
    c, s, d = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array(
        [
            [c + x * x * d, x * y * d - z * s, x * z * d + y * s],
            [y * x * d + z * s, c + y * y * d, y * z * d - x * s],
            [z * x * d - y * s, z * y * d + x * s, c + z * z * d],
        ]
    )


class SoleKinematics:
    def __init__(self, leg_fk, hull_path=HULL_PATH, hull_sha256=HULL_SHA256):
        data = Path(hull_path).read_bytes()
        if hull_sha256 is not None and hashlib.sha256(data).hexdigest() != hull_sha256:
            raise ValueError("Sole hull file changed")
        hull = json.loads(data)
        self.leg = leg_fk
        self.vertices = {
            n: np.asarray(hull[n]["hull_vertices_ankle_m"], float) for n in FOOT_NAMES
        }
        self.chains = {}
        for n in FOOT_NAMES:
            chain, site = leg_fk.paths[n]
            self.chains[n] = (
                [
                    (
                        np.asarray(node["R"], float),
                        np.asarray(node["p"], float),
                        node["joint"],
                        (
                            None
                            if node["joint"] is None
                            else np.asarray(node["axis"], float)
                        ),
                    )
                    for node in chain
                ],
                np.asarray(site, float),
            )

    def frame(self, name, q, joint_names):
        """Ankle-body rotation/origin in the trunk frame and the actuated hinges (anchor, unit axis, joint index)."""
        idx = {j: i for i, j in enumerate(joint_names)}
        R = np.eye(3)
        p = np.zeros(3)
        hinges = []
        chain, site = self.chains[name]
        for Rn, pn, joint, axis in chain:
            p = p + R @ pn
            R = R @ Rn
            if joint is not None:
                hinges.append((p.copy(), R @ axis, idx[joint]))
                R = R @ _rot(axis, float(q[idx[joint]]))
        return R, p, hinges, site

    def point_velocity(self, hinges, point, qd):
        """Velocity of an ankle-body material point relative to the trunk, trunk frame (m/s)."""
        v = np.zeros(3)
        for anchor, axis, j in hinges:
            v += np.cross(axis, point - anchor) * qd[j]
        return v

    def foot_angular_velocity(self, hinges, qd):
        w = np.zeros(3)
        for _, axis, j in hinges:
            w += axis * qd[j]
        return w

    def foot(self, name, q, qd, joint_names, up, patch_m):
        """Site and lowest-sole contact patch of one foot, all in the trunk frame."""
        R, p, hinges, site = self.frame(name, q, joint_names)
        site_b = p + R @ site
        verts = p + self.vertices[name] @ R.T
        h = verts @ up
        hmin = float(h.min())
        patch = verts[h <= hmin + patch_m]
        contact = patch.mean(0)
        return dict(
            site=site_b,
            contact=contact,
            lowest_height=hmin,
            site_height=float(site_b @ up),
            patch_size=len(patch),
            v_site=self.point_velocity(hinges, site_b, qd),
            v_contact=self.point_velocity(hinges, contact, qd),
            w_foot=self.foot_angular_velocity(hinges, qd),
            R=R,
        )
