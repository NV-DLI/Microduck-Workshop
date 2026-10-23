"""Contact odometry: Microduck simulation and control."""

import math
import numpy as np
from sole_kinematics import SoleKinematics, FOOT_NAMES

HEIGHT_BAND_M = 0.001
PATCH_M = 0.0005
DOUBLE_DISAGREEMENT_M_S = 0.05  # same diagnostic thresholds as frozen LegFK
VERTICAL_RELATIVE_M_S = 0.08


def _vector(v, n):
    a = np.asarray(v, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError("Finite vector with declared size required")
    return a


class SoleContactLegOdometry:
    """estimate() mirrors LegFK.estimate's contract; `point`/`height`/`band_m` exist only for ablations/tests."""

    def __init__(
        self,
        leg_fk,
        *,
        band_m=HEIGHT_BAND_M,
        patch_m=PATCH_M,
        point="contact",
        height="lowest",
        kinematics=None
    ):
        if point not in ("contact", "site") or height not in ("lowest", "site"):
            raise ValueError("Unknown rule")
        if not (0 <= band_m <= 0.01 and 0 < patch_m <= 0.005):
            raise ValueError("Bounded geometric thresholds required")
        self.leg = leg_fk
        self.kin = kinematics or SoleKinematics(leg_fk)
        self.band_m, self.patch_m, self.point, self.height = (
            float(band_m),
            float(patch_m),
            point,
            height,
        )
        self.paths = leg_fk.paths  # read-only, same chain as the frozen LegFK

    def foot(self, name, q, joint_names):
        return self.leg.foot(name, q, joint_names)

    def estimate(self, q14, qd14, gyro3, projected_gravity3, joint_names):
        q, qd, w, g = (
            _vector(q14, 14),
            _vector(qd14, 14),
            _vector(gyro3, 3),
            _vector(projected_gravity3, 3),
        )
        if len(joint_names) != 14 or len(set(joint_names)) != 14:
            raise ValueError("14 declared unique channels required")
        gn = float(np.linalg.norm(g))
        if gn < 0.5:
            raise ValueError("Invalid gravity direction")
        up = -g / gn
        feet = {}
        for name in FOOT_NAMES:
            f = self.kin.foot(name, q, qd, joint_names, up, self.patch_m)
            point = f["contact"] if self.point == "contact" else f["site"]
            v_rel = f["v_contact"] if self.point == "contact" else f["v_site"]
            inertial = v_rel + np.cross(w, point)
            feet[name] = {
                "position_body_m": point.tolist(),
                "site_body_m": f["site"].tolist(),
                "relative_height_m": float(
                    f["lowest_height"] if self.height == "lowest" else f["site_height"]
                ),
                "site_relative_height_m": float(f["site_height"]),
                "lowest_sole_height_m": float(f["lowest_height"]),
                "contact_patch_vertices": int(f["patch_size"]),
                "relative_inertial_velocity_body_m_s": inertial.tolist(),
                "relative_vertical_velocity_m_s": float(up @ inertial),
                "parameters_root_velocity_body_m_s": (-inertial).tolist(),
            }
        lowest = min(v["relative_height_m"] for v in feet.values())
        parameterss = [
            k for k, v in feet.items() if v["relative_height_m"] - lowest <= self.band_m
        ]
        estimate = np.mean(
            [feet[k]["parameters_root_velocity_body_m_s"] for k in parameterss], axis=0
        )
        disagreement = math.dist(
            feet["left_foot"]["parameters_root_velocity_body_m_s"],
            feet["right_foot"]["parameters_root_velocity_body_m_s"],
        )
        ambiguous = len(parameterss) == 2 and disagreement > DOUBLE_DISAGREEMENT_M_S
        high_vertical = any(
            abs(feet[k]["relative_vertical_velocity_m_s"]) > VERTICAL_RELATIVE_M_S
            for k in parameterss
        )
        return {
            "velocity_body_m_s": estimate.tolist(),
            "feet": feet,
            "support_hypothesis": parameterss,
            "near_equal_height": len(parameterss) == 2,
            "double_support_velocity_disagreement_m_s": disagreement,
            "double_support_ambiguous": ambiguous,
            "high_relative_vertical_speed": high_vertical,
            "diagnostic_low_confidence": ambiguous or high_vertical,
            "contact_observed": False,
            "actual_slip_identified": False,
            "rule": dict(
                height=self.height,
                band_m=self.band_m,
                point=self.point,
                patch_m=self.patch_m,
            ),
        }
