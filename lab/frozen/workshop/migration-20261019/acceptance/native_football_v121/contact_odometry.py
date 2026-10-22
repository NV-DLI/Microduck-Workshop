"""Candidate public leg odometry: CAD sole contact patch + tight lowest-sole support (CPU, not installed).

Drop-in replacement for the `leg_fk` object given to the unchanged
football/public_ball_memory_v1/memory.py BallMemory (and therefore the unchanged
public_ball_read_adapter_v1/v2). It implements the same `estimate(q14, qd14,
gyro3, projected_gravity3, joint_names)` contract and returns the same keys.
Integration, yaw (body gyro), duplicate/stale/gap/expiry handling stay in the
frozen memory/adapter code and are not touched.

Differences from the frozen LegFK.estimate (selected on DEV runs only, see
outputs/migration-20261019/claude-football-odometry-v97-v1/REPORT.zh.md):
  1. Contact point: centroid of the USD sole collision hull vertices lying within
     PATCH_M of the lowest sole point along public gravity "up", instead of the
     fixed CAD site. For a flat planted sole this changes nothing; while the sole
     rolls on its heel/toe/side edge it is the instantaneous pivot (no-slip point).
  2. Support: a foot is a support candidate only if its lowest sole point is within
     HEIGHT_BAND_M (1 mm) of the lowest foot, instead of the site within 3 mm.
     Candidates are averaged as before.
Inputs: only delivered encoders, body gyro and projected gravity. No pose, contact,
force, commanded velocity or simulator value is read. No calibration constant was
fitted to outcomes: band/patch are geometric thresholds; no scale or offset term.
"""
import math
import numpy as np
from sole_kinematics import SoleKinematics, FOOT_NAMES

HEIGHT_BAND_M = .001
PATCH_M = .0005
DOUBLE_DISAGREEMENT_M_S = .05   # same diagnostic thresholds as frozen LegFK
VERTICAL_RELATIVE_M_S = .08


def _vector(v, n):
    a = np.asarray(v, dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all(): raise ValueError('Finite vector with declared size required')
    return a


class SoleContactLegOdometry:
    """estimate() mirrors LegFK.estimate's contract; `point`/`height`/`band_m` exist only for ablations/tests."""
    def __init__(self, leg_fk, *, band_m=HEIGHT_BAND_M, patch_m=PATCH_M, point='contact', height='lowest', kinematics=None):
        if point not in ('contact', 'site') or height not in ('lowest', 'site'): raise ValueError('Unknown rule')
        if not (0 <= band_m <= .01 and 0 < patch_m <= .005): raise ValueError('Bounded geometric thresholds required')
        self.leg = leg_fk; self.kin = kinematics or SoleKinematics(leg_fk)
        self.band_m, self.patch_m, self.point, self.height = float(band_m), float(patch_m), point, height
        self.paths = leg_fk.paths  # read-only, same chain as the frozen LegFK

    def foot(self, name, q, joint_names):
        return self.leg.foot(name, q, joint_names)

    def estimate(self, q14, qd14, gyro3, projected_gravity3, joint_names):
        q, qd, w, g = _vector(q14, 14), _vector(qd14, 14), _vector(gyro3, 3), _vector(projected_gravity3, 3)
        if len(joint_names) != 14 or len(set(joint_names)) != 14: raise ValueError('14 declared unique channels required')
        gn = float(np.linalg.norm(g))
        if gn < .5: raise ValueError('Invalid gravity direction')
        up = -g/gn
        feet = {}
        for name in FOOT_NAMES:
            f = self.kin.foot(name, q, qd, joint_names, up, self.patch_m)
            point = f['contact'] if self.point == 'contact' else f['site']
            v_rel = f['v_contact'] if self.point == 'contact' else f['v_site']
            inertial = v_rel+np.cross(w, point)
            feet[name] = {'position_body_m': point.tolist(), 'site_body_m': f['site'].tolist(),
                          'relative_height_m': float(f['lowest_height'] if self.height == 'lowest' else f['site_height']),
                          'site_relative_height_m': float(f['site_height']), 'lowest_sole_height_m': float(f['lowest_height']),
                          'contact_patch_vertices': int(f['patch_size']),
                          'relative_inertial_velocity_body_m_s': inertial.tolist(),
                          'relative_vertical_velocity_m_s': float(up@inertial),
                          'candidate_root_velocity_body_m_s': (-inertial).tolist()}
        lowest = min(v['relative_height_m'] for v in feet.values())
        candidates = [k for k, v in feet.items() if v['relative_height_m']-lowest <= self.band_m]
        estimate = np.mean([feet[k]['candidate_root_velocity_body_m_s'] for k in candidates], axis=0)
        disagreement = math.dist(feet['left_foot']['candidate_root_velocity_body_m_s'], feet['right_foot']['candidate_root_velocity_body_m_s'])
        ambiguous = len(candidates) == 2 and disagreement > DOUBLE_DISAGREEMENT_M_S
        high_vertical = any(abs(feet[k]['relative_vertical_velocity_m_s']) > VERTICAL_RELATIVE_M_S for k in candidates)
        return {'velocity_body_m_s': estimate.tolist(), 'feet': feet, 'support_hypothesis': candidates,
                'near_equal_height': len(candidates) == 2, 'double_support_velocity_disagreement_m_s': disagreement,
                'double_support_ambiguous': ambiguous, 'high_relative_vertical_speed': high_vertical,
                'diagnostic_low_confidence': ambiguous or high_vertical,
                'contact_observed': False, 'actual_slip_identified': False,
                'rule': dict(height=self.height, band_m=self.band_m, point=self.point, patch_m=self.patch_m)}
