"""Discardable encoder/IMU-only foot-site velocity feasibility estimator.

No world pose, contact, simulator, model, future sample, or labels are inputs.
Foot sites are fixed CAD points; no-slip at these points is only a hypothesis.
"""
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

MJCF_SHA = '7a6fdf437f5a80c7348ad801f43f906b997a834389cb70cca5e1e8517ba38044'
I = ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))
HEIGHT_BAND_M = .003
DOUBLE_DISAGREEMENT_M_S = .05
VERTICAL_RELATIVE_M_S = .08


def vector(v, n):
    if len(v) != n or not all(math.isfinite(x) for x in v):
        raise ValueError('Finite vector with declared size required')
    return list(v)


def add(a, b): return [x+y for x, y in zip(a, b)]
def sub(a, b): return [x-y for x, y in zip(a, b)]
def dot(a, b): return sum(x*y for x, y in zip(a, b))
def cross(a, b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
def mv(r, v): return [dot(row, v) for row in r]
def mm(a, b): return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))
def transpose(r): return tuple(zip(*r))


def quaternion(wxyz):
    w, x, y, z = vector(wxyz, 4)
    n = math.hypot(w, x, y, z)
    if not n: raise ValueError('Zero quaternion')
    w, x, y, z = [v/n for v in (w, x, y, z)]
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def axis_rotation(axis, angle):
    x, y, z = axis
    c, s, d = math.cos(angle), math.sin(angle), 1-math.cos(angle)
    return ((c+x*x*d, x*y*d-z*s, x*z*d+y*s),
            (y*x*d+z*s, c+y*y*d, y*z*d-x*s),
            (z*x*d-y*s, z*y*d+x*s, c+z*z*d))


def numbers(node, key, default):
    return [float(x) for x in node.get(key, default).split()]


class LegFK:
    def __init__(self, xml_path, *, require_pinned=True):
        data = Path(xml_path).read_bytes()
        if require_pinned and hashlib.sha256(data).hexdigest() != MJCF_SHA:
            raise ValueError('Expected retained old CAD MJCF')
        root = ET.fromstring(data)
        if root.find('compiler').get('angle') != 'radian': raise ValueError('Radians required')
        trunk = root.find("worldbody/body[@name='trunk_base']")
        if trunk is None: raise ValueError('Missing trunk')
        self.paths = {}

        def walk(body, chain):
            # Root freejoint/pose explicitly excluded: all transforms are in trunk.
            for child in body.findall('body'):
                node = {'R': quaternion(numbers(child, 'quat', '1 0 0 0')),
                        'p': numbers(child, 'pos', '0 0 0'), 'joint': None}
                joints = child.findall('joint')
                if len(joints) > 1: raise ValueError('Only one hinge per old CAD link')
                if joints:
                    joint = joints[0]
                    if joint.get('type') != 'hinge' or numbers(joint, 'pos', '0 0 0') != [0., 0., 0.] or float(joint.get('ref', '0')) != 0:
                        raise ValueError('Unexpected joint anchor/reference/type')
                    axis = numbers(joint, 'axis', '0 0 1')
                    norm = math.hypot(*axis)
                    if not norm: raise ValueError('Zero hinge axis')
                    node.update(joint=joint.get('name'), axis=[v/norm for v in axis])
                path = chain+[node]
                for site in child.findall('site'):
                    if site.get('name') in ('left_foot', 'right_foot'):
                        self.paths[site.get('name')] = (path, numbers(site, 'pos', '0 0 0'))
                walk(child, path)
        walk(trunk, [])
        if set(self.paths) != {'left_foot', 'right_foot'}: raise ValueError('Both original foot sites required')

    def foot(self, name, q, joint_names):
        if len(q) != len(joint_names) or len(set(joint_names)) != len(joint_names):
            raise ValueError('Explicit unique joint order required')
        named = dict(zip(joint_names, vector(q, len(joint_names))))
        R, p, hinges = I, [0., 0., 0.], []
        chain, site = self.paths[name]
        for node in chain:
            p = add(p, mv(R, node['p']))
            R = mm(R, node['R'])
            if node['joint']:
                axis = mv(R, node['axis'])
                hinges.append((node['joint'], p[:], axis))
                R = mm(R, axis_rotation(node['axis'], named[node['joint']]))
        point = add(p, mv(R, site))
        columns = {joint: cross(axis, sub(point, anchor)) for joint, anchor, axis in hinges}
        J = [[columns.get(joint, [0., 0., 0.])[i] for joint in joint_names] for i in range(3)]
        return {'position_body_m': point, 'jacobian_m_per_rad': J, 'rotation': R}

    def estimate(self, q14, qd14, gyro3, projected_gravity3, joint_names):
        vector(q14, 14); vector(qd14, 14); vector(gyro3, 3); vector(projected_gravity3, 3)
        if len(joint_names) != 14: raise ValueError('14 declared channels required')
        gn = math.hypot(*projected_gravity3)
        if gn < .5: raise ValueError('Invalid gravity direction')
        up = [-v/gn for v in projected_gravity3]
        feet = {}
        for name in ('left_foot', 'right_foot'):
            value = self.foot(name, q14, joint_names)
            point = value['position_body_m']
            joint_velocity = mv(value['jacobian_m_per_rad'], qd14)
            relative_inertial_velocity = add(joint_velocity, cross(gyro3, point))
            feet[name] = {'position_body_m': point,
                          'relative_height_m': dot(up, point),
                          'relative_inertial_velocity_body_m_s': relative_inertial_velocity,
                          'relative_vertical_velocity_m_s': dot(up, relative_inertial_velocity),
                          'parameters_root_velocity_body_m_s': [-v for v in relative_inertial_velocity]}
        lowest = min(v['relative_height_m'] for v in feet.values())
        parameterss = [k for k, v in feet.items() if v['relative_height_m']-lowest <= HEIGHT_BAND_M]
        estimate = [sum(feet[k]['parameters_root_velocity_body_m_s'][i] for k in parameterss)/len(parameterss) for i in range(3)]
        disagreement = math.dist(feet['left_foot']['parameters_root_velocity_body_m_s'], feet['right_foot']['parameters_root_velocity_body_m_s'])
        ambiguous = len(parameterss) == 2 and disagreement > DOUBLE_DISAGREEMENT_M_S
        high_vertical = any(abs(feet[k]['relative_vertical_velocity_m_s']) > VERTICAL_RELATIVE_M_S for k in parameterss)
        return {'velocity_body_m_s': estimate, 'feet': feet, 'support_hypothesis': parameterss,
                'near_equal_height': len(parameterss) == 2, 'double_support_velocity_disagreement_m_s': disagreement,
                'double_support_ambiguous': ambiguous, 'high_relative_vertical_speed': high_vertical,
                'diagnostic_low_confidence': ambiguous or high_vertical,
                'contact_observed': False, 'actual_slip_identified': False}
