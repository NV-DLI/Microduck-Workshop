"""Encoder-only head FK for the course's pinned, October jaw_soft robot asset.

No simulator/root/world state is accepted. Transforms are relative to the
trunk_base link (not its centre of mass). Input angles are absolute joint q in
radians, not policy actions or q minus HOME. This is asset kinematics, not an
extrinsic/encoder calibration for a physical unit.
"""
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

def finite(v,name):
    if isinstance(v,bool) or not math.isfinite(float(v)): raise ValueError(name+' must be finite')
    return float(v)
def rotate(R,v):return tuple(sum(a*b for a,b in zip(row,v)) for row in R)

MJCF_SHA256 = 'ec9d2f731596cfc847a7fc3898ef8b5e67a548928abc23216537f1da8af2729c'
MJCF_COMMIT = 'b02fdea4902c2b93b7d34a83a6f05da4ececb7d3'
MJCF_PATH = Path(__file__).resolve().parent.parent/'sensors/sources/robot_allcollisions.xml'
HEAD_JOINTS = ('neck_pitch','head_pitch','head_yaw','head_roll')
HEAD_BODIES = ('neck','neck_pitch','yaw_roll_motion','jaw_soft')
IDENTITY = ((1.,0.,0.),(0.,1.,0.),(0.,0.,1.))


def matrix_product(a,b):
    return tuple(tuple(sum(a[i][k]*b[k][j]for k in range(3))for j in range(3))for i in range(3))


def quaternion_wxyz(q):
    if len(q)!=4:raise ValueError('Quaternion requires WXYZ')
    w,x,y,z=(finite(v,'quaternion')for v in q)
    norm=math.sqrt(w*w+x*x+y*y+z*z)
    if norm==0:raise ValueError('Zero quaternion')
    w,x,y,z=(v/norm for v in (w,x,y,z))
    return ((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)),
            (2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)),
            (2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)))


@dataclass(frozen=True)
class SE3:
    rotation: tuple = IDENTITY
    translation_m: tuple = (0.,0.,0.)

    def then(self,local):
        v=rotate(self.rotation,local.translation_m)
        return SE3(matrix_product(self.rotation,local.rotation),tuple(self.translation_m[i]+v[i]for i in range(3)))

    def to_dict(self):
        return {'rotation':self.rotation,'translation_m':self.translation_m}


def _numbers(element,name,default):
    return tuple(float(v)for v in element.get(name,default).split())


def _mjcf_transform(element):
    return SE3(quaternion_wxyz(_numbers(element,'quat','1 0 0 0')),_numbers(element,'pos','0 0 0'))


class CourseHeadFK:
    """Load only four joints and two sites from an exact, retained MJCF."""

    def __init__(self,mjcf_path=MJCF_PATH):
        data=Path(mjcf_path).read_bytes()
        if hashlib.sha256(data).hexdigest()!=MJCF_SHA256:
            raise ValueError('FK requires the pinned October RL jaw_soft CAD')
        root=ET.fromstring(data)
        if root.find('compiler').get('angle')!='radian':raise ValueError('Expected radians')
        body=root.find("worldbody/body[@name='trunk_base']")
        if body is None:raise ValueError('Expected trunk_base link')
        self.static_links=[]
        for body_name,joint_name in zip(HEAD_BODIES,HEAD_JOINTS):
            body=body.find(f"body[@name='{body_name}']")
            if body is None:raise ValueError('Missing head body')
            joint=body.find('joint')
            if (joint is None or joint.get('name')!=joint_name or joint.get('type')!='hinge'
                or _numbers(joint,'axis','0 0 1')!=(0,0,1)
                or _numbers(joint,'pos','0 0 0')!=(0,0,0)
                or float(joint.get('ref','0'))!=0):
                raise ValueError('Unexpected hinge axis, anchor or reference')
            self.static_links.append(_mjcf_transform(body))
        self.static_links=tuple(self.static_links)
        self.sites={name:_mjcf_transform(body.find(f"site[@name='{name}']"))for name in ('tof','head_camera')}

    def forward(self,joint_positions):
        """Four named encoder q values only; no base/world pose or sensor truth."""
        if set(joint_positions)!=set(HEAD_JOINTS):
            raise ValueError('Supply exactly the four named absolute head encoder positions')
        head=SE3()
        for name,static in zip(HEAD_JOINTS,self.static_links):
            angle=finite(joint_positions[name],name);c,s=math.cos(angle),math.sin(angle)
            # MJCF hinge axis lies in the child body; rotate after its static transform.
            head=head.then(static).then(SE3(((c,-s,0.),(s,c,0.),(0.,0.,1.))))
        site_to_upright=SE3(quaternion_wxyz((.5,-.5,.5,-.5)))
        optical_from_flu=SE3(((0.,-1.,0.),(0.,0.,-1.),(1.,0.,0.)))
        upright_from_raw=SE3() # Beta native0-degree: remove old alpha CW90
        tof_site=head.then(self.sites['tof']);camera_site=head.then(self.sites['head_camera'])
        upright=camera_site.then(site_to_upright)
        return {'jaw_soft':head,'tof_site':tof_site,
                'tof_flu':tof_site.then(site_to_upright).then(optical_from_flu),
                'camera_site':camera_site,'camera_upright_optical':upright,
                'camera_raw_optical':upright.then(upright_from_raw)}

    def forward_policy14(self,positions,*,joint_names):
        """Explicit channel names prevent silently assuming simulation joint order."""
        if len(positions)!=14 or len(joint_names)!=14 or len(set(joint_names))!=14:
            raise ValueError('Require 14 unique declared joint names and 14 encoder values')
        named={name:finite(q,'encoder')for name,q in zip(joint_names,positions)}
        if not set(HEAD_JOINTS)<=set(named):raise ValueError('Head encoder channels missing')
        return self.forward({name:named[name]for name in HEAD_JOINTS})

