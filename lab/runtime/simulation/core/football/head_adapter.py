"""Named encoder FK, body-relative camera IK and head-IMU transport.

No world/root/ball coordinates are accepted. Native head commands are offsets
from home, while the FK consumes absolute effective output joint positions.
"""
import math,xml.etree.ElementTree as ET
import numpy as np
from hardware_head_fk import CourseHeadFK,HEAD_JOINTS,MJCF_PATH
from beta_head_imu import rotate,inverse
HOME=np.asarray([.3491,.3491,0.,0.])

def vee(R):return np.asarray([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])*.5
def rotation_error(desired,current):
    d=desired@current.T;angle=math.acos(float(np.clip((np.trace(d)-1)/2,-1,1)))
    if angle<1e-7:return vee(d)
    return vee(d)*angle/max(math.sin(angle),1e-8)

class HeadAdapter:
    def __init__(self):
        self.fk=CourseHeadFK();root=ET.parse(MJCF_PATH).getroot()
        self.limits=np.asarray([[float(v) for v in root.find(f'.//joint[@name="{name}"]').get('range').split()] for name in HEAD_JOINTS])
        self.last=HOME.copy()
    def rotation(self,q):return np.asarray(self.fk.forward(dict(zip(HEAD_JOINTS,map(float,q))))['camera_raw_optical'].rotation)
    def solve(self,yaw_pitch,*,max_iterations=40):
        yaw,pitch=map(float,yaw_pitch)
        if not all(math.isfinite(v) for v in (yaw,pitch)):raise ValueError('Finite body-relative gaze required')
        z=np.asarray([math.cos(yaw)*math.cos(pitch),math.sin(yaw)*math.cos(pitch),-math.sin(pitch)])
        x=np.cross(z,[0.,0.,1.]);x/=np.linalg.norm(x);y=np.cross(z,x);desired=np.column_stack([x,y,z])
        q=self.last.copy();eps=1e-5
        for i in range(max_iterations):
            R=self.rotation(q);error=rotation_error(desired,R)
            if np.linalg.norm(error)<1e-5:break
            J=np.column_stack([rotation_error(self.rotation(q+np.eye(4)[j]*eps),R)/eps for j in range(4)])
            # Damped orientation solve with a very small home prior resolves the
            # redundant neck/head pitch pair without inventing fixed joint axes.
            delta=np.linalg.solve(J.T@J+np.eye(4)*1e-3,J.T@error+1e-5*(HOME-q))
            q=np.clip(q+np.clip(delta,-.12,.12),self.limits[:,0]+.02,self.limits[:,1]-.02)
        self.last=q
        residual=float(np.linalg.norm(rotation_error(desired,self.rotation(q))))
        return {'gaze_body_yaw_pitch_rad':[yaw,pitch],'desired_head_joint_rad':q.tolist(),
                'head_pose_native_offsets_rad':(q-HOME).tolist(),'joint_names':list(HEAD_JOINTS),
                'orientation_residual_rad':residual,'accepted':residual<.03,
                'world_stabilized':False,'native_home_rad':HOME.tolist()}
    def body_imu(self,head_sample,joint_names,q_rad,qd_rad_s,*,capture_s):
        if head_sample is None:raise ValueError('Head IMU not warmed up')
        if head_sample.get('orientation_valid') is not True:raise ValueError('Head IMU gravity initialization is not valid')
        if head_sample.get('quaternion_order')!='wxyz' or head_sample.get('axes')!='head FLU' or head_sample.get('absolute_world_yaw') is not False:raise ValueError('Expected local Beta head IMU packet')
        if abs(float(head_sample['timestamp_s'])-capture_s)>1e-6:raise ValueError('Acquisition-time head IMU required for sensor-frame conversion')
        named=dict(zip(joint_names,q_rad));vel=dict(zip(joint_names,qd_rad_s));q=np.asarray([named[n]for n in HEAD_JOINTS]);qd=np.asarray([vel[n]for n in HEAD_JOINTS])
        # The virtual hardware IMU FLU uses the CAD camera/tof site rotation.
        def imu_rotation(v):return np.asarray(self.fk.forward(dict(zip(HEAD_JOINTS,map(float,v))))['camera_site'].rotation)
        R=imu_rotation(q);eps=1e-5;Rnext=imu_rotation(q+qd*eps)
        head_joint_omega_body=vee((Rnext-R)@R.T/eps)
        gyro=R@np.asarray(head_sample['gyro'])-head_joint_omega_body
        w,x,y,z=head_sample['quat'];gravity=R@np.asarray(rotate(inverse((x,y,z,w)),(0.,0.,-1.)));gravity/=np.linalg.norm(gravity)
        return {'gyro_body_rad_s':gyro.tolist(),'gravity_body':gravity.tolist(),
                'source':'local head IMU + same-time encoder FK and head joint angular velocity; no absolute root pose',
                'head_imu_timestamp_s':head_sample['timestamp_s']}
