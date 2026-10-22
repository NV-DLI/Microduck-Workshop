"""Gaze tracking: Microduck simulation and control."""
import math
import numpy as np


def observed_ball_gaze(point_body_m, camera_origin_body_m):
    point=np.asarray(point_body_m,dtype=float);origin=np.asarray(camera_origin_body_m,dtype=float)
    if point.shape!=(3,)or origin.shape!=(3,)or not np.isfinite(point).all()or not np.isfinite(origin).all():
        raise ValueError('Finite observed ball and encoder camera origin required')
    ray=point-origin
    if np.linalg.norm(ray)<.02:raise ValueError('Ball ray too short for gaze')
    return [math.atan2(ray[1],ray[0]),math.atan2(-ray[2],math.hypot(ray[0],ray[1]))]


class GazeSlew:
    def __init__(self,rate_rad_s=.6):
        if not math.isfinite(rate_rad_s)or rate_rad_s<=0:raise ValueError('Positive finite gaze rate required')
        self.rate=rate_rad_s;self.current=np.zeros(2);self.timestamp=None;self.last=None

    def update(self,target,t):
        raw=np.asarray(target,dtype=float)
        if raw.shape!=(2,)or not np.isfinite(raw).all()or not math.isfinite(t):raise ValueError('Finite gaze and timestamp required')
        if self.timestamp is not None and t<self.timestamp:raise ValueError('Gaze clock cannot rewind')
        dt=0. if self.timestamp is None else t-self.timestamp
        limited=np.clip(raw,[-.65,-.15],[.65,1.45])
        self.current+=np.clip(limited-self.current,-self.rate*dt,self.rate*dt)
        self.timestamp=t
        self.last={'requested_body_yaw_pitch_rad':raw.tolist(),'bounded_body_yaw_pitch_rad':limited.tolist(),
                   'executed_body_yaw_pitch_rad':self.current.tolist(),'slew_limit_rad_s':self.rate,
                   'limit_clipped':bool(np.any(raw!=limited)),'world_stabilized':False,
                   'gait_head_command_generalization_validated':False}
        return self.current.tolist()
