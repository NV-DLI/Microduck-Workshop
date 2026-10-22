"""Image-only head feedback, correcting persistent command-to-encoder error.

A cropped detection authorizes attention only. It cannot supply metric range,
move the base, establish ball position, or trigger a kick. The feedback uses
the visible box centre, current encoder optical frame and the existing head
command. The local angular Jacobian does not assume ideal command tracking.
"""
import math
import numpy as np
from beta_perception_geometry import raw_pixel_to_unit_ray


def attention_correction(attention, *, command, rotation_body_from_raw,
                         camera, now_s, dt_s, gain=2., maximum_rate=.45):
    if attention is None or not 0 <= now_s-attention['capture_timestamp_s'] <= .2:
        return None
    if not 0 < dt_s <= .2:
        return None
    command=np.asarray(command,float);R=np.asarray(rotation_body_from_raw,float)
    if command.shape!=(2,) or R.shape!=(3,3) or not np.isfinite(command).all() or not np.isfinite(R).all():
        raise ValueError('Finite current command and same-capture encoder rotation required')
    pixel=attention['observed_visible_box_center_px']
    local=np.asarray(raw_pixel_to_unit_ray(pixel,camera));ray=R@local
    if local[2]<=.1:raise ValueError('Forward observed image ray required')
    # Camera angular sensitivity: yaw about body Z; positive down-pitch
    # about the horizontal left axis at the current commanded yaw.
    yaw=float(command[0]);axes=np.array([[0.,0.,1.],[-math.sin(yaw),math.cos(yaw),0.]])
    derivatives=np.stack([-R.T@np.cross(axis,ray) for axis in axes],axis=1)
    J=(derivatives[:2]*local[2]-local[:2,None]*derivatives[2])/local[2]**2
    error=local[:2]/local[2]
    delta=-np.linalg.solve(J.T@J+np.eye(2)*.025,J.T@error)*gain*dt_s
    delta=np.clip(delta,-maximum_rate*dt_s,maximum_rate*dt_s)
    desired=np.clip(command+delta,[-.4,0.],[.4,1.4])
    return {'gaze_body_yaw_pitch_rad':desired.tolist(),
            'capture_timestamp_s':attention['capture_timestamp_s'],
            'observed_visible_box_center_px':list(pixel),
            'image_error_normalized':error.tolist(),'local_image_jacobian':J.tolist(),
            'unclipped_increment_rad':delta.tolist(),'old_command_rad':command.tolist(),
            'cropped':attention['cropped'],'maximum_command_rate_rad_s':maximum_rate,
            'control':'incremental visible-pixel error with same-time encoder rotation',
            'distance_estimated':False,'map_ball_update_allowed':False,
            'base_motion_authorized':False,'kick_authorized':False,
            'uncalibrated_local_jacobian':True}
