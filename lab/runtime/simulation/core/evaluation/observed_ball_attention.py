"""Image direction only: a partial ball box cannot authorize range or a kick."""
import math
import numpy as np
from beta_perception_geometry import raw_pixel_to_unit_ray


def observed_attention(detections, *, camera, rotation_body_from_raw,
                       capture_s, now_s):
    if not 0 <= now_s-capture_s <= .2:
        return None
    parameterss=[r for r in detections or [] if r.get('class_name') in ('ball','football')
                and r.get('confidence',0.)>=.45]
    if not parameterss:return None
    row=max(parameterss,key=lambda r:r['confidence']);box=row['bbox_xyxy_px']
    if len(box)!=4 or not all(math.isfinite(v) for v in box):raise ValueError('Finite observed box required')
    x1,y1,x2,y2=box
    if not 0<=x1<x2<camera.width_px or not 0<=y1<y2<camera.height_px:raise ValueError('Original raster box required')
    pixel=[(x1+x2)*.5,(y1+y2)*.5]
    ray=np.asarray(rotation_body_from_raw)@np.asarray(raw_pixel_to_unit_ray(pixel,camera))
    return {'gaze_body_yaw_pitch_rad':[math.atan2(ray[1],ray[0]),math.atan2(-ray[2],math.hypot(ray[0],ray[1]))],
            'observed_bbox_xyxy_px':list(box),'observed_visible_box_center_px':pixel,
            'capture_timestamp_s':capture_s,'cropped':min(x1,y1,camera.width_px-1-x2,camera.height_px-1-y2)<2.,
            'distance_estimated':False,'map_ball_update_allowed':False,'kick_authorized':False,
            'scope':'Attention toward actually visible pixels only; hidden extent and sphere centre are not inferred'}
