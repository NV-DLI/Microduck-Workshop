"""Goal points: Microduck simulation and control."""
import math
import numpy as np
import cv2
from beta_sensor_profile import CAMERA_PROFILE
from beta_perception_geometry import upright_to_raw, raw_pixel_to_unit_ray

MIN_BASE_CROP_MARGIN_PX = 2.


def extract_goal_bases(rgb_upright, bbox, gravity_raw_optical, camera=CAMERA_PROFILE):
    result={'kind':'RGB_goal_post_base_parameters_v3','accepted':False,
            'reason':None,'footpoints_upright_px':None,'task_qualified':False,
            'parameters_only':True,'footpoint_visibility_unverified':True,
            'ground_assignment_allowed':False,
            'caller_requires_additional_geometry_or_temporal_confirmation':True,
            'minimum_observed_base_crop_margin_px':MIN_BASE_CROP_MARGIN_PX,
            'assumption':'two complete white course goal posts on contrasting floor'}
    def reject(reason):result['reason']=reason;return result
    image=np.asarray(rgb_upright)
    if image.shape!=(camera.height_px,camera.width_px,3) or image.dtype!=np.uint8:
        return reject('need_original_upright_RGB_uint8')
    b=np.asarray(bbox,dtype=float);g=np.asarray(gravity_raw_optical,dtype=float)
    if b.shape!=(4,) or not np.isfinite(b).all() or g.shape!=(3,) or not np.isfinite(g).all():
        return reject('invalid_box_or_capture_gravity')
    if abs(np.linalg.norm(g)-1)>1e-5 or np.linalg.norm(g[:2])<.1:
        return reject('gravity_projection_undefined')
    x0,y0,x1,y1=b;h,w=image.shape[:2]
    if x0<1 or y0<1 or x1>w-2 or y1>h-2:
        return reject('goal_box_touches_image_border')
    if x1-x0<18 or y1-y0<18:
        return reject('goal_box_too_small')
    x0,y0=int(math.floor(x0)),int(math.floor(y0));x1,y1=int(math.ceil(x1)),int(math.ceil(y1))
    crop=image[y0:y1+1,x0:x1+1].astype(np.int16)
    # Fixed design thresholds; no per-frame threshold fitting or teacher access.
    white=((crop.min(axis=2)>=190)&((crop.max(axis=2)-crop.min(axis=2))<=35)).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(white,8)
    if count<2:return reject('no_white_component')
    selected=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    yy,xx=np.nonzero(labels==selected)
    if len(xx)<80:return reject('insufficient_observed_white_pixels')
    pixels=np.column_stack((xx+x0,yy+y0)).astype(np.float64)
    # Beta native pixel centres; no alpha CW90 inverse; never use box or network coordinates as K pixels.
    raw=pixels.copy()
    K=np.array([[camera.fx,0,camera.cx],[0,camera.fy,camera.cy],[0,0,1]],dtype=np.float64)
    D=np.array(camera.distortion_k1_k2_p1_p2_k3,dtype=np.float64)
    undistorted=cv2.undistortPointsIter(raw[:,None,:],K,D,None,None,
                                     (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-12))[:,0,:]
    if not np.isfinite(undistorted).all():return reject('distortion_inverse_failed')
    # Rotate the undistorted image plane so the projected gravity points down.
    down=g[:2]/np.linalg.norm(g[:2]);right=np.array([down[1],-down[0]])
    rect=np.column_stack((undistorted@right,undistorted@down))*1000.
    lower=rect.min(axis=0);upper=rect.max(axis=0);extent=upper-lower
    if min(extent)<8:return reject('component_has_no_two_dimensional_extent')
    groups=[np.flatnonzero(rect[:,0]<=lower[0]+.28*extent[0]),
            np.flatnonzero(rect[:,0]>=upper[0]-.28*extent[0])]
    feet=[];details=[]
    for group in groups:
        if len(group)<20:return reject('one_goal_post_not_observed')
        points=rect[group]
        if np.ptp(points[:,1])<.55*extent[1]:return reject('one_goal_post_not_full_height')
        base=group[points[:,1]>=points[:,1].max()-2.]
        if len(base)<3:return reject('post_base_support_insufficient')
        # A detector crop can truncate a perfectly visible post. Do not call
        # that artificial crop edge a footpoint or invent its continuation.
        observed_base=pixels[base]
        crop_margin=float(np.min(np.column_stack((observed_base[:,0]-x0,x1-observed_base[:,0],
                                                   observed_base[:,1]-y0,y1-observed_base[:,1]))))
        if crop_margin<=MIN_BASE_CROP_MARGIN_PX:
            return reject('post_base_at_detector_crop_boundary')
        pixel=pixels[base].mean(axis=0)
        feet.append(pixel.tolist());details.append({'white_pixels':len(group),'base_band_pixels':len(base),
                                                   'observed_base_crop_margin_px':crop_margin,
                                                   'rectified_vertical_span':float(np.ptp(points[:,1]))})
    # A retained open frame must have two separated bases, not one white blob.
    base_rect=[rect[group[rect[group,1]>=rect[group,1].max()-2.]].mean(axis=0) for group in groups]
    if abs(base_rect[1][0]-base_rect[0][0])<.5*extent[0]:return reject('post_bases_not_separated')
    # A filled white rectangle is not the course's open goal frame.
    central=(rect[:,0]>lower[0]+.35*extent[0])&(rect[:,0]<upper[0]-.35*extent[0])&(
        rect[:,1]>lower[1]+.4*extent[1])&(rect[:,1]<upper[1]-.15*extent[1])
    # A field marking visible THROUGH a genuine goal can share the white
    # component with its crossbar. Measure the central WINDOW occupancy, not
    # its fraction of all white goal pixels (which overweights thin posts).
    # This still rejects a filled rectangle and never repairs an occlusion.
    ay,ax=np.indices(crop.shape[:2])
    all_raw=np.column_stack(((ax+x0).reshape(-1),(ay+y0).reshape(-1))).astype(np.float64)
    all_undistorted=cv2.undistortPointsIter(all_raw[:,None,:],K,D,None,None,
        (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-12))[:,0,:]
    all_rect=np.column_stack((all_undistorted@right,all_undistorted@down))*1000.
    central_window=(all_rect[:,0]>lower[0]+.35*extent[0])&(all_rect[:,0]<upper[0]-.35*extent[0])&(
        all_rect[:,1]>lower[1]+.4*extent[1])&(all_rect[:,1]<upper[1]-.15*extent[1])
    area=int(central_window.sum())
    if area<10:return reject('central_window_insufficient_pixels')
    occupancy=float(central.sum())/area
    result['central_window_pixels']=area
    result['central_white_occupancy']=occupancy
    result['central_white_occupancy_rejection_threshold']=.5
    if occupancy>=.5:return reject('central_region_not_open')
    rays=[raw_pixel_to_unit_ray(upright_to_raw(p,camera),camera) for p in feet]
    result.update(accepted=True,reason='two_appearance_based_post_base_parameterss',footpoints_upright_px=feet,
                  footpoint_raw_optical_rays=rays,observed_component_pixels=len(pixels),posts=details,
                  full_visibility_verified=False,semantic_correctness_verified=False,
                  subpixel_accuracy_calibrated=False)
    return result
