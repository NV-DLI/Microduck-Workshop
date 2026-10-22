"""Diagnostic grounded-sphere position from silhouette direction and a plane.

Unlike angular-size ranging, this keeps only the fitted cone direction. It
requires an independently estimated ground plane and a ball resting on it.
It does not infer contact, height, FK or pose from simulator state.
"""
import math
import numpy as np
from .sphere_geometry import fit_sphere_rays


def estimate_grounded_direction(rays, *, radius_m, pose, ground,
                                capture_timestamp_s, now_s, max_age_s=.2,
                                max_range_m=4., min_downward_incidence=.08):
    """pose/ground are CaptureCameraPose/PlaneEstimate in the same frame.

    Plane normal is upward. A known radius offsets the centre plane. The
    supplied pose must be at capture time; the resulting point remains in
    that capture reference frame, not the current moving robot frame.
    """
    result={'kind':'grounded_sphere_from_cone_direction_v1','accepted':False,
            'reason':None,'point_reference_m':None,
            'assumes_ground_contact':True,'ground_contact_verified':False,
            'point_retimed_to_current_robot':False,'task_or_hardware_validated':False}
    def reject(reason):
        result['reason']=reason
        return result
    numbers=(radius_m,capture_timestamp_s,now_s,max_age_s,max_range_m,min_downward_incidence)
    if any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in numbers):
        return reject('invalid_numeric_input')
    if radius_m<=0 or max_age_s<0 or max_range_m<=0 or not 0<min_downward_incidence<1:
        return reject('invalid_limits')
    try:
        rotation=np.asarray(pose.rotation_reference_from_raw,dtype=float)
        origin=np.asarray(pose.translation_m,dtype=float)
        normal=np.asarray(ground.normal,dtype=float)
        if rotation.shape!=(3,3) or origin.shape!=(3,) or normal.shape!=(3,):
            return reject('invalid_pose_or_plane_shape')
        if not all(np.isfinite(v).all() for v in (rotation,origin,normal)) or not math.isfinite(ground.offset_m) or not math.isfinite(pose.timestamp_s):
            return reject('nonfinite_pose_or_plane')
        if not np.allclose(rotation@rotation.T,np.eye(3),rtol=0,atol=1e-6) or abs(np.linalg.det(rotation)-1)>1e-6 or abs(np.linalg.norm(normal)-1)>1e-6:
            return reject('invalid_rotation_or_plane_normal')
        if any(not isinstance(v,str) or not v.strip() for v in (pose.source,ground.source,pose.reference_frame,ground.reference_frame)):
            return reject('missing_pose_or_plane_provenance')
    except (AttributeError,TypeError,ValueError):
        return reject('invalid_pose_or_plane')
    age=now_s-capture_timestamp_s;result['age_s']=age
    if age < -1e-9 or age>max_age_s+1e-9:
        return reject('future_or_stale_image')
    if abs(pose.timestamp_s-capture_timestamp_s)>1e-6:
        return reject('pose_not_at_image_capture')
    if pose.reference_frame!=ground.reference_frame:
        return reject('ground_pose_frame_mismatch')
    fitted=fit_sphere_rays(rays,radius_m)
    result['cone_fit']=fitted
    if not fitted['accepted']:
        return reject('silhouette_rejected:'+fitted['reason'])
    direction=[x/fitted['center_distance_m'] for x in fitted['center_camera_m']]
    ray=[sum(x*y for x,y in zip(row,direction)) for row in pose.rotation_reference_from_raw]
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    incidence=dot(ground.normal,ray)
    result['downward_incidence']=-incidence
    if incidence > -min_downward_incidence:
        return reject('direction_not_sufficiently_toward_ground')
    offset=ground.offset_m+radius_m
    height=dot(ground.normal,pose.translation_m)-offset
    if height<=0:
        return reject('camera_not_above_ball_center_plane')
    distance=-height/incidence
    if distance>max_range_m:
        return reject('intersection_beyond_max_range')
    point=[x+distance*d for x,d in zip(pose.translation_m,ray)]
    result.update(accepted=True,reason='geometry_consistent_under_declared_ground_contact_assumption',
                  point_reference_m=point,reference_frame=pose.reference_frame,
                  capture_timestamp_s=capture_timestamp_s,pose_source=pose.source,
                  ground_source=ground.source,slant_range_m=distance,
                  cone_direction_camera_unit=direction,
                  angular_size_range_used=False,
                  point_derivative_wrt_ground_offset=[d/incidence for d in ray],
                  plane_offset_error_distance_gain=1/abs(incidence))
    return result
