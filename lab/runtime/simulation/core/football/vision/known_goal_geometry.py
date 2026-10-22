"""Known-landmark baseline range from two OBSERVED goal-post base bearings.

The declared/surveyed .265m post spacing sets metric scale. IMU defines a common
horizontal plane; encoder-estimated ground height only checks consistency.
Neither simulation root/object pose nor depth/teacher masks is accepted.
"""
import math
import numpy as np
from beta_perception_geometry import raw_pixel_to_unit_ray

POST_SEPARATION_M=.265

def project_known_post_pair(parameterss,*,pose,ground,camera,capture_timestamp_s,now_s):
 r={'kind':'known_goal_baseline_two_bearings_v1','accepted':False,'reason':None,'goal_center_reference_m':None,'truth_used':False,'scale_source':'declared surveyed post centre separation','known_center_separation_m':POST_SEPARATION_M,'ground_height_used_for_scale':False,'map_pose_used_for_scale':False,'ground_contact_assumed':True,'task_qualified':False,'missing_pixels_reconstructed':False}
 def reject(reason):r['reason']=reason;return r
 if not parameterss.get('accepted')or parameterss.get('kind')!='RGB_beta_goal_post_base_parameters_v1':return reject('requires_two_observed_Beta_post_bases')
 if not 0<=now_s-capture_timestamp_s<=.2 or abs(pose.timestamp_s-capture_timestamp_s)>1e-6:return reject('stale_image_or_FK')
 points=parameterss['footpoints_upright_px'];R=np.asarray(pose.rotation_reference_from_raw);normal=np.asarray(ground.normal);origin=np.asarray(pose.translation_m)
 rays=[R@np.asarray(raw_pixel_to_unit_ray(p,camera))for p in points];incidence=np.asarray([normal@v for v in rays])
 if np.any(incidence>=-.015):return reject('post_base_not_below_estimated_horizon')
 ratio=incidence[0]/incidence[1];denominator=np.linalg.norm(rays[0]-ratio*rays[1])
 if denominator<.02:return reject('post_bearings_too_close')
 ranges=np.array([1.,ratio])*(POST_SEPARATION_M/denominator)
 if np.min(ranges)<.15 or np.max(ranges)>3:return reject('post_range_outside_course')
 p=[origin+distance*ray for distance,ray in zip(ranges,rays)];height_error=float(np.mean([normal@v-ground.offset_m for v in p]));r['encoder_ground_height_discrepancy_m']=height_error
 if abs(height_error)>.025:return reject('known_baseline_conflicts_with_encoder_ground_height')
 # Pixel localization uncertainty is stated, not tuned using runtime truth.
 angular=math.acos(float(np.clip(np.dot(rays[0],rays[1]),-1,1)));pixel_angle=2./min(camera.fx,camera.fy);range_uncertainty=float(np.max(ranges)*pixel_angle/max(math.sin(angular),1e-6))
 r.update(accepted=True,reason='observed_goal_bearings_match_declared_baseline_and_height',post_points_reference_m=[v.tolist()for v in p],goal_center_reference_m=((p[0]+p[1])/2).tolist(),reference_frame=pose.reference_frame,capture_timestamp_s=capture_timestamp_s,post_slant_ranges_m=ranges.tolist(),estimated_range_uncertainty_m=range_uncertainty,uncertainty_scope='2px bearing allowance; baseline/FK/IMU systematic errors not calibrated')
 return r
