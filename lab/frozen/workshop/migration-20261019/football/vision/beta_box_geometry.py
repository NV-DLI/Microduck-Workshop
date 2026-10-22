"""Beta YOLO-box sphere candidate with actual-image edge support, no teacher API.

Metric scale uses the declared .035m sphere radius. Box/edge errors and radius
are explicit assumptions; partial/cropped/unsupported boundaries are rejected.
"""
import math
import cv2,numpy as np

def estimate_box_sphere(rgb,box,*,radius_m,pose,ground,camera,capture_timestamp_s,now_s):
 r={'kind':'beta_YOLO_box_known_radius_v1','accepted':False,'reason':None,'point_reference_m':None,'box_used_for_metric_size':True,'observed_boundary_required':True,'missing_boundary_reconstructed':False,'truth_used':False,'task_or_hardware_validated':False,'radius_assumption_m':radius_m,'box_edge_uncertainty_px':None}
 def reject(reason):r['reason']=reason;return r
 if not 0<=now_s-capture_timestamp_s<=.2 or abs(pose.timestamp_s-capture_timestamp_s)>1e-6:return reject('stale_image_or_FK')
 b=np.asarray(box,float);x0,y0,x1,y1=b;w=x1-x0;h=y1-y0;cx=(x0+x1)/2;cy=(y0+y1)/2
 if min(w,h)<8 or not .65<w/h<1.5:return reject('unsupported_ball_box_shape')
 if x0<4 or y0<4 or x1>camera.width_px-5 or y1>camera.height_px-5:return reject('cropped_ball')
 # Canny evidence is used only to reject missing silhouette sectors. Returned
 # location is from measured detector extents, not an invented/full circle mask.
 pad=math.ceil(4+.08*max(w,h));left=max(0,int(x0)-pad);top=max(0,int(y0)-pad);right=min(camera.width_px,int(x1)+pad+1);bottom=min(camera.height_px,int(y1)+pad+1)
 crop=rgb[top:bottom,left:right];edges=cv2.Canny(cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY),35,90);yy,xx=np.nonzero(edges);dx=(xx+left-cx)/(w/2);dy=(yy+top-cy)/(h/2);rad=np.sqrt(dx*dx+dy*dy)
 uncertainty=max(2.,.04*max(w,h));r['box_edge_uncertainty_px']=uncertainty
 on=np.abs(rad-1)<=2*uncertainty/min(w,h);angles=np.arctan2(dy[on],dx[on]);bins=np.floor((angles+math.pi)/(2*math.pi)*24).astype(int)%24;supported=len(set(bins.tolist()));r['supported_boundary_sectors']=supported;r['required_boundary_sectors']=21
 if supported<21:return reject('partial_occlusion_or_box_boundary_unsupported')
 points=np.array([[cx,cy],[x0,cy],[x1,cy],[cx,y0],[cx,y1]],float);uv=cv2.undistortPoints(points[:,None,:],np.asarray(camera.K),np.asarray(camera.distortion_k1_k2_p1_p2_k3))[:,0,:];rays=np.column_stack((uv,np.ones(5)));rays/=np.linalg.norm(rays,axis=1)[:,None]
 half_angles=[math.acos(float(np.clip(np.dot(rays[0],v),-1,1)))for v in rays[1:]];alpha=float(np.mean(half_angles));distance=radius_m/math.sin(alpha)
 if not .06<distance<3.:return reject('size_range_outside_course')
 point=np.asarray(pose.translation_m)+np.asarray(pose.rotation_reference_from_raw)@(rays[0]*distance);height=float(np.dot(ground.normal,point)-ground.offset_m);r['estimated_center_height_above_plane_m']=height
 # A sphere should be near its declared support plane; no exact contact is
 # asserted. This checks scale/pose disagreement, never sets height by truth.
 if abs(height-radius_m)>.035:return reject('size_range_inconsistent_with_conditional_ground')
 relative_uncertainty=2*uncertainty/min(w,h);r.update(accepted=True,reason='YOLO_extent_and_observed_edges_consistent',point_reference_m=point.tolist(),reference_frame=pose.reference_frame,capture_timestamp_s=capture_timestamp_s,slant_range_m=distance,estimated_range_uncertainty_m=distance*relative_uncertainty,uncertainty_scope='box extent allowance only; lens, radius and FK systematic errors are additional',assumes_unoccluded_sphere=True,ground_contact_verified=False)
 return r
