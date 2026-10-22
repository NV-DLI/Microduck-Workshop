"""Training-only geometry/depth labels. Never import into controller execution.

Uses actual RTX depth to exclude occluded projected surfaces. World geometry
and camera poses are privileged labels, kept in the dataset's teacher folder.
"""
import math
import numpy as np,cv2

def quaternion_matrix(q):
    x,y,z,w=np.asarray(q,float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def rays(profile):
    xx,yy=np.meshgrid(np.arange(profile.width_px),np.arange(profile.height_px));pixels=np.stack([xx,yy],axis=-1).astype(float)
    xy=cv2.undistortPoints(pixels.reshape(-1,1,2),np.asarray(profile.K),np.asarray(profile.distortion_k1_k2_p1_p2_k3))[:,0]
    d=np.column_stack([xy,np.ones(len(xy))]);return d/np.linalg.norm(d,axis=1)[:,None]

def sphere_depth(d,center,radius):
    projection=d@center;discriminant=projection**2-(np.dot(center,center)-radius**2)
    t=projection-np.sqrt(np.maximum(discriminant,0))
    return np.where((discriminant>=0)&(t>.03),t,np.inf)

def box_depth(origin,d,center,size):
    lo=np.asarray(center)-np.asarray(size)/2;hi=np.asarray(center)+np.asarray(size)/2
    safe=np.where(np.abs(d)<1e-12,np.copysign(1e-12,d+1e-20),d)
    a=(lo-origin)/safe;b=(hi-origin)/safe;near=np.minimum(a,b).max(axis=1);far=np.maximum(a,b).min(axis=1)
    return np.where((far>=near)&(near>.03),near,np.inf)

def labels(profile,camera_pose,measured_ray_depth,ball_world,*,ball_radius=.035):
    depth=np.asarray(measured_ray_depth,float)
    if depth.shape!=(profile.height_px,profile.width_px):raise ValueError('Actual raw-camera raster depth required')
    R=quaternion_matrix(camera_pose['quaternion_ros_xyzw']);origin=np.asarray(camera_pose['position_world_m']);d=rays(profile);world_d=d@R.T
    c=R.T@(np.asarray(ball_world)-origin);ball=sphere_depth(d,c,ball_radius)
    posts=[((.93,.1325,.09625),(.025,.025,.1925)),((.93,-.1325,.09625),(.025,.025,.1925)),((.93,0.,.1925),(.025,.29,.025))]
    goal=np.minimum.reduce([box_depth(origin,world_d,center,size)for center,size in posts])
    results=[];diagnostics=[];measured=depth.reshape(-1)
    for class_id,(name,expected) in enumerate([('ball',ball),('goal',goal)]):
        projected=np.isfinite(expected);comparable=projected&np.isfinite(measured)
        residual=np.full(measured.shape,np.inf);residual[comparable]=np.abs(measured[comparable]-expected[comparable]);visible=comparable&(residual<.015)
        projected_count=int(projected.sum());visible_count=int(visible.sum());fraction=visible_count/max(1,projected_count)
        diagnostic={'class_id':class_id,'class_name':name,'projected_pixels':projected_count,'visible_depth_matched_pixels':visible_count,'visible_fraction':fraction,'accepted':False}
        if visible_count>=20 and fraction>=.25:
            yy,xx=np.nonzero(visible.reshape(depth.shape));x0=max(0,int(xx.min())-1);x1=min(profile.width_px,int(xx.max())+2);y0=max(0,int(yy.min())-1);y1=min(profile.height_px,int(yy.max())+2)
            if x1-x0>=6 and y1-y0>=6:
                box=[x0,y0,x1,y1];result={'class_id':class_id,'class_name':name,'bbox_xyxy_px':box,'yolo_xywh':[(x0+x1)/2/profile.width_px,(y0+y1)/2/profile.height_px,(x1-x0)/profile.width_px,(y1-y0)/profile.height_px],'teacher_only':True}
                results.append(result);diagnostic.update(accepted=True,bbox_xyxy_px=box)
        diagnostics.append(diagnostic)
    return results,diagnostics
