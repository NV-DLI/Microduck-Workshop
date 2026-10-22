"""Floor observation from64real zone distances under the declared virtual model.

This reconstructs the existing nearest-zone sampling geometry, not raw depth.
Its assumptions are explicit:32x32 source raster/50deg,16subrays per45deg8x8
zone. It is not an ST photon/target-selection firmware model. Real hardware
requires independent zone-response calibration before enabling this estimator.
No root/object state, contact truth, or privileged camera pose is accepted.
"""
import math
import numpy as np

def fit_virtual_floor(tof_frame, proprio, head_fk):
    p=proprio
    if set(p)!={'timestamp_s','joint_names','q_rad','qd_rad_s','gyro_body_rad_s','gravity_body'}:raise ValueError('Encoder/IMU input only; no extra privileged fields')
    if abs(tof_frame['timestamp_s']-p['timestamp_s'])>1e-6:raise ValueError('Same-capture ToF/encoder input required')
    if len(tof_frame['cells'])!=8 or any(len(r)!=8 for r in tof_frame['cells']):raise ValueError('Exactly8x8 actual zones required')
    fk=head_fk.forward_policy14(p['q_rad'],joint_names=p['joint_names'])['tof_flu']
    R=np.asarray(fk.rotation);origin=np.asarray(fk.translation_m);points=[]
    normal=-np.asarray(p['gravity_body'],dtype=float)
    if normal.shape!=(3,)or not np.isfinite(normal).all()or not .5<np.linalg.norm(normal)<1.5:raise ValueError('Measured finite gravity direction required')
    normal/=np.linalg.norm(normal)
    frame=tof_frame;model=frame['model'];profile=frame['profile']
    if model.get('samples_per_zone_axis')!=4 or model.get('source_width_px')!=32 or model.get('source_height_px')!=32 or model.get('source_horizontal_fov_deg')!=50. or model.get('source_vertical_fov_deg')!=50. or profile.get('horizontal_fov_deg')!=45. or profile.get('vertical_fov_deg')!=45.:raise ValueError('Declared virtual32raster/16subray/45degree zone profile required')
    if model['implementation']!='angular-zone nearest-valid geometric approximation':raise ValueError('Explicit virtual sensor model required; this is not a generic ST firmware correction')
    for cell in sum(tof_frame['cells'],[]):
        if cell['status']=='valid'and not cell.get('contains_too_close',False)and cell['range_m']is not None and .1<=cell['range_m']<=2:
            if cell.get('no_return_samples')or cell.get('invalid_samples'):continue
            rays=[];count=model['samples_per_zone_axis']
            for subrow in range(count):
                for subcol in range(count):
                    az=math.radians(profile['horizontal_fov_deg']*(.5-(cell['col']+(subcol+.5)/count)/8))
                    el=math.radians(profile['vertical_fov_deg']*(.5-(cell['row']+(subrow+.5)/count)/8))
                    f=math.cos(el)*math.cos(az);left=math.cos(el)*math.sin(az);up=math.sin(el)
                    width=model['source_width_px'];height=model['source_height_px'];fx=width/(2*math.tan(math.radians(model['source_horizontal_fov_deg'])/2));fy=height/(2*math.tan(math.radians(model['source_vertical_fov_deg'])/2));cx=(width-1)/2;cy=(height-1)/2
                    col=math.floor(cx-fx*left/f+.5);rr=math.floor(cy-fy*up/f+.5)
                    if not 0<=col<width or not 0<=rr<height:continue
                    ray=np.array([1.,(cx-col)/fx,(cy-rr)/fy]);ray/=np.linalg.norm(ray);rays.append(ray.tolist())
            rays=np.asarray(rays);world=rays@R.T
            selected=world[np.argmin(world@normal)]
            points.append(origin+selected*cell['range_m'])
    points=np.asarray(points);normal=-np.asarray(p['gravity_body']);normal/=np.linalg.norm(normal)
    if len(points)<12:return {'accepted':False,'reason':'insufficient_returns','valid_returns':len(points)}
    rng=np.random.default_rng(979031);best=None
    for _ in range(256):
        a,b,c=points[rng.choice(len(points),3,replace=False)];n=np.cross(b-a,c-a);size=np.linalg.norm(n)
        if size<1e-5:continue
        n/=size
        if n@normal<0:n=-n
        d=n@a
        if n@normal<np.cos(np.deg2rad(25))or not -.22<d<-.05:continue
        residual=np.abs(points@n-d);inliers=residual<.005
        score=(int(inliers.sum()),-float(np.median(residual[inliers]))if inliers.any()else-1)
        if best is None or score>best[0]:best=(score,inliers)
    if best is None or best[0][0]<12:return {'accepted':False,'reason':'no_consistent_floor','valid_returns':len(points)}
    q=points[best[1]];center=q.mean(axis=0);_,sv,v=np.linalg.svd(q-center);n=v[-1]
    if n@normal<0:n=-n
    d=float(n@center);residual=np.abs(q@n-d)
    return {'accepted':True,'normal_body':n.tolist(),'offset_m':d,'valid_returns':len(points),'inliers':len(q),
            'residual_rms_m':float(np.sqrt(np.mean(residual**2))),'second_spread_m':float(sv[1]/np.sqrt(len(q))),
            'angle_from_imu_deg':float(np.rad2deg(np.arccos(np.clip(n@normal,-1,1)))),
            'truth_used':False,'tof_aggregation_bias_calibrated':False,'virtual_forward_model':'nearest-zone rasterized subray from declared virtual sensor projection; no raw depth array consumed'}
