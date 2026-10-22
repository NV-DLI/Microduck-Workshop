"""Static model-kinematic view reachability; scoring truth never enters control."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

M=Path(__file__).resolve().parents[2];O=M.parents[1]/'outputs/migration-20261019'
D=O/'football-v37-closer-observation20s'
sys.path[:0]=[str(M/'football'),str(M/'sensors')]
from head_adapter import HeadAdapter,HOME
from hardware_head_fk import CourseHeadFK,HEAD_JOINTS,MJCF_PATH,quaternion_wxyz
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda name:[json.loads(x)for x in(D/name).read_text().splitlines()]


def main():
    sensor=read('sensor-inputs.jsonl')[-1];control=read('control.jsonl')[-1]
    detection=read('detections.jsonl')[-1]
    physical=min(read('physics-trace.jsonl'),key=lambda r:abs(r['timestamp_s']-sensor['timestamp_s']))
    manifest=json.loads((D/'manifest.json').read_text());camera=manifest['sensors']['camera'];K=np.array(camera['K']);width,height=camera['width_px'],camera['height_px']
    qroot=physical['root_quaternion_xyzw'];R=np.array(quaternion_wxyz([qroot[3],*qroot[:3]]))
    ball=R.T@(np.array(physical['ball_position_world_m'])-np.array(physical['root_position_world_m']))
    fk=CourseHeadFK();adapter=HeadAdapter();radius=.035
    planes=np.array([[K[0,0],0,K[0,2]],[-K[0,0],0,width-1-K[0,2]],
                     [0,K[1,1],K[1,2]],[0,-K[1,1],height-1-K[1,2]]])
    planes/=np.linalg.norm(planes,axis=1)[:,None]
    def project(q):
        pose=fk.forward(dict(zip(HEAD_JOINTS,map(float,q))))['camera_raw_optical']
        local=np.array(pose.rotation).T@(ball-np.array(pose.translation_m))
        margins=np.r_[planes@local-radius,local[2]-radius]
        pixel=K@local;pixel=pixel[:2]/pixel[2]
        return margins,pixel
    actual=np.array([sensor['proprio']['q_rad'][sensor['proprio']['joint_names'].index(name)]for name in HEAD_JOINTS])
    commanded=np.array(control['head_mapping']['desired_head_joint_rad']);native_caps=np.array([1.1,1.1,1.4,.31])
    model_bounds=adapter.limits
    command_bounds=np.column_stack((np.maximum(model_bounds[:,0],HOME-native_caps),np.minimum(model_bounds[:,1],HOME+native_caps)))
    def item(q):
        margins,pixel=project(q)
        return {'absolute_effective_q_rad':q.tolist(),'native_home_offsets_rad':(q-HOME).tolist(),
            'sphere_plane_margins_m':margins.tolist(),'minimum_margin_m':float(margins.min()),
            'sphere_fully_inside':bool((margins>=0).all()),'center_pixel':pixel.tolist()}
    def optimize(bounds):
        starts=[np.clip(actual,bounds[:,0],bounds[:,1]),np.clip(commanded,bounds[:,0],bounds[:,1]),np.clip(HOME,bounds[:,0],bounds[:,1])]
        starts.extend(qmc.scale(qmc.Sobol(4,scramble=True,seed=975720).random_base2(3),bounds[:,0],bounds[:,1]))
        outcomes=[]
        for initial in starts:
            margin=float(project(initial)[0].min())
            result=minimize(lambda v:-v[4],np.r_[initial,margin],method='SLSQP',
                bounds=[tuple(x)for x in bounds]+[(-1.,1.)],
                constraints={'type':'ineq','fun':lambda v:project(v[:4])[0]-v[4]},
                options={'maxiter':100,'ftol':1e-10})
            candidate=item(result.x[:4]);candidate['optimizer_converged']=bool(result.success)
            outcomes.append(candidate)
        best=max(outcomes,key=lambda r:r['minimum_margin_m'])
        return {'bounds_absolute_q_rad':bounds.tolist(),'starts':len(starts),'best_found':best,
                'full_visibility_candidates':sum(r['sphere_fully_inside']for r in outcomes)}
    result={'schema':'microduck.v37-static-view-command-envelope-review.v1','qualification_claimed':False,
        'timestamp_s':sensor['timestamp_s'],'truth_scope':'Frozen actual body and ball only for offline geometry scoring; no controller inputs or changes',
        'last_box':[r['bbox_xyxy_px']for r in detection['detections']if r['class_name']=='ball'],
        'actual_measured_head':item(actual),'commanded_if_perfect_tracking':item(commanded),
        'commanded_native_outside_source_envelope_rad':np.maximum(abs(commanded-HOME)-native_caps,0).tolist(),
        'source_command_envelope_offsets_rad':native_caps.tolist(),
        'current_MJCF_joint_range_reachability':optimize(model_bounds),
        'source_command_envelope_intersect_MJCF_reachability':optimize(command_bounds),
        'limitations':['A constructive positive-frustum-margin candidate proves static geometric view existence, not global optimization or dynamic tracking.',
            'MJCF joint ranges describe the current simulation model, not hardware-certified safe limits.',
            'Source command envelope is not identical to effective joint coordinates; this diagnostic assumes perfect command tracking only to ask if an in-envelope full view exists.',
            'Actual native policy follows commands indirectly and can produce different joints. No static candidate is an approved physical command.',
            'Camera K/FOV/extrinsics are simulation assumptions. Lens distortion, self-occlusion, motion, and electrical limits are not certified.'],
        'evidence_sha256':{str(p):sha(p)for p in [D/'manifest.json',D/'sensor-inputs.jsonl',D/'control.jsonl',D/'physics-trace.jsonl',D/'detections.jsonl',MJCF_PATH,M/'football/head_adapter.py',M/'football/hardware_head_fk.py',M/'football/sources/kick-controller-review/v8-bound/scripts__infer_policy.py',Path(__file__)]}}
    output=O/'football-v37-fov-independent-review-v1.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(output),'actual':result['actual_measured_head'],'commanded':result['commanded_if_perfect_tracking'],'out_of_envelope':result['commanded_native_outside_source_envelope_rad'],
        'model_best':result['current_MJCF_joint_range_reachability']['best_found'],'source_best':result['source_command_envelope_intersect_MJCF_reachability']['best_found']}))


if __name__=='__main__':main()
