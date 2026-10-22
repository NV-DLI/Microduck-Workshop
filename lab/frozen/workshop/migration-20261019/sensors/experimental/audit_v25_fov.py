"""Offline scoring-only camera geometry audit; never changes the controller."""
import hashlib, importlib.util, json, math, sys
from pathlib import Path
import numpy as np
M=Path(__file__).resolve().parents[2]
O=M.parents[1]/'outputs/migration-20261019'
D=O/'football-v25-guarded-near-ball40s'
sys.path[:0]=[str(M/'football'),str(M/'sensors')]
from head_adapter import HeadAdapter
from hardware_head_fk import CourseHeadFK,quaternion_wxyz
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda f:[json.loads(l) for l in (D/f).read_text().splitlines()]
sensor=read('sensor-inputs.jsonl')[-1];detection=read('detections.jsonl')[-1];control=read('control.jsonl')[-1]
physics=min(read('physics-trace.jsonl'),key=lambda r:abs(r['timestamp_s']-sensor['timestamp_s']))
manifest=json.loads((D/'manifest.json').read_text());camera=manifest['sensors']['camera'];K=np.array(camera['K']);w,h=camera['width_px'],camera['height_px']
fk=CourseHeadFK();p=sensor['proprio'];cam=fk.forward_policy14(p['q_rad'],joint_names=p['joint_names'])['camera_raw_optical'];R=np.array(cam.rotation);c=np.array(cam.translation_m)
box=detection['detections'][0]['bbox_xyxy_px'];pixel=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2,1.]);ray=R@np.linalg.solve(K,pixel)
angles=lambda v:[float(np.arctan2(v[1],v[0])),float(np.arctan2(-v[2],np.linalg.norm(v[:2])))]
q=physics['root_quaternion_xyzw'];RB=np.array(quaternion_wxyz([q[3],*q[:3]]));ball=RB.T@(np.array(physics['ball_position_world_m'])-np.array(physics['root_position_world_m']))
# True ball used ONLY for this offline diagnostic; never sent to navigation.
radius=.035
# Halfspace margins are exact sphere-vs-perspective-frustum distances.
planes=np.array([[K[0,0],0,K[0,2]],[-K[0,0],0,w-1-K[0,2]],[0,K[1,1],K[1,2]],[0,-K[1,1],h-1-K[1,2]]]);planes/=np.linalg.norm(planes,axis=1)[:,None]
def projection(pose):
 local=np.array(pose.rotation).T@(ball-np.array(pose.translation_m));pixel=K@local;pixel=pixel[:2]/pixel[2]
 return {'ball_raw_optical_m':local.tolist(),'ball_center_pixel_scoring_only':pixel.tolist(),'sphere_frustum_margin_m':(planes@local-radius).tolist(),'minimum_margin_m':float((planes@local-radius).min()),'all_sphere_inside':bool((planes@local>=radius).all() and local[2]>radius)}
actual=projection(cam);adapter=HeadAdapter();grid=[]
for pitch in np.linspace(0,1.4,29):
 for yaw in np.linspace(-.4,.4,17):
  solution=adapter.solve((yaw,pitch));pose=fk.forward(dict(zip(solution['joint_names'],solution['desired_head_joint_rad'])))['camera_raw_optical'];v=projection(pose)
  grid.append({'command':[float(yaw),float(pitch)],'IK_accepted':solution['accepted'],'joint_q':solution['desired_head_joint_rad'],**v})
valid=[r for r in grid if r['IK_accepted']];best=max(valid,key=lambda r:r['minimum_margin_m']);head=[p['q_rad'][p['joint_names'].index(n)]for n in control['head_mapping']['joint_names']]
result={'schema':'microduck.v25-camera-fov-offline-audit.v1','inputs':{str(D/n):sha(D/n)for n in ('sensor-inputs.jsonl','detections.jsonl','control.jsonl','physics-trace.jsonl','manifest.json')},'audit_source_sha256':sha(__file__),'timestamp_s':sensor['timestamp_s'],'physics_timestamp_s':physics['timestamp_s'],'camera_body_translation_m':c.tolist(),'actual_optical_axis_body':R[:,2].tolist(),'actual_optical_axis_yaw_pitch_rad':angles(R[:,2]),'box':box,'visible_center_ray_gaze':angles(ray),'recorded_requested_gaze':control['trained_gaze_domain']['requested'],'fresh_pixel_FK_reproduces_request':bool(np.allclose(angles(ray),control['trained_gaze_domain']['requested'],atol=1e-12,rtol=0)),'measured_head_q_rad':head,'commanded_head_q_rad':control['head_mapping']['desired_head_joint_rad'],'actual':actual,'truth_only_camera_to_ball_gaze':angles(ball-c),'actual_camera_ball_angular_radius_deg':float(np.degrees(np.arcsin(radius/np.linalg.norm(ball-c)))),'reachable_grid':{'yaw_domain_rad':[-.4,.4],'pitch_domain_rad':[0,1.4],'count':len(grid),'accepted_IK_count':len(valid),'full_sphere_inside_count':sum(r['all_sphere_inside']for r in valid),'best':best},'limits':['K/FOV are explicit simulation assumptions, not physical Beta calibration.','Kinematic perfect-tracking grid at frozen actual body/ball state; does not establish dynamical attainability.','True world ball/root pose are scoring-only inputs. No controller/action/sensor file changed.','Near-vertical optical axes make yaw ill-conditioned; optical gaze does not equal head_yaw joint.'],'findings':['The late request is exactly reproducible from fresh visible pixels and same-frame encoders, not stale gaze.','Actual head pitch differs from commanded target, so repeating absolute observed bearing can settle at a persistent image error.','The top-clipped ball needs recentering; a partial visible-box center is not the hidden sphere center.']}
out=O/'football-v25-fov-independent-audit-v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed_fresh_FK':result['fresh_pixel_FK_reproduces_request'],'actual':actual,'grid':result['reachable_grid']}))
