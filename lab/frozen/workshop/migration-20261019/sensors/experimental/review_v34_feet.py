"""Offline actual28q -> named effective14 -> source CAD sole kinematics.

Private root pose is used only to score motion. No force/contact is inferred.
"""
from pathlib import Path
import json, hashlib, sys
import numpy as np
from scipy.spatial.transform import Rotation

M=Path(__file__).resolve().parents[2];O=M.parents[1]/'outputs/migration-20261019'
sys.path.insert(0,str(M/'football'))
from leg_odometry import LegFK
from hardware_head_fk import MJCF_PATH,MJCF_SHA256
R=O/'football-v34-waypoint-position20s';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    assert sha(MJCF_PATH)==MJCF_SHA256
    manifest=json.loads((R/'manifest.json').read_text());allnames=manifest['joint_names'];names=manifest['policy_joint_names']
    active=[allnames.index(n)for n in names];passive=[allnames.index('passive_'+n+'_backlash')for n in names]
    physics=[json.loads(x)for x in (R/'physics-trace.jsonl').read_text().splitlines()]
    turns=json.loads((O/'football-v34-turn-independent-review-v1.json').read_text())['segments']
    geometry_path=M/'football/sources/sole-collision-geometry.json';geometry=json.loads(geometry_path.read_text())
    fk=LegFK(MJCF_PATH,require_pinned=False)
    vertices={n:np.asarray(g['vertices_ankle_m'])for n,g in geometry.items()}
    long_axes={}
    for n,v in vertices.items():
        _,_,vh=np.linalg.svd(v-v.mean(0),full_matrices=False);long_axes[n]=vh[0]
    def pose(row):
        q=np.asarray(row['joint_positions_absolute_rad']);effective=q[active]+q[passive]
        rootR=Rotation.from_quat(row['root_quaternion_xyzw']).as_matrix();rootp=np.asarray(row['root_position_world_m']);feet={}
        for n,v in vertices.items():
            f=fk.foot(n,effective.tolist(),names);footR=np.asarray(f['rotation']);site=np.asarray(fk.paths[n][1])
            ankle=np.asarray(f['position_body_m'])-footR@site
            world=(ankle+v@footR.T)@rootR.T+rootp
            direction=rootR@footR@long_axes[n]
            feet[n]={'sole_centroid_world':world.mean(0),'sole_min_z':float(world[:,2].min()),
                     'long_axis_yaw':float(np.arctan2(direction[1],direction[0]))}
        return effective,feet
    out=[]
    for seg in turns:
        start=seg['turn']['start_s'];cap=seg['cap_timestamp_s'];end=seg['following_zero_command_window']['end_s']
        rows=[r for r in physics if start-1e-8<=r['timestamp_s']<=end+1e-8];computed=[pose(r)for r in rows]
        t=np.array([r['timestamp_s']for r in rows]);q=np.stack([c[0]for c in computed]);j=int(np.argmin(abs(t-cap)));details={}
        for n in vertices:
            p=np.stack([c[1][n]['sole_centroid_world']for c in computed]);z=np.array([c[1][n]['sole_min_z']for c in computed]);yaw=np.unwrap([c[1][n]['long_axis_yaw']for c in computed])
            details[n]={'centroid_xy_displacement_to_cap_m':(p[j,:2]-p[0,:2]).tolist(),
                        'centroid_xy_displacement_after_stop_m':(p[-1,:2]-p[j,:2]).tolist(),
                        'centroid_xy_max_excursion_during_turn_m':float(np.linalg.norm(p[:j+1,:2]-p[0,:2],axis=1).max()),
                        'sole_lowest_vertex_z_range_during_turn_m':[float(z[:j+1].min()),float(z[:j+1].max())],
                        'sole_lowest_vertex_height_span_m':float(np.ptp(z[:j+1])),
                        'sole_long_axis_yaw_delta_during_turn_rad':float(yaw[j]-yaw[0]),
                        'sole_long_axis_yaw_delta_after_stop_rad':float(yaw[-1]-yaw[j])}
        joint={n:{'start_rad':float(q[0,i]),'cap_rad':float(q[j,i]),'after_stop_rad':float(q[-1,i]),
                  'delta_to_cap_rad':float(q[j,i]-q[0,i]),'delta_after_stop_rad':float(q[-1,i]-q[j,i]),
                  'range_during_turn_rad':float(np.ptp(q[:j+1,i]))}for i,n in enumerate(names)if 'hip' in n or 'knee'in n or 'ankle'in n}
        out.append({'start_s':start,'cap_s':cap,'stop_end_s':end,'body_yaw_delta_rad':seg['turn']['actual_yaw_delta_rad'],
                    'body_yaw_after_stop_rad':seg['following_zero_command_window']['actual_yaw_delta_rad'],
                    'feet':details,'effective_joint_coordinates':joint})
    report={'schema':'microduck.v34-actual-joint-sole-kinematics-review.v1','qualification_claimed':False,
            'scope':'Offline scoring only: measured28joint coordinates collapsed by source-proven coaxial passive mapping, exact pinned October CAD and actual rootlink pose.',
            'contact_force_observed':False,'ground_contact_or_no_slip_claimed':False,
            'method':{'effective_q':'named q_servo+q_passive in declared14joint order','foot_points':'Actual source sole mesh vertices in ankle coordinates transformed through pinned LegFK','foot_heading':'Major principal axis of sole mesh; only within-segment unwrapped changes used, so sign ambiguity cancels','height':'WorldZ of lowest transformed authored sole vertex; not contact force or calibrated runtime collision margin.'},
            'segments':out,'limits':['No ground-pair trace exists in this run; cannot certify planted feet, no-slip contact, contact order or successful stepping.','CAD FK is a kinematic reconstruction. Solver deformation/margins and real hardware flex are absent.','A small height/excursion is evidence about motion size, not a categorical contact detector.'],
            'source_sha256':{str(p):sha(p)for p in [R/'manifest.json',R/'physics-trace.jsonl',MJCF_PATH,geometry_path,M/'football/leg_odometry.py',O/'official-joint-semantics-review-v1.json',O/'football-v34-turn-independent-review-v1.json',Path(__file__)]}}
    p=O/'football-v34-foot-kinematics-independent-review-v1.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'report':str(p),'segments':[{'start':s['start_s'],'body_yaw':s['body_yaw_delta_rad'],'hip_yaw':{n:s['effective_joint_coordinates'][n]['delta_to_cap_rad']for n in ['left_hip_yaw','right_hip_yaw']},'feet':s['feet']}for s in out]},indent=2))

if __name__=='__main__':main()
