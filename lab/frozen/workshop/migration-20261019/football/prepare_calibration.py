#!/usr/bin/env python3
"""Freeze model-bound development requests; does not simulate or qualify them."""
import argparse,hashlib,itertools,json
from pathlib import Path
F=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def make(gait,kick):
    commands=[{'case':'zero','raw_policy_command':[0.,0.,0.]}]
    for axis,values in [('vx',[-.24,-.18,-.12,.08,.12,.16,.20,.24,.28]),('vy',[-.20,-.12,-.06,.06,.12,.20]),('yaw',[-.8,-.6,-.4,-.2,.2,.4,.6,.8])]:
        index=['vx','vy','yaw'].index(axis)
        for v in values:
            c=[0.,0.,0.];c[index]=v;commands.append({'case':f'{axis}{v:+.2f}','raw_policy_command':c})
    placements=[{'case':f'right-kick-{i:02}','ball_initial_body_xy_m':[x,y],'kick_duration_s':duration,'post_kick_recovery_s':4.}for i,(x,y,duration)in enumerate(itertools.product([.085,.105,.125],[-.015,-.035,-.055],[.5,1.]))]
    files={'gait':gait,'kick':kick,'head_fk':F/'hardware_head_fk.py','head_adapter':F/'head_adapter.py','foot_geometry':F/'sources/sole-collision-geometry-backlash.json','startup_support':F/'startup_support.py','body_target_bounds':F.parent/'gait/bounded-action-limits.json','target_limiter':F.parent/'gait/bounded_actions.py'}
    return {'schema':'microduck.beta-football-calibration-plan.v1','runtime_id':'newton-bam-20261019','hardware_profile_id':'beta-gc2093-l8cx-v1','scope':'development only; never reuse these seeds for frozen goal qualification','executed':False,'qualified':False,
        'sources':{n:{'path':str(p.resolve()),'sha256':sha(p)}for n,p in files.items()},'development_seeds':[975011,975113,975229,975347,975461],
        'gait':{'command_cases':commands,'seconds_per_case':10.,'measure_full_trace':True,'diagnostic_steady_window_s':[2.,10.],'physical_control_targets':{'approach_m_s':.06,'near_ball_m_s':.025,'turn_rad_s':[-.2,.2]},
                'fit_requirements':['all command signs measured independently','zero input measured separately; no hidden standing dead zone','record vx/vy/yaw cross-coupling and actual physical200Hz state','model/source/limiter/runtime hashes exact','no reuse of v7 or pre-BAM velocity and stop regressors','fit on development data only; include every failure'],
                'followup':'Run independent stop/restart validation with the same adapter and bounded action settings; do not convert mean tracking into frozen instantaneous-RMSE pass'},
        'kick':{'official_v8_source_is_same_hash_as_previous_known_clip':True,'not_enabled_in_controller':True,'prepared_pose_cases':placements,'development_seeds':[975509,975613,975719],
                'prepared_pose_scope':'Explicit initial ball placement for contact calibration only; final task must approach the ball using sensors and may not teleport',
                'preconditions':['measured IMU valid, all startup support fully removed','same frozen walk/stop model reaches a real stationary stance','kick actor61obs14action order/scale exactly verified','last-action observation is actually executed preceding target offset; not fabricated zero history'],
                'record':['actual200Hz right/left-foot-ball contact matrices','sphere motion and whole-ball goal geometry in evaluator only','raw and bounded target plus BAM delayed target','servo/effective output/passive soft-limit violation separately','fall, miss, contact impulse and post-kick recovery'],
                'selection':'Fit reachable ball-body placement envelope from all development trials; then verify on independent placements and full sensor-driven approach. Single lucky hit is not deployment evidence'},
        'truth_boundary':'Simulation poses and contact forces are calibration labels/independent scores only; deployed controller accepts real RGB/ToF/encoder/IMU and known map',
        'release_requirement':'Frozen180s empty and known-one-obstacle protocols remain unchanged; initialization failures stay in denominator'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--gait',type=Path,required=True);p.add_argument('--kick',type=Path,default=F/'sources/ball_kick_right-v8.onnx');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve model-bound plan revisions')
    data=make(a.gait,a.kick);a.output.write_text(json.dumps(data,indent=2)+'\n');print(json.dumps({'plan':str(a.output),'sha256':sha(a.output),'gait_cases':len(data['gait']['command_cases']),'kick_cases':len(data['kick']['prepared_pose_cases']),'executed':False}))
if __name__=='__main__':main()
