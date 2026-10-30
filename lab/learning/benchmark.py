"""Plan paired task evaluations and summarize complete physical attempt logs."""
import argparse
import json
import math
import random
from pathlib import Path

METRICS = ['independent_success','unseen_start_success','data_efficiency','inference_latency','training_cost','multitask_extension']

def wilson(passed, total):
    if not total:return None
    z=1.95996398454;p=passed/total;d=1+z*z/total
    mid=(p+z*z/(2*total))/d
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0.,mid-half),min(1.,mid+half)]

def plan(seed, episodes):
    if not 1<=episodes<=1000:raise ValueError('Use 1–1000 episodes to keep seed blocks disjoint')
    rng=random.Random(seed)
    return dict(version=1,metrics=METRICS,
        methods=['traditional_rgb_tof','traditional_tof_missing','visual_resnet18_gru','groot_skill_policy'],
        learned_inputs=dict(rgb_history_lags=[72,48,24,0],state_dimension=127,task='Walk to the football, kick it into the goal with the right foot, and recover to standing.'),
        actions=dict(prediction_horizon=16,dimensions=8,execute_prefix=4,decision_hz=12.5,motor_hz=50),
        fixed=[dict(seed=seed+i,root_xy=[-.55,.04],ball_xy=[.02,0.],root_yaw_rad=0.)for i in range(episodes)],
        unseen=[dict(seed=seed+1000+i,root_xy=[round(rng.uniform(-.58,-.52),5),round(rng.uniform(.02,.06),5)],ball_xy=[round(rng.uniform(-.005,.045),5),round(rng.uniform(-.025,.025),5)],root_yaw_rad=0.)for i in range(episodes)],
        data_efficiency=dict(episode_budgets=[5,10,20,33],training_seeds=[seed+2001,seed+2002,seed+2003],nested_subsets=True,split_unit='complete physical episode',initialization='same independent pretrained starting weights at every budget; no warm-start from larger budget'),
        latency=dict(report=['prediction_median','prediction_p95','sensor_to_actuator_median','sensor_to_actuator_p95'],exclude='warm-up; report queued action retrieval separately',clock='wall time, not simulator time'),
        training_cost=dict(report=['feature_cache_seconds','training_seconds','peak_memory_bytes','training_device','all_attempts_compute'],pretraining_cost='not included; disclose prior pretraining'),
        multitask=dict(instructions=['Kick the football into the goal.','Walk near the football and stop without touching it.'],same_initial_observation=True,small_baseline_conditioning='task-ID for known tasks; text encoder for paraphrase tests',requires='separate demonstrations and task-specific success scorers; single-task scores do not establish language generalization'),
        rules=['Retain every planned attempt, including process failures.','Freeze the checkpoint before opening held-out results.','No detector, privileged object pose or expert fallback in learned policy evaluation.','Sensor ablation is a dependency test, not a redesigned monocular controller.','Separate familiar-range unseen combinations from out-of-range starts.','Simulation waiting for inference is not a real-time hardware test.'])

def summarize(attempts, expected, fixtures=None):
    if len(set(expected))!=len(expected):raise ValueError('Repeated planned seed')
    if [r['seed']for r in attempts]!=expected:raise ValueError('Incomplete, reordered or unexpected attempt denominator')
    successes=0;errors=0
    for row in attempts:
        if row.get('audit_error'):raise ValueError('Unresolved score audit error')
        if row.get('split')!='test':raise ValueError('Only independent test episodes belong in this report')
        if row.get('exit_code')!=0:errors+=1;continue
        s=row.get('score')
        if not isinstance(s,dict) or type(s.get('passed'))is not bool:raise ValueError('Missing physical task score')
        if s.get('seed')!=row['seed']:raise ValueError('Recorded score seed differs')
        try:
            from .walking_data import score as original_score
        except ImportError:
            from walking_data import score as original_score
        path=Path(row['path']).resolve()
        if original_score(path)!=s:raise ValueError('Recorded score differs from original physical evidence')
        if fixtures is not None:
            fixture=next(f for f in fixtures if f['seed']==row['seed'])
            actual=json.loads((path/'manifest.json').read_text())['fixture']
            pairs=list(zip(actual['root_pose_xyzw'][:2],fixture['root_xy']))+list(zip(actual['ball_pose_xyzw'][:2],fixture['ball_xy']))
            yaw=fixture['root_yaw_rad'];q=[0.,0.,math.sin(yaw/2),math.cos(yaw/2)]
            pairs+=list(zip(actual['root_pose_xyzw'][3:],q))
            if any(not math.isclose(a,b,abs_tol=1e-7)for a,b in pairs):raise ValueError('Actual physical start differs from planned fixture')
            scene=dict(ball_radius_m=.05,ball_mass_kg=.03,goal_plane_x_m=.68,goal_opening_width_m=.388,goal_opening_height_m=.244,scene_profile='indoor-trapezoid-official-ball-v2',layout='empty')
            if any(actual.get(k)!=v for k,v in scene.items()):raise ValueError('Physical scene differs')
        successes+=s['passed']
    return dict(attempted=len(expected),passed=successes,success_rate=successes/len(expected)if expected else None,
                wilson95=wilson(successes,len(expected)),process_failures=errors,
                note='Complete task score; detector accuracy and offline action error are not task success.')

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('plan');q.add_argument('--seed',type=int,default=120000);q.add_argument('--episodes',type=int,default=20)
    q=s.add_parser('report');q.add_argument('--attempts',type=Path,required=True);q.add_argument('--plan',type=Path,required=True);q.add_argument('--suite',choices=['fixed','unseen'],default='fixed')
    for q in s.choices.values():q.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.command=='plan':result=plan(a.seed,a.episodes)
    else:
        spec=json.loads(a.plan.read_text());result=summarize(json.loads(a.attempts.read_text()),[f['seed']for f in spec[a.suite]],spec[a.suite])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x')as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
