"""CPU reconstruction of recorded first-episode reward mass; never run physics.

Two unrecorded inputs (foot width, self-contact pair count) are NOT invented:
those negative terms are bounded using actual first-episode aggregate metrics.
All remaining terms use the saved 200 Hz physical trajectory and frozen source.
"""
from pathlib import Path
import hashlib, json, sys, importlib.util
import numpy as np
import torch

M=Path('@COURSE@/workshop/migration-20261019')
O=M.parents[1]/'outputs/migration-20261019';E=M/'skills/experiments'
sys.path[:0]=[str(E/'jump_release_v8'),str(E/'jump_reward_v7'),str(E/'jump_phase_v6'),str(M/'skills')]
from progress import Progress,Frame
from contract import HOME_POSE

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def audit(version):
 path=O/f'jump-{version}-short100-final-trace16-v1.json'
 evpath=O/f'jump-{version}-short100-final-eval16-v1.json'
 archive=np.load(path.with_suffix('.npz'));z={k:archive[k]for k in archive.files};meta=json.loads(path.read_text())['metadata'];ev=json.loads(evpath.read_text())
 n=z['active'].shape[1];footids=meta['foot_contact_body_indices'];nonfeet=[i for i in range(z['body_ground_force_world'].shape[2])if i not in footids]
 p=Progress(n);home=torch.tensor(HOME_POSE);previous=torch.zeros(n,dtype=torch.bool)
 known={};perstep=[];maskall=[];phase_mismatches={k:0 for k in ('takeoff','landed','valid_takeoff','phase_continuous_air')};known_raw={}
 for t in range(len(z['active'])):
  a={k:torch.from_numpy(z[k][t])for k in ('preparation_ready','active','COM_world','COM_velocity_world','sole_height','body_ground_force_world','root_height','actual_linear_velocity','actual_angular_velocity','COM_foot_distance','q_physical','qd_physical','tilt')}
  ready=a['preparation_ready']&a['active'];fresh=ready&~previous
  p.reset(torch.nonzero(~ready|fresh).flatten());p.supported[fresh]=True;p.support_time[fresh]=.5
  force=a['body_ground_force_world'];foot=force[:,footids].norm(dim=-1)>.08;nf=force[:,nonfeet]
  f=Frame(a['COM_world'][:,2],a['COM_velocity_world'][:,2],a['sole_height'],foot,nf.norm(dim=-1).amax(-1)>.08,nf.sum(1).norm(dim=-1),torch.zeros(n),a['tilt'].cos(),a['root_height'],a['actual_linear_velocity'],a['actual_angular_velocity'],a['COM_foot_distance'],a['q_physical']-home,a['qd_physical'],torch.full((n,),.07))
  old_launch_paid=p.launch_paid.clone();old_visible_paid=p.visible_paid.clone()
  p.update(f)
  if version=='v7':
   p.last_terms['launch_velocity']=2*(p.launch_paid-old_launch_paid).clamp(min=0)/.6
   p.last_terms['visible_height']=6*(p.visible_paid-old_visible_paid).clamp(min=0)/.012
  for k in phase_mismatches:
   value=p.phase.air_s if k=='phase_continuous_air'else getattr(p,k)
   ref=torch.from_numpy(z[k][t]);err=(value-ref).abs()>1e-6 if value.dtype!=torch.bool else value!=ref
   phase_mismatches[k]+=int((err&ready).sum())
  for k,v in p.last_terms.items():
   known.setdefault(k,torch.zeros(n));known[k]+=v*ready
   known_raw.setdefault(k,torch.zeros(n));known_raw[k]+=v*a['active']
  perstep.append(torch.stack([v*ready for v in p.last_terms.values()],-1));maskall.append(ready)
  previous=ready
 assert not any(phase_mismatches.values()),phase_mismatches
 arrays=torch.stack(perstep).numpy();mask=np.stack([x.numpy()for x in maskall]);terms=list(p.last_terms)
 ctrl=np.arange(3,len(mask),4);active=z['active'][ctrl];ready=z['preparation_ready'][ctrl]&active
 cost=(z['raw_mixed_action'][ctrl]-z['executed_action'][ctrl])**2
 gap=-.02*ev['bounded_controller']['options']['action_gap_cost']*cost.mean(-1)*ready
 physical=ev['qualification']['physical_integrity']
 assert max(physical['servo14_limit_violation_rad'])==0 and max(physical['physical14_output_limit_violation_rad'])==0
 count=int(active.sum());agg={k:float(v['mean']*count)for k,v in ev['first_episode_metrics'].items()if k.startswith('progress_reward_')}
 # Actual aggregate metrics are 4-tick reward masses. They must NOT be multiplied by control_dt again.
 missing=('post_touchdown_width','self_contact');omitted_negative_lower=sum(agg['progress_reward_'+k]for k in missing)
 # v7 uses grounded velocity/old height, but all other source terms and states are the same.
 validate={}
 for k in terms:
  if k in missing:continue
  actual=agg['progress_reward_'+k];replay=float(known_raw[k].sum());validate[k]={'actual_aggregate':actual,'replayed_aggregate':replay,'abs_error':abs(actual-replay)}
 assert max(x['abs_error']for x in validate.values())<2e-4,validate
 rows=[]
 for w in range(n):
  idx=np.flatnonzero(mask[:,w]);take=np.flatnonzero(mask[:,w]&z['takeoff'][:,w]);delta=z['projection_delta'][idx,w]
  positive=sum(float(known[k][w])for k in ('launch_velocity','valid_takeoff','visible_height','airtime','touchdown','settle','stable_landing','durable_landing','durable_visible_height'))
  net=float(arrays[:,w].sum()+gap[:,w].sum())
  vals={k:float(known[k][w])for k in terms if k not in missing};vals['raw_executed_target_cost']=float(gap[:,w].sum());vals['physical_violation_cost']=0.
  c=z['command'][idx,w,0];alpha=z['handoff_alpha'][idx,w];post=z['landed'][idx,w]
  rows.append({'world':w,'category':'confirmed_flight_failed'if len(take)else'no_confirmed_flight',
   'prepared':bool(len(idx)),'task_recorded_duration_s':len(idx)*.005,'task_clock_last_s':float(z['episode_time_s'][idx[-1],w]),
   'formal_success':bool(ev['qualification']['trial_success'][w]),'actual_base_fall':bool(np.any(z['tilt'][idx,w]>.8)|np.any(z['root_height'][idx,w]<.07)),
   'source_valid_takeoff':bool(z['valid_takeoff'][idx,w].any()),'known_rewards':vals,
   'positive_reward_total':positive,'net_return_upper_bound':net,'net_return_lower_bound':net+omitted_negative_lower,
   'bound_reason':'Only unknown per-world width/self-contact costs; actual aggregate negative mass is conservatively assigned in full to each world for the lower bound. Physical violation cost is proven zero.',
   'jump_command_on_seconds':float((c>0).sum()*.005),'actor_some_authority_seconds':float((alpha<1).sum()*.005),
   'authority_before_first_air':None if not len(take)else float(z['handoff_alpha'][take[0],w]),
   'residual_leg_fraction_near_0p12':float((np.abs(np.delete(delta,np.s_[5:9],axis=1))>.114).mean())})
 groups={}
 for cat in ('no_confirmed_flight','confirmed_flight_failed'):
  rr=[r for r in rows if r['category']==cat]
  groups[cat]={'worlds':[r['world']for r in rr],'n':len(rr),'mean_return_upper_bound':float(np.mean([r['net_return_upper_bound']for r in rr])),'mean_positive_reward':float(np.mean([r['positive_reward_total']for r in rr])),'failed_landing_cost_total':sum(r['known_rewards']['failed_landing']for r in rr)}
 return {'version':version,'reward_formula_used':'Actual frozen formula for this stage; v7 grounded-launch/height frontier or v8 confirmed-release/retrospective-height as applicable',
   'all_worlds_retained':True,'per_world':rows,'groups':groups,'phase_state_mismatches':phase_mismatches,
   'aggregate_actual_log_consistency':validate,'unknown_per_world_actual_cohort_costs':{k:agg['progress_reward_'+k]for k in missing},
   'cohort_width_self_costs_may_include_preparation_but_are_conservative_bounds':True,
   'environment_return_scope':'Undiscounted first-task environment reward, not value-bootstrap/GAE or success. Continuous costs already contain physics_dt; event rewards have no extra control_dt factor.',
   'evidence':{str(q):sha(q)for q in (path,path.with_suffix('.npz'),evpath)}}

def main():
 stages=[audit(v)for v in ('v7','v8')]
 sources=[Path(__file__),E/'jump_release_v8/progress.py',E/'jump_release_v8/release_rewards.py',E/'jump_reward_v7/progress.py',E/'jump_reward_v7/landing_gate.py',E/'jump_phase_v6/progress.py',E/'jump_progress_v3/environment.py',E/'jump_projected_v5/projected_env.py',E/'jump_progress_v3/actor_authority.py',E/'bounded/controller.py',E/'jump_residual_v4/handoff.py',M/'skills/isaac_skills.py']
 report={'schema':'microduck.jump-v8-actual-reward-budget-review.v2','source_sha256':{str(p):sha(p)for p in sources},'stages':stages,
  'dt_semantics':{'physical':.005,'control':.02,'physical_substeps_per_control':4,'progress_events_extra_dt_factor':1.,'continuous_terms_include_physics_dt':True,'bound_cost_extra_factor':.02,'environment_after_get_rewards_extra_dt':False},
  'findings':['There is no positive stand-still reward in this active jump recipe. Stable non-jumps avoid the -8 failed-landing event that tiny actual flights incur.','All formal landing bonuses are zero in these trajectories; the old positive recovery exploitation was already removed in v7.','v8 removes grounded launch credit, producing sparser small positive flight reward while retaining failed-landing cost. Actual behavior regresses toward no launch.','The jump command is only active for .75 s; fallback standing blending begins at .80 s and fully delegates at .94 s. Residual is therefore time-limited, though measured first-release alpha is zero.','This diagnosis does not authorize larger residual amplitude or another training run.'],
  'qualification_thresholds_unchanged':True,'new_optimizer_updates':0}
 path=O/'jump-v8-actual-reward-budget-review-v2.json'
 if path.exists():raise FileExistsError(path)
 path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps([{'version':x['version'],'groups':x['groups'],'omitted':x['unknown_per_world_actual_cohort_costs']}for x in stages]))
if __name__=='__main__':main()
