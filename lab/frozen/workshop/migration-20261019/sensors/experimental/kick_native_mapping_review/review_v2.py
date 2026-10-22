"""Independent CPU review; imports no Isaac runtime and changes no evaluated source."""
import ast,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
M=Path(__file__).resolve().parents[3];K=M/'football/kick_native_target_ab_v1';G=M/'gait';O=M.parents[1]/'outputs/migration-20261019'
sys.path.insert(0,str(G));from bounded_actions import BoundedActions
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Bias:
    def process_actions(self,actions):self.processed_actions=actions*self._scale+self._offset
class Bounded(Bias):
    def process_actions(self,actions):raise AssertionError('Forbidden extra bounded processing')
module=ast.parse((K/'mapping.py').read_text());module.body=[x for x in module.body if isinstance(x,ast.ClassDef)]
ns={'torch':torch,'T':lambda x:x,'BiasedJointPositionAction':Bias,'BoundedTrainingAction':Bounded}
exec(compile(module,str(K/'mapping.py'),'exec'),ns);Candidate=ns['NativePreparationAction']
gen=torch.Generator().manual_seed(976181);N=64
q=torch.randn(N,28,generator=gen)*.01;limits=torch.tensor([[-1.,1.]]*14).expand(N,-1,-1).clone();limits28=torch.cat((limits,limits),1)
raw=torch.randn(N,14,generator=gen)*3;home=torch.randn(14,generator=gen)*.1;bias=torch.randn(N,14,generator=gen)*.01
phase=torch.where(torch.arange(N)%2==0,torch.tensor(-1.),torch.tensor(.1));checks={}
def make(mapping):
 t=Candidate();t.mapping=mapping;t.num_envs=N;t.device='cpu';t._env=NS(kick_training=NS(clock=NS(phase=phase)))
 t._asset=NS(data=NS(joint_pos=q.clone(),joint_pos_limits=limits28.clone()));t._joint_ids=list(range(14));t.passive_ids=list(range(14,28));t._scale=1.;t._offset=home;t.encoder_bias=bias;t.target_excess=torch.zeros(N)
 t.limiter=BoundedActions([[-1.,1.]]*14,margin_rad=.035,max_target_rate_rad_s=8.);return t
native=make('official_raw');native.process_actions(raw)
checks['native_process_raw_scale_home_bitexact']=torch.equal(native.processed_actions,raw+home)
checks['native_previous_is_actual_home_raw_minus_bias']=torch.equal(native.limiter.previous,raw+home-bias)
checks['native_diagnostic_excess_zero']=bool((native.target_excess==0).all())
project=make('course_projected');project.process_actions(raw);active=phase>=0
checks['preparation_rows_unaltered_bitexact']=torch.equal(project.processed_actions[~active],(raw+home)[~active])
checks['task_output_bounded']=bool(((project.processed_actions-bias)[active].abs()<=.965001).all())
checks['physical_state_untouched']=torch.equal(project._asset.data.joint_pos,q)
checks['previous_bias_applied_once']=torch.equal(project.limiter.previous,project.processed_actions-bias)
evalsrc=(K/'evaluate.py').read_text();envsrc=(K/'environment.py').read_text()
checks.update(zero_residual_forced='args.zero_residual=True' in evalsrc,
 learner_inference_disabled='and not args.zero_residual' in evalsrc,
 raw_history_restored='self.body.previous_raw.copy_(torch.where(env.ready_mask[:,None],merged,torch.zeros_like(merged)))' in envsrc,
 task_only_teacher_projection='torch.where((self.clock.phase>=0)[:,None],requested,merged)' in envsrc,
 scale_one_asserted="torch.all(torch.as_tensor(env.action_manager.get_term('joint_pos')._scale)==1.)" in envsrc,
 finite_sensor_before_compile=evalsrc.index('install_before_first_tick(hook.body.producer)')<evalsrc.index('compiled=compile_sensor(hook.body.producer)'),
 no_automatic_resets="if bool((term|trunc).any()):raise RuntimeError" in evalsrc,
 all_worlds_saved='finish_tick=finish_tick.cpu().numpy()' in evalsrc)
checks['upstream_clip_none_runtime_guard']='cfg.actions.joint_pos.clip is not None' in evalsrc
files=[Path(__file__),K/'mapping.py',K/'environment.py',K/'evaluate.py',K/'contract.py',K/'target_authority.py',K/'protocol.json',G/'bounded_actions.py',G/'precision_recipe_v3.py']
r={'schema':'microduck.kick-native-mapping-independent-cpu-review.v2','state':'passed'if all(checks.values())else'failed','checks':checks,'source_sha256':{str(p):sha(p)for p in files},'scope':'CPU exact class-body tests with original BoundedActions and an explicitly emulated upstream affine process. Full native BAM and runtime traces require independent GPU evidence. No evaluated source modified.',
'findings':['Raw native arm calls original BiasedJointPositionAction.process_actions directly, so BoundedTrainingAction.process_actions does not run; scale1/home mapping remains, encoder bias is subtracted exactly once later by official apply_actions.','Both arms use raw teacher during phase<0; projected task arm first projects individual experts then uses existing target slew/output box. Original raw mixed history is restored for actor observations.','The limiter previous field is a target history in native preparation, not measured state. It initializes projected task from the actual last native request.','Zero residual is enforced regardless of CLI; loaded K7 learner is provenance only and its forward is never used.'],
'caveats':['Finished worlds continue physical evolution while others finish; all scoring must truncate by each finish_tick. This is already explicit in the collector conditions, not after-failure recovery evidence.','The final collector asserts clip=None at runtime. Actual traces must still verify common pre-task state parity; this CPU review cannot establish physical parity.']}
out=O/'kick-native-target-ab-independent-cpu-review-v2.json';assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'state':r['state'],'checks':checks}));assert all(checks.values())
