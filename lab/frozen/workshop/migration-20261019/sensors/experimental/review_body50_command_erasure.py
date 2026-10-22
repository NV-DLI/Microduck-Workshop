"""CPU extraction of actual command methods; no Kit, source edits or GPU."""
from pathlib import Path
import ast,hashlib,json,types
import torch

M=Path(__file__).resolve().parents[2]
L=M/'runtime/IsaacLab-5088b326c7fc81caa7899395b0176ad08b89e9e1'
MAN=L/'source/isaaclab/isaaclab/managers/command_manager.py'
VEL=L/'source/isaaclab/isaaclab/envs/mdp/commands/velocity_command.py'
CMD=L/'source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/mdp/commands.py'
CFG=L/'source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/flat_env_cfg.py'
HOOK=M/'gait/experiments/body50_v9/environment_hook.py'

def function(path, cls, name, namespace):
 tree=ast.parse(path.read_text());scope=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name==cls)
 node=next(n for n in scope.body if isinstance(n,ast.FunctionDef)and n.name==name)
 node.decorator_list=[];node.returns=None
 for arg in node.args.args+node.args.kwonlyargs:arg.annotation=None
 unit=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));exec(compile(unit,str(path),'exec'),namespace)
 return namespace[name]

def run():
 ns={'torch':torch,'index_fill_':lambda value,ids,fill:value.__setitem__(ids,fill)}
 compute=function(MAN,'CommandTerm','compute',ns);resample=function(MAN,'CommandTerm','_resample',ns)
 velcmd=function(VEL,'UniformVelocityCommand','command',ns);velupdate=function(VEL,'UniformVelocityCommand','_update_command',ns);velresample=function(VEL,'UniformVelocityCommand','_resample_command',ns)
 posecmd=function(CMD,'UniformPoseDeltaCommand','command',ns);poseupdate=function(CMD,'UniformPoseDeltaCommand','_update_command',ns);poseresample=function(CMD,'UniformPoseDeltaCommand','_resample_command',ns)
 Base=type('ActualCommandMethods',(),dict(compute=compute,_resample=resample,_update_metrics=lambda self:None))
 V=type('Velocity',(Base,),dict(command=property(velcmd),_update_command=velupdate,_resample_command=velresample))
 P=type('Pose',(Base,),dict(command=property(posecmd),_update_command=poseupdate,_resample_command=poseresample))
 v=V();v.num_envs=2;v.device='cpu';v.vel_command_b=torch.zeros(2,3);v.is_standing_env=torch.tensor([False,True]);v.cfg=types.SimpleNamespace(heading_command=False,rel_standing_envs=0.,resampling_time_range=(3.,3.),ranges=types.SimpleNamespace(lin_vel_x=(.22,.22),lin_vel_y=(.1,.1),ang_vel_z=(.3,.3)))
 def pose(dim):
  p=P();p.num_envs=2;p.device='cpu';p.dim=dim;p._command=torch.zeros(2,dim);p.cfg=types.SimpleNamespace(resampling_time_range=(2.,2.),ranges=[(.2,.2)]*dim);return p
 terms={'base_velocity':v,'head_pose':pose(4),'body_pose':pose(6)}
 for term in terms.values():term.time_left=torch.zeros(2);term.command_counter=torch.zeros(2,dtype=torch.long);term._resample(torch.arange(2))
 # Preserve an explicit standing row to distinguish the source's own intended
 # standing mask from the hook's accidental erasure of every inactive row.
 v.is_standing_env[1]=True
 mgr=types.SimpleNamespace(_terms=terms)
 mgr.compute=types.MethodType(function(MAN,'CommandManager','compute',ns),mgr)
 mgr.get_command=types.MethodType(function(MAN,'CommandManager','get_command',ns),mgr)
 env=types.SimpleNamespace(command_manager=mgr,task_active_mask=torch.zeros(2,dtype=torch.bool))
 tree=ast.parse(HOOK.read_text());node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='compute_commands')
 context={'commands':mgr.compute,'env':env};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(HOOK),'exec'),context)
 actual_hook=context['compute_commands'];timeline=[]
 for tick in range(1,201):
  t=tick*.02;env.task_active_mask[:]=t>=1.5
  v.is_standing_env[1]=True
  actual_hook(.02)
  timeline.append({'time_s':t,'task_active':bool(env.task_active_mask[0]),'base':v.command.tolist(),'head':terms['head_pose'].command.tolist(),'body':terms['body_pose'].command.tolist(),'base_standing_mask':v.is_standing_env.tolist()})
 first=lambda k:next(row['time_s']for row in timeline if row['task_active']and any(abs(x)>0 for x in row[k][0]))
 checks={'get_command_aliases_live_buffer':mgr.get_command('base_velocity').data_ptr()==v.vel_command_b.data_ptr(),
  'nonstanding_row_erased_after_release':timeline[74]['base'][0]==[0.,0.,0.]and not timeline[74]['base_standing_mask'][0],
  'head_erased_after_release':timeline[74]['head'][0]==[0.]*4,
  'body_erased_after_release':timeline[74]['body'][0]==[0.]*6,
  'velocity_only_restored_at_resample':first('base')>=3.,
  'head_and_body_only_restored_at_resample':first('head')>=2. and first('body')>=2.}
 sources=[MAN,VEL,CMD,CFG,HOOK,Path(__file__),M/'gait/experiments/native_from_scratch_v1/recipe.py']
 report={'schema':'microduck.body50-command-erasure-independent-cpu.v1','passed':all(checks.values()),'checks':checks,
  'source_methods_executed':'AST extraction unchanged method bodies: CommandTerm.compute/_resample, CommandManager.compute/get_command, UniformVelocity command/update/resample, UniformPoseDelta command/update/resample, actual hook compute_commands. Only annotations/decorators removed; CPU index_fill helper uses equivalent tensor indexed assignment.',
  'controlled_timers':{'task_release_s':1.5,'base_resample_s':3.,'head_body_resample_s':2.,'standing_env0':False,'cpu_only':True},
  'observed_first_nonzero_after_release_s':{k:first(k)for k in ['base','head','body']},
  'source_actual_ranges_s':{'base_velocity':[3,8],'head_pose':[2,5],'body_pose':[2,5]},
  'conclusion':'The hook mutates each persistent command buffer during startup. Transition to task_active does not restore the sampled command. Nonstanding rows remain zero until next resample; a resample during startup is erased again. This is separate from the intentional standing mask.',
  'limitations':['This proves source mechanism, not the percentage of a specific GPU run affected. Existing trace/actor command histories must quantify exposure.','No frozen file or active run altered.','MicroDuck-specific forward/turn resampling buckets inherit the same buffer and compute semantics; their sampling probabilities were not exercised in this minimal reproduction.'],
  'timeline':timeline,'source_sha256':{str(q):hashlib.sha256(q.read_bytes()).hexdigest()for q in sources}}
 out=M.parents[1]/'outputs/migration-20261019/body50-command-startup-erasure-independent-v1.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'output':str(out),'checks':checks,'first_nonzero':report['observed_first_nonzero_after_release_s']}))

if __name__=='__main__':run()
