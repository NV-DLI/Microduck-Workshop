"""Independent CPU-only recipe comparison; no scene or simulation is launched."""
from pathlib import Path
import copy,hashlib,inspect,json,sys
import torch
C=Path('@COURSE@');M=C/'workshop/migration-20261019';G=M/'gait';D=G/'experiments/deep_standing_d1';O=C/'outputs/migration-20261019'
sys.path[:0]=[str(D),str(G)]
import isaaclab_tasks
from isaaclab_tasks.utils import resolve_task_config
from isaaclab.utils import to_dict
from stop_gaze_recipe_v7 import configure as old
from commands import configure as new
from rsl_rl.algorithms import PPO
cfg,agent=resolve_task_config('IsaacContrib-Velocity-Flat-MicroDuck','rsl_rl_cfg_entry_point',overrides=['presets=backlash'])
other=copy.deepcopy(cfg);old(cfg);new(other)
def differences(a,b,path=''):
    if type(a)!=type(b):return[path+':type']
    if isinstance(a,dict):return sum((differences(a[k],b[k],path+'.'+k)for k in a),[])if a.keys()==b.keys()else[path+':keys']
    if isinstance(a,(list,tuple)):return sum((differences(v,w,path+f'[{i}]')for i,(v,w)in enumerate(zip(a,b))),[])if len(a)==len(b)else[path+':len']
    return []if a==b else[path]
diff=differences(to_dict(cfg),to_dict(other))
grid=json.loads((D/'command-grid.json').read_text());source=Path(grid['source_table']);original=json.loads(source.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pairs={tuple(r['gaze_body_yaw_pitch_rad'])for r in grid['rows']}
expected={(y,p)for y in (-.2,0.,.2)for p in (0.,.7,1.,1.2)}
train=(D/'train.py').read_text();hook=G/'experiments/body50_v9/environment_hook.py';h=hook.read_text()
checks={'only_declared_base_and_head_command_config_diff':bool(diff)and all(p.startswith('.commands.base_velocity.')or p.startswith('.commands.head_pose.')for p in diff),
 'no_reward_action_physics_DR_or_observation_config_change':all(not p.startswith(('.rewards.','.actions.','.sim.','.events.','.observations.'))for p in diff),
 'all12_grid_pairs_exact':pairs==expected and len(grid['rows'])==12,
 'grid_rows_are_exact_parent_IK_rows':all(r in original['rows']for r in grid['rows']),
 'source_grid_hash_matches':sha(source)==grid['source_sha256'],
 'dwell20to24':other.commands.head_pose.resampling_time_range==(20.,24.),
 'body_pose_remains_zero':all(a==b==0 for a,b in cfg.commands.body_pose.ranges),
 'body50_inactive_commands_are_zeroed':"for key in ('base_velocity', 'head_pose', 'body_pose'):"in h and 'value[~env.task_active_mask] = 0'in h,
 'actual_actor_hook_asserts_prep_head_zero_and_body_zero':"command_audit['preparation_head_nonzero'] or command_audit['body_nonzero']"in train,
 'parent_all_states_initial_equality_required':"('actor_state_dict','critic_state_dict','optimizer_state_dict')"in train and "if not all(ledger['initial_state_exact'].values()):raise RuntimeError"in train,
 'installed_RSL_restores_adaptive_LR_from_Adam': 'self.learning_rate = self.optimizer.param_groups[0]["lr"]'in inspect.getsource(PPO.load),
 'prewarm_precedes_learn_and_no_optimizer_in_prewarm':train.index('for _ in range(400):')<train.index('runner.learn(') and 'optimizer.step'not in train[train.index('for _ in range(400):'):train.index('runner.learn(')],
 'same_masked_PPO_implementation_imported':'from ppo_mask import install'in train and 'masking = install(runner.alg, base)'in train}
files=[Path(__file__),D/'train.py',D/'commands.py',D/'command-grid.json',D/'plan.json',hook,G/'experiments/body50_v9/ppo_mask.py',G/'gaze_recipe_v6.py',G/'stop_gaze_recipe_v7.py',source,Path(inspect.getfile(PPO))]
r={'schema':'microduck.deep-standing-d1-independent-cpu-review.v1','all_checks_passed':all(checks.values()),'checks':checks,'actual_cpu_config_differences':diff,'evidence':{str(p):sha(p)for p in files},'no_gpu_job_started':True,'scope':'Independent actual CPU config comparison and source-level contract review. Actual initialized actor/critic/Adam equality, startup-head readback, nonzero PPO updates and dynamics require the separately run GPU smoke; this report does not claim those have occurred.'}
out=O/'deep-standing-d1-independent-cpu-review-v1.json';out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'output':str(out),'passed':r['all_checks_passed'],'checks':checks}))
