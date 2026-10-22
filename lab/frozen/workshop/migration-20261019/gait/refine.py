"""Fine-tune a selected native-BAM checkpoint for course precision; experimental until evaluated."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab.app import launch_simulation
from isaaclab.utils import to_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_tasks.utils import resolve_task_config
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--num-envs',type=int,default=1024)
    ap.add_argument('--iterations',type=int,default=2500)
    ap.add_argument('--seed',type=int,default=970111)
    ap.add_argument('--resume',type=Path,required=True)
    args=ap.parse_args()
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    task='IsaacContrib-Velocity-Flat-MicroDuck'
    cfg,agent=resolve_task_config(task,'rsl_rl_cfg_entry_point',overrides=['presets=backlash'])
    from precision_recipe import configure
    configure(cfg)
    cfg.seed=agent.seed=args.seed
    cfg.scene.num_envs=args.num_envs
    cfg.commands.base_velocity.debug_vis=False
    agent.device=cfg.sim.device='cuda:0'
    agent.save_interval=250
    agent.logger='tensorboard'
    cfg.log_dir=str(args.output)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    ledger={'schema':'microduck.training-ledger.v1','run_id':args.output.name,'skill':'walking',
        'runtime_id':'newton-bam-20261019','hardware_profile_id':'beta-gc2093-l8cx-v1',
        'initialization':'continuation' if args.resume else 'random',
        'parent_checkpoint':str(args.resume) if args.resume else None,
        'parent_checkpoint_sha256':sha(args.resume) if args.resume else None,
        'env_count':args.num_envs,'rollout_steps':agent.num_steps_per_env,
        'iterations_planned':args.iterations,'seed':args.seed,'task':task,'presets':['backlash'],
        'source_sha256':sha(__file__),'recipe_sha256':sha(Path(__file__).with_name('precision_recipe.py')),'recipe':'precision-camera-v1','domain_randomization':'official parameter coverage with final CoM ranges; course command/reward specialization; complete env.yaml',
        'control_hz':50,'physics_hz':200,'state':'starting',
        'shared_gpu':True,'evaluation_data_used_for_training':False}
    atomic(args.output/'training-ledger.json',ledger)
    with launch_simulation(cfg,{'device':'cuda:0','visualizer':[]}):
        base=gym.make(task,cfg=cfg).unwrapped
        env=RslRlVecEnvWrapper(base,clip_actions=agent.clip_actions)
        dump_yaml(str(args.output/'env.yaml'),cfg)
        dump_yaml(str(args.output/'agent.yaml'),agent)
        runner=OnPolicyRunner(env,to_dict(agent),log_dir=str(args.output),device='cuda:0')
        if args.resume:runner.load(str(args.resume))
        start_iteration=int(runner.current_learning_iteration)
        parent_updates=start_iteration+1
        ledger['parent_completed_updates']=parent_updates
        ledger['parent_run_id']=args.resume.parent.name
        runner.save(str(args.output/'initial.pt'))
        ledger['initial_checkpoint_sha256']=sha(args.output/'initial.pt')
        ledger['started_at_unix']=time.time();ledger['state']='training'
        atomic(args.output/'training-ledger.json',ledger)
        started=time.monotonic()
        original_save=runner.save
        def save_with_evidence(path,infos=None):
            original_save(path,infos)
            ledger.update(state='training',latest_checkpoint=str(path),latest_checkpoint_sha256=sha(path),
                          training_wall_seconds=time.monotonic()-started,
                          iterations_this_run=int(runner.current_learning_iteration)-start_iteration+1,
                          iterations_cumulative=parent_updates+int(runner.current_learning_iteration)-start_iteration+1)
            atomic(args.output/'training-ledger.json',ledger)
        runner.save=save_with_evidence
        try:
            runner.learn(num_learning_iterations=args.iterations,init_at_random_ep_len=True)
            runner.export_policy_to_onnx(str(args.output/'exported'),'policy.onnx')
            runner.export_policy_to_jit(str(args.output/'exported'),'policy.pt')
            ledger.update(state='completed',iterations_this_run=args.iterations,
                          selected_checkpoint_sha256=ledger['latest_checkpoint_sha256'],
                          training_wall_seconds=time.monotonic()-started,
                          exported_onnx_sha256=sha(args.output/'exported/policy.onnx'))
        except BaseException as exc:
            ledger.update(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                          error=repr(exc),training_wall_seconds=time.monotonic()-started)
            raise
        finally:
            atomic(args.output/'training-ledger.json',ledger)
            env.close()

if __name__=='__main__':main()
