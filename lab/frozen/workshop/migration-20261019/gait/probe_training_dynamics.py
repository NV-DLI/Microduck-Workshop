"""Independently check compiled dynamics after real startup/reset DR events."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from precision_recipe_v4 import configure

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError('Keep prior diagnostics')
sys.path.insert(0,str(Path(__file__).parents[1]/'skills/experiments/built_com'))
from validate_mass_matrix import compare
task='IsaacContrib-Velocity-Flat-MicroDuck'
cfg=parse_env_cfg(task,device='cuda:0',num_envs=16,overrides=['presets=backlash'])
configure(cfg);cfg.seed=974111;cfg.commands.base_velocity.debug_vis=False
with launch_simulation(cfg,{'device':'cuda:0','visualizer':[]}):
    env=gym.make(task,cfg=cfg).unwrapped
    solver=env.sim.backend_manager._solver
    arrays={};reports=[]
    for index in range(3):
        env.reset()
        for _ in range(5):env.step(torch.zeros(16,14,device='cuda:0'))
        result,matrices=compare(solver)
        reports.append({'reset_index':index,**result})
        arrays.update({f'reset_{index}_{k}':v for k,v in matrices.items()})
        for name in ['body_mass','body_ipos','body_inertia','body_iquat','dof_armature']:
            arrays[f'reset_{index}_{name}']=getattr(solver.mjw_model,name).numpy().copy()
        arrays[f'reset_{index}_actual_commands']=env.command_manager.get_command('base_velocity').cpu().numpy().copy()
    report={'state':'passed' if all(x['all_passed']for x in reports)else'failed',
        'scope':'Actual training environment startup mass/inertia and reset armature/BAM events, then Newton MJWarp mass matrix against fresh independent CPU MuJoCo',
        'skill_success_claimed':False,'resets':reports,'num_envs':16,'recipe':'bounded-axis-low-speed-v4',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'recipe_sha256':hashlib.sha256(Path(__file__).with_name('precision_recipe_v4.py').read_bytes()).hexdigest()}
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'post-events-arrays.npz',**arrays)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'state':report['state'],'max_errors':[max(c['max_absolute_error']for c in r['cases'])for r in reports]}),flush=True)
    env.close()
    if report['state']!='passed':raise RuntimeError('Post-event compiled dynamics mismatch')
