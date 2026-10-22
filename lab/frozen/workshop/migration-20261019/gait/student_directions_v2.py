"""Independent student direction/release development collector v2. Original v1 retained.

Record first-episode Newton gait evidence at 200 Hz; no automatic acceptance.

The independent audit consumes the raw trace. Development and held-out runs
are separate; held-out execution requires a pre-existing, checkpoint-bound plan.
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES
from batch_policy import BatchPolicy

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor(v): return getattr(v,'torch',v)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',choices=['fixed','stop'],required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--per-command',type=int,default=64)
    p.add_argument('--split',choices=['development','heldout'],default='development')
    p.add_argument('--test-plan',type=Path)
    p.add_argument('--adapter',type=Path,help='Explicit 3-vector command mapping per named case')
    p.add_argument('--randomized',action='store_true')
    p.add_argument('--policy-recipe',choices=['official'],default='official')
    a=p.parse_args();model_sha=sha(a.model)
    cases=['forward','backward','turn_left','turn_right']
    desired_values={'forward':[.12,0.,0.],'backward':[-.12,0.,0.],'turn_left':[0.,0.,.4],'turn_right':[0.,0.,-.4]}
    # Four movement windows verify all three restarts after the three holds.
    seconds=20.
    if a.protocol!='fixed' or a.split!='development' or a.per_command!=3: p.error('This frozen supplemental check is 4 directions ×3 development environments only')
    if a.split=='heldout':
        if not a.test_plan: p.error('Held-out evaluation needs a pre-frozen test plan')
        plan=json.loads(a.test_plan.read_text())
        if not plan.get('frozen') or plan.get('checkpoint_sha256')!=model_sha or a.seed not in plan['seeds']:
            p.error('Frozen plan/checkpoint/seed mismatch')
        if plan.get('policy_recipe','official')!=a.policy_recipe:p.error('Frozen plan/action recipe mismatch')
        if a.per_command!=(64 if a.protocol=='fixed' else 32):p.error('Frozen trial count mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    task='IsaacContrib-Velocity-Flat-MicroDuck'
    cfg=parse_env_cfg(task,device='cuda:0',num_envs=a.per_command*len(cases),overrides=['presets=backlash'])
    recipe_hashes={}
    cfg.seed=a.seed;cfg.episode_length_s=seconds+5;cfg.curriculum=None
    cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.resampling_time_range=(1e8,1e8)
    cfg.commands.head_pose.ranges=tuple((0.,0.)for _ in range(4))
    cfg.commands.body_pose.ranges=tuple((0.,0.)for _ in range(6))
    # Nominal tests retain random initial placement/heading, but no force pushes.
    if not a.randomized:
        for name in ['foot_friction','encoder_bias','mass_inertia','randomize_com','randomize_head_com','randomize_joint_friction','randomize_armature','push_robot']:
            setattr(cfg.events,name,None)
        cfg.scene.robot.actuators['servos'].vin_range=None
        cfg.scene.robot.actuators['servos'].vin_drop_gain_range=None
        cfg.scene.robot.actuators['servos'].min_delay=4;cfg.scene.robot.actuators['servos'].max_delay=4
        cfg.observations.policy.enable_corruption=False
        cfg.observations.policy.imu.params.update(max_angle_deg=0,min_lag=0,max_lag=0)
    adapter=json.loads(a.adapter.read_text()) if a.adapter else desired_values
    if set(adapter)!=set(desired_values) or any(len(v)!=3 or not all(np.isfinite(v))for v in adapter.values()):
        raise ValueError('Adapter must declare forward/lateral/turn finite raw policy commands')
    started=time.monotonic()
    with launch_simulation(cfg,{'device':'cuda:0','visualizer':[]}):
        env=gym.make(task,cfg=cfg).unwrapped;robot=env.scene['robot'];n=env.num_envs
        if cfg.actions.joint_pos.scale!=1.0 or not cfg.actions.joint_pos.use_default_offset:raise RuntimeError('Student official action adapter mismatch')
        env.sim.backend_manager.set_decimation(1)
        env._physics_handles_decimation=False
        obs,_=env.reset();model=BatchPolicy(a.model,model_sha,'cuda:0')
        parity=model.parity(obs['policy'][:8],tolerance=1e-5)
        servo_ids,_=robot.find_joints(MICRODUCK_JOINT_NAMES,preserve_order=True)
        passive_ids,_=robot.find_joints(['passive_'+x+'_backlash'for x in MICRODUCK_JOINT_NAMES],preserve_order=True)
        limits=tensor(robot.data.joint_pos_limits).clone()
        groups=[name for name in cases for _ in range(a.per_command)]
        desired=torch.tensor([desired_values[name]for name in groups],device=env.device)
        mapped=torch.tensor([adapter[name]for name in groups],device=env.device)
        cmd=env.command_manager.get_term('base_velocity')
        fell=torch.zeros(n,dtype=torch.bool,device=env.device)
        rows=[];controls=[];targets=[];head_angular=[];tick=0
        head_ids,_=robot.find_bodies(['jaw_soft'],preserve_order=True)
        action_term=env.action_manager.get_term('joint_pos')
        step_start=time.monotonic()
        with torch.inference_mode():
            for step in range(round(seconds/.02)):
                moving=step*.02<8.  # 8 seconds commanded motion, then 12 seconds command release
                cmd.is_standing_env[:]=False;cmd.vel_command_b[:]=mapped if moving else 0.
                obs=env.observation_manager.compute();action=model(obs['policy'])
                controls.append(torch.cat([desired if moving else torch.zeros_like(desired),
                                           cmd.vel_command_b,action],dim=1).clone())
                env.action_manager.process_action(action)
                targets.append((action_term.processed_actions-action_term.encoder_bias).clone())
                for _ in range(4):
                    env._sim_step_counter+=1
                    env.action_manager.apply_action();env.scene.write_data_to_sim()
                    env.sim.step(render=False);env.scene.update(dt=.005);tick+=1
                    data=robot.data
                    gravity=tensor(data.projected_gravity_b)
                    fell|=gravity[:,2]>-.5
                    head_angular.append(tensor(data.body_ang_vel_w)[:,head_ids[0]].clone())
                    rows.append(torch.cat([tensor(data.root_pos_w),tensor(data.root_quat_w),
                        tensor(data.root_lin_vel_b),tensor(data.root_ang_vel_b),
                        tensor(data.joint_pos),gravity,fell[:,None].float()],dim=1).clone())
                env.episode_length_buf+=1;env.common_step_counter+=1
                # No autoreset: every trial remains its original physical trajectory.
                if 'interval' in env.event_manager.active_terms:
                    env.event_manager.apply(mode='interval',dt=.02)
                if step%500==0:print(json.dumps({'step':step,'falls':int(fell.sum()),'wall_s':time.monotonic()-step_start}),flush=True)
        trace=torch.stack(rows).cpu().numpy();control=torch.stack(controls).cpu().numpy()
        raw=a.output/'raw-trace.npz'
        np.savez_compressed(raw,state=trace,control=control,applied_output_target=torch.stack(targets).cpu().numpy(),head_angular_velocity_world=torch.stack(head_angular).cpu().numpy(),joint_limits=limits.cpu().numpy(),
            servo_ids=np.asarray(servo_ids),passive_ids=np.asarray(passive_ids),
            physics_dt=np.asarray(.005),control_dt=np.asarray(.02))
        report={'schema':'microduck.gait-physical-trace.v1','qualification_claimed':False,
            'runtime_id':'newton-bam-20261019','hardware_profile_id':'beta-gc2093-l8cx-v1',
            'protocol_id':'student-direction-release-v2','command_seconds':8.,'release_seconds':12.,
            'split':a.split,'seed':a.seed,'randomized':a.randomized,'groups':groups,
            'model':str(a.model.resolve()),'model_sha256':model_sha,'adapter':adapter,
            'adapter_sha256':sha(a.adapter)if a.adapter else None,
            'policy_recipe':a.policy_recipe,'recipe_hashes':recipe_hashes,'last_action_observation':'raw network action stored by official ActionManager; scale 1/default offset/no wrapper clip',
            'test_plan_sha256':sha(a.test_plan)if a.test_plan else None,
            'physics_hz':200,'policy_hz':50,'actual_physics_frames':tick,'joint_names':robot.joint_names,
            'servo_names':MICRODUCK_JOINT_NAMES,'state_columns':{'root_pos':[0,3],'quat_xyzw':[3,7],
            'linear_velocity_body':[7,10],'angular_velocity_body':[10,13],'joint_pos':[13,13+robot.num_joints],
            'projected_gravity':[13+robot.num_joints,16+robot.num_joints],'fell_latched':16+robot.num_joints},
            'control_columns':{'desired':[0,3],'policy_command':[3,6],'action':[6,20]},
            'parity':parity,'raw_trace':str(raw),'raw_trace_sha256':sha(raw),
            'source_sha256':sha(__file__),'step_wall_seconds':time.monotonic()-step_start,
            'wall_seconds':time.monotonic()-started,'no_autoreset':True}
        (a.output/'trace-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        env.close()

if __name__=='__main__':main()
