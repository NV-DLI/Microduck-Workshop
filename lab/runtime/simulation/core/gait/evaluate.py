"""Evaluate: Microduck simulation and control."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
import isaaclab_tasks
from isaaclab.app import launch_simulation
from isaaclab_tasks.utils import parse_env_cfg
from batch_policy import BatchPolicy

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--num-envs',type=int,default=24)
    ap.add_argument('--seconds',type=float,default=20.)
    ap.add_argument('--seed',type=int,default=970031)
    ap.add_argument('--backlash',action='store_true')
    ap.add_argument('--randomized',action='store_true')
    ap.add_argument('--sweep',action='store_true',help='Development command calibration; never held-out evaluation')
    args=ap.parse_args()
    task='IsaacContrib-Velocity-Flat-MicroDuck'
    cfg=parse_env_cfg(task,device='cuda:0',num_envs=args.num_envs,
                      overrides=['presets=backlash'] if args.backlash else [])
    cfg.seed=args.seed
    cfg.episode_length_s=args.seconds+5
    cfg.commands.base_velocity.debug_vis=False
    cfg.commands.base_velocity.resampling_time_range=(1e8,1e8)
    cfg.commands.head_pose.ranges=tuple((0.,0.) for _ in range(4))
    cfg.commands.body_pose.ranges=tuple((0.,0.) for _ in range(6))
    cfg.curriculum=None
    cfg.events.reset_base.params['pose_range']={}
    # DR includes voltage, friction, mass/CoM, encoder error, sensor noise and push.
    if not args.randomized:
        for name in ['foot_friction','encoder_bias','mass_inertia','randomize_com',
                     'randomize_head_com','randomize_joint_friction','randomize_armature','push_robot']:
            setattr(cfg.events,name,None)
        cfg.scene.robot.actuators['servos'].vin_range=None
        cfg.scene.robot.actuators['servos'].vin_drop_gain_range=None
        cfg.scene.robot.actuators['servos'].min_delay=4
        cfg.scene.robot.actuators['servos'].max_delay=4
        cfg.observations.policy.enable_corruption=False
        cfg.observations.policy.imu.params.update(max_angle_deg=0,min_lag=0,max_lag=0)
    args.output.mkdir(parents=True,exist_ok=False)
    sha=hashlib.sha256(args.model.read_bytes()).hexdigest()
    started=time.time()
    with launch_simulation(cfg,{'headless':True,'device':'cuda:0','visualizer':[]}):
        env=gym.make(task,cfg=cfg).unwrapped
        robot=env.scene['robot']
        model=BatchPolicy(args.model,sha,'cuda:0')
        obs,_=env.reset()
        parity=model.parity(obs['policy'][:min(8,args.num_envs)],tolerance=1e-5)
        command=env.command_manager.get_term('base_velocity')
        command_values=([[x,0.,0.] for x in [0.,.08,.12,.16,.2,.24,.3,.4]]+
            [[0.,0.,x] for x in [-1.,-.85,-.7,-.55,-.4,-.25,.25,.4,.55,.7,.85,1.]]) if args.sweep else [
            [0.,0.,0.],[.12,0.,0.],[.22,0.,0.],[0.,0.,.5],[0.,0.,-.5],[.1,0.,.4]]
        values=torch.tensor(command_values,device='cuda:0')
        commands=values[torch.arange(args.num_envs,device='cuda:0')%len(values)]
        command.is_standing_env[:]=False
        command.vel_command_b[:]=commands
        obs=env.observation_manager.compute()
        frames=[];failed=torch.zeros(args.num_envs,dtype=torch.bool,device='cuda:0')
        begin=time.time()
        with torch.inference_mode():
            for step in range(round(args.seconds/.02)):
                action=model(obs['policy'])
                obs,rew,terminated,truncated,info=env.step(action)
                failed|=terminated
                command.is_standing_env[:]=False
                command.vel_command_b[:]=commands
                data=getattr(robot.data,'torch',robot.data)
                if step % 5 == 0:
                    frames.append({'t':(step+1)*.02,'root_pos':data.root_pos_w.cpu().numpy(),
                        'root_quat':data.root_quat_w.cpu().numpy(),'lin_vel_b':data.root_lin_vel_b.cpu().numpy(),
                        'ang_vel_b':data.root_ang_vel_b.cpu().numpy(),
                        'joint_pos':data.joint_pos.cpu().numpy(),'action':action.cpu().numpy(),
                        'observed_command':obs['policy'][:,48:51].cpu().numpy(),
                        'fell':failed.cpu().numpy().copy()})
                if step % 250 == 0: print(json.dumps({'step':step,'falls':int(failed.sum()),'elapsed':time.time()-begin}),flush=True)
        trace={k:np.stack([f[k] for f in frames]) for k in frames[0]}
        np.savez_compressed(args.output/'raw-trace.npz',**trace)
        mask=trace['t']>2.
        vx=trace['lin_vel_b'][mask,:,0].mean(axis=0)
        yaw=trace['ang_vel_b'][mask,:,2].mean(axis=0)
        results=[]
        for i in range(args.num_envs):
            c=commands[i].cpu().tolist()
            results.append({'env':i,'command':c,'fell':bool(failed[i]),'mean_vx':float(vx[i]),'mean_yaw':float(yaw[i]),
                'drift_xy_m':float(np.linalg.norm(trace['root_pos'][-1,i,:2]-trace['root_pos'][10,i,:2]))})
        report={'schema':'microduck.gait-development.v1','qualification_claimed':False,
            'runtime_id':'newton-bam-20261019','hardware_profile_id':'beta-gc2093-l8cx-v1',
            'model':str(args.model),'model_sha256':sha,'seed':args.seed,'backlash':args.backlash,
            'randomized':args.randomized,'development_sweep':args.sweep,'parity':parity,'episodes':results,
            'step_wall_seconds':time.time()-begin,'wall_seconds':time.time()-started,
            'trace_sha256':hashlib.sha256((args.output/'raw-trace.npz').read_bytes()).hexdigest()}
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
        env.close()

if __name__=='__main__':main()
