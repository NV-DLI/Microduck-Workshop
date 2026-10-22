#!/usr/bin/env python3
"""Community IMU latency A/B at matched8rad/s adapter,3s hold then vx=.22. Development only.

Run each200/400/800Hz candidate in a new directory.50Hz control and20ms BAM
delay stay fixed in seconds. Raw, limited and actual delayed targets are saved.
"""
import argparse,hashlib,json,time
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor(x):return getattr(x,'torch',x)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--physics-hz',type=int,choices=[200,400,800],default=200);p.add_argument('--seconds',type=float,default=20);p.add_argument('--per-command',type=int,default=4);p.add_argument('--seed',type=int,default=974001);p.add_argument('--margin-rad',type=float,default=.035);p.add_argument('--max-target-rate',type=float,default=3.);p.add_argument('--unbounded-control',action='store_true');p.add_argument('--command',type=float,nargs=3,help='Optional single development command instead of forward/lateral/turn cases');a=p.parse_args()
 if a.output.exists():raise FileExistsError('Retain every experiment')
 if a.seconds<=0 or a.per_command<1:raise ValueError('Positive duration/batch required')
 import numpy as np,torch,gymnasium as gym,isaaclab_tasks
 from isaaclab.app import launch_simulation
 from isaaclab_tasks.utils import parse_env_cfg
 from batch_policy import BatchPolicy
 from bounded_actions import BoundedActions,load_limits,JOINT_NAMES
 dt=1/a.physics_hz;decimation=a.physics_hz//50
 cases={name:(a.command or [.22,0,0])for name in ('lag0','lag1','lag3','lag4')}
 groups=[name for name in cases for _ in range(a.per_command)];cfg=parse_env_cfg('IsaacContrib-Velocity-Flat-MicroDuck',device='cuda:0',num_envs=len(groups),overrides=['presets=backlash'])
 cfg.seed=a.seed;cfg.sim.dt=dt;cfg.decimation=decimation;cfg.sim.render_interval=decimation;cfg.episode_length_s=a.seconds+5;cfg.curriculum=None
 cfg.commands.base_velocity.debug_vis=False;cfg.commands.base_velocity.resampling_time_range=(1e8,1e8);cfg.commands.head_pose.ranges=tuple((0.,0.)for _ in range(4));cfg.commands.body_pose.ranges=tuple((0.,0.)for _ in range(6))
 for name in ('foot_friction','encoder_bias','mass_inertia','randomize_com','randomize_head_com','randomize_joint_friction','randomize_armature','push_robot'):setattr(cfg.events,name,None)
 servo_cfg=cfg.scene.robot.actuators['servos'];servo_cfg.vin_range=None;servo_cfg.vin_drop_gain_range=None;servo_cfg.min_delay=round(.020/dt);servo_cfg.max_delay=round(.020/dt)
 cfg.observations.policy.enable_corruption=False;cfg.observations.policy.imu.params.update(max_angle_deg=0,min_lag=0,max_lag=0)
 bound_source=load_limits();a.output.mkdir(parents=True);report={'schema':'microduck.community-lag-ab-development.v1','runtime_id':'newton-bam-20261019','hardware_profile_id':'beta-gc2093-l8cx-v1','model_sha256':sha(a.model),'development_only':True,'qualification_claimed':False,'frozen_protocol_changed':False,'physics_hz':a.physics_hz,'control_hz':50,'bam_delay_seconds':.020,'bam_delay_physics_steps':servo_cfg.min_delay,'target_margin_rad':a.margin_rad,'target_slew_rad_s':a.max_target_rate,'bounded':not a.unbounded_control,'bound_source':bound_source,'groups':groups,'seed':a.seed,'status':'starting'}
 rows=[];controls=[];delayed=[];imu_history=[];imu_actual=[];imu_supplied=[];started=time.monotonic();env=None
 try:
  with launch_simulation(cfg,{'device':'cuda:0','visualizer':[]}):
   env=gym.make('IsaacContrib-Velocity-Flat-MicroDuck',cfg=cfg).unwrapped;robot=env.scene['robot'];manager=env.sim.backend_manager
   manager.set_decimation(1);env._physics_handles_decimation=False
   if manager.backend_name!='newton'or not env.sim.cfg.use_newton_actuators:raise RuntimeError('Native Newton BAM required')
   env.reset();ids,names=robot.find_joints(JOINT_NAMES,preserve_order=True);pids,_=robot.find_joints(['passive_'+n+'_backlash'for n in JOINT_NAMES],preserve_order=True)
   if list(names)!=JOINT_NAMES:raise ValueError('Named joint order mismatch')
   limits=tensor(robot.data.joint_pos_limits).clone();term=env.action_manager.get_term('joint_pos');cmd=env.command_manager.get_term('base_velocity');model=BatchPolicy(a.model,report['model_sha256'],env.device)
   limiters={name:BoundedActions(bound_source['physical_output_limits_rad'],margin_rad=a.margin_rad,max_target_rate_rad_s=rate)for name,rate in [('lag0',8.),('lag1',8.),('lag3',8.),('lag4',8.)]}
   # Pair each repetition's reset across adapters BEFORE scored physics begins.
   # Absolute root XY is translated by each environment origin only.
   for group in range(1,4):
    src=slice(0,a.per_command);dst=torch.arange(group*a.per_command,(group+1)*a.per_command,device=env.device);state=torch.cat([tensor(robot.data.root_pos_w),tensor(robot.data.root_quat_w),tensor(robot.data.root_lin_vel_w),tensor(robot.data.root_ang_vel_w)],dim=1)[src].clone();state[:,:3]-=env.scene.env_origins[src];state[:,:3]+=env.scene.env_origins[dst]
    robot.write_root_pose_to_sim_index(root_pose=state[:,:7],env_ids=dst);robot.write_root_velocity_to_sim_index(root_velocity=state[:,7:],env_ids=dst)
    robot.write_joint_state_to_sim_index(position=tensor(robot.data.joint_pos)[src].clone(),velocity=tensor(robot.data.joint_vel)[src].clone(),env_ids=dst)
   env.scene.update(dt)
   report['matched_initialization']='identical first-group reset states, translated only by environment origin; before any scored step'
   report['adapter_groups']={name:{'margin':a.margin_rad,'rate_rad_s':8.,'body_IMU_lag_control_steps':lag,'body_IMU_lag_s':lag*.02}for name,lag in [('lag0',0),('lag1',1),('lag3',3),('lag4',4)]}
   solver=manager._solver;m=manager.get_model();drives=[x for x in m.actuators if type(x.drive).__name__=='DriveBam']
   if not drives:raise RuntimeError('Actual native BAM drive not found')
   report['actual_solver']={'joint_solref':solver.mjw_model.jnt_solref.numpy().tolist(),'joint_solimp':solver.mjw_model.jnt_solimp.numpy().tolist(),'newton_joint_labels':m.joint_label,'joint_to_newton_dof':solver.mjc_jnt_to_newton_dof.numpy().tolist(),'native_bam':[{'class':type(x.drive).__module__+'.'+type(x.drive).__name__,'pos_indices':x.pos_indices.numpy().tolist(),'target_pos_indices':x.target_pos_indices.numpy().tolist(),'min_delay':x.drive.min_delay,'max_delay':x.drive.max_delay}for x in drives]}
   command=torch.tensor([cases[g]for g in groups],device=env.device);fell=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device);counts={'boundary_limited':0,'rate_limited':0,'boundary_overrode_rate':0};origin=env.sim.get_physics_step_count()
   with torch.inference_mode():
    for step in range(round(a.seconds*50)):
     cmd.is_standing_env[:]=False;cmd.vel_command_b[:]=command if step*.02>=3. else 0.;obs=env.observation_manager.compute();policy_observation=obs['policy'].clone();imu_history.append(policy_observation[:,:6].clone());imu_actual.append(policy_observation[:,:6].cpu().numpy())
     for group,lag in enumerate((0,1,3,4)):
      idx=slice(group*a.per_command,(group+1)*a.per_command);policy_observation[idx,:6]=imu_history[max(0,len(imu_history)-1-lag)][idx]
     imu_supplied.append(policy_observation[:,:6].cpu().numpy());raw=model(policy_observation);q=tensor(robot.data.joint_pos)
     action=raw.clone();requested=raw*term._scale+term._offset-term.encoder_bias;d={'requested_output_target_rad':requested.clone(),'executed_output_target_rad':requested.clone(),**{k:torch.zeros_like(raw,dtype=torch.bool)for k in ('boundary_limited','rate_limited','boundary_overrode_rate')}}
     for group,name in enumerate(('lag0','lag1','lag3','lag4')):
      idx=slice(group*a.per_command,(group+1)*a.per_command)
      take=lambda value:value[idx]if isinstance(value,torch.Tensor)and value.ndim>=2 and value.shape[0]==env.num_envs else value
      limited,diag=limiters[name].apply(raw[idx],scale=take(term._scale),offset=take(term._offset),encoder_bias=take(term.encoder_bias),q_servo=q[idx][:,ids],q_passive=q[idx][:,pids],servo_limits=limits[idx][:,ids]);action[idx]=limited
      for key in d:d[key][idx]=diag[key]
     for key in counts:counts[key]+=int(d[key].sum())
     env.action_manager.process_action(action)
     if not torch.allclose(term.processed_actions-term.encoder_bias,d['executed_output_target_rad'],atol=1e-6,rtol=0):raise ValueError('Native action manager target differs from logged execution target')
     controls.append(torch.stack([raw,action,d['requested_output_target_rad'],d['executed_output_target_rad'],d['boundary_limited'].float(),d['rate_limited'].float(),d['boundary_overrode_rate'].float()],dim=1).cpu().numpy())
     for _ in range(decimation):
      env._sim_step_counter+=1;env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(dt)
      data=robot.data;gravity=tensor(data.projected_gravity_b);fell|=gravity[:,2]>-.5
      rows.append(torch.cat([tensor(data.root_pos_w),tensor(data.root_quat_w),tensor(data.root_lin_vel_b),tensor(data.root_ang_vel_b),tensor(data.joint_pos),tensor(data.joint_vel),gravity,fell[:,None].float()],dim=1).cpu().numpy())
      delayed.append([x.drive._delayed_target.numpy().copy()for x in drives])
     env.episode_length_buf+=1;env.common_step_counter+=1
     if step%250==0:print(json.dumps({'step':step,'physics_hz':a.physics_hz,'falls':int(fell.sum()),'clip_counts':counts}),flush=True)
   if env.sim.get_physics_step_count()-origin!=len(rows):raise ValueError('Recorded rows do not represent actual single physics ticks')
   arrays={'ideal_policy_imu_current':np.stack(imu_actual),'policy_imu_supplied':np.stack(imu_supplied),'state':np.stack(rows),'control':np.stack(controls),'joint_limits':limits.cpu().numpy(),'servo_ids':np.asarray(ids),'passive_ids':np.asarray(pids),'physics_dt':np.asarray(dt),'control_dt':np.asarray(.02)}
   for i in range(len(drives)):arrays['bam_actual_delayed_target_'+str(i)]=np.stack([r[i]for r in delayed])
   np.savez_compressed(a.output/'raw-trace.npz',**arrays)
   q=arrays['state'][:,:,13:13+robot.num_joints];effective=q[:,:,ids]+q[:,:,pids];physical=np.asarray(bound_source['physical_output_limits_rad'])
   violated=lambda data,lim:np.maximum(lim[...,0]-data,data-lim[...,1]).clip(min=0)
   report['initial_hold_s']=3.
   report['group_metrics']={}
   for group,name in enumerate(cases):
    sl=slice(group*a.per_command,(group+1)*a.per_command);moving=arrays['state'][round(3*a.physics_hz):,sl];report['group_metrics'][name]={'survived':int((moving[:,:,-1].max(axis=0)==0).sum()),'episodes':a.per_command,'mean_body_velocity':moving[:,:,7:10].mean(axis=(0,1)).tolist(),'mean_body_yaw':float(moving[:,:,12].mean()),'raw_to_executed_target_rms_rad':float(np.sqrt(np.mean((arrays['control'][:,sl,2]-arrays['control'][:,sl,3])**2))),'servo_peak_violation_rad':float(violated(q[:,sl][:,:,ids],arrays['joint_limits'][sl][:,ids]).max()),'effective_output_peak_violation_rad':float(violated(effective[:,sl],physical).max())}
   report.update(status='development_complete',physics_samples=len(rows),control_samples=len(controls),clip_counts=counts,all_survived=not bool(fell.any()),servo_peak_violation_rad=float(violated(q[:,:,ids],arrays['joint_limits'][:,ids]).max()),effective_output_peak_violation_rad=float(violated(effective,physical).max()),passive_soft_limit_peak_violation_rad=float(violated(q[:,:,pids],arrays['joint_limits'][:,pids]).max()),joint_names=robot.joint_names,trace_sha256=sha(a.output/'raw-trace.npz'),control_fields=['raw_action','executed_action','requested_output_target','executed_output_target','boundary_limited','rate_limited','boundary_overrode_rate'],state_fields={'root_pos':[0,3],'quat_xyzw':[3,7],'linear_velocity_body':[7,10],'angular_velocity_body':[10,13],'joint_pos':[13,13+robot.num_joints],'joint_vel':[13+robot.num_joints,13+2*robot.num_joints],'gravity':[13+2*robot.num_joints,16+2*robot.num_joints],'fell':16+2*robot.num_joints})
   env.close();env=None
 except BaseException as error:
  report.update(status='failed',error=repr(error))
  if rows:np.savez_compressed(a.output/'partial-trace.npz',state=np.stack(rows),control=np.stack(controls))
  raise
 finally:
  report['wall_seconds']=time.monotonic()-started;report['clip_count_scope']='candidate would-clip counts only; adapter bypassed'if a.unbounded_control else'counts applied to actual target before native BAM delay';report['source_sha256']={n:sha(Path(__file__).with_name(n))for n in ('bounded_actions.py','community_lag_ab.py','bounded-action-limits.json')};(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k]for k in ('status','physics_hz','clip_counts','all_survived','servo_peak_violation_rad','effective_output_peak_violation_rad','passive_soft_limit_peak_violation_rad')}))
if __name__=='__main__':main()
