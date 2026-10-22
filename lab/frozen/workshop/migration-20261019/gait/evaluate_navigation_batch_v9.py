#!/usr/bin/env python3
"""Record actual BODY50 delivery and every real200Hz task tick after physical startup.

Prospective collector: formal runs require a pre-frozen model bundle and complete case cohort.

No rendering, automatic reset, state clipping or reward-based pass proxy.
A formal part has32episodes; development uses disjoint seeds and never qualifies.
"""
import argparse,copy,hashlib,importlib,json,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
M=Path(__file__).resolve().parents[1];sys.path.insert(0,str(M/'acceptance'))
from navigation_metrics_v2 import protocol,schedule,evaluate,PROTOCOL_SHA

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(value):
 if isinstance(value,dict):return{k:js(v)for k,v in value.items()}
 if isinstance(value,(list,tuple)):return[js(v)for v in value]
 if value is None or isinstance(value,(str,int,float,bool)):return value
 if callable(value):return value.__module__+'.'+value.__qualname__
 return str(value)
def digest(value):return hashlib.sha256(json.dumps(js(value),sort_keys=True).encode()).hexdigest()
def T(x):return getattr(x,'torch',x)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--family',choices=['fixed','stop'],default='fixed');p.add_argument('--case',default='forward_+0.060');p.add_argument('--condition',choices=['nominal','full_official_DR'],default='nominal');p.add_argument('--adapter',choices=['trained-head-reference-v8'],required=True);p.add_argument('--recipe-dir',type=Path);p.add_argument('--model-line',choices=['community-adapter','from-scratch'],required=True);p.add_argument('--seed-index',type=int,choices=[0,1,2],default=0);p.add_argument('--qualification-bundle',type=Path);p.add_argument('--development',action='store_true');p.add_argument('--seed',type=int,default=976101);p.add_argument('--num-envs',type=int,default=32);p.add_argument('--seconds',type=float);p.add_argument('--cases',nargs='+',help='Development case subset; default is every case in the family');a=p.parse_args()
 frozen=protocol();duration=20 if a.family=='fixed'else 78
 if not a.development and a.cases is not None:raise ValueError('Formal batch records every case in its family')
 cases=a.cases or list(frozen['fixed_commands'if a.family=='fixed'else'stop_restart']['cases'])
 if len(set(cases))!=len(cases):raise ValueError('Duplicate case')
 for case in cases:schedule(a.family,case,0.)
 per_case=a.num_envs;batch_size=per_case*len(cases)
 if a.output.exists():raise FileExistsError('Keep every failed attempt')
 if a.development:
  if a.seed in frozen['group_rule']['seeds']:raise ValueError('Qualification seeds forbidden in development')
  duration=duration if a.seconds is None else a.seconds
 else:
  if a.qualification_bundle is None or a.num_envs!=32 or a.seconds is not None:raise ValueError('Formal run requires pre-frozen model bundle,32envs,unmodified duration')
  a.seed=frozen['group_rule']['seeds'][a.seed_index]
 if a.num_envs<1 or duration<5.:raise ValueError('Positive batch/duration required')
 if a.adapter=='trained-head-reference-v8'and a.recipe_dir is None:raise ValueError('Frozen training recipe directory required; do not replace its action history')
 import numpy as np,torch,gymnasium as gym,isaaclab_tasks
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 from isaaclab.app import launch_simulation
 from isaaclab_tasks.utils import parse_env_cfg
 import isaaclab.sim as sim_utils
 from isaaclab.utils.io import dump_yaml
 from batch_policy import BatchPolicy
 export_report=json.loads(a.model.with_suffix('.json').read_text());checkpoint=Path(export_report['checkpoint'])
 if checkpoint.parent.resolve()!=a.recipe_dir.resolve() or sha(checkpoint)!=export_report['checkpoint_sha256']:raise ValueError('Export/recipe mismatch')
 initial=torch.load(a.recipe_dir/'initial.pt',map_location='cpu',weights_only=False)
 training_ledger=json.loads((a.recipe_dir/'training-ledger.json').read_text())
 if training_ledger.get('recipe')!='body50-stop-gaze-v9' or training_ledger.get('action_mapping',{}).get('head_reference_blend')!=1.:raise ValueError('BODY50 v9 checkpoint and final head mapping required')
 if int(export_report['checkpoint_iteration'])<int(initial['iter']):raise ValueError('Checkpoint precedes declared BODY50 training')
 body_source=a.recipe_dir/'source_snapshot/body50_v9'
 source_hashes={str((body_source/Path(p).name).resolve()):h for p,h in training_ledger['source_sha256'].items()}
 if any(sha(p)!=h for p,h in source_hashes.items()):raise ValueError('Frozen BODY50 training sources changed')
 sys.path.insert(0,str(body_source))
 from environment_hook import Body50TrainingHook, T as body_tensor
 from compiled_sensor import install as compile_sensor
 if a.recipe_dir:
  a.recipe_dir=a.recipe_dir.resolve();sys.path.insert(0,str(a.recipe_dir))
  for name in ['bounded_actions','precision_recipe','precision_recipe_v2','precision_recipe_v3']:
   if name in sys.modules:raise RuntimeError('Frozen recipe import was contaminated: '+name)
 from bounded_actions import BoundedActions,load_limits,JOINT_NAMES
 cfg=parse_env_cfg('IsaacContrib-Velocity-Flat-MicroDuck',device='cuda:0',num_envs=batch_size,overrides=['presets=backlash'])
 cfg.seed=a.seed;cfg.sim.dt=.005;cfg.decimation=4;cfg.sim.render_interval=4;cfg.episode_length_s=duration+5;cfg.curriculum=None
 cfg.commands.base_velocity.debug_vis=False;cfg.commands.base_velocity.resampling_time_range=(1e8,1e8);cfg.commands.head_pose.ranges=tuple((0.,0.)for _ in range(4));cfg.commands.body_pose.ranges=tuple((0.,0.)for _ in range(6))
 base_obs=js(cfg.observations.policy.to_dict());base_bam=js(cfg.scene.robot.actuators['servos'].to_dict());base_dr=js(cfg.events.to_dict());limits_source=load_limits()
 adapter_file_sources={str(Path(importlib.import_module('bounded_actions').__file__).resolve()):sha(Path(importlib.import_module('bounded_actions').__file__))};adapter_sources={'bounded_actions.py':next(iter(adapter_file_sources.values()))}
 if a.adapter=='trained-head-reference-v8':
  action_class=importlib.import_module('head_reference_curriculum_v7').FinalHeadResidualAction;cfg.actions.joint_pos.class_type=action_class
  for name in ['precision_recipe.py','precision_recipe_v2.py','precision_recipe_v3.py','precision_recipe_v4.py','head_reference_curriculum_v7.py','head_reference_v7.py','bounded-action-limits.json']:
   if not(a.recipe_dir/name).is_file():raise ValueError('Missing frozen recipe dependency '+name)
   adapter_sources[name]=sha(a.recipe_dir/name);adapter_file_sources[str((a.recipe_dir/name).resolve())]=sha(a.recipe_dir/name)
  adapter_sources['training-env.yaml']=sha(a.recipe_dir/'env.yaml');adapter_file_sources[str((a.recipe_dir/'env.yaml').resolve())]=sha(a.recipe_dir/'env.yaml')
 # Use the independently read-back16 precompiled mass/CoM variants. No runtime
 # center-of-mass callback is falsely counted as applying to Newton's solver.
 com_manifest=M/'skills/experiments/built_com/generated/seed-974001/manifest.json';com=json.loads(com_manifest.read_text());variants=[]
 base_usd=Path(com['source_usd']).resolve()
 if sha(base_usd)!=com['source_sha256']:raise ValueError('Compiled CoM base source USD changed')
 cfg.scene.robot.spawn.usd_path=str(base_usd)
 for v in com['variants']:
  f=Path(v['usd']);variants.append({'path':str(f),'sha256':sha(f)})
 spawn_source={'base':str(base_usd),'base_sha256':sha(base_usd),'variants':variants,'manifest':str(com_manifest),'manifest_sha256':sha(com_manifest)}
 model_bundle={'network_sha256':sha(a.model),'action_adapter_sha256':digest(adapter_sources),'action_parameters_sha256':digest({'adapter':a.adapter,'margin_rad':.035,'rate_rad_s':8 if a.adapter=='trained-head-reference-v8'else 3,'action_history':'raw','head_reference_blend':1.,'head_residual_rad':.15,'official_action':cfg.actions.joint_pos.to_dict()}),'observation_parameters_sha256':digest({'base_command_fields':base_obs,'body50_sources':source_hashes,'body_tau_s':2.,'encoder_same_read':True,'nominal_extra_corruption':False,'full_DR_extra_corruption':True,'initialization_fixture':'physical, <=8s; no reset or state projection; task begins only after full withdrawal'}),'native_bam_parameters_sha256':digest(base_bam),'robot_usd_sha256':digest(spawn_source),'DR_recipe_sha256':digest({'official_events':base_dr,'compiled_com':spawn_source,'runtime_com_callbacks_disabled':True})}
 if not a.development:
  requested=json.loads(a.qualification_bundle.read_text())
  if requested.get('model_bundle')!=model_bundle or requested.get('protocol_sha256')!=PROTOCOL_SHA:raise ValueError('Formal candidate differs from pre-frozen model bundle')
  if datetime.fromisoformat(requested['frozen_at_utc'])>=datetime.now(timezone.utc):raise ValueError('Model bundle must be frozen before qualification')
 if a.condition=='nominal':
  for name in ('foot_friction','encoder_bias','mass_inertia','randomize_com','randomize_head_com','randomize_joint_friction','randomize_armature','push_robot'):setattr(cfg.events,name,None)
  cfg.scene.robot.actuators['servos'].vin_range=None;cfg.scene.robot.actuators['servos'].vin_drop_gain_range=None
  cfg.scene.robot.actuators['servos'].min_delay=4;cfg.scene.robot.actuators['servos'].max_delay=4
  cfg.observations.policy.enable_corruption=False;cfg.observations.policy.imu.params.update(max_angle_deg=0,min_lag=0,max_lag=0)
 else:
  spawns=[]
  for v in variants:
   spawn=copy.deepcopy(cfg.scene.robot.spawn);spawn.usd_path=v['path'];spawns.append(spawn)
  cfg.scene.robot.spawn=sim_utils.MultiAssetSpawnerCfg(assets_cfg=spawns);cfg.events.randomize_com=None;cfg.events.randomize_head_com=None
 a.output.mkdir(parents=True);dump_yaml(str(a.output/'collector-env.yaml'),cfg)
 report={'schema':'microduck.navigation-v2-part.v1','runtime_id':frozen['runtime_id'],'hardware_profile_id':frozen['hardware_profile_id'],'protocol_sha256':PROTOCOL_SHA,'development_only':a.development,'qualification_claimed':False,'cohort_definition':'One independent RNG stream per frozen seed/condition; all family cases coexist in disjoint native worlds,32 episodes per case','model_line':a.model_line,'model_file':str(a.model.resolve()),'model_bundle':model_bundle,'adapter':a.adapter,'adapter_sources':adapter_sources,'adapter_file_sources':adapter_file_sources,'physical_output_bound_source':limits_source,'family':a.family,'case':'batch','cases':cases,'batch_size':batch_size,'condition':a.condition,'seed':a.seed,'seed_index':a.seed_index,'expected_episodes':a.num_envs,'expected_seconds':duration,'automatic_resets':False,'state_projection':False,'support_during_task':False,'physics_hz':200,'policy_hz':50,'inference_precision':'FP32, TF32 disabled','created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'starting','body50_sources':source_hashes,'body50_observation_contract':{'actor_dimensions':61,'body_tau_s':2.,'same_read_encoders':True,'readiness_required':True,'extra_observation_dr':a.condition=='full_official_DR','initialization_max_seconds':8.,'actor_world_state':False},'robot_asset_sources':spawn_source,'qualification_bundle_sha256':sha(a.qualification_bundle)if a.qualification_bundle else None,'qualification_bundle_file':str(a.qualification_bundle.resolve())if a.qualification_bundle else None}
 report_path=a.output/'report.json';report_path.write_text(json.dumps(report,indent=2)+'\n');rows=[];controls=[];bam_rows=[];actor_rows=[];packet_rows=[];wrench_rows=[];sensor_fields=('gyro_body_rad_s','gravity_body','encoder_q_rad','encoder_qd_rad_s','gyro_counts','quaternion_xyz_half_bits','read_timestamp_s','encoder_read_timestamp_s','orientation_valid');started=time.monotonic();env=None
 try:
  with launch_simulation(cfg,{'device':'cuda:0','visualizer':[]}):
   env=gym.make('IsaacContrib-Velocity-Flat-MicroDuck',cfg=cfg).unwrapped;robot=env.scene['robot'];manager=env.sim.backend_manager;manager.set_decimation(1);env._physics_handles_decimation=False
   if manager.backend_name!='newton'or not env.sim.cfg.use_newton_actuators:raise RuntimeError('Native Newton BAM required')
   hook=Body50TrainingHook(env,seed=a.seed,extra_observation_dr=a.condition=='full_official_DR');compiled_sensor=compile_sensor(hook.producer)
   env.reset();ids,names=robot.find_joints(JOINT_NAMES,preserve_order=True);pids,_=robot.find_joints(['passive_'+n+'_backlash'for n in JOINT_NAMES],preserve_order=True)
   if list(names)!=JOINT_NAMES or robot.num_joints!=28:raise ValueError('Named28joint BAM contract required')
   limits=T(robot.data.joint_pos_limits).clone();term=env.action_manager.get_term('joint_pos');cmd=env.command_manager.get_term('base_velocity');model=BatchPolicy(a.model,report['model_bundle']['network_sha256'],env.device)
   external=BoundedActions(limits_source['physical_output_limits_rad'])if a.adapter=='community-external3'else None
   if external is None and(type(term).__name__!='FinalHeadResidualAction'or term.limiter.rate!=8.):raise ValueError('Frozen internal8 action mapping required')
   solver=manager._solver;native=manager.get_model();drives=[x for x in native.actuators if type(x.drive).__name__=='DriveBam']
   if not drives:raise RuntimeError('Native BAM drive absent')
   report['actual_solver']={'joint_solref':solver.mjw_model.jnt_solref.numpy().tolist(),'joint_solimp':solver.mjw_model.jnt_solimp.numpy().tolist(),'newton_joint_labels':native.joint_label,'joint_to_newton_dof':solver.mjc_jnt_to_newton_dof.numpy().tolist(),'native_bam':[{'class':type(x.drive).__module__+'.'+type(x.drive).__name__,'pos_indices':x.pos_indices.numpy().tolist(),'target_pos_indices':x.target_pos_indices.numpy().tolist(),'min_delay':x.drive.min_delay,'max_delay':x.drive.max_delay,'parameters':{k:getattr(x.drive,k).numpy().tolist()for k in x.drive.PER_DOF_PARAMS if getattr(x.drive,k,None)is not None}}for x in drives]}
   # Keep evaluated post-reset physical parameter values, not only desired cfg.
   parameter_arrays={}
   for name in ['body_mass','body_inertia','joint_armature','joint_friction_coeff','default_mass','default_inertia']:
    try:parameter_arrays[name]=T(getattr(robot.data,name)).cpu().numpy()
    except AttributeError:pass
   parameter_arrays['encoder_bias']=T(term.encoder_bias).cpu().numpy();np.savez_compressed(a.output/'post-reset-parameters.npz',**parameter_arrays)
   report['post_reset_parameter_sha256']=sha(a.output/'post-reset-parameters.npz')
   # Physical initialization is recorded separately and never resets or selects worlds.
   warmup_origin=env.sim.get_physics_step_count();startup_peak=0.
   with torch.inference_mode():
    for warmup_step in range(400):
     cmd.is_standing_env[:]=False;cmd.vel_command_b[:]=0
     env.command_manager.get_term('head_pose').command[:]=0;env.command_manager.get_term('body_pose').command[:]=0
     obs=env.observation_manager.compute(update_history=True);ready=env.ready_mask
     raw=torch.zeros((batch_size,14),device=env.device)
     if ready.any():raw[ready]=model(obs['policy'][ready])
     env.action_manager.process_action(raw)
     for _ in range(4):
      env._sim_step_counter+=1;env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(.005)
     env.common_step_counter+=1
     if bool(env.task_active_mask.all()):break
   startup_completed=env.task_active_mask.clone()
   report['physical_initialization']={**hook.summary(),'physical_ticks':env.sim.get_physics_step_count()-warmup_origin,'automatic_resets':0,'task_measurement_excludes_initialization':True,'completed_mask':startup_completed.cpu().tolist(),'failed_worlds_retained':True}
   # Failed initialization is a trial failure, not a reason to erase the
   # cohort. The frozen fixture withdraws all forces on failure by its5s
   # timeout; verify that actual buffer before recording unsupported motion.
   if bool(body_tensor(robot.data._sim_bind_body_external_wrench).abs().any()):raise RuntimeError('Nonzero actual wrench remains after the startup deadline')
   initial_previous_raw=hook.previous_raw.cpu().numpy().copy();native_home14=T(robot.data.default_joint_pos)[:,ids].cpu().numpy().copy()
   fell=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device);origin=env.sim.get_physics_step_count();obs=env.observation_manager.compute();report['onnx_parity']=model.parity(obs['policy'][env.ready_mask],tolerance=1e-5);inference_rows=[]
   with torch.inference_mode():
    for step in range(round(duration*50)):
     command=torch.tensor([schedule(a.family,case,step*.02)for case in cases for _ in range(per_case)],device=env.device,dtype=torch.float32);cmd.is_standing_env[:]=False;cmd.vel_command_b[:]=command
     env.command_manager.get_term('head_pose').command[:]=0;env.command_manager.get_term('body_pose').command[:]=0
     obs=env.observation_manager.compute(update_history=True);ready=env.ready_mask.clone() & startup_completed;obs['policy']=torch.where(ready[:,None],obs['policy'],torch.zeros_like(obs['policy']));actor_rows.append(obs['policy'].cpu().numpy().copy());packet=hook.producer.observe();packet_rows.append({k:packet[k].cpu().numpy().copy()for k in sensor_fields})
     # A lost fusion lock remains a failed trial in the original cohort. It
     # must not become a network inference from an initialization sentinel.
     inference_rows.append(ready.cpu().numpy().copy());raw=torch.zeros((batch_size,14),device=env.device)
     if ready.any():raw[ready]=model(obs['policy'][ready])
     q=T(robot.data.joint_pos);requested=raw*term._scale+term._offset-term.encoder_bias
     if external is not None:
      action,diagnostic=external.apply(raw,scale=term._scale,offset=term._offset,encoder_bias=term.encoder_bias,q_servo=q[:,ids],q_passive=q[:,pids],servo_limits=limits[:,ids]);env.action_manager.process_action(action)
     else:
      env.action_manager.process_action(raw)
      requested=term.reference_combined_action*term._scale+term._offset-term.encoder_bias
     actual_target=term.processed_actions-term.encoder_bias;executed=(actual_target+term.encoder_bias-term._offset)/term._scale
     if not torch.allclose(env.action_manager.action,raw if external is None else executed,atol=1e-6,rtol=0):raise ValueError('Action history differs from trained adapter contract')
     controls.append(torch.cat([raw,executed,requested,actual_target,env.action_manager.action,command],dim=-1).cpu().numpy())
     for _ in range(4):
      env._sim_step_counter+=1;env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step(render=False);env.scene.update(.005)
      d=robot.data;gravity=T(d.projected_gravity_b);fell|=(gravity[:,2]>-.6)|(T(d.root_pos_w)[:,2]<.065)
      rows.append(torch.cat([T(d.root_pos_w),T(d.root_quat_w),T(d.root_lin_vel_b),T(d.root_ang_vel_b),T(d.joint_pos),T(d.joint_vel),gravity,fell[:,None].float()],dim=1).cpu().numpy())
      wrench_rows.append(body_tensor(robot.data._sim_bind_body_external_wrench).reshape(batch_size,-1).abs().amax(-1).cpu().numpy().copy())
      bam_rows.append([{k:getattr(x.drive,k).numpy().copy()for k in ('motor_torque','effective_vin','_delayed_target')}for x in drives])
     env.episode_length_buf+=1;env.common_step_counter+=1;hook.task_steps.add_(1)
     if 'interval'in env.event_manager.available_modes:env.event_manager.apply(mode='interval',dt=.02)
     if step%250==0:print(json.dumps({'step':step,'sim_s':(step+1)*.02,'falls':int(fell.sum()),'case':a.case,'adapter':a.adapter}),flush=True)
   report['body50_task_readback']=hook.summary();report['compiled_sensor']=compiled_sensor.report()
   if report['body50_task_readback']['actual_task_external_wrench_peak']!=0.:raise RuntimeError('Actual task assistance detected')
   if env.sim.get_physics_step_count()-origin!=len(rows):raise ValueError('Actual tick count differs from captured samples')
   arrays={'state':np.stack(rows),'time_s':np.arange(1,len(rows)+1)/200,'control':np.stack(controls),'joint_limits':limits.cpu().numpy(),'physical_output_limits':np.asarray(limits_source['physical_output_limits_rad']),'servo_ids':np.asarray(ids),'passive_ids':np.asarray(pids)}
   for j in range(len(drives)):
    for k in ('motor_torque','effective_vin','_delayed_target'):arrays['bam_'+str(j)+'_'+k]=np.stack([r[j][k]for r in bam_rows])
   arrays['actor_observation']=np.stack(actor_rows);arrays['initial_previous_raw']=initial_previous_raw;arrays['native_home14']=native_home14;arrays['external_wrench_peak']=np.stack(wrench_rows);arrays['actor_inferred_mask']=np.stack(inference_rows);arrays['initial_startup_completed_mask']=startup_completed.cpu().numpy().copy()
   for key in sensor_fields:arrays['body50_'+key]=np.stack([row[key]for row in packet_rows])
   result=[]
   # Each raw BAM buffer contains14 consecutive servo channels per environment.
   for name,value in arrays.items():
    if name.startswith('bam_')and value.shape!=(len(rows),batch_size*14):raise ValueError('Unexpected BAM world/channel mapping: '+name+str(value.shape))
   part_reports=[]
   for ci,case in enumerate(cases):
    start=ci*per_case;stop=start+per_case;part_dir=a.output/case;part_dir.mkdir()
    part_arrays={}
    for name,value in arrays.items():
     if name in ('state','control','actor_observation','external_wrench_peak','actor_inferred_mask')or name.startswith('body50_'):part_arrays[name]=value[:,start:stop]
     elif name in ('joint_limits','initial_previous_raw','native_home14','initial_startup_completed_mask'):part_arrays[name]=value[start:stop]
     elif name.startswith('bam_'):part_arrays[name]=value[:,start*14:stop*14]
     else:part_arrays[name]=value
    np.savez_compressed(part_dir/'raw-trace.npz',**part_arrays)
    parameters={k:(v[start:stop]if v.ndim and v.shape[0]==batch_size else v)for k,v in parameter_arrays.items()}
    np.savez_compressed(part_dir/'post-reset-parameters.npz',**parameters)
    (part_dir/'collector-env.yaml').write_bytes((a.output/'collector-env.yaml').read_bytes())
    part_result=evaluate(part_arrays,family=a.family,case=case,seconds=duration)
    for item in part_result:
     count=int((~part_arrays['actor_inferred_mask'][:,item['episode_index']]).sum());item['sensor_unready_control_steps']=count;item['uninterrupted_sensor_control']=count==0
    result.extend(part_result)
    part=copy.deepcopy(report);part.update(status='recorded',case=case,physics_samples=len(rows),control_samples=len(controls),trace_sha256=sha(part_dir/'raw-trace.npz'),episodes=part_result,passed_episodes=sum(r['passed']for r in part_result),joint_names=robot.joint_names,post_reset_parameter_sha256=sha(part_dir/'post-reset-parameters.npz'),collector_env_sha256=sha(part_dir/'collector-env.yaml'),batch_environment_indices=list(range(start,stop)),solver_parameter_scope='Full batch readback; batch_environment_indices identifies this part',collector_sources={},wall_seconds=time.monotonic()-started)
    for file in [Path(__file__),M/'acceptance/navigation_metrics_v2.py']:
     target=part_dir/('source-'+file.name);target.write_bytes(file.read_bytes());part['collector_sources'][str(target.resolve())]=sha(target)
    (part_dir/'report.json').write_text(json.dumps(part,indent=2)+'\n');part_reports.append(str((part_dir/'report.json').resolve()))
   report['part_reports']=part_reports;report['schema']='microduck.navigation-v2-development-batch.v1'if a.development else'microduck.navigation-v2-frozen-batch.v1';report['expected_episodes']=batch_size
   report.update(status='recorded',physics_samples=len(rows),control_samples=len(controls),episodes=result,passed_episodes=sum(r['passed']for r in result),joint_names=robot.joint_names,action_fields={'raw':[0,14],'executed':[14,28],'requested_target':[28,42],'actual_target':[42,56],'history':[56,70],'command':[70,73]},state_fields={'root_pos':[0,3],'quat_xyzw':[3,7],'linear_velocity_body':[7,10],'angular_velocity_body':[10,13],'joint_pos':[13,41],'joint_vel':[41,69],'gravity':[69,72],'fell':72})
   env.close();env=None
 except BaseException as error:
  report.update(status='failed',error=repr(error),traceback=traceback.format_exc())
  if rows:np.savez_compressed(a.output/'partial-trace.npz',state=np.stack(rows),control=np.stack(controls),actor_observation=np.stack(actor_rows))
  raise
 finally:
  report['wall_seconds']=time.monotonic()-started;report['collector_sources']={}
  for p in [Path(__file__),M/'acceptance/navigation_metrics_v2.py']:
   target=a.output/('source-'+p.name);target.write_bytes(p.read_bytes());report['collector_sources'][str(target.resolve())]=sha(target);report['collector_env_sha256']=sha(a.output/'collector-env.yaml');report_path.write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'status':report['status'],'passed':report.get('passed_episodes'),'episodes':batch_size,'development':a.development}))
if __name__=='__main__':main()
