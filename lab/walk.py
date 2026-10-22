"""A bounded native-Newton walking observation lab. No automatic reset on a fall."""
from pathlib import Path
import argparse,json,hashlib,time,sys
p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--velocity',nargs=3,type=float,required=True);p.add_argument('--seconds',type=float,default=12);p.add_argument('--seed',type=int,default=42);p.add_argument('--headless',action='store_true');a=p.parse_args()
sys.path.insert(0,str(a.runtime/'gait'))
import numpy as np,torch,gymnasium as gym
from isaaclab.app import launch_simulation
from batch_policy import BatchPolicy
a.output.mkdir(parents=True,exist_ok=False);digest=hashlib.sha256(a.model.read_bytes()).hexdigest();report=dict(status='starting',seed=a.seed,model_sha256=digest,command=a.velocity,physics_hz=200,policy_hz=50,scope='nominal walking observation; not full-task evaluation');rows=[]
try:
 with launch_simulation(None,{'device':'cuda:0','require_kit':True,'headless':a.headless}):
  import isaaclab_tasks
  from isaaclab_tasks.utils import parse_env_cfg
  cfg=parse_env_cfg('IsaacContrib-Velocity-Flat-MicroDuck',device='cuda:0',num_envs=1,overrides=['presets=backlash']);cfg.seed=a.seed;cfg.curriculum=None;cfg.episode_length_s=a.seconds+10
  cfg.scene.robot.spawn.usd_path=str(a.runtime/'visuals/assets/microduck_white_orange_v2.usda')
  cfg.commands.base_velocity.resampling_time_range=(1e8,1e8);cfg.commands.base_velocity.debug_vis=False
  cfg.commands.head_pose.ranges=tuple((0.,0.)for _ in range(4));cfg.commands.body_pose.ranges=tuple((0.,0.)for _ in range(6));cfg.events.reset_base.params['pose_range']={}
  for name in ['foot_friction','encoder_bias','mass_inertia','randomize_com','randomize_head_com','randomize_joint_friction','randomize_armature','push_robot']:setattr(cfg.events,name,None)
  cfg.scene.robot.actuators['servos'].vin_range=None;cfg.scene.robot.actuators['servos'].vin_drop_gain_range=None;cfg.scene.robot.actuators['servos'].min_delay=4;cfg.scene.robot.actuators['servos'].max_delay=4
  cfg.observations.policy.enable_corruption=False;cfg.observations.policy.imu.params.update(max_angle_deg=0,min_lag=0,max_lag=0)
  env=gym.make('IsaacContrib-Velocity-Flat-MicroDuck',cfg=cfg).unwrapped;env.reset();robot=env.scene['robot'];manager=env.sim.backend_manager
  if manager.backend_name!='newton' or not env.sim.cfg.use_newton_actuators:raise RuntimeError('Native Newton actuators required')
  manager.set_decimation(1);env._physics_handles_decimation=False
  policy=BatchPolicy(a.model,digest,'cuda:0');cmd=env.command_manager.get_term('base_velocity');fallen=False
  with torch.inference_mode():
   for step in range(round(a.seconds/.02)):
    cmd.is_standing_env[:]=False;cmd.vel_command_b[:]=torch.tensor(a.velocity,device='cuda:0')if step>=100 else 0
    env.command_manager.get_term('head_pose').command[:]=0;env.command_manager.get_term('body_pose').command[:]=0
    action=policy(env.observation_manager.compute(update_history=True)['policy']);env.action_manager.process_action(action)
    for tick in range(4):
     env._sim_step_counter+=1;env.action_manager.apply_action();env.scene.write_data_to_sim();env.sim.step(render=not a.headless and tick==3);env.scene.update(.005)
    env.episode_length_buf+=1;env.common_step_counter+=1
    def T(x):return getattr(x,'torch',x)
    fallen=bool((T(robot.data.projected_gravity_b)[0,2]>-.6)|(T(robot.data.root_pos_w)[0,2]<.065))
    if step%5==0:rows.append(dict(t=(step+1)*.02,position=T(robot.data.root_pos_w)[0].cpu().tolist(),velocity=T(robot.data.root_lin_vel_b)[0].cpu().tolist(),fell=fallen))
    if fallen:break
  report.update(status='completed',fell=fallen,simulated_seconds=(step+1)*.02,automatic_resets=False)
  (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'trajectory.json').write_text(json.dumps(rows)+'\n');print(json.dumps(report,indent=2),flush=True)
  env.close()
except BaseException as error:report.update(status='failed',error=repr(error));raise
finally:
 (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');(a.output/'trajectory.json').write_text(json.dumps(rows)+'\n')
 print(json.dumps(report,indent=2))
