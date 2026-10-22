"""Prospective stop/gaze refinement, not a qualified student recipe.

Half the environments retain normal command sampling; half alternate motion
with12–24s stops. Only zero-command rewards penalize real oscillation and slip.
The deployment observation remains61D and receives no simulation position.
"""
import torch
from gaze_recipe_v6 import configure as previous_configure
from gaze_recipe_v6 import configure_agent
from precision_recipe_v4 import AxisCommand, COMMAND_CASES
from precision_recipe_v5 import planar_score, yaw_score
from precision_recipe_v3 import filtered_velocity, T
from head_reference_curriculum_v7 import ProgressiveHeadResidualAction


class LongStopCommand(AxisCommand):
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        self.next_long_stop=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
        self.long_stop_durations=torch.zeros(self.num_envs,device=self.device)

    def _resample_command(self,env_ids):
        if isinstance(env_ids,slice):ids=torch.arange(self.num_envs,device=self.device)[env_ids]
        else:ids=torch.as_tensor(env_ids,dtype=torch.long,device=self.device)
        super()._resample_command(ids)
        long_ids=ids[ids%2==0]
        if not len(long_ids):return
        stopping=self.next_long_stop[long_ids]
        table=torch.tensor(COMMAND_CASES,device=self.device)
        chosen=torch.randint(1,len(table),(len(long_ids),),device=self.device)
        chosen[stopping]=0
        self.vel_command_b[long_ids]=table[chosen]
        self.is_standing_env[long_ids]=stopping
        self.sampled_case_indices[long_ids]=chosen
        duration=torch.empty(len(long_ids),device=self.device).uniform_(3.,7.)
        duration[stopping]=torch.empty(int(stopping.sum()),device=self.device).uniform_(12.,24.)
        self.time_left[long_ids]=duration
        self.long_stop_durations[long_ids]=torch.where(stopping,duration,torch.zeros_like(duration))
        self.next_long_stop[long_ids]=~stopping


def stopped(env):
    return env.command_manager.get_command('base_velocity').abs().amax(dim=-1)<1e-6


def planar(env):
    mean=filtered_velocity(env);command=env.command_manager.get_command('base_velocity')
    regular=planar_score(mean,command)
    return torch.where(stopped(env),12./(1.+mean[:,:2].square().sum(-1)/(.004**2)),regular)


def yaw(env):
    mean=filtered_velocity(env);command=env.command_manager.get_command('base_velocity')
    return torch.where(stopped(env),12./(1.+(mean[:,2]/.02).square()),yaw_score(mean,command))


def stationary_motion(env):
    data=env.scene['robot'].data
    linear=T(data.root_lin_vel_b);angular=T(data.root_ang_vel_b)
    cost=(linear[:,:2]/.02).square().sum(-1)+(angular[:,2]/.08).square()+(angular[:,:2]/.3).square().sum(-1)
    return stopped(env).float()*8./(1.+cost)


def stationary_contacts(env,asset_cfg,sensor_cfg):
    contact=T(env.scene.sensors[sensor_cfg.name].data.current_contact_time)[:,sensor_cfg.body_ids]>0
    return stopped(env).float()*contact.all(dim=1).float()


def stationary_slip(env,asset_cfg,sensor_cfg):
    contact=T(env.scene.sensors[sensor_cfg.name].data.current_contact_time)[:,sensor_cfg.body_ids]>0
    velocity=T(env.scene[asset_cfg.name].data.body_link_lin_vel_w)[:,asset_cfg.body_ids,:2]
    cost=((velocity/.03).square().sum(-1)*contact).sum(-1).clamp(max=10.)
    return stopped(env).float()*cost


def configure(cfg):
    from isaaclab.managers import RewardTermCfg
    previous_configure(cfg)
    cfg.episode_length_s=60.
    cfg.commands.base_velocity.class_type=LongStopCommand
    cfg.actions.joint_pos.class_type=ProgressiveHeadResidualAction
    cfg.rewards.track_lin_vel=RewardTermCfg(func=planar,weight=1.)
    cfg.rewards.track_ang_vel=RewardTermCfg(func=yaw,weight=1.)
    cfg.rewards.stationary_motion=RewardTermCfg(func=stationary_motion,weight=1.)
    params={key:cfg.rewards.foot_slip.params[key] for key in ('asset_cfg','sensor_cfg')}
    cfg.rewards.stationary_contacts=RewardTermCfg(func=stationary_contacts,weight=2.,params=params)
    cfg.rewards.stationary_slip=RewardTermCfg(func=stationary_slip,weight=-.5,params=params)
    return cfg
