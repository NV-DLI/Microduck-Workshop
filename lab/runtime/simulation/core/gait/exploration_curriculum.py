"""Precision refinement with bounded exploration and unchanged physical DR.

The inherited official forward bucket has a0.3m/s minimum. Changing the range
alone does not train low speeds. This sampler replaces that bucket explicitly.
"""
import torch
from isaaclab_tasks.contrib.microduck.mdp.commands import MicroDuckVelocityCommand
from command_curriculum import configure as previous_configure,AxisCommand,COMMAND_CASES
from cycle_tracking import BoundedTrainingAction,filtered_velocity,T

def physical_output_cost(position,limits):
    """Penalize entering a50mrad interior buffer; never modify physical state."""
    distance=torch.maximum(limits[:,0]+.05-position,position-(limits[:,1]-.05)).clamp(min=0)
    return (distance/.05).square().sum(-1)

def physical_output_proximity(env):
    term=env.action_manager.get_term('joint_pos');q=T(term._asset.data.joint_pos)
    output=q[:,term._joint_ids]+q[:,term.passive_ids]
    limits=torch.as_tensor(term.limiter.limits,device=output.device,dtype=output.dtype)
    return physical_output_cost(output,limits)

def planar_score(mean,command):
    target=command[:,:2];actual=mean[:,:2];speed=target.norm(dim=-1);moving=speed>1e-6
    direction=target/speed.clamp(min=1e-6)[:,None]
    progress=((actual*direction).sum(-1)/speed.clamp(min=.012)).clamp(0.,1.)
    tolerance=(speed*.2).clamp(min=.012)
    precision=1./(1.+((actual-target).square().sum(-1)/tolerance.square()))
    standing=12./(1.+actual.square().sum(-1)/(.012**2))
    return torch.where(moving,4.*progress+12.*precision,standing)

def yaw_score(mean,command):
    target=command[:,2];actual=mean[:,2];speed=target.abs();moving=speed>1e-6
    progress=(actual*target.sign()/speed.clamp(min=.04)).clamp(0.,1.)
    tolerance=(speed*.2).clamp(min=.04)
    precision=1./(1.+((actual-target)/tolerance).square())
    standing=12./(1.+(actual/.07).square())
    return torch.where(moving,4.*progress+12.*precision,standing)

def planar(env):return planar_score(filtered_velocity(env),env.command_manager.get_command('base_velocity'))
def yaw(env):return yaw_score(filtered_velocity(env),env.command_manager.get_command('base_velocity'))

def configure(cfg):
    from isaaclab.managers import RewardTermCfg
    previous_configure(cfg)
    cfg.commands.base_velocity.class_type=AxisCommand
    cfg.commands.base_velocity.forward_min_speed=0.;cfg.commands.base_velocity.rel_forward_envs=0.;cfg.commands.base_velocity.rel_turn_in_place_envs=0.
    cfg.commands.base_velocity.resampling_time_range=(4.,8.)
    cfg.commands.base_velocity.ranges.lin_vel_x=(-.1,.1);cfg.commands.base_velocity.ranges.lin_vel_y=(-.06,.06);cfg.commands.base_velocity.ranges.ang_vel_z=(-.4,.4)
    cfg.commands.head_pose.ranges=tuple((0.,0.)for _ in range(4));cfg.commands.body_pose.ranges=tuple((0.,0.)for _ in range(6))
    cfg.rewards.track_lin_vel=RewardTermCfg(func=planar,weight=1.)
    cfg.rewards.track_ang_vel=RewardTermCfg(func=yaw,weight=1.)
    cfg.rewards.camera_angular.weight=-.01
    cfg.rewards.raw_target_excess.weight=-2.
    cfg.rewards.physical_output_proximity=RewardTermCfg(func=physical_output_proximity,weight=-2.)
    return cfg


def configure_agent(agent):
    # Keep every inherited actor/normalizer tensor byte-identical. The sampling
    # distribution has an explicit cap; no post-hoc action or state projection.
    agent.actor.distribution_cfg.std_range=(.03,.20)
    agent.algorithm.entropy_coef=0.
    return agent
