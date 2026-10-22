"""Independent Newton foot/ball contact audit, never an executor observation.

The paired contact force is privileged simulation evaluation data. Its only
consumer is the goal/contact scorer. Real locomotion, aiming and policy inputs
must never read this object or its outputs.
"""
from __future__ import annotations
import math


class ContactAudit:
    def __init__(self, force_threshold_n=.001):
        if not math.isfinite(force_threshold_n) or force_threshold_n<=0:
            raise ValueError('Positive finite physical contact threshold required')
        self.threshold=force_threshold_n
        self.reset()

    def reset(self):
        self.kick_contact_observed=False
        self.pre_kick_ball_contact_observed=False
        self.kick_contact_impulse_ns=0.
        self.max_kick_contact_force_n=0.
        self.samples=0

    def update(self, forces, foot_names, active_policy, kick_requested, dt):
        if not math.isfinite(dt) or dt<=0 or len(forces)!=len(foot_names):
            raise ValueError('Invalid contact sample dimensions or timestep')
        magnitudes=[]
        for force in forces:
            if len(force)!=3 or not all(math.isfinite(v) for v in force):
                raise ValueError('Physical contact force must be a finite XYZ vector')
            magnitudes.append(math.hypot(*force))
        kicking_foot='ankle_left' if active_policy=='kick_left' else ('ankle_right' if active_policy=='kick_right' else None)
        active_force=sum(f for f,name in zip(magnitudes,foot_names) if name==kicking_foot)
        observed=bool(kick_requested and kicking_foot is not None and active_force>self.threshold)
        self.kick_contact_observed |= observed
        self.pre_kick_ball_contact_observed |= (not kick_requested and any(f>self.threshold for f in magnitudes))
        if kicking_foot is not None:
            self.kick_contact_impulse_ns+=active_force*dt
            self.max_kick_contact_force_n=max(self.max_kick_contact_force_n,active_force)
        self.samples+=1
        return {'forces_on_ball_world_n':forces, 'foot_names':list(foot_names),
                'active_policy':active_policy,'kick_requested':bool(kick_requested),
                'physical_kick_contact_this_step':observed,
                'kick_contact_observed':self.kick_contact_observed,
                'purpose':'independent physical contact evaluation only; never planner input'}

    def summary(self):
        return {'kick_contact_observed':self.kick_contact_observed,
                'pre_kick_ball_contact_observed':self.pre_kick_ball_contact_observed,
                'kick_contact_impulse_ns':self.kick_contact_impulse_ns,
                'max_kick_contact_force_n':self.max_kick_contact_force_n,
                'contact_samples':self.samples,'contact_threshold_n':self.threshold}


from scoring import goal_crossing,GOAL_SCORING_VERSION
class PhysicalGoalScorer:
    """Score raw Newton substeps independently; never pass this to executor."""
    def __init__(self):self.contact=ContactAudit();self.previous=None;self.goal_time=None;self.fell=False;self.obstacle_collision=False;self.max_joint_violation=0.;self.last_t=None
    def update(self,*,timestamp_s,ball_world_xyz,paired_foot_forces_world_n,foot_names,active_policy,kick_requested,fell=False,obstacle_force_n=0.,joint_violation_rad=0.):
        if not math.isfinite(timestamp_s) or timestamp_s<0:raise ValueError('Finite nonnegative timestamp required')
        if not all(math.isfinite(v) and v>=0 for v in (obstacle_force_n,joint_violation_rad)):raise ValueError('Finite nonnegative scorer metrics required')
        if len(set(foot_names))!=len(foot_names) or set(foot_names)!={'ankle_left','ankle_right'}:raise ValueError('Exactly both paired ankle/ball contact channels required')
        if self.last_t is not None and timestamp_s<=self.last_t:raise ValueError('Monotone physical substeps required')
        dt=.005 if self.last_t is None else timestamp_s-self.last_t
        if dt>.005001:raise ValueError('Record contact at200Hz physical substeps, not50Hz controls')
        if len(ball_world_xyz)!=3 or not all(math.isfinite(v) for v in ball_world_xyz):raise ValueError('Finite world ball trajectory required for scoring')
        row=self.contact.update(paired_foot_forces_world_n,foot_names,active_policy,kick_requested,dt)
        crossing=self.previous is not None and goal_crossing(self.previous,ball_world_xyz)
        if crossing and self.contact.kick_contact_observed and self.goal_time is None:self.goal_time=timestamp_s
        self.fell|=bool(fell);self.obstacle_collision|=obstacle_force_n>.001;self.max_joint_violation=max(self.max_joint_violation,joint_violation_rad)
        self.previous=list(ball_world_xyz);self.last_t=timestamp_s
        return {**row,'timestamp_s':timestamp_s,'whole_ball_crossed_this_step':bool(crossing),'qualified_goal_seen':self.goal_time is not None,'scoring_only':True}
    def summary(self):
        return {'scoring_version':GOAL_SCORING_VERSION,'goal_time_s':self.goal_time,'real_foot_ball_contact':self.contact.kick_contact_observed,'whole_ball_crossed_inside_goal':self.goal_time is not None,'fell':self.fell,'obstacle_collision':self.obstacle_collision,'joint_limit_violation_rad':self.max_joint_violation,'contact':self.contact.summary(),'controller_must_not_read':True}
