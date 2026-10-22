"""Named physical outputs and passive soft constraints are distinct quantities."""
from pathlib import Path
import hashlib,json
import numpy as np
from contact_scoring import PhysicalGoalScorer
PROTOCOL=Path(__file__).resolve().parents[1]/'evaluation/football-physics-semantics.json'
PROTOCOL_SHA='a2aaa6ae660d9f8d8ad2fb30fa62d133cccdcb08a04a6cf54f3cce5c6a948c08'

class PhysicalGoalScorerV2(PhysicalGoalScorer):
    """Preserve every legacy field; add independent v2 named safety gates."""
    def __init__(self,joint_names):
        super().__init__()
        if hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()!=PROTOCOL_SHA:raise ValueError('Frozen football protocol hash changed')
        self.named_joints=list(joint_names)
        if len(self.named_joints)!=14 or len(set(self.named_joints))!=14:raise ValueError('14 distinct physical output names required')
        self.servo_peak=0.;self.output_peak=0.;self.passive=[];self.sample_dt=[]
    def update(self,*,servo_coordinate_violation_rad,physical_output_violation_rad,passive_soft_limit_residual_rad,**kwargs):
        for value in (servo_coordinate_violation_rad,physical_output_violation_rad,passive_soft_limit_residual_rad):
            a=np.asarray(value,dtype=float)
            if a.shape!=(14,) or not np.isfinite(a).all() or (a<0).any():raise ValueError('Finite nonnegative per-joint14 metrics required')
        dt=.005 if self.last_t is None else kwargs['timestamp_s']-self.last_t
        row=super().update(**kwargs)
        self.servo_peak=max(self.servo_peak,max(servo_coordinate_violation_rad));self.output_peak=max(self.output_peak,max(physical_output_violation_rad))
        self.passive.append(list(passive_soft_limit_residual_rad));self.sample_dt.append(dt)
        return {**row,'servo_coordinate_peak_violation_rad':float(max(servo_coordinate_violation_rad)),'physical_output_peak_violation_rad':float(max(physical_output_violation_rad)),'passive_soft_limit_residual_rad':list(passive_soft_limit_residual_rad),'legacy_all28_peak_violation_rad':kwargs['joint_violation_rad'],'physics_semantics_id':'football-beta-180s-mechanical-v2'}
    def summary(self):
        old=super().summary();a=np.asarray(self.passive);dt=np.asarray(self.sample_dt);diagnostics=[]
        if len(a):
            for j,n in enumerate(self.named_joints):diagnostics.append({'joint':n,'peak_rad':float(a[:,j].max()),'p95_rad':float(np.percentile(a[:,j],95)),'p99_rad':float(np.percentile(a[:,j],99)),'over_range_fraction':float((a[:,j]>0).mean()),'over_range_duration_s':float(dt[a[:,j]>0].sum())})
        safety=self.servo_peak<=.001 and self.output_peak<=.001
        return {**old,'physics_semantics_id':'football-beta-180s-mechanical-v2','protocol_sha256':PROTOCOL_SHA,'servo_coordinate_peak_violation_rad':float(self.servo_peak),'physical_output_peak_violation_rad':float(self.output_peak),'legacy_all28_peak_violation_rad':old['joint_limit_violation_rad'],'passive_soft_limit_peak_residual_rad':float(a.max())if len(a)else None,'passive_soft_limit_per_joint':diagnostics,'passive_fit_status':'not calibrated to physical hardware','mechanical_output_bounds_passed':bool(safety and len(a)),'physical_episode_success':bool(len(a)and safety and old['whole_ball_crossed_inside_goal']and old['real_foot_ball_contact']and not old['fell']and not old['obstacle_collision']and old['goal_time_s']<=180.),'qualification_claimed':False}
