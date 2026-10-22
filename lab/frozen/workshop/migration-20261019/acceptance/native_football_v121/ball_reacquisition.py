"""Bounded public head-only recovery; old ball/map data is attention seed only."""
import math
import numpy as np
from task_navigation import TaskNavigator
from goal_reacquisition_native import GoalAwareGazeSlew
from image_head_servo import attention_correction as image_geometry
from bounded_image_head_servo import bounded_step
from native_head_domain import HeadCommandDomainAdapter
from attention_owner import AttentionOwner
from attention_head_permission import head_only_permission
from ball_view_hint import BallViewHint

LOWER=np.array([-1.6,0.]);UPPER=np.array([1.6,1.4])

class BallSearch:
    def __init__(self):
        self.started=None;self.targets=[];self.index=0;self.arrived=None;self.failed=None;self.domain_checks=[]
    def clear(self):self.__init__()
    def begin(self,t,seed):
        if self.started is not None:return
        q=np.asarray(seed,float)
        if not math.isfinite(t)or t<0 or q.shape!=(2,)or not np.isfinite(q).all():raise ValueError('Finite search seed/time required')
        self.started=t;gate=HeadCommandDomainAdapter()
        offsets=[(0,0),(-.3,0),(.3,0),(0,.2),(-.3,.2),(.3,.2),(0,-.25),(-.3,-.25),(.3,-.25)]
        # A long approach can leave the remembered gaze aimed at the old,
        # farther ball. Head-only scan must include the existing near-ball
        # pitch domain. No old metric estimate grants body/kick permission.
        if q[1]<.85:
            offsets=[(0,0),(0,1.1-q[1]),(0,1.3-q[1]),(-.3,1.1-q[1]),(.3,1.1-q[1]),(-.3,1.3-q[1]),(.3,1.3-q[1])]+offsets[1:]
        for dy,dp in offsets:
            target=np.clip(q+[dy,dp],LOWER,UPPER);check=gate.solve(target)
            self.domain_checks.append({'target_gaze_rad':target.tolist(),'accepted':check['accepted'],'head_offsets':check['head_pose_native_offsets_rad'],'pointing_residual_rad':check['requested_pointing_residual_rad'],'source_command_bounds_rad':check['source_head_command_bounds_rad']})
            if check['accepted'] and not any(np.array_equal(target,x)for x in self.targets):self.targets.append(target)
        if not self.targets:self.failed='ball_search_no_source_domain_target'
    def target(self,t,current):
        if self.started is None:raise ValueError('Search has not begun')
        if not math.isfinite(t)or t<self.started:raise ValueError('Monotone finite search clock required')
        if t-self.started>=24.-1e-9:self.failed='ball_reacquisition_timeout'
        if self.failed:return np.asarray(current,float)
        target=self.targets[self.index]
        if np.max(abs(np.asarray(current)-target))<=.03:
            if self.arrived is None:self.arrived=t
            if t-self.arrived>=1.-1e-9:
                self.index+=1;self.arrived=None
                if self.index>=len(self.targets):self.failed='ball_search_targets_exhausted';return np.asarray(current,float)
                target=self.targets[self.index]
        else:self.arrived=None
        return target.copy()

class BallAwareGaze(GoalAwareGazeSlew):
    def __init__(self,owner):
        super().__init__(owner);self.candidate_domain=HeadCommandDomainAdapter()

    def update(self,target,t):
        owner=self.owner
        if getattr(owner,'external_fine_step_head_owned',False):
            if not math.isfinite(t)or t<0 or self.timestamp is not None and t<self.timestamp:raise ValueError('Monotone external head-owner time required')
            self.timestamp=t
            self.last={'executed_body_yaw_pitch_rad':self.current.tolist(),'source':'neutral fine-step owns native head command; navigation head targets paused, attention deadlines continue','world_stabilized':False}
            owner.ball_recovery_diagnostic={'mode':'neutral fine-step head owner; no search waypoint credited','base_motion_authorized':False,'kick_authorized':False,'truth_used':False}
            return self.current.tolist()
        permission=getattr(owner,'attention_head_permission',None) or {}
        fresh_head_context=bool(owner.ball_recovery_context_valid and permission.get('accepted')
            and permission.get('capture_s')==t and permission.get('owner')==owner.attention_owner.state)
        if owner.attention_owner.state=='GOAL_REFRESH':
            # Commit attention until a new accepted pose fix, with no body
            # permission. NearBallNavigator may have selected another state;
            # this explicit owner is the sole head-attention decision here.
            owner.state='REFRESH_GOAL'
            old_context,old_input=owner.goal_attention_context_valid,owner.goal_attention_input
            if not fresh_head_context:
                owner.goal_attention_context_valid=False;owner.goal_attention_input=None
            try:result=super().update(target,t)
            finally:owner.goal_attention_context_valid,owner.goal_attention_input=old_context,old_input
            owner.ball_recovery_diagnostic={'mode':'committed goal refresh; ball absence grants no body authority',
                'started_s':owner.ball_search.started,'failed':None,'base_motion_authorized':False,'kick_authorized':False,
                'source_domain_checks':owner.ball_search.domain_checks,'truth_used':False}
            return result
        g=getattr(owner,'last_geometry',None);ball=g.get('ball') if g else None
        if (getattr(owner,'final_entry',False) and owner.attention_owner.state=='NONE'
                and owner.state!='REFRESH_GOAL' and g and g.get('timestamp_s')==t
                and ball and ball.get('accepted') and g.get('floor_observation',{}).get('accepted')
                and abs(self.current[0])>.4):
            # Search has already admitted this gaze through native IK. Keep
            # the successful actual view through the static measurement window
            # instead of snapping its target back into the legacy +/- .4 box.
            # No head motion, body permission, or observation timestamp renewal.
            if not math.isfinite(t) or t<0 or self.timestamp is not None and t<self.timestamp:raise ValueError('Monotone held-view time required')
            self.timestamp=t
            self.last={'executed_body_yaw_pitch_rad':self.current.tolist(),'source':'hold current complete-ball view during near observation','capture_s':t,'world_stabilized':False,'new_motion_authority':False}
            return self.current.tolist()
        if not owner.ball_recovery_active:return super().update(target,t)
        if not math.isfinite(t)or t<0 or self.timestamp is not None and t<self.timestamp:raise ValueError('Finite monotone gaze time required')
        dt=0. if self.timestamp is None else t-self.timestamp
        search=owner.ball_search;search.begin(t,self.current)
        candidate=search.target(t,self.current);mode='bounded search seeded by previous public gaze only'
        hint=owner.ball_view_hint.target(now_s=t,current=self.current) if not search.failed else None
        pixel=owner.ball_recovery_pixel_input
        if hint is not None:
            candidate=hint;mode='return to last complete-ball gaze command; head-only angular hint'
        elif pixel is not None and not search.failed:
            attention,rotation=pixel
            measured=image_geometry(attention,command=self.current,rotation_body_from_raw=rotation,camera=owner.vision.camera,now_s=t,dt_s=dt)
            if measured is not None:
                lower=np.maximum(-.45*dt,LOWER-self.current);upper=np.minimum(.45*dt,UPPER-self.current)
                delta,box=bounded_step(measured['local_image_jacobian'],np.array(measured['image_error_normalized'])*2*dt,lower,upper)
                candidate=self.current+delta;mode='fresh visible ball pixels only; same-capture encoder FK'
        if not fresh_head_context:candidate=self.current.copy();mode='hold head; fresh public head pose or explicit attention owner unavailable'
        raw=np.asarray(candidate,float);limited=np.clip(raw,LOWER,UPPER)
        proposed=self.current+np.clip(limited-self.current,-self.rate*dt,self.rate*dt)
        previous_ik=self.candidate_domain.last.copy()
        domain=self.candidate_domain.solve(proposed)
        if domain['accepted']:self.current=proposed
        else:
            self.candidate_domain.last=previous_ik
            mode='hold head; proposed gaze rejected by original native command/physical IK domain'
        self.timestamp=t
        self.last={'requested_body_yaw_pitch_rad':raw.tolist(),'bounded_body_yaw_pitch_rad':limited.tolist(),'executed_body_yaw_pitch_rad':self.current.tolist(),'slew_limit_rad_s':self.rate,'limit_clipped':bool(np.any(raw!=limited)),'world_stabilized':False,'gait_head_command_generalization_validated':False,'scope':'Head attention only; actual native command and physical IK gates remain in executor'}
        owner.ball_recovery_diagnostic={'mode':mode,'started_s':search.started,'elapsed_s':t-search.started,'deadline_s':24.,'waypoint_index':search.index,'waypoint_count':len(search.targets),'failed':search.failed,'last_complete_ball_capture_s':owner.ball_time,'current_complete_ball_age_s':t-owner.ball_time,'source_domain_checks':search.domain_checks,'base_motion_authorized':False,'kick_authorized':False,'truth_used':False,'fresh_pixel_capture_s':None if pixel is None else pixel[0]['capture_timestamp_s'],
            'candidate_native_domain':domain,'proposed_gaze_rad':proposed.tolist(),
            'local_J_contract':'ideal body yaw/pitch axes local approximation; uncalibrated command-to-actual-camera response, ignores camera translation'}
        return self.current.tolist()

class BallReacquisitionNavigator(TaskNavigator):
    def reset(self):
        super().reset();self.ball_recovery_active=False;self.ball_recovery_context_valid=False;self.ball_recovery_pixel_input=None;self.ball_recovery_diagnostic=None;self.attention_head_permission=None;self.ball_search=BallSearch();self.attention_owner=AttentionOwner();self.gaze=BallAwareGaze(self)
        self.ball_view_hint=BallViewHint()
    def _output(self,t,command=(0.,0.,0.),reason='',gaze=None):
        g=self.last_geometry;ball=g.get('ball')if g else None
        fresh=bool(g and g.get('timestamp_s')==t and ball and ball.get('accepted')and g.get('floor_observation',{}).get('accepted'))
        # The image was captured before this call changes the gaze target.
        # Store the prior issued gaze, not the new goal-refresh output.
        self.ball_view_hint.observe(now_s=t,capture_s=None if g is None else g.get('timestamp_s'),
            gaze=self.gaze.current,complete=fresh)
        near=bool(self.final_entry or self.public_near_context(t))
        request_goal=bool(near and reason=='visual_localization_refresh' and t-self.goal_time>8.)
        owner=self.attention_owner.update(t=t,near=near,complete_capture_s=t if fresh else None,
            goal_fix_s=self.goal_time,request_goal=request_goal,
            defer_complete_release=bool(getattr(self,'external_fine_step_active',False)
                and not getattr(self,'external_fine_step_head_owned',False)))
        if self.attention_owner.transition=='new_accepted_goal_fix':
            self.ball_view_hint.begin_return(t)
        if owner in ('NONE','FAILED'):
            self.ball_view_hint.cancel_return()
        attention_active=owner!='NONE'
        self.ball_recovery_active=owner in ('BALL_REACQUIRE','FAILED')
        if attention_active:
            self.ball_search.begin(self.attention_owner.started,self.gaze.current)
            if self.attention_owner.failed:self.ball_search.failed=self.attention_owner.failed
        self.ball_recovery_context_valid=False;self.ball_recovery_pixel_input=None;self.attention_head_permission=None
        if owner=='NONE' and fresh:self.ball_search.clear()
        if attention_active and self.current_sensor_pair is not None:
            proprio,_=self.current_sensor_pair
            self.attention_head_permission=head_only_permission(t=t,owner=owner,proprio=proprio,geometry=g)
            self.ball_recovery_context_valid=self.attention_head_permission['accepted']
            attention=self.ball_attention
            if self.ball_recovery_context_valid and attention is not None and 0<=t-attention['capture_timestamp_s']<=.2:
                fk=self.vision.head.forward_policy14(proprio['q_rad'],joint_names=proprio['joint_names'])['camera_raw_optical']
                self.ball_recovery_pixel_input=(attention,fk.rotation)
            command=(0.,0.,0.)
        out=super()._output(t,command,reason,gaze)
        if attention_active:
            self.native_commands.reset();self.final_approach.invalidate(t,'ball_reacquisition_requires_new_complete_metric')
            out.update(state='REFRESH_GOAL' if owner=='GOAL_REFRESH' else 'REACQUIRE_BALL',velocity_body_xyw=[0.,0.,0.],kick_request=False,reason='bounded_public_attention_owner',ball_reacquisition=self.ball_recovery_diagnostic)
            out['motion_authorization']={**out.get('motion_authorization',{}),'allowed':False,'output_reason':'head_only_ball_reacquisition'}
            out['legacy_gaze_domain_diagnostic']=out.pop('trained_gaze_domain',None)
            out['native_ball_gaze_domain']={'candidate_gaze_lower_rad':LOWER.tolist(),'candidate_gaze_upper_rad':UPPER.tolist(),'source_head_offsets_checked_by':'HeadCommandDomainAdapter: pinned native head bounds and physical IK limits intersection','learned_tracking_validated':False}
        out['public_attention_owner']=self.attention_owner.diagnostic(t)
        out['public_ball_view_hint']=self.ball_view_hint.diagnostic(t)
        out['public_attention_head_permission']=self.attention_head_permission
        return out
    def step(self,*args,**kwargs):
        out=super().step(*args,**kwargs)
        if self.attention_owner.state!='NONE':
            out.update(state='REFRESH_GOAL' if self.attention_owner.state=='GOAL_REFRESH' else 'REACQUIRE_BALL',velocity_body_xyw=[0.,0.,0.],kick_request=False,ball_reacquisition=self.ball_recovery_diagnostic)
        return out
