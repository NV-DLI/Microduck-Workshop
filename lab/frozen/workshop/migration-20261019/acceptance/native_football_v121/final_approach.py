"""Public point-estimate development alignment; no qualification/hidden state.

The old bbox uncertainty remains reported and is NOT a calibrated bound on the
new grounded center. This separate development controller never authorizes a
kick. All fresh complete RGB/ToF/pose gates precede its finite motion budget.
"""
from collections import deque
import math
import numpy as np
from turn_prediction import near_turn_stop_hint


def wrap(x):return math.atan2(math.sin(x),math.cos(x))
def rotation(x):return np.array([[math.cos(x),-math.sin(x)],[math.sin(x),math.cos(x)]])

def in_training_box(point):
    p=np.asarray(point,float)
    return bool(p.shape==(3,) and np.isfinite(p).all() and .075<=p[0]<=.105 and -.05<=p[1]<=-.02)


class FinalApproach:
    """Finite turn/forward pulses followed by actual fresh quiet observations."""
    def __init__(self):
        self.turns=0;self.advances=0;self.last_t=None;self.phase='OBSERVE'
        self.started=None;self.sign=0.;self.duration=0.;self.samples=deque()
        self.failed=None;self.acceptable_since=None;self.forward_command=.3
        self.staging_phase="POSITION"; self.turn_stop_limit=.06; self.turn_goal=False
        self.turn_amplitude={-1.:.4,1.:.4};self.turn_rate={-1.:.25,1.:.25};self.pending_turn=None;self.pulse_stop_s=None;self.policy_stop_capture=None

    def policy_command(self,*,t,packet,velocity):
        """50 Hz pulse expiry. No world state, new motion, or stale-frame reuse."""
        if self.policy_stop_capture is not None and self.last_t<=self.policy_stop_capture:
            return [0.,0.,0.],{'held_zero_until_new_capture':True,'truth_used':False}
        if self.phase!='TURN' or not self.turn_goal:return velocity,None
        from neutral_fine_step import NeutralFineStep
        valid=NeutralFineStep.packet_valid(packet,t) and self.last_t is not None and 0<=t-self.last_t<=.2
        expired=self.pulse_stop_s is not None and t>=self.pulse_stop_s-1e-9
        trace={'timestamp_s':t,'deadline_s':self.pulse_stop_s,'fresh_public_packet':valid,'expired':expired,'truth_used':False}
        if not valid or expired:
            self.policy_stop_capture=self.last_t
            self.invalidate(t,'goal_pulse_expired' if expired else 'goal_pulse_stale_input')
            return [0.,0.,0.],trace
        return velocity,trace

    def invalidate(self,t,reason):
        self.phase='OBSERVE';self.started=t;self.samples.clear();self.acceptable_since=None
        return [0.,0.,0.],{'phase':self.phase,'reason':reason,'truth_used':False,'kick_authorized':False,'final_alignment_candidate':False}

    def update(self,*,t,point,gyro_body,gyro_level,goal_heading,authorized,uncertainty):
        if isinstance(t,bool) or not math.isfinite(t) or t<0:
            raise ValueError('Finite nonnegative capture clock required before any state write')
        p=np.asarray(point,float) if point is not None else np.full(3,np.nan)
        w=np.asarray(gyro_body,float)
        valid=(math.isfinite(t) and t>=0 and p.shape==(3,) and w.shape==(3,) and np.isfinite(p).all()
               and np.isfinite(w).all() and math.isfinite(gyro_level) and math.isfinite(goal_heading)
               and math.isfinite(uncertainty) and uncertainty>=0)
        if not valid:return self.invalidate(t,'invalid_public_final_input')
        if self.last_t is not None and not 0<t-self.last_t<=.2:
            self.last_t=t;return self.invalidate(t,'final_capture_clock_gap')
        self.last_t=t
        if self.failed:return self.invalidate(t,self.failed)
        if not authorized:return self.invalidate(t,'final_upstream_guard')
        if p[0]<=.06:return self.invalidate(t,'final_point_too_close_or_behind')
        if self.started is None:self.started=t
        shot_point=rotation(-goal_heading)@p[:2]
        standoff_error=shot_point-np.array([.26,-.035])
        # Establish a shot-line station while the ball is still observable.
        # Latch only from observation, never retarget an active turn/advance.
        distance=float(np.linalg.norm(p[:2]))
        foot_bearing=wrap(math.atan2(p[1],p[0])+math.asin(min(1.,.035/max(distance,.035))))
        predicted_goal_heading=wrap(goal_heading-foot_bearing)
        if self.staging_phase=='POSITION' and self.phase=='OBSERVE' and ((.13<=p[0]<=.36 and abs(predicted_goal_heading)<=.08) or in_training_box(p)):
            self.staging_phase='GOAL_ALIGN'
        stage_position=self.staging_phase=='POSITION'
        target_x=.26 if stage_position else .09
        error=p[:2]-rotation(goal_heading)@np.array([target_x,-.035])
        norm=float(np.linalg.norm(error));inside=in_training_box(p)
        desired=bool(abs(p[0]-.09)<=.005 and abs(p[1]+.035)<=.005 and abs(goal_heading)<=.08)
        acceptable=inside and abs(goal_heading)<=.08
        # First approach the physical strike point. Aim the body independently
        # once close; changing yaw changes the measured body-relative point.
        final_yaw=self.turn_goal if self.phase=='TURN' else (not stage_position or inside)
        bearing=wrap(goal_heading-max(-.06,min(.06,predicted_goal_heading))) if final_yaw else math.atan2(error[1],error[0])
        d={'phase':self.phase,'point_level_m':p.tolist(),'goal_heading_rad':goal_heading,'waypoint_error_m':error.tolist(),
           'bearing_rad':bearing,'alignment_mode':'final_goal_heading' if final_yaw else 'direct_visual_preparation_point',
           'old_bbox_uncertainty_m':uncertainty,'uncertainty_envelope_inside':False,
           'uncalibrated_target_authorization':True,'training_box_contains_point':inside,'desired_center_5mm':desired,
           'final_alignment_candidate':False,'turn_attempts':self.turns,'advance_attempts':self.advances,
           'kick_authorized':False,'truth_used':False,
           'staging_phase':self.staging_phase,'shot_frame_ball_xy_m':shot_point.tolist(),
           'standoff_error_m':standoff_error.tolist(),'standoff_target_m':[.26,-.035],'foot_alignment_bearing_rad':foot_bearing,'predicted_goal_heading_after_foot_alignment_rad':predicted_goal_heading,'goal_mouth_heading_gate_rad':.08}
        if self.phase=='TURN' and self.turn_goal:
            if t>=self.pulse_stop_s-1e-9:
                out,h=self.invalidate(t,'goal_pulse_expired');return out,{**d,**h}
            return [0.,0.,self.turn_amplitude[self.sign]*self.sign],{**d,'pulse_deadline_s':self.pulse_stop_s}
        if self.phase=='TURN':
            hint=near_turn_stop_hint(bearing,gyro_level,self.sign)
            if self.turn_stop_limit<.06:
                hint['signed_remaining_target_rad']=self.turn_stop_limit
                hint['stop_hint']=bool(gyro_level*self.sign>.12 and bearing*self.sign>0 and (bearing-.15*gyro_level)*self.sign<=self.turn_stop_limit)
            elapsed=t-self.started;minimum_elapsed=elapsed>=.4-1e-9
            hint={**hint,'turn_elapsed_s':elapsed,'minimum_duration_s':.4,'minimum_duration_elapsed':minimum_elapsed}
            if minimum_elapsed and (bearing*self.sign<=0 or abs(bearing)<=self.turn_stop_limit or hint['stop_hint']) or elapsed>=2.4:
                out,h=self.invalidate(t,'final_turn_stop');return out,{**d,**h,'turn_stop_prediction':hint}
            return [0.,0.,.8*self.sign],{**d,'elapsed_s':t-self.started,'turn_stop_prediction':hint}
        if self.phase=='ADVANCE':
            if not stage_position and (abs(goal_heading)>.08 or abs(foot_bearing)>.08):
                out,h=self.invalidate(t,'standoff_heading_drift_stop');return out,{**d,**h}
            if self.forward_command>.2 and np.linalg.norm(p[:2])<=.22:
                out,h=self.invalidate(t,'final_close_range_transition_stop');return out,{**d,**h,'forward_command':self.forward_command,'close_range_threshold_m':.22}
            if abs(bearing)>.15 or error[0]<=.005 or p[0]<=.075 or t-self.started>=self.duration-1e-9:
                out,h=self.invalidate(t,'final_advance_stop');return out,{**d,**h}
            return [self.forward_command,0.,0.],{**d,'elapsed_s':t-self.started,'duration_s':self.duration,'forward_command':self.forward_command,'close_range_threshold_m':.22}
        self.samples.append((t,p[:2].copy(),float(w@w)))
        while self.samples and t-self.samples[0][0]>.45:self.samples.popleft()
        xyz=np.stack([z[1]for z in self.samples]);rms=float(np.sqrt(np.mean([z[2]for z in self.samples])))
        diameter=float(np.linalg.norm(xyz[:,None]-xyz[None,:],axis=-1).max())
        covered=len(self.samples)>=6 and t-self.samples[0][0]>=.33
        quiet=covered and rms<=.12 and diameter<=.02 and t-self.started>=.4
        d.update(measured_quiet=quiet,gyro_3D_RMS_rad_s=rms,ball_XY_diameter_m=diameter,quiet_sample_times=[z[0]for z in self.samples])
        if not quiet:return [0.,0.,0.],d
        if self.pending_turn is not None:
            record=self.pending_turn;self.pending_turn=None
            net=wrap(record['heading']-goal_heading)*record['sign']
            elapsed=record['duration']
            if .003<net<.9 and elapsed>0:
                observed=min(2.,max(.05,net/elapsed));sign=record['sign']
                self.turn_rate[sign]=.25*self.turn_rate[sign]+.75*observed
            elif abs(net)<.005 and elapsed>=.5:
                sign=record['sign'];self.turn_amplitude[sign]=min(.8,self.turn_amplitude[sign]+.2)
            d['completed_public_pulse']={**record,'settled_heading':goal_heading,'net_signed_yaw_rad':net,'updated_rates':dict(self.turn_rate)}
        if acceptable and not stage_position:
            if self.acceptable_since is None:self.acceptable_since=t
            # Prefer the center. A bounded one-second stable acceptable point is
            # also usable for this explicit uncalibrated development attempt.
            if desired or t-self.acceptable_since>=1.:
                return [0.,0.,0.],{**d,'phase':'PREPARED','reason':'final_point_and_heading_development_candidate','final_alignment_candidate':True}
            return [0.,0.,0.],{**d,'reason':'observe_acceptable_point'}
        self.acceptable_since=None
        # Close neutral-head preparation targets the existing .06 turn-stop
        # neighborhood before head return. The later actual movement gate
        # remains .15; this stricter entry margin is not a guaranteed drift bound.
        close_preparation=bool(np.linalg.norm(p[:2])<=.22 and not final_yaw)
        turn_entry_limit=(.005 if abs(goal_heading)>.065 else .03 if np.linalg.norm(p[:2])<=.22 else .06) if final_yaw else (.06 if close_preparation else .15)
        d.update(turn_entry_bearing_limit_rad=turn_entry_limit,close_headreturn_direction_margin=close_preparation)
        if abs(bearing)>turn_entry_limit:
            if self.turns>=24:self.failed='final_turn_budget_exhausted';return self.invalidate(t,self.failed)
            self.turn_goal=final_yaw
            self.turn_stop_limit=.015 if final_yaw and not inside else .06
            self.turns+=1;self.phase='TURN';self.sign=math.copysign(1.,bearing);self.started=t;self.samples.clear()
            if final_yaw:
                duration=min(.8,max(.04,abs(bearing)/self.turn_rate[self.sign]))
                # Round to policy cadence and record the prescribed pulse.
                duration=round(duration/.02)*.02
                self.pulse_stop_s=t+duration
                self.pending_turn={'heading':goal_heading,'sign':self.sign,'duration':duration,'start_s':t,'amplitude':self.turn_amplitude[self.sign]}
                d['goal_pulse']={**self.pending_turn,'deadline_s':self.pulse_stop_s}

            return [0.,0.,(self.turn_amplitude[self.sign] if final_yaw else .8)*self.sign],{**d,'phase':'TURN','reason':'final_public_heading','turn_attempts':self.turns}
        if not stage_position and abs(goal_heading)<=.065 and abs(foot_bearing)<=.03:
            # Hold a fresh zero-body OBSERVE frame for the stricter fine entry.
            # No forward pulse may replace a rejected fine-step handoff.
            if np.linalg.norm(p[:2])<=.22:
                return [0.,0.,0.],{**d,'reason':'standoff_aligned_observe','fine_standoff_candidate':True}
        if not stage_position and np.linalg.norm(p[:2])<=.22:
            return [0.,0.,0.],{**d,'reason':'fine_standoff_wait_for_entry_guards'}
        if not stage_position and abs(predicted_goal_heading)>.08:
            return self.invalidate(t,'right_foot_line_outside_goal_mouth_development_domain')
        if error[0]<=.005:
            return self.invalidate(t,'final_reverse_or_lateral_reposition_unvalidated')
        if self.advances>=20:self.failed='final_forward_budget_exhausted';return self.invalidate(t,self.failed)
        close=bool(np.linalg.norm(p[:2])<=.22)
        self.forward_command=.2 if close else .3
        self.advances+=1;self.phase='ADVANCE';self.started=t;self.duration=.2 if close else (.4 if norm>.06 else .2);self.samples.clear()
        return [self.forward_command,0.,0.],{**d,'phase':'ADVANCE','reason':'final_public_forward','duration_s':self.duration,'forward_command':self.forward_command,'close_range_threshold_m':.22,'pulse_started_in_close_range':close,'advance_attempts':self.advances}
