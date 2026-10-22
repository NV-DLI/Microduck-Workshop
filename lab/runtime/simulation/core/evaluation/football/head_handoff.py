"""Head handoff: Microduck simulation and control."""
import math
import numpy as np
from ball_status_adapter import SameReadMemoryAdapter
from final_approach import in_training_box


class HeadHandoff:
    def __init__(self,leg_fk,joint_names):
        self.adapter=SameReadMemoryAdapter(leg_fk,joint_names)
        self.state='APPROACH';self.started=None;self.last_t=None;self.head=None
        self.quiet_start=None;self.kick_start=None;self.reason=None;self.attempts=0
        self.latest_readback=None;self.last_goal_heading=None;self.last_goal_s=None
        self.initial_capture_s=None

    def read(self,packet,*,now_s):
        self.latest_readback=self.adapter.consume(packet,now_s=now_s)
        return self.latest_readback

    def capture(self,*,now_s,geometry,head_offsets,parameters,goal_heading,goal_fix_s):
        # Anchor only before return; an actual reobservation during return is
        # reported separately and never hides a stale anchor or resets its age.
        if self.state!='APPROACH':
            if math.isfinite(goal_heading) and math.isfinite(goal_fix_s) and 0<=now_s-goal_fix_s<=.5:
                self.last_goal_heading=float(goal_heading);self.last_goal_s=float(goal_fix_s)
            return self.status(now_s)
        ball=geometry.get('ball') if geometry else None
        if not(ball and ball.get('accepted')):return self.status(now_s)
        g=ball['geometry'];read_s=self.adapter.last_read_s
        if read_s is None or not 0<=now_s-read_s<=.02000001:return self.status(now_s)
        anchor=self.adapter.memory.observe_complete_ball(timestamp_s=now_s,point_body_m=g['point_reference_m'],
            estimated_range_uncertainty_m=g['estimated_range_uncertainty_m'],complete=bool(g['observed_boundary_required']),
            geometry_accepted=bool(g['accepted']),capture_read_age_s=now_s-read_s)
        if parameters:
            head=np.asarray(head_offsets,float)
            if head.shape!=(4,)or not np.isfinite(head).all():raise ValueError('Actual current four head commands required')
            if not in_training_box(anchor['point_level_m'])or not math.isfinite(goal_heading)or abs(goal_heading)>.08:
                return self.status(now_s)
            self.head=head.copy();self.started=now_s;self.last_t=now_s;self.initial_capture_s=now_s
            self.last_goal_heading=float(goal_heading);self.last_goal_s=float(goal_fix_s)
            self.state='HEAD_RETURN';self.attempts+=1
        return self.status(now_s)

    def fail(self,reason):self.state='FAILED';self.reason=reason

    def policy(self,*,now_s,packet):
        self.read(packet,now_s=now_s)
        state=self.adapter.memory.status(now_s)
        if self.state=='APPROACH':return self.status(now_s)
        if self.state in ('FAILED','COMPLETE'):return self.status(now_s)
        if self.state=='KICK':
            valid=bool(packet is not None and packet.get('orientation_valid') is True
                and 0<=now_s-packet['read_timestamp_s']<=.08
                and 0<=now_s-packet['timestamp_s']<=.08)
            if not valid:self.fail('sensor_lost_during_kick')
            elif not 0<now_s-self.last_t<=.025:self.fail('kick_policy_clock_gap')
            elif now_s-self.kick_start>=4.-1e-8:self.state='COMPLETE'
            self.last_t=now_s
            return self.status(now_s)
        if not state['memory_available']:
            self.fail('headreturn_ball_memory_expired');return self.status(now_s)
        if now_s-self.started>4.5:
            self.fail('headreturn_deadline');return self.status(now_s)
        dt=now_s-self.last_t
        if not 0<=dt<=.025:self.fail('headreturn_policy_clock_gap');return self.status(now_s)
        self.last_t=now_s;self.head+=np.clip(-self.head,-.3*dt,.3*dt)
        if packet is None:self.fail('headreturn_public_sensor_unavailable');return self.status(now_s)
        gravity=np.asarray(packet['gravity_body'],float);gyro=np.asarray(packet['gyro_body_rad_s'],float)
        qd=np.asarray(packet['encoder_qd_rad_s'],float)
        tilt=math.atan2(float(np.linalg.norm(gravity[:2])),-float(gravity[2]))
        quiet=(np.max(np.abs(self.head))<=1e-9 and np.linalg.norm(gyro)<=.1 and np.sqrt(np.mean(qd**2))<=.25 and tilt<=.35)
        if quiet:
            if self.quiet_start is None:self.quiet_start=now_s
        else:self.quiet_start=None
        if self.quiet_start is not None and now_s-self.quiet_start>=.5-1e-8:
            goal_fresh=self.last_goal_s is not None and 0<=now_s-self.last_goal_s<=.5
            if not in_training_box(state['point_level_m']):self.fail('propagated_point_outside_training_box')
            elif not goal_fresh:self.fail('fresh_goal_fix_required_after_headreturn')
            elif abs(self.last_goal_heading)>.08:self.fail('goal_heading_outside_development_gate')
            else:self.state='KICK';self.kick_start=now_s
        return self.status(now_s)

    def status(self,now_s):
        memory=self.adapter.memory.status(now_s)
        return {'state':self.state,'reason':self.reason,'head_offsets':None if self.head is None else self.head.tolist(),
            'override_body_zero':self.state!='APPROACH','head_return_complete':self.state in ('KICK','COMPLETE'),
            'kick_runtime_authorized':self.state=='KICK','kick_phase_s':None if self.kick_start is None else now_s-self.kick_start,
            'kick_onset_s':self.kick_start,'attempts':self.attempts,'memory':memory,
            'last_goal_heading_rad':self.last_goal_heading,'last_goal_fix_s':self.last_goal_s,
            'headreturn_start_s':self.started,'headreturn_quiet_start_s':self.quiet_start,
            'uncertainty_envelope_inside':False,'uncalibrated_target_authorization':True,'qualification_claimed':False,
            'only_point_estimate_development_permission':True,'truth_used':False}
