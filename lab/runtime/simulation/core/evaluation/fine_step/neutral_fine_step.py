"""Neutral fine step: Microduck simulation and control."""
import math
import numpy as np

class NeutralFineStep:
    def __init__(self):
        self.state='IDLE';self.reason=None;self.started=None;self.last_policy=None
        self.head=None;self.quiet_start=None;self.step_start=None;self.step_end=None
        self.capture_s=None;self.capture_safe=False;self.goal_s=None;self.goal_heading=None
        self.anchor_s=None;self.attempts=0;self.last_finished=None;self.step_goal_heading=None
    @property
    def active(self):return self.state!='IDLE'
    @property
    def hold_navigation_head(self):return self.state in ('RETURN','STEP','SETTLE','FAILED')
    def fail(self,reason):
        self.state='FAILED';self.reason=reason
        if hasattr(self,'last_sent'):self.head=self.last_sent.copy()
    @staticmethod
    def packet_valid(packet,t):
        try:
            return bool(packet is not None and packet.get('orientation_valid') is True
                and isinstance(packet['seq'],int)and not isinstance(packet['seq'],bool)and packet['seq']>=0
                and all(math.isfinite(packet[k])and packet[k]>=0 and 0<=t-packet[k]<=.08 for k in ('timestamp_s','read_timestamp_s','encoder_read_timestamp_s'))
                and packet['encoder_read_timestamp_s']==packet['read_timestamp_s']
                and np.asarray(packet['gyro_body_rad_s']).shape==(3,)and np.asarray(packet['gravity_body']).shape==(3,)
                and np.asarray(packet['encoder_q_rad']).shape==(14,)and np.asarray(packet['encoder_qd_rad_s']).shape==(14,)
                and all(np.isfinite(packet[k]).all()for k in ('encoder_q_rad','encoder_qd_rad_s','gyro_body_rad_s','gravity_body')))
        except (ValueError,KeyError,TypeError):return False
    @staticmethod
    def waypoint_bearing(point,goal_heading):
        p=np.asarray(point,float)
        if p.shape!=(3,)or not np.isfinite(p).all()or goal_heading is None or not math.isfinite(goal_heading):return None
        c,s=math.cos(goal_heading),math.sin(goal_heading)
        target=np.array([c*.09+s*.035,s*.09-c*.035])
        error=p[:2]-target
        return float(math.atan2(error[1],error[0]))
    @staticmethod
    def angle_delta(current,reference):
        return math.atan2(math.sin(current-reference),math.cos(current-reference))
    @staticmethod
    def quiet(packet):
        gravity=np.asarray(packet['gravity_body'],float);gyro=np.asarray(packet['gyro_body_rad_s'],float);qd=np.asarray(packet['encoder_qd_rad_s'],float)
        tilt=math.atan2(float(np.linalg.norm(gravity[:2])),-float(gravity[2]))
        return bool(np.linalg.norm(gyro)<=.1 and np.sqrt(np.mean(qd**2))<=.25 and tilt<=.35)
    def capture(self,*,t,guard_safe,complete,trigger,actual_head,memory,goal_heading,goal_fix_s):
        if not math.isfinite(t)or t<0 or self.capture_s is not None and t<=self.capture_s:raise ValueError('Fresh monotone capture required')
        self.capture_s=t;self.capture_safe=bool(guard_safe)
        if math.isfinite(goal_heading)and math.isfinite(goal_fix_s)and 0<=t-goal_fix_s<=8:
            self.goal_heading=float(goal_heading);self.goal_s=float(goal_fix_s)
        if self.state=='IDLE' and trigger:
            p=np.asarray(memory.get('point_level_m'),float);head=np.asarray(actual_head,float)
            valid=bool(complete and guard_safe and memory.get('memory_available') and memory.get('ball_capture_timestamp_s')==t
                and p.shape==(3,)and np.isfinite(p).all()and .13<=p[0]and np.linalg.norm(p[:2])<=.22
                and -.10<=p[1]<=.08 and self.goal_s is not None and 0<=t-self.goal_s<=8
                and abs(self.waypoint_bearing(p,self.goal_heading))<=.15 and head.shape==(4,)and np.isfinite(head).all())
            if not valid:self.fail('neutral_step_initial_public_domain_rejected')
            else:
                self.state='RETURN';self.started=t;self.last_policy=t;self.head=head.copy();self.anchor_s=t
                self.quiet_start=None;self.step_start=None;self.step_end=None;self.step_goal_heading=None;self.attempts+=1
        elif self.state=='REACQUIRE' and complete and guard_safe and t>self.step_end:
            self.state='IDLE';self.last_finished=t;self.quiet_start=None
        return self.status(t,memory)
    def policy(self,*,t,packet,memory,actual_head):
        if not math.isfinite(t)or t<0 or self.last_policy is not None and t<self.last_policy:raise ValueError('Finite monotone policy clock required')
        self.last_sent=np.asarray(actual_head,float).copy()
        if self.last_sent.shape!=(4,)or not np.isfinite(self.last_sent).all():raise ValueError('Actual sent four-head command required')
        if self.state in ('IDLE','FAILED'):return self.status(t,memory)
        dt=t-self.last_policy
        if not 0<=dt<=.025:self.fail('neutral_step_policy_clock_gap');return self.status(t,memory)
        self.last_policy=t
        if t-self.started>=24.-1e-9:self.fail('neutral_step_total_deadline');return self.status(t,memory)
        if not self.packet_valid(packet,t):self.fail('neutral_step_invalid_same_read');return self.status(t,memory)
        if self.capture_s is None or not 0<=t-self.capture_s<=.2 or not self.capture_safe:
            self.fail('neutral_step_stale_or_hazard_capture');return self.status(t,memory)
        if self.state=='REACQUIRE':return self.status(t,memory)
        # SETTLE has irrevocably zero body authority. Old ball memory cannot
        # authorize another step here; fresh current packet/capture and the
        # original quiet/attention clocks still govern head-only reacquisition.
        if self.state=='SETTLE':
            if self.quiet(packet):
                if self.quiet_start is None:self.quiet_start=t
            else:self.quiet_start=None
            if self.quiet_start is not None and t-self.quiet_start>=.5-1e-9:self.state='REACQUIRE'
            return self.status(t,memory)
        if not memory.get('memory_available')or memory.get('ball_capture_timestamp_s')!=self.anchor_s or not 0<=t-self.anchor_s<=4.5:
            self.fail('neutral_step_memory_expired_or_reanchored');return self.status(t,memory)
        p=np.asarray(memory['point_level_m'],float)
        if p.shape!=(3,)or not np.isfinite(p).all()or not(.075<p[0]<=.25 and -.10<=p[1]<=.08):
            self.fail('neutral_step_propagated_point_invalid');return self.status(t,memory)
        if self.state=='RETURN':
            self.head+=np.clip(-self.head,-.3*dt,.3*dt)
            neutral=np.max(abs(self.head))<=1e-9 and np.max(abs(np.asarray(actual_head)))<=1e-9
            if neutral and self.quiet(packet):
                if self.quiet_start is None:self.quiet_start=t
            else:self.quiet_start=None
            if self.quiet_start is not None and t-self.quiet_start>=.5-1e-9:
                if self.goal_s is None or not 0<=t-self.goal_s<=.5:self.fail('neutral_step_fresh_goal_required')
                elif abs(self.waypoint_bearing(p,self.goal_heading))>.15:self.fail('neutral_step_waypoint_direction_required')
                elif p[0]<.13:self.fail('neutral_step_start_range_too_close')
                else:self.state='STEP';self.step_start=t;self.step_goal_heading=self.goal_heading;self.quiet_start=None
        elif self.state=='STEP':
            if t-self.step_start>=.6-1e-9 or p[0]<=.105:
                self.state='SETTLE';self.step_end=t;self.quiet_start=None
            elif self.goal_s is None or not 0<=t-self.goal_s<=1.1 or self.step_goal_heading is None or abs(self.angle_delta(self.goal_heading,self.step_goal_heading))>.15:
                self.fail('neutral_step_goal_or_heading_guard')
        return self.status(t,memory)
    def status(self,t,memory):
        return {'state':self.state,'reason':self.reason,'started_s':self.started,'step_start_s':self.step_start,'step_end_s':self.step_end,
            'anchor_capture_s':self.anchor_s,'head_offsets':None if self.state in ('IDLE','REACQUIRE')or self.head is None else self.head.tolist(),
            'velocity':[.2,0.,0.]if self.state=='STEP'else[0.,0.,0.],
            'override':self.active,'head_owned':self.hold_navigation_head,'quiet_started_s':self.quiet_start,
            'attempts':self.attempts,'last_finished_s':self.last_finished,'capture_guard_s':self.capture_s,'capture_safe':self.capture_safe,
            'goal_fix_s':self.goal_s,'goal_heading_rad':self.goal_heading,
            'waypoint_bearing_rad':self.waypoint_bearing(memory.get('point_level_m'),self.goal_heading),
            'step_goal_heading_rad':self.step_goal_heading,
            'step_heading_drift_rad':None if self.goal_heading is None or self.step_goal_heading is None else self.angle_delta(self.goal_heading,self.step_goal_heading),
            'direction_contract':'entry/RETURN preparation-waypoint bearing; STEP change in goal bearing from STEP onset; final kick absolute goal heading remains separate',
            'memory':memory,'maximum_memory_age_s':4.5,
            'maximum_attention_s':24.,'maximum_step_s':.6,'kick_authorized':False,'qualification_claimed':False,'truth_used':False}
