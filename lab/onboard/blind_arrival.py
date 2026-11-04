"""Near-ball visibility loss: require continuous public memory, then head return and measured quiet."""
import math

PARAMETERS = dict(x_m=[.07,.15], y_m=[-.10,.01], heading_rad=math.pi/12,
                  memory_max_age_s=6.5, goal_max_age_s=2., guard_max_age_s=.2,
                  sensor_max_age_s=.08, missing_captures=3)

def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def vec(v,n):return isinstance(v,(list,tuple)) and len(v)==n and all(finite(x) for x in v)
def inside(point):return vec(point,3) and .07<=point[0]<=.15 and -.10<=point[1]<=.01

class BlindArrival:
    def __init__(self):
        self.epoch=None;self.last_capture=None;self.missing_count=0
        self.last_visible_s=None;self.latch=None;self.attempted=False

    def reset_epoch(self,epoch):
        if epoch!=self.epoch:
            self.__init__();self.epoch=epoch

    def observe_capture(self,*,t,epoch,ball_metric_valid):
        self.reset_epoch(epoch)
        if not finite(t) or self.last_capture is not None and t<=self.last_capture:
            return False
        if self.last_capture is not None and t-self.last_capture>.2+1e-9:
            self.missing_count=0
        self.last_capture=t
        if ball_metric_valid:
            self.last_visible_s=t;self.missing_count=0
            self.latch=None  # Visible-ball control returns to its ordinary domain.
        else:self.missing_count+=1
        return True

    def common_checks(self,*,t,epoch,memory,packet,guard_safe,guard_s,goal_heading,
                      goal_s,body_command,odometry_continuous,kicked):
        packet = packet or {}
        p=memory.get('point_level_m');g=packet.get('gravity_body');read=packet.get('read_timestamp_s')
        anchor=memory.get('ball_capture_timestamp_s');age=memory.get('ball_age_s')
        tilt=math.atan2(math.hypot(*g[:2]),-g[2]) if vec(g,3) else math.inf
        return dict(same_epoch=epoch==self.epoch,not_kicked=not kicked,
            current_capture=finite(self.last_capture) and 0<=t-self.last_capture<=.2+1e-9,
            current_floor_guard=guard_safe is True and finite(guard_s) and 0<=t-guard_s<=.2+1e-9,
            memory_available=memory.get('memory_available') is True,
            original_anchor_recent=finite(anchor) and finite(age) and 0<=age<=6.5 and abs((t-anchor)-age)<=1e-6,
            continuous_odometry=odometry_continuous is True and memory.get('samples_since_capture',0)>=8,
            current_memory_read=finite(memory.get('sensor_read_age_s')) and 0<=memory['sensor_read_age_s']<=.08,
            current_body_IMU=packet.get('orientation_valid') is True and finite(read) and 0<=t-read<=.08 and tilt<=.35,
            arrival_point=inside(p),goal_heading=finite(goal_heading) and abs(goal_heading)<=math.pi/12,
            recent_goal=finite(goal_s) and 0<=t-goal_s<=2.,
            body_command_stopped=vec(body_command,3) and max(abs(x) for x in body_command)<=1e-9)

    def consider(self,*,handoff_state,fine_state,**inputs):
        checks=self.common_checks(**inputs)
        checks.update(observed_in_epoch=self.last_visible_s is not None,
                      missing_metric_ball=self.missing_count>=3,
                      eligible_owner=handoff_state in ('APPROACH','HEAD_RETURN') and fine_state in ('IDLE','SETTLE','REACQUIRE'),
                      single_attempt=not self.attempted)
        allowed=all(checks.values())
        trace=dict(t=inputs['t'], epoch=inputs['epoch'], handoff_state=handoff_state, fine_state=fine_state,
                   allowed=allowed,checks=checks,point_level_m=inputs['memory'].get('point_level_m'),
                   heading_rad=inputs['goal_heading'],anchor_capture_s=inputs['memory'].get('ball_capture_timestamp_s'),
                   original_anchor_age_s=inputs['memory'].get('ball_age_s'),last_metric_visible_s=self.last_visible_s,
                   missing_capture_count=self.missing_count,reason='near_ball_metric_unavailable_after_continuous_public_approach',
                   original_range_uncertainty_m=inputs['memory'].get('original_range_uncertainty_m'),
                   parameters=dict(PARAMETERS),truth_used=False,validated_kick_domain=False,
                   point_estimate_only=True,requested_action='existing_head_return_then_existing_measured_quiet')
        if allowed:
            self.attempted=True;self.latch=dict(trace,entry_s=inputs['t'],epoch=inputs['epoch'])
        return trace

    def exit_gate(self,**inputs):
        checks=self.common_checks(**inputs)
        checks.update(latched=self.latch is not None,
                      original_epoch=self.latch is not None and self.latch['epoch']==inputs['epoch'],
                      original_anchor=self.latch is not None and self.latch['anchor_capture_s']==inputs['memory'].get('ball_capture_timestamp_s'))
        # Losing the arrival domain or current safety proof withdraws the blind
        # authorization. No automatic retry based on the same blind latch.
        entry_s=None if self.latch is None else self.latch['entry_s']
        if not all(checks.values()):
            self.latch=None  # Irreversible for this attempt; attempted remains true.
        return dict(t=inputs['t'], epoch=inputs['epoch'], guard_s=inputs['guard_s'], guard_safe=inputs['guard_safe'],
                    goal_s=inputs['goal_s'], anchor_capture_s=inputs['memory'].get('ball_capture_timestamp_s'),
                    allowed=all(checks.values()),checks=checks,point_level_m=inputs['memory'].get('point_level_m'),
                    heading_rad=inputs['goal_heading'],anchor_age_s=inputs['memory'].get('ball_age_s'),
                    entry_s=entry_s,truth_used=False,
                    measured_quiet_required_elsewhere_s=.5,validated_kick_domain=False)


PROFILE = 'near-blind-v1'
OBSERVATION_FAILURES = frozenset({'ball_search_no_source_domain_target', 'ball_reacquisition_timeout',
    'ball_search_targets_exhausted', 'final_turn_budget_exhausted', 'final_forward_budget_exhausted'})

class BlindArrivalSupervisor(BlindArrival):
    """Instance-owned bridge into the existing public handoff, not a kick policy."""
    def __init__(self):
        super().__init__()
        self.capture_context = {}
        self.control_context = {}
        self.last_decision = None
        self.last_exit = None
        self.entry_record = None
        self.last_safety = None

    def capture(self, *, t, epoch, complete, guard_safe, goal_heading, goal_s):
        # Base reset must not discard the just-established capture context.
        self.observe_capture(t=t, epoch=epoch, ball_metric_valid=complete)
        self.capture_context = dict(capture_s=t, epoch=epoch, complete_metric_ball=bool(complete),
                                    guard_safe=bool(guard_safe), guard_s=t,
                                    # Localization represents an unavailable heading
                                    # with NaN. Keep that unavailable (never fresh),
                                    # while making strict JSON evidence serializable.
                                    goal_heading=goal_heading if finite(goal_heading) else None,
                                    goal_s=goal_s if finite(goal_s) else None)

    def control(self, *, epoch, fine, actual_head, actual_velocity):
        self.control_context = dict(epoch=epoch, fine=fine, actual_head=list(actual_head),
                                    actual_velocity=list(actual_velocity))

    def inputs(self, handoff, t, packet, memory):
        c=self.capture_context; k=self.control_context
        gap=bool((handoff.latest_readback or {}).get('sequence_gap'))
        return dict(t=t, epoch=k.get('epoch'), memory=memory, packet=packet,
            guard_safe=c.get('guard_safe',False), guard_s=c.get('guard_s'),
            goal_heading=c.get('goal_heading'), goal_s=c.get('goal_s'),
            body_command=k.get('actual_velocity',[1.,0.,0.]),
            odometry_continuous=bool(memory.get('memory_available') and not gap),
            kicked=handoff.kick_start is not None)

    def actual_head_neutral(self):
        h=self.control_context.get('actual_head')
        return vec(h,4) and max(abs(x) for x in h)<=1e-9

    def before_policy(self, handoff, t, packet):
        if handoff.state not in ('APPROACH','HEAD_RETURN'):
            return
        memory=handoff.adapter.memory.status(t)
        inputs=self.inputs(handoff,t,packet,memory)
        fine=self.control_context.get('fine')
        if fine is None:
            return
        if self.latch is not None and handoff.state=='HEAD_RETURN':
            checks=self.common_checks(**inputs)
            # Position is decided at the original quiet exit, not while feet
            # finish settling. Sensor/obstacle freshness cannot wait for that.
            safety={k:v for k,v in checks.items() if k not in ('arrival_point','goal_heading','recent_goal')}
            safety['original_anchor']=self.latch['anchor_capture_s']==memory.get('ball_capture_timestamp_s')
            self.last_safety=dict(t=t,checks=safety,allowed=all(safety.values()))
            if not all(safety.values()):
                self.latch=None
                handoff.fail('blind_arrival_public_guard_lost')
                return
        if self.attempted:
            return
        decision=self.consider(handoff_state=handoff.state,fine_state=fine.state,**inputs)
        self.last_decision=decision
        if not decision['allowed']:
            return
        self.entry_record=dict(decision,t=t,epoch=self.epoch)
        self.last_safety=dict(t=t,allowed=True,checks=decision['checks'])
        fine.state='IDLE'
        fine.reason='released_to_blind_arrival_handoff'
        fine.track_velocity=[0.,0.,0.]
        fine.quiet_start=None
        if handoff.state=='APPROACH':
            handoff.enter_from_fine(now_s=t,goal_heading=inputs['goal_heading'],goal_fix_s=inputs['goal_s'])
            # enter_from_fine's legacy <=.5s condition must not erase a true
            # <=2s observation. Keep its real time; never synthesize a refresh.
            handoff.last_goal_heading=float(inputs['goal_heading'])
            handoff.last_goal_s=float(inputs['goal_s'])
            import numpy as np
            handoff.head=np.asarray(self.control_context['actual_head'],float).copy()

    def final_domain(self, handoff, t, memory, packet, ordinary_gate):
        if self.latch is None:
            return ordinary_gate(memory.get('point_level_m'))
        inputs=self.inputs(handoff,t,packet,memory)
        # Use the latest genuine goal update on the handoff, never a retimed one.
        inputs['goal_heading']=handoff.last_goal_heading
        inputs['goal_s']=handoff.last_goal_s
        self.last_exit=self.exit_gate(**inputs)
        return self.last_exit['allowed']

    def owns_observation_wait(self,handoff):
        return bool(self.latch is not None and handoff.state=='HEAD_RETURN'
                    and self.last_safety and self.last_safety['allowed'])

    def filter_observation_failure(self, reason, handoff):
        if self.owns_observation_wait(handoff) and reason in OBSERVATION_FAILURES:
            return None
        return reason

    def status(self):
        control={k:v for k,v in self.control_context.items() if k!='fine'}
        control['fine_state']=getattr(self.control_context.get('fine'),'state',None)
        return dict(profile=PROFILE,parameters=dict(PARAMETERS),epoch=self.epoch,
                    last_capture_s=self.last_capture,capture_context=dict(self.capture_context),control_context=control,
                    latched=self.latch is not None,attempted=self.attempted,
                    missing_capture_count=self.missing_count,last_visible_s=self.last_visible_s,
                    entry=self.entry_record,decision=self.last_decision,exit=self.last_exit,
                    current_safety=self.last_safety,truth_used=False,validated_kick_domain=False)


def policy_before(handoff,t,packet):
    supervisor=getattr(handoff,'blind_arrival',None)
    if supervisor is not None:supervisor.before_policy(handoff,t,packet)

def quiet_actual_head(handoff):
    supervisor=getattr(handoff,'blind_arrival',None)
    return supervisor is None or supervisor.latch is None or supervisor.actual_head_neutral()

def final_domain(handoff,t,memory,packet,ordinary_gate):
    supervisor=getattr(handoff,'blind_arrival',None)
    return ordinary_gate(memory.get('point_level_m')) if supervisor is None else supervisor.final_domain(handoff,t,memory,packet,ordinary_gate)
