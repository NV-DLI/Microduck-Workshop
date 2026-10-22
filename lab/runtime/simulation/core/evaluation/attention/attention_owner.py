"""Public bounded goal/ball attention ownership; never body or kick authority."""
import math


class AttentionOwner:
    def __init__(self):
        self.state = 'NONE'
        self.started = None
        self.phase_started = None
        self.goal_completed = None
        self.failed = None
        self.last_t = None
        self.transition = 'not_started'

    def update(self, *, t, near, complete_capture_s, goal_fix_s, request_goal):
        if not math.isfinite(t) or t < 0 or self.last_t is not None and t < self.last_t:
            raise ValueError('Finite monotone attention clock required')
        if complete_capture_s is not None and (not math.isfinite(complete_capture_s) or not 0 <= t-complete_capture_s <= .2):
            raise ValueError('Current complete ball capture required')
        if not (math.isfinite(goal_fix_s) or goal_fix_s == -math.inf) or goal_fix_s > t:
            raise ValueError('Causal goal fix required')
        self.last_t = t
        self.transition = 'owner_held'
        if self.failed:
            return self.state
        if self.state == 'NONE':
            if near and (request_goal or complete_capture_s is None):
                self.started = self.phase_started = t
                self.goal_completed = None
                self.state = 'GOAL_REFRESH' if request_goal else 'BALL_REACQUIRE'
                self.transition = 'goal_refresh_committed' if request_goal else 'complete_ball_missing'
        elif t-self.started >= 24.-1e-9:
            self.failed = 'attention_reacquisition_total_timeout'
            self.state = 'FAILED'
            self.transition = self.failed
        elif self.state == 'GOAL_REFRESH':
            if goal_fix_s > self.phase_started and 0 <= t-goal_fix_s <= .2:
                self.goal_completed = goal_fix_s
                self.phase_started = t
                self.state = 'BALL_REACQUIRE'
                self.transition = 'new_accepted_goal_fix'
            elif t-self.phase_started >= 6.-1e-9:
                self.failed = 'goal_refresh_committed_timeout'
                self.state = 'FAILED'
                self.transition = self.failed
        elif self.state == 'BALL_REACQUIRE':
            after = self.goal_completed if self.goal_completed is not None else self.phase_started
            if complete_capture_s is not None and complete_capture_s > after:
                if request_goal and self.goal_completed is None:
                    self.state = 'GOAL_REFRESH'
                    self.phase_started = t
                    self.transition = 'new_ball_then_expired_goal_refresh'
                else:
                    self.state = 'NONE'
                    self.transition = 'new_complete_ball_after_attention_phase'
        return self.state

    def diagnostic(self, t):
        return {'owner': self.state, 'episode_started_s': self.started,
                'phase_started_s': self.phase_started, 'new_goal_fix_s': self.goal_completed,
                'total_elapsed_s': None if self.started is None else t-self.started,
                'goal_phase_deadline_s': 6., 'total_deadline_s': 24.,
                'transition': self.transition, 'failed': self.failed,
                'body_authorized': False, 'kick_authorized': False, 'truth_used': False}
