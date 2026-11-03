EXPECTED='15ac24528cf99519cf74786d771b3a6d2ac5f8f50cd30d59c4809d504a2ad0c3'
"""Bounded zero-motion wait for a new goal after neutral head settling."""
import inspect,textwrap,hashlib
MAX_WAIT_S=1.0
REASON='waiting_for_fresh_goal_while_stationary'
def install(module):
    if hasattr(module, "_onboard_install_record"):
        return module._onboard_install_record
    assert hashlib.sha256(open(module.__file__,'rb').read()).hexdigest()==EXPECTED
    cls=module.NeutralFineStep;source=textwrap.dedent(inspect.getsource(cls.policy))
    old='            self.fail("neutral_step_fresh_goal_required")'
    new='            key = (self.started, self.anchor_s)\n            if getattr(self, "goal_wait_key", None) != key:\n                self.goal_wait_key = key\n                self.goal_wait_started = t\n            if t - self.goal_wait_started >= GOAL_MAX_WAIT_S - 1e-9:\n                self.fail("neutral_step_fresh_goal_required")\n            else:\n                self.reason = GOAL_WAIT_REASON'
    old="    "+old;new=textwrap.indent(new,"    ")
    assert source.count(old)==1;changed=source.replace(old,new)
    anchor='    if self.state == "RETURN":\n'
    clear='    if self.state == "RETURN" and self.reason == GOAL_WAIT_REASON and self.goal_s is not None and 0 <= t - self.goal_s <= .5:\n        self.reason = None\n        self.goal_wait_key = None\n        self.goal_wait_started = None\n'
    expiry='    if self.state == "RETURN" and self.reason == GOAL_WAIT_REASON and getattr(self, "goal_wait_started", None) is not None and t - self.goal_wait_started >= GOAL_MAX_WAIT_S - 1e-9:\n        self.fail("neutral_step_fresh_goal_required")\n        return self.status(t, memory)\n'
    assert changed.count(anchor)==1;changed=changed.replace(anchor,clear+expiry+anchor)
    scope=dict(cls.policy.__globals__,GOAL_MAX_WAIT_S=MAX_WAIT_S,GOAL_WAIT_REASON=REASON)
    exec(compile(changed,'<private bounded goal refresh wait>','exec'),scope);cls.policy=scope['policy']
    original_status=cls.status
    def status(self,t,memory):
        out=original_status(self,t,memory)
        out['goal_refresh_wait_started_s']=getattr(self,'goal_wait_started',None)
        out['goal_refresh_wait_max_s']=MAX_WAIT_S
        return out
    cls.status=status
    module._onboard_install_record = dict(max_wait_s=MAX_WAIT_S,state='RETURN',body_velocity=[0,0,0],fresh_goal_age_max_s=.5,original_memory_and_attention_deadlines_unchanged=True,source_sha256=EXPECTED,executed_policy_sha256=hashlib.sha256(changed.encode()).hexdigest())

    return module._onboard_install_record
