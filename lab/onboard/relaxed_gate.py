"""Final handoff tolerance; approach target and sensor guards remain unchanged."""
import math,hashlib,inspect,textwrap
import numpy as np
EXPECTED_SHA = 'ad9e401db7cd02f44bd8d05623c344c396089eba28b2ed617953d094b97f8fa4'
MARGIN_M=.005
HEADING_MAX_RAD=math.radians(10.0)
def exit_box(point):
    p=np.asarray(point,float)
    return bool(p.shape==(3,) and np.isfinite(p).all() and .070<=p[0]<=.110 and -.055<=p[1]<=-.015)
def install(module):
    if hasattr(module, "_onboard_install_record"):
        return module._onboard_install_record
    actual=hashlib.sha256(open(module.__file__,'rb').read()).hexdigest()
    if actual!=EXPECTED_SHA:raise RuntimeError('Head handoff source changed')
    original=module.HeadHandoff.policy
    source=textwrap.dedent(inspect.getsource(original))
    old='elif abs(self.last_goal_heading) > 0.08:'
    assert source.count(old)==1
    changed=source.replace(old,'elif abs(self.last_goal_heading) > HANDOFF_HEADING_MAX_RAD:')
    scope=dict(original.__globals__,in_training_box=exit_box,HANDOFF_HEADING_MAX_RAD=HEADING_MAX_RAD)
    exec(compile(changed,'<private final-handoff-tolerance>','exec'),scope)
    module.HeadHandoff.policy=scope['policy']
    record=dict(course_onboard=True,entry_box_unchanged=True,approach_unchanged=True,original_x_m=[.075,.105],original_y_m=[-.05,-.02],final_x_m=[.070,.110],final_y_m=[-.055,-.015],margin_m=MARGIN_M,original_heading_rad=.08,final_heading_rad=HEADING_MAX_RAD,fresh_goal_max_age_s=.5,quiet_duration_s=.5,headreturn_deadline_s=4.5,source_sha256=actual,executed_policy_sha256=hashlib.sha256(changed.encode()).hexdigest(),scope='Only HEAD_RETURN final position/yaw checks; no truth input, model change or forced kick; not a validated policy domain')
    import neutral_fine_step
    from goal_wait import install as install_goal_wait
    record["stationary_goal_refresh"] = install_goal_wait(neutral_fine_step)
    module._onboard_install_record = record
    return record
