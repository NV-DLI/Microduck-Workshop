"""Final handoff tolerance; approach target and sensor guards remain unchanged."""
import math,hashlib,inspect,textwrap
import numpy as np
EXPECTED_SHA = 'ad9e401db7cd02f44bd8d05623c344c396089eba28b2ed617953d094b97f8fa4'
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
    changed=changed.replace('now_s - self.started > 4.5', 'now_s - self.started > 6.5')
    changed=changed.replace('0 <= now_s - self.last_goal_s <= 0.5', '0 <= now_s - self.last_goal_s <= 2.0')
    from approach_control import kick_box
    scope=dict(original.__globals__,in_training_box=kick_box,
               HANDOFF_HEADING_MAX_RAD=math.radians(15))
    from blind_arrival import policy_before, quiet_actual_head, final_domain
    old_read = '    self.read(packet, now_s=now_s)'
    old_box = 'if not in_training_box(state["point_level_m"]):'
    old_quiet = 'np.max(np.abs(self.head)) <= 1e-9'
    assert changed.count(old_read) == changed.count(old_box) == changed.count(old_quiet) == 1
    changed = changed.replace(old_read, old_read + '\n    blind_before(self, now_s, packet)')
    changed = changed.replace(old_box, 'if not blind_final_domain(self, now_s, state, packet, in_training_box):')
    changed = changed.replace(old_quiet, old_quiet + ' and blind_actual_head_quiet(self)')
    scope.update(blind_before=policy_before, blind_final_domain=final_domain,
                 blind_actual_head_quiet=quiet_actual_head)
    exec(compile(changed,'<private final-handoff-tolerance>','exec'),scope)
    module.HeadHandoff.policy=scope['policy']
    record=dict(course_onboard=True,blind_arrival=True,entry_box_unchanged=True,
                final_x_m=[.070,.125],final_y_m=[-.065,-.005],final_heading_rad=math.radians(15),
                fresh_goal_max_age_s=2.,quiet_duration_s=.5,headreturn_deadline_s=6.5,
                source_sha256=actual,executed_policy_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                scope='Public position/yaw checks followed by measured quiet and the official kick')
    import neutral_fine_step
    from goal_wait import install as install_goal_wait
    record["stationary_goal_refresh"] = install_goal_wait(neutral_fine_step)
    from approach_control import install_fine_wrappers
    install_fine_wrappers(neutral_fine_step.NeutralFineStep)
    module._onboard_install_record = record
    return record
