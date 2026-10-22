"""Test fine transition latch: Microduck simulation and control."""
import copy,json,math
from pathlib import Path
import unittest
import numpy as np
from fine_transition_latch import FineTransitionLatch,original_trigger
from neutral_fine_step import NeutralFineStep

RUN=Path('@COURSE@/configuration/core/football-v63-full-task180s')
GOAL_MAP=np.array([.93,0.])


def snapshot(i,c,prev):
    """Executor locals at the fine-trigger line, derived only from the logged control row."""
    t=c['timestamp_s'];fine=c.get('public_neutral_fine_step')or{};fd=c.get('final_approach')or{}
    prev_state=(prev.get('public_neutral_fine_step')or{}).get('state')if prev else None
    pose=c.get('estimated_pose_map_xyyaw');point=fd.get('point_level_m');gh=float('nan')
    if pose is not None and point is not None:
        cy,sy=math.cos(-pose[2]),math.sin(-pose[2]);v=GOAL_MAP-np.asarray(pose[:2]);rel=(cy*v[0]-sy*v[1],sy*v[0]+cy*v[1])
        gh=math.atan2(rel[1]-point[1],rel[0]-point[0])
    fma=c.get('final_motion_authorization')or{};nbp=c.get('near_ball_phase')or{}
    age=fma.get('goal_memory_age_s',nbp.get('goal_fix_age_s'))   # controller.goal_time = t-age; absent -> NaN (fail closed)
    return dict(capture_index=i,t=t,handoff_state=(c.get('public_handoff')or{}).get('state'),fine_active=prev_state not in(None,'IDLE'),
        fine_state=prev_state or 'IDLE',final_diag=fd,velocity_body_xyw=c.get('velocity_body_xyw'),complete=c.get('public_metric_ball')is not None,
        guard_safe=fine.get('capture_safe'),memory=fine.get('memory'),goal_heading=gh,goal_fix_s=t-age if age is not None else float('nan'),
        actual_head=(c.get('head_feedback')or{}).get('actual_native_command'))


def _rows():
    out={};prev=None
    with open(RUN/'control.jsonl')as f:
        for i,line in enumerate(f):
            c=json.loads(line)
            if 484<=i<=488:out[i]=snapshot(i,c,prev)
            prev=c
            if i>488:break
    return out


ROWS=_rows();A=ROWS[486];B=ROWS[487]   # actual 32.47 transition stop, actual 32.535 next capture


def capture_entry(s):
    return NeutralFineStep().capture(t=s['t'],guard_safe=s['guard_safe'],complete=s['complete'],trigger=True,actual_head=s['actual_head'],
        memory=s['memory'],goal_heading=s['goal_heading'],goal_fix_s=s['goal_fix_s'])


def nxt(base,**over):
    s=copy.deepcopy(base);s.update(over);return s


def mem(base,**over):
    m=copy.deepcopy(base['memory']);m.update(over);return m


class Latch(unittest.TestCase):
    def armed(self):
        L=FineTransitionLatch();r=L.evaluate(**A);self.assertFalse(r['trigger']);self.assertIsNotNone(r['armed']);return L

    def test_actual_sequence_only_next_capture_enters_return_with_zero_body(self):
        L=FineTransitionLatch()
        for i in(484,485):self.assertFalse(L.evaluate(**ROWS[i])['trigger'])
        r=L.evaluate(**A)
        self.assertFalse(r['trigger']);self.assertEqual(r['armed'],{'capture_index':486,'timestamp_s':32.47})
        self.assertEqual(r['reason'],'armed_for_next_capture_only:entry_failed_now:original_entry_domain')
        r=L.evaluate(**B)
        self.assertTrue(r['trigger']);self.assertEqual(r['kind'],'close_range_transition_latched_next_capture')
        st=capture_entry(B)
        self.assertEqual(st['state'],'RETURN');self.assertEqual(st['velocity'],[0.,0.,0.]);self.assertTrue(st['override'])
        self.assertEqual(st['anchor_capture_s'],B['t']);self.assertFalse(st['kick_authorized'])
        self.assertFalse(L.evaluate(**ROWS[488])['trigger'])          # consumed

    def test_arming_grants_nothing(self):
        r=FineTransitionLatch().evaluate(**A)
        self.assertFalse(r['trigger']);self.assertFalse(r['body_authorized']);self.assertFalse(r['kick_authorized'])
        self.assertEqual(capture_entry(A)['state'],'FAILED')    # why entry must never be forced at 32.47

    def test_one_frame_ttl(self):
        L=self.armed();self.assertFalse(L.evaluate(**nxt(B,guard_safe=False))['trigger'])
        later=nxt(B,capture_index=488,t=B['t']+.065,memory=mem(B,ball_capture_timestamp_s=B['t']+.065))
        r=L.evaluate(**later);self.assertFalse(r['trigger']);self.assertEqual(r['reason'],'no_pending_transition')

    def test_gap_and_index(self):
        for over,check in(({'t':A['t']+.0801,'memory':mem(B,ball_capture_timestamp_s=A['t']+.0801)},'gap_gt0_le_0p08s'),
                          ({'capture_index':488},'very_next_capture_index')):
            L=self.armed();r=L.evaluate(**nxt(B,**over));self.assertFalse(r['trigger']);self.assertEqual(r['reason'],'pending_expired:'+check)

    def test_duplicate_or_backward_capture_cancels(self):
        for over in({'t':A['t'],'memory':mem(B,ball_capture_timestamp_s=A['t'])},{'capture_index':486},{'t':A['t']-.01}):
            L=self.armed();r=L.evaluate(**nxt(B,**over))
            self.assertFalse(r['trigger']);self.assertEqual(r['reason'],'rejected:capture_index_and_time_strictly_increasing')
            self.assertFalse(L.evaluate(**B)['trigger'])           # pending already consumed

    def test_cancel_and_reset(self):
        L=self.armed();L.cancel(capture_index=487,t=B['t'],reason='control_exception');self.assertFalse(L.evaluate(**B)['trigger'])
        L=self.armed();L.reset();self.assertFalse(L.evaluate(**B)['trigger'])

    def test_reanchoring_and_pending_point_never_used(self):
        for m,check in((mem(B,ball_capture_timestamp_s=A['t']),'memory_anchored_this_capture'),(mem(B,memory_available=False),'memory_anchored_this_capture'),
                       (None,'memory_anchored_this_capture'),(mem(B,point_level_m=[.129,-.04,-.08]),'original_entry_domain'),
                       (mem(B,point_level_m=[.2,.0801,-.08]),'original_entry_domain'),(mem(B,point_level_m=[.2,-.1001,-.08]),'original_entry_domain'),
                       (mem(B,point_level_m=[math.nan,0.,0.]),'memory_point_finite')):
            L=self.armed();r=L.evaluate(**nxt(B,memory=m));self.assertFalse(r['trigger']);self.assertEqual(r['reason'],'pending_expired:'+check)

    def test_guards_on_next_capture(self):
        turn=dict(B['final_diag'],phase='TURN',reason='final_public_heading')
        for over,check in(({'guard_safe':False},'body_guard_safe'),({'complete':False},'complete_same_capture_ball'),
                          ({'final_diag':turn},'ordinary_observe'),({'final_diag':dict(B['final_diag'],reason='final_upstream_guard')},'ordinary_observe'),
                          ({'final_diag':{k:v for k,v in B['final_diag'].items()if k!='measured_quiet'}},'ordinary_observe'),({'final_diag':{}},'ordinary_observe'),
                          ({'velocity_body_xyw':[0.,0.,1e-12]},'exact_zero_body_command'),({'velocity_body_xyw':[.2,0.,0.]},'exact_zero_body_command'),
                          ({'handoff_state':'HEAD_RETURN'},'handoff_approach'),({'fine_active':True},'fine_idle_not_active'),({'fine_state':'FAILED'},'fine_idle_not_active'),
                          ({'goal_fix_s':B['t']-8.001},'goal_fix_age_0_to_8s'),({'memory':mem(B,point_level_m=[.20,.02,-.08])},'waypoint_bearing_le_0p15'),
                          ({'goal_heading':math.nan},'goal_heading_finite'),({'actual_head':[0.,math.inf,0.,0.]},'native_head_finite_in_bounds'),
                          ({'actual_head':[0.,0.,0.,.32]},'native_head_finite_in_bounds')):
            L=self.armed();r=L.evaluate(**nxt(B,**over));self.assertFalse(r['trigger'],over);self.assertEqual(r['reason'],'pending_expired:'+check,over)

    def test_arbitrary_observe_never_triggers(self):
        self.assertEqual(FineTransitionLatch().evaluate(**B)['reason'],'no_pending_transition')
        L=FineTransitionLatch()
        for i in(484,485,487,488):self.assertFalse(L.evaluate(**ROWS[i])['trigger'])

    def test_transition_not_armed_without_clean_state(self):
        for over,check in(({'guard_safe':False},'body_guard_safe'),({'complete':False},'complete_same_capture_ball'),
                          ({'velocity_body_xyw':[.3,0.,0.]},'exact_zero_body_command'),({'handoff_state':'KICK'},'handoff_approach'),
                          ({'fine_active':True},'fine_idle_not_active'),({'memory':mem(A,ball_capture_timestamp_s=A['t']-.065)},'memory_anchored_this_capture'),
                          ({'final_diag':dict(A['final_diag'],forward_command=.2)},'stop_came_from_0p3_pulse_at_0p22'),
                          ({'goal_fix_s':math.nan},'finite_public_inputs'),({'actual_head':[0.,0.,math.nan,0.]},'finite_public_inputs')):
            L=FineTransitionLatch();r=L.evaluate(**nxt(A,**over))
            self.assertFalse(r['trigger']);self.assertIsNone(r['armed']);self.assertEqual(r['reason'],'transition_not_armed:'+check,over)
            self.assertFalse(L.evaluate(**B)['trigger'])

    def test_immediate_transition_when_entry_valid_now(self):
        s=nxt(A,memory=mem(B,ball_capture_timestamp_s=A['t']))
        r=FineTransitionLatch().evaluate(**s)
        self.assertTrue(r['trigger']);self.assertEqual(r['kind'],'close_range_transition_immediate');self.assertIsNone(r['armed'])
        self.assertEqual(capture_entry(s)['state'],'RETURN')

    def test_new_transition_while_pending_rearms_once(self):
        L=self.armed()
        s=nxt(A,capture_index=487,t=B['t'],memory=mem(A,ball_capture_timestamp_s=B['t']))
        r=L.evaluate(**s);self.assertFalse(r['trigger']);self.assertEqual(r['armed']['capture_index'],487)
        self.assertEqual(r['consumed_pending']['capture_index'],486)

    def test_original_0p2_path_bitwise(self):
        for hs in('APPROACH','KICK',None):
            for act in(False,True):
                for reason in('final_public_forward','final_close_range_transition_stop',None):
                    for fc in(.2,.3,None,.20000001):
                        fd={'reason':reason,'forward_command':fc}
                        expect=bool(hs=='APPROACH' and not act and fd.get('reason')=='final_public_forward' and fd.get('forward_command')==.2)
                        self.assertEqual(original_trigger(handoff_state=hs,fine_active=act,final_diag=fd),expect)
                        r=FineTransitionLatch().evaluate(**nxt(B,handoff_state=hs,fine_active=act,final_diag=fd))
                        self.assertEqual(r['original_trigger'],expect)
                        if expect:self.assertEqual((r['trigger'],r['kind']),(True,'original_0p2_forward_pulse'))
        r=FineTransitionLatch().evaluate(**nxt(B,final_diag={'reason':'final_public_forward','forward_command':.2},guard_safe=False,memory=None))
        self.assertTrue(r['trigger'])          # original never pre-checked; unchanged

    def test_entry_equivalence_with_original_capture(self):
        rng=np.random.default_rng(64);hits=0
        for k in range(2000):
            s=copy.deepcopy(B);p=list(s['memory']['point_level_m']);p[0]+=rng.uniform(-.1,.03);p[1]+=rng.uniform(-.08,.12)
            s['memory']['point_level_m']=p;s['goal_heading']+=rng.uniform(-.4,.4);s['goal_fix_s']=s['t']-rng.uniform(-.5,9.)
            L=self.armed();r=L.evaluate(**s)
            self.assertEqual(r['trigger'],capture_entry(s)['state']=='RETURN');hits+=r['trigger']
        self.assertGreater(hits,50)

    def test_never_authorizes(self):
        L=self.armed();r=L.evaluate(**B)
        self.assertTrue(r['trigger']);self.assertFalse(r['body_authorized']);self.assertFalse(r['kick_authorized']);self.assertFalse(r['feedback_bias_used'])


if __name__=='__main__':unittest.main()
