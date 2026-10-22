import copy,unittest
import numpy as np
from head_feedback import ObservationHeadFeedback,head_context_allowed
from head_feedback_core import HOME


class HeadFeedbackTests(unittest.TestCase):
    def call(self,h,i,**changes):
        t=i*.02;q=np.zeros(14);q[5:9]=HOME+[.2,0,0,0]
        args=dict(now_s=t,nominal=np.zeros(4),packet=dict(seq=i,read_timestamp_s=t,timestamp_s=t,
            encoder_read_timestamp_s=t,encoder_q_rad=q.tolist(),orientation_valid=True),
            velocity=[0,0,0],state='REACQUIRE_BALL',handoff_state='APPROACH',
            navigation_enabled=True,head_motion_allowed=True)
        args.update(changes);return h.update(**args)
    def active(self):
        h=ObservationHeadFeedback()
        for i in range(350):self.call(h,i)
        self.assertLess(h.core.bias[0,0],-.1)
        return h
    def test_five_seconds_then_new_read_only(self):
        h=ObservationHeadFeedback()
        for i in range(250):self.call(h,i)
        self.assertFalse(h.core.bias.any())
        self.call(h,250);self.assertTrue(h.core.bias.any())
        old=h.core.bias.copy();p=dict(seq=250,read_timestamp_s=5.,timestamp_s=5.,encoder_read_timestamp_s=5.,encoder_q_rad=np.r_[np.zeros(5),HOME+[.2,0,0,0],np.zeros(5)].tolist(),orientation_valid=True)
        self.call(h,251,packet=p);np.testing.assert_array_equal(old,h.core.bias)
    def test_motion_waits_for_actual_withdrawal_no_timer_restart(self):
        h=self.active();start=None;holds=[]
        for i in range(350,390):
            sent,d=self.call(h,i,velocity=[.3,0,0],state='NAV2_APPROACH')
            self.assertFalse(d['integration_enabled'])
            if d['body_zero_for_withdrawal']:
                start=d['withdrawal_started_s']if start is None else start
                self.assertEqual(start,d['withdrawal_started_s']);holds.append(i)
            else:
                np.testing.assert_allclose(sent,[0,0,0,0],atol=1e-9);break
        self.assertTrue(holds);self.assertIsNone(h.withdraw_started)
    def test_hazard_freezes_head_and_deadline_still_expires(self):
        h=self.active();prior=h.core.sent.copy()
        for i in range(350,451):
            sent,d=self.call(h,i,velocity=[.3,0,0],state='NAV2_APPROACH',head_motion_allowed=False)
            np.testing.assert_array_equal(sent,prior[0])
        self.assertEqual(h.failure,'head_feedback_withdrawal_timeout')
    def test_invalid_packet_freezes_and_no_interval_bridging(self):
        h=self.active();b=h.core.bias.copy();s=h.core.sent.copy()
        self.call(h,350,packet=None);np.testing.assert_array_equal(s,h.core.sent);np.testing.assert_array_equal(b,h.core.bias)
        _,d=self.call(h,351);self.assertFalse(d['integrated_new_read'])
    def test_head_return_never_integrates_and_kick_requires_real_zero(self):
        h=self.active();_,d=self.call(h,350,handoff_state='HEAD_RETURN',nominal=h.core.sent[0]+.006)
        self.assertFalse(d['integration_enabled']);self.assertFalse(h.core.bias.any())
        _,d=self.call(h,351,handoff_state='KICK');self.assertEqual(d['failure'],'head_feedback_actual_command_not_neutral_at_kick')
    def test_disallowed_phase_never_integrates(self):
        h=ObservationHeadFeedback()
        for i in range(350):self.call(h,i,state='CONTROL_ERROR')
        self.assertFalse(h.core.bias.any())
    def test_bad_clock_does_not_mutate(self):
        h=self.active();before=copy.deepcopy(h.__dict__)
        with self.assertRaises(ValueError):self.call(h,350,now_s=float('nan'))
        self.assertEqual(h.observe_started,before['observe_started']);np.testing.assert_array_equal(h.core.bias,before['core'].bias)
    def test_stale_encoder_or_fractional_sequence_rejected(self):
        h=ObservationHeadFeedback()
        for key,value in [('encoder_read_timestamp_s',-.1),('seq',1.5)]:
            q=np.r_[np.zeros(5),HOME,np.zeros(5)].tolist();p=dict(seq=1,read_timestamp_s=.02,timestamp_s=.02,encoder_read_timestamp_s=.02,encoder_q_rad=q,orientation_valid=True);p[key]=value
            _,d=self.call(h,1,packet=p);self.assertFalse(d['packet_valid']);h=ObservationHeadFeedback()
    def test_nominal_changes_cannot_extend_withdrawal_deadline(self):
        h=self.active();start=None
        for i in range(350,451):
            nominal=[-.9 if i%2 else .9,0,0,0]
            _,d=self.call(h,i,nominal=nominal,velocity=[.3,0,0],state='NAV2_APPROACH')
            if start is None:start=d['withdrawal_started_s']
            self.assertEqual(start,d['withdrawal_started_s'])
        self.assertEqual(h.failure,'head_feedback_withdrawal_timeout')
    def test_context_preserves_floor_tooclose_and_original_hold(self):
        kw=dict(now_s=1.,capture_s=1.,control={'state':'REACQUIRE_BALL','reason':'bounded_public_attention_owner',
            'public_attention_owner':{'owner':'BALL_REACQUIRE'}},geometry={'timestamp_s':1.,'floor_observation':{'accepted':True}},
            too_close=False,ball_context=True,goal_context=True)
        self.assertTrue(head_context_allowed(**kw))
        for change in ({'too_close':True},{'ball_context':False},{'capture_s':.7},
                       {'geometry':{'timestamp_s':1.,'floor_observation':{'accepted':False}}},
                       {'control':{'state':'REFRESH_GOAL','reason':'actual_ToF_near_obstacle_or_too_close'}}):
            self.assertFalse(head_context_allowed(**{**kw,**change}))


if __name__=='__main__':unittest.main()
