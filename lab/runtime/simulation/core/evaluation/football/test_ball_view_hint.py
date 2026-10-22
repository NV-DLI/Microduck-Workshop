import copy
import json
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from ball_view_hint import BallViewHint
from ball_reacquisition import BallReacquisitionNavigator
from controller import ControllerConfig
from visual_localization import rotate


class HintTests(unittest.TestCase):
    def test_only_complete_current_capture_stores_original_gaze(self):
        h=BallViewHint()
        self.assertFalse(h.observe(now_s=1.,capture_s=1.,gaze=[-.4,1.],complete=False))
        self.assertTrue(h.observe(now_s=2.,capture_s=2.,gaze=[-1.5,1.1],complete=True))
        self.assertEqual(h.capture_s,2.)
        self.assertTrue(np.array_equal(h.gaze,[-1.5,1.1]))

    def test_partial_failed_old_future_bad_inputs_cannot_cover_hint(self):
        for args in (
            dict(complete=False),dict(capture_s=3.1),dict(capture_s=1.),
            dict(capture_s=float('nan')),dict(now_s=-1.),dict(gaze=[0.,float('nan')]),
            dict(gaze=[2.,1.]),dict(gaze=[0.])):
            h=BallViewHint();h.observe(now_s=2.,capture_s=2.,gaze=[-.4,1.],complete=True)
            before=copy.deepcopy(h.__dict__)
            call=dict(now_s=3.,capture_s=3.,gaze=[-1.,1.],complete=True);call.update(args)
            self.assertFalse(h.observe(**call))
            self.assertEqual(h.capture_s,before['capture_s'])
            self.assertTrue(np.array_equal(h.gaze,before['gaze']))
            self.assertEqual(h.reason,before['reason'])

    def test_return_keeps_capture_age_and_has_no_metric_or_motion(self):
        h=BallViewHint();h.observe(now_s=2.,capture_s=2.,gaze=[-1.5,1.1],complete=True)
        self.assertTrue(h.begin_return(10.))
        self.assertTrue(np.array_equal(h.target(now_s=10.1,current=[0.,0.]),[-1.5,1.1]))
        d=h.diagnostic(10.1)
        self.assertEqual(d['capture_timestamp_s'],2.)
        self.assertAlmostEqual(d['original_capture_age_s'],8.1)
        self.assertIsNone(d['metric_position'])
        self.assertFalse(d['body_authorized'] or d['kick_authorized'] or d['refreshes_observation_time'])

    def test_arrival_only_command_not_actual_tracking_restores_fallback(self):
        h=BallViewHint();h.observe(now_s=2.,capture_s=2.,gaze=[-1.5,1.1],complete=True)
        h.begin_return(10.)
        self.assertIsNone(h.target(now_s=11.,current=[-1.49,1.1]))
        self.assertFalse(h.returning)
        self.assertEqual(h.capture_s,2.)

    def test_reset_clears_hint(self):
        n=BallReacquisitionNavigator(None,config=ControllerConfig(ground_mode='tof-virtual-floor'))
        n.ball_view_hint.observe(now_s=2.,capture_s=2.,gaze=[-1.,1.],complete=True)
        n.ball_view_hint.begin_return(10.);n.reset()
        self.assertIsNone(n.ball_view_hint.capture_s)
        self.assertFalse(n.ball_view_hint.returning)

    def test_hint_does_not_bypass_stale_head_permission_or_original_timeout(self):
        import test_ball_reacquisition as fixture
        fixture.Recovery.setUpClass();f=fixture.Recovery()
        n,t=f.nav(60.);f.actual(n,t)
        n.ball_view_hint.observe(now_s=t,capture_s=t,gaze=[-1.5,1.1],complete=True)
        n.ball_view_hint.begin_return(t)
        before=n.gaze.current.copy();original_start=n.ball_search.started;old_ball=n.ball_time
        n.gaze.update([0.,0.],t+.1)
        self.assertTrue(np.array_equal(n.gaze.current,before))
        n.gaze.update([0.,0.],t+24.)
        self.assertEqual(n.ball_search.failed,'ball_reacquisition_timeout')
        self.assertEqual(n.ball_search.started,original_start)
        self.assertEqual(n.ball_time,old_ball)
        self.assertTrue(np.array_equal(n.gaze.current,before))

    def test_actual_complete_capture_stores_pre_goal_output_gaze(self):
        root=Path('@COURSE@/configuration/core/football-v60-full-task180s')
        read=lambda name:[json.loads(x)for x in (root/name).read_text().splitlines()]
        controls=read('control.jsonl');sensors={x['timestamp_s']:x for x in read('sensor-inputs.jsonl')}
        detections={x['timestamp_s']:x for x in read('detections.jsonl')}
        t=48.27;original=next(x for x in controls if x['timestamp_s']==t)
        before=next(x for x in controls if x['timestamp_s']==48.2)
        n=BallReacquisitionNavigator(None,config=ControllerConfig(ground_mode='tof-virtual-floor'))
        n.pose=np.array(original['estimated_pose_map_xyyaw']);n.final_entry=True
        n.first_time=2.34;n.last_time=48.2;n.goal_time=31.8
        n.ball_time=35.4;n.ball=n.pose[:2]+rotate(n.pose[2],original['public_metric_ball']['point_level_m'][:2])
        n.attention_owner.state='BALL_REACQUIRE';n.attention_owner.started=35.47
        n.attention_owner.phase_started=35.47;n.attention_owner.last_t=48.2
        n.gaze.current=np.array(before['gaze_body_yaw_pitch_rad']);n.gaze.timestamp=48.2
        n.ball_search.begin(35.47,[-.17133256514572132,1.3904477599867453])
        sr=sensors[t]
        out=n.step(np.asarray(Image.open(root/sr['rgb'])),sr['proprio'],sr['tof_frame'],
                   now_s=t,detections=detections[t]['detections'])
        self.assertEqual(out['public_ball_view_hint']['capture_timestamp_s'],48.27)
        self.assertTrue(np.array_equal(n.ball_view_hint.gaze,before['gaze_body_yaw_pitch_rad']))
        self.assertFalse(np.array_equal(n.ball_view_hint.gaze,n.gaze.current))
        self.assertEqual(n.attention_owner.started,35.47)
        self.assertEqual(out['velocity_body_xyw'],[0.,0.,0.]);self.assertFalse(out['kick_request'])
        # A later actual goal fix can aim the head using the retained hint.
        # Neither a new metric nor a new attention clock follows from it.
        n.goal_time=51.87;n.last_geometry['timestamp_s']=51.87
        n.last_geometry['ball']['accepted']=False
        n.current_sensor_pair=(dict(sr['proprio'],timestamp_s=51.87),sr['tof_frame'])
        n._output(51.87,reason='fresh_ball_required')
        self.assertEqual(n.attention_owner.state,'BALL_REACQUIRE')
        self.assertEqual(n.attention_owner.started,35.47)
        self.assertEqual(n.ball_view_hint.capture_s,48.27)
        self.assertTrue(n.ball_view_hint.returning)
        self.assertEqual(n.ball_time,48.27)


if __name__=='__main__':
    unittest.main()
