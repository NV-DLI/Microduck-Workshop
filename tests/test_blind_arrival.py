"""Near-blind handoff needs recent observations and live public safety evidence."""
from pathlib import Path
import copy,json,math,sys,unittest
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'lab/onboard'))
from blind_arrival import BlindArrival,BlindArrivalSupervisor

class BlindArrivalTest(unittest.TestCase):
    def ready(self):
        gate=BlindArrival()
        for t,seen in [(1.,True),(1.07,False),(1.14,False),(1.21,False)]:
            gate.observe_capture(t=t,epoch=0,ball_metric_valid=seen)
        inputs=dict(t=1.22,epoch=0,memory=dict(memory_available=True,point_level_m=[.13,-.04,-.07],
                ball_capture_timestamp_s=1.,ball_age_s=.22,samples_since_capture=11,sensor_read_age_s=.02),
            packet=dict(gravity_body=[0,0,-1],read_timestamp_s=1.2,orientation_valid=True),
            guard_safe=True,guard_s=1.21,goal_heading=.1,goal_s=1.1,body_command=[0,0,0],
            odometry_continuous=True,kicked=False)
        return gate,inputs

    def enter(self,gate,inputs):return gate.consider(handoff_state='APPROACH',fine_state='SETTLE',**inputs)

    def test_recent_near_occlusion_latches_once(self):
        gate,inputs=self.ready();self.assertTrue(self.enter(gate,inputs)['allowed'])
        self.assertTrue(gate.exit_gate(**inputs)['allowed'])
        self.assertFalse(self.enter(gate,inputs)['allowed'])

    def test_never_seen_far_wrong_side_and_insufficient_missing_do_not_latch(self):
        gate,inputs=self.ready();gate.last_visible_s=None
        self.assertFalse(self.enter(gate,inputs)['allowed'])
        for point in [[.151,-.04,-.07],[.069,-.04,-.07],[.13,.011,-.07],[.13,-.101,-.07]]:
            gate,inputs=self.ready();inputs['memory']['point_level_m']=point
            self.assertFalse(self.enter(gate,inputs)['allowed'])
        gate,inputs=self.ready();gate.missing_count=2
        self.assertFalse(self.enter(gate,inputs)['allowed'])

    def test_live_sensor_floor_goal_and_motion_guards_remain_required(self):
        changes=[dict(packet=None),dict(guard_safe=False),dict(guard_s=.9),dict(goal_s=-1.),
            dict(goal_heading=math.radians(15.01)),dict(body_command=[.01,0,0]),
            dict(odometry_continuous=False),dict(kicked=True)]
        for change in changes:
            gate,inputs=self.ready();inputs.update(change)
            with self.subTest(change):self.assertFalse(self.enter(gate,inputs)['allowed'])

    def test_failed_exit_revokes_latch_permanently_in_epoch(self):
        gate,inputs=self.ready();self.assertTrue(self.enter(gate,inputs)['allowed'])
        broken=copy.deepcopy(inputs);broken['guard_safe']=False
        self.assertFalse(gate.exit_gate(**broken)['allowed']);self.assertIsNone(gate.latch)
        self.assertFalse(gate.exit_gate(**inputs)['allowed']);self.assertFalse(self.enter(gate,inputs)['allowed'])

    def test_reanchor_and_epoch_change_cannot_reuse_authority(self):
        gate,inputs=self.ready();self.enter(gate,inputs)
        inputs['memory'].update(ball_capture_timestamp_s=1.1,ball_age_s=.12)
        self.assertFalse(gate.exit_gate(**inputs)['allowed'])
        gate,inputs=self.ready();self.enter(gate,inputs);gate.reset_epoch(1)
        self.assertIsNone(gate.latch);self.assertFalse(self.enter(gate,inputs)['allowed'])

    def test_new_visible_metric_revokes_blind_latch(self):
        gate,inputs=self.ready();self.enter(gate,inputs)
        gate.observe_capture(t=1.28,epoch=0,ball_metric_valid=True)
        self.assertIsNone(gate.latch)

    def test_supervisor_requires_actual_head_neutral_and_serializes_missing_heading(self):
        gate=BlindArrivalSupervisor()
        gate.control_context={'actual_head':[0,0,.01,0]}
        self.assertFalse(gate.actual_head_neutral())
        gate.control_context={'actual_head':[0,0,0,0]}
        self.assertTrue(gate.actual_head_neutral())
        gate.capture(t=1.,epoch=0,complete=False,guard_safe=True,goal_heading=float('nan'),goal_s=None)
        json.dumps(gate.status(),allow_nan=False)
        self.assertIsNone(gate.capture_context['goal_heading'])

if __name__=='__main__':unittest.main()
