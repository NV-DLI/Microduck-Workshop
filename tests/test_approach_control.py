from pathlib import Path
import sys, unittest
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'lab/onboard'))
from approach_control import remembered_kick_gate, kick_box
from shot_completion import ShotCompletion

class NavigationTest(unittest.TestCase):
    def args(self):
        return dict(t=10., memory=dict(memory_available=True,point_level_m=[.10,-.035,.05],
                ball_age_s=5.,original_range_uncertainty_m=.01),heading=.1,goal_s=9.5,
                guard=True,moving=False,kicked=False)
    def test_recent_occluded_ball_can_start_handoff(self):
        self.assertTrue(remembered_kick_gate(**self.args())['allowed'])
    def test_lost_ball_alone_never_means_ready(self):
        a=self.args();a['memory']=dict(memory_available=False)
        self.assertFalse(remembered_kick_gate(**a)['allowed'])
    def test_wrong_side_far_stale_uncertain_and_motion_rejected(self):
        for key,val in [('point_level_m',[.1,.1,.05]),('point_level_m',[.25,-.035,.05]),
                        ('ball_age_s',7.),('original_range_uncertainty_m',.04)]:
            a=self.args();a['memory'][key]=val
            self.assertFalse(remembered_kick_gate(**a)['allowed'])
        for key,val in [('heading',.5),('goal_s',7.),('guard',False),('moving',True),('kicked',True)]:
            a=self.args();a[key]=val
            self.assertFalse(remembered_kick_gate(**a)['allowed'])
    def test_unlimited_has_no_deadline(self):
        s=ShotCompletion(0)
        self.assertIsNone(s.update(now_s=100000.,task_s=100000.,ball_xyz=[0,0,0],kick_completed=False,kick_contact=False))
    def test_30_second_deadline(self):
        s=ShotCompletion(30)
        self.assertEqual(s.update(now_s=33.,task_s=30.,ball_xyz=[0,0,0],kick_completed=False,kick_contact=False),'task_duration_limit')

class ContinuousTransferTest(unittest.TestCase):
    def run_transfer(self, *, point=(.277,-.08,-.07), guard=True, heading=.05,
                     goal_s=6.74, memory_available=True, state='TRACK'):
        from types import SimpleNamespace
        from copy import deepcopy
        from approach_control import transfer_continuous_track
        fine=SimpleNamespace(coarse_track=True,state=state,step_start=4.6,anchor_s=3.8,
            head=[-.1,.1,0,0],track_velocity=[.3,-.2,.1],quiet_start=None,
            step_end=None,reason=None,track_brake_start=None)
        before=deepcopy(vars(fine))
        result=transfer_continuous_track(fine,t=6.74,
            memory=dict(memory_available=memory_available,point_level_m=list(point)),
            heading=heading,goal_s=goal_s,guard=guard)
        return fine,before,result

    def test_in_domain_transfer_continues_without_reset(self):
        fine,before,result=self.run_transfer()
        self.assertTrue(result['transferred'])
        self.assertFalse(fine.coarse_track)
        for key,value in before.items():
            if key!='coarse_track':self.assertEqual(vars(fine)[key],value,key)

    def test_previous_lateral_boundary_case_is_rejected(self):
        fine,before,result=self.run_transfer(point=(.277,-.10349,-.07))
        self.assertFalse(result['transferred'])
        self.assertFalse(result['checks']['near_domain'])
        self.assertEqual(fine.state,'SETTLE')
        self.assertEqual(fine.track_velocity,[0.,0.,0.])
        self.assertTrue(fine.coarse_track)
        for key in ('step_start','anchor_s','head','track_brake_start'):
            self.assertEqual(vars(fine)[key],before[key],key)

    def test_lateral_boundaries_and_current_safety_are_enforced(self):
        import math
        for y in [-.10,.08]:
            self.assertTrue(self.run_transfer(point=(.27,y,-.07))[2]['transferred'])
        for kwargs in [dict(point=(.27,-.100001,-.07)),dict(point=(.27,.080001,-.07)),
                       dict(point=(.1299,-.03,-.07)),dict(guard=False),
                       dict(memory_available=False),dict(goal_s=4.739),dict(heading=math.radians(10.01))]:
            with self.subTest(kwargs):
                fine,_,result=self.run_transfer(**kwargs)
                self.assertFalse(result['transferred'])
                self.assertEqual(fine.state,'SETTLE')
                self.assertEqual(fine.track_velocity,[0,0,0])

    def test_other_owners_and_far_point_are_unchanged(self):
        for state in ['IDLE','RETURN','TURN','HEAD_RETURN','KICK']:
            fine,before,result=self.run_transfer(state=state)
            self.assertIsNone(result);self.assertEqual(vars(fine),before)
        fine,before,result=self.run_transfer(point=(.280001,-.03,-.07))
        self.assertIsNone(result);self.assertEqual(vars(fine),before)

if __name__=='__main__':unittest.main()
