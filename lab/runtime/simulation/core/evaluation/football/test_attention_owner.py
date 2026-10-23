import unittest
import math
import copy
import numpy as np
from attention_owner import AttentionOwner
import test_ball_reacquisition as recovery_fixture


class OwnerTests(unittest.TestCase):
    def test_goal_commit_survives_missing_ball(self):
        a = AttentionOwner()
        self.assertEqual(
            a.update(
                t=10.0,
                near=True,
                complete_capture_s=10.0,
                goal_fix_s=1.0,
                request_goal=True,
            ),
            "GOAL_REFRESH",
        )
        for t in (10.1, 11.0, 14.0):
            self.assertEqual(
                a.update(
                    t=t,
                    near=True,
                    complete_capture_s=None,
                    goal_fix_s=1.0,
                    request_goal=True,
                ),
                "GOAL_REFRESH",
            )
        self.assertEqual(a.started, 10.0)

    def test_goal_must_be_new_accepted_fix_then_new_complete_ball(self):
        a = AttentionOwner()
        a.update(
            t=10.0,
            near=True,
            complete_capture_s=10.0,
            goal_fix_s=1.0,
            request_goal=True,
        )
        a.update(
            t=11.0,
            near=True,
            complete_capture_s=None,
            goal_fix_s=10.0,
            request_goal=True,
        )
        self.assertEqual(a.state, "GOAL_REFRESH")
        a.update(
            t=11.1,
            near=True,
            complete_capture_s=11.1,
            goal_fix_s=11.1,
            request_goal=False,
        )
        self.assertEqual(a.state, "BALL_REACQUIRE")
        self.assertEqual(a.started, 10.0)
        a.update(
            t=11.2,
            near=True,
            complete_capture_s=11.1,
            goal_fix_s=11.1,
            request_goal=False,
        )
        self.assertEqual(a.state, "BALL_REACQUIRE")
        a.update(
            t=11.3,
            near=True,
            complete_capture_s=11.3,
            goal_fix_s=11.1,
            request_goal=False,
        )
        self.assertEqual(a.state, "NONE")

    def test_goal_timeout_not_reset_by_fresh_ball(self):
        a = AttentionOwner()
        a.update(
            t=10.0,
            near=True,
            complete_capture_s=10.0,
            goal_fix_s=1.0,
            request_goal=True,
        )
        for t in (11.0, 12.0, 15.0):
            a.update(
                t=t, near=True, complete_capture_s=t, goal_fix_s=1.0, request_goal=True
            )
        a.update(
            t=16.0,
            near=True,
            complete_capture_s=16.0,
            goal_fix_s=1.0,
            request_goal=True,
        )
        self.assertEqual(a.failed, "goal_refresh_committed_timeout")

    def test_total_deadline_survives_goal_refresh(self):
        a = AttentionOwner()
        a.update(
            t=10.0,
            near=True,
            complete_capture_s=None,
            goal_fix_s=1.0,
            request_goal=False,
        )
        a.update(
            t=20.0,
            near=True,
            complete_capture_s=20.0,
            goal_fix_s=1.0,
            request_goal=True,
        )
        a.update(
            t=21.0,
            near=True,
            complete_capture_s=None,
            goal_fix_s=21.0,
            request_goal=False,
        )
        self.assertEqual(a.started, 10.0)
        a.update(
            t=34.0,
            near=True,
            complete_capture_s=None,
            goal_fix_s=21.0,
            request_goal=True,
        )
        self.assertEqual(a.failed, "attention_reacquisition_total_timeout")

    def test_invalid_clock_or_observation_no_state_mutation(self):
        for values in (
            {"t": math.nan},
            {"t": -1.0},
            {"complete_capture_s": math.nan},
            {"goal_fix_s": 11.0},
        ):
            a = AttentionOwner()
            before = copy.deepcopy(a.__dict__)
            args = dict(
                t=10.0,
                near=True,
                complete_capture_s=None,
                goal_fix_s=1.0,
                request_goal=True,
            )
            args.update(values)
            with self.assertRaises(ValueError):
                a.update(**args)
            self.assertEqual(a.__dict__, before)

    def test_far_state_never_takes_ownership(self):
        a = AttentionOwner()
        a.update(
            t=10.0,
            near=False,
            complete_capture_s=None,
            goal_fix_s=1.0,
            request_goal=True,
        )
        self.assertEqual(a.state, "NONE")


class ActualGoalOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = recovery_fixture.Recovery()
        recovery_fixture.Recovery.setUpClass()

    def test_actual_missing_ball_frames_allow_goal_head_only(self):
        for t in (60.0, 100.0, 140.0, 180.0):
            n, t = self.fixture.nav(t)
            n.goal_time = t - 9.0
            before = n.gaze.current.copy()
            old_ball = n.ball_time
            out = self.fixture.actual(n, t)
            self.assertEqual(out["public_attention_owner"]["owner"], "GOAL_REFRESH")
            self.assertEqual(out["state"], "REFRESH_GOAL")
            self.assertEqual(out["velocity_body_xyw"], [0.0, 0.0, 0.0])
            self.assertFalse(out["kick_request"])
            self.assertFalse(out["motion_authorization"]["allowed"])
            self.assertEqual(n.ball_time, old_ball)
            self.assertLess(n.gaze.current[1], before[1])

    def test_committed_goal_holds_without_new_head_pose_capture(self):
        n, t = self.fixture.nav(60.0)
        n.goal_time = t - 9.0
        self.fixture.actual(n, t)
        n.refresh_too_close = True
        before = n.gaze.current.copy()
        n.gaze.update([0.0, 0.0], t + 0.1)
        self.assertTrue(np.array_equal(before, n.gaze.current))

    def test_committed_goal_preserves_invalid_public_pose_hold(self):
        n, t = self.fixture.nav(60.0)
        n.goal_time = t - 9.0
        self.fixture.actual(n, t)
        n.ball_recovery_context_valid = False
        before = n.gaze.current.copy()
        n.gaze.update([0.0, 0.0], t + 0.1)
        self.assertTrue(np.array_equal(before, n.gaze.current))


if __name__ == "__main__":
    unittest.main()
