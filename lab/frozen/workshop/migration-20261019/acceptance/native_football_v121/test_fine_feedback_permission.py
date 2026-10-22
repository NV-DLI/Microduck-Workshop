import copy
import unittest
import numpy as np
from fine_feedback_permission import fine_reacquire_feedback_allowed
from head_feedback import ObservationHeadFeedback
from head_feedback_core import HOME
import test_head_feedback as base_tests


class PermissionTests(unittest.TestCase):
    def context(self):
        return dict(
            now_s=8., capture_s=7.935, handoff_state="APPROACH",
            head_motion_allowed=True, velocity=[0., 0., 0.], kick_requested=False,
            fine_status={"state": "REACQUIRE", "started_s": 2., "step_end_s": 6.,
                         "quiet_started_s": 6.5, "velocity": [0., 0., 0.], "kick_authorized": False},
            control={"state": "NEUTRAL_FINE_REACQUIRE", "velocity_body_xyw": [0., 0., 0.],
                     "kick_request": False, "public_attention_owner": {"owner": "BALL_REACQUIRE"},
                     "public_attention_head_permission": {"accepted": True, "owner": "BALL_REACQUIRE",
                         "capture_s": 7.935, "body_authorized": False, "kick_authorized": False}})

    def test_phase_matrix(self):
        self.assertTrue(fine_reacquire_feedback_allowed(**self.context()))
        for phase in ("IDLE", "RETURN", "STEP", "SETTLE", "FAILED"):
            args = self.context()
            args["fine_status"]["state"] = phase
            self.assertFalse(fine_reacquire_feedback_allowed(**args))
        for handoff in ("HEAD_RETURN", "KICK", "COMPLETE", "FAILED"):
            self.assertFalse(fine_reacquire_feedback_allowed(**{**self.context(), "handoff_state": handoff}))

    def test_capture_clock_owner_body_and_quiet_bounds(self):
        for patch in ({"now_s": float("nan")}, {"capture_s": 7.7}, {"capture_s": 8.01},
                      {"now_s": 26.}, {"head_motion_allowed": False}, {"velocity": [.2, 0., 0.]},
                      {"velocity": [float("nan"), 0, 0]}, {"kick_requested": True}):
            self.assertFalse(fine_reacquire_feedback_allowed(**{**self.context(), **patch}))
        for key, value in (("quiet_started_s", None), ("quiet_started_s", 7.6), ("quiet_started_s", 5.9),
                           ("started_s", float("nan")), ("velocity", [.2, 0., 0.]), ("kick_authorized", True)):
            args = self.context()
            args["fine_status"][key] = value
            self.assertFalse(fine_reacquire_feedback_allowed(**args))
        for key, value in (("accepted", False), ("owner", "GOAL_REFRESH"), ("capture_s", 7.9),
                           ("body_authorized", True), ("kick_authorized", True)):
            args = self.context()
            args["control"]["public_attention_head_permission"][key] = value
            self.assertFalse(fine_reacquire_feedback_allowed(**args))
        args = self.context()
        before = copy.deepcopy(args)
        fine_reacquire_feedback_allowed(**args)
        self.assertEqual(args, before)

    def call(self, h, i, **changes):
        args = dict(state="NEUTRAL_FINE_REACQUIRE", handoff_state="NEUTRAL_FINE_STEP",
                    fine_reacquire_head_only=True)
        args.update(changes)
        return base_tests.HeadFeedbackTests().call(h, i, **args)

    def test_original5s_delay_and_no_duplicate_integration(self):
        h = ObservationHeadFeedback()
        for i in range(250):
            self.assertFalse(self.call(h, i)[1]["integration_enabled"])
        self.assertTrue(self.call(h, 250)[1]["integration_enabled"])
        bias = h.core.bias.copy()
        q = np.r_[np.zeros(5), HOME + [.2, 0., 0., 0.], np.zeros(5)].tolist()
        packet = dict(seq=250, read_timestamp_s=5., timestamp_s=5., encoder_read_timestamp_s=5.,
                      encoder_q_rad=q, orientation_valid=True)
        self.call(h, 251, packet=packet)
        np.testing.assert_array_equal(h.core.bias, bias)

    def test_invalid_packet_or_permission_cannot_integrate(self):
        for change in ({"packet": None}, {"head_motion_allowed": False}, {"fine_reacquire_head_only": False}):
            h = ObservationHeadFeedback()
            for i in range(300):
                self.assertFalse(self.call(h, i, **change)[1]["integration_enabled"])
            self.assertFalse(h.core.bias.any())

    def test_all_other_phases_remain_disabled_and_KICK_must_be_neutral(self):
        for phase, handoff in [
            ("NEUTRAL_FINE_RETURN", "NEUTRAL_FINE_STEP"), ("NEUTRAL_FINE_STEP", "NEUTRAL_FINE_STEP"),
            ("NEUTRAL_FINE_SETTLE", "NEUTRAL_FINE_STEP"), ("NEUTRAL_FINE_REACQUIRE", "HEAD_RETURN"),
            ("NEUTRAL_FINE_REACQUIRE", "KICK"),
        ]:
            h = ObservationHeadFeedback()
            for i in range(300):
                self.assertFalse(self.call(h, i, state=phase, handoff_state=handoff)[1]["integration_enabled"])

    def test_new_body_request_waits_for_actual_compensation_withdrawal(self):
        h = ObservationHeadFeedback()
        for i in range(350):
            self.call(h, i)
        self.assertTrue(h.core.bias.any())
        first = None
        held = 0
        for i in range(350, 450):
            sent, d = self.call(h, i, state="FINAL_ALIGN", handoff_state="APPROACH",
                                fine_reacquire_head_only=False, velocity=[.2, 0., 0.])
            self.assertFalse(d["integration_enabled"])
            if d["body_zero_for_withdrawal"]:
                held += 1
                first = d["withdrawal_started_s"] if first is None else first
                self.assertEqual(first, d["withdrawal_started_s"])
            else:
                self.assertLessEqual(np.max(abs(np.asarray(sent))), 1e-9)
                self.assertFalse(h.core.bias.any())
                break
        self.assertGreater(held, 0)
        self.assertIsNone(h.failure)


if __name__ == "__main__":
    unittest.main()
