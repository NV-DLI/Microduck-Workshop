import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from head_feedback_core import HeadSetpointFeedback, HOME, BOUND
from head_feedback import ObservationHeadFeedback
import test_head_feedback as base_tests

_path = Path(__file__).parents[1] / "head_feedback/head_feedback_core.py"
_spec = importlib.util.spec_from_file_location("frozen_v58_pitch_test_oracle", _path)
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)


class PitchRedistribution(unittest.TestCase):
    def blocked(self, sign=-1):
        h = HeadSetpointFeedback(1)
        nominal = np.array([[sign * .8, .8, 0., 0.]])
        q = nominal + HOME
        q[0, 0] -= sign * .2
        h.bias[0, 0] = sign * .3
        h.sent[:] = nominal + h.bias
        h.last_seq[:] = 1
        h.last_read[:] = 1.
        h.last_now = 1.
        args = dict(now_s=1.02, nominal=nominal, measured_q=q, seq=[2],
                    read_s=[1.02], valid=[True], enabled=[True])
        return h, args

    def test_lower_and_upper_neck_limits_transfer_only_pitch_error(self):
        for sign in (-1, 1):
            h, args = self.blocked(sign)
            baseline = _old.HeadSetpointFeedback(1)
            baseline.__dict__ = copy.deepcopy(h.__dict__)
            old_sent, _ = baseline.update(**copy.deepcopy(args))
            sent, d = h.update(**args)
            self.assertTrue(d["pitch_error_redistributed"][0])
            self.assertAlmostEqual(d["integral_error_rad"][0, 1], -sign * .2)
            self.assertAlmostEqual(h.bias[0, 1], -sign * .2 * .35 * .02)
            np.testing.assert_array_equal(h.bias[:, [0, 2, 3]], baseline.bias[:, [0, 2, 3]])
            np.testing.assert_array_equal(sent[:, [0, 2, 3]], old_sent[:, [0, 2, 3]])
            self.assertLessEqual(np.max(abs(sent)), 1.1)

    def test_not_yet_at_limit_or_inward_error_does_not_transfer(self):
        h, args = self.blocked()
        h.sent[0, 0] = -1.09
        _, d = h.update(**args)
        self.assertFalse(d["pitch_error_redistributed"][0])
        self.assertEqual(h.bias[0, 1], 0.)
        h, args = self.blocked()
        args["measured_q"][0, 0] -= .4
        _, d = h.update(**args)
        self.assertFalse(d["pitch_error_redistributed"][0])
        self.assertGreater(h.bias[0, 0], -.3)
        self.assertEqual(h.bias[0, 1], 0.)

    def test_duplicate_and_invalid_packet_cannot_transfer(self):
        for patch in ({"seq": [1], "read_s": [1.]}, {"valid": [False]}):
            h, args = self.blocked()
            before = h.bias.copy()
            _, d = h.update(**{**args, **patch})
            np.testing.assert_array_equal(h.bias, before)
            self.assertFalse(d["pitch_error_redistributed"][0])
        h, args = self.blocked()
        h.update(**{**args, "valid": [False]})
        before = h.bias.copy()
        _, d = h.update(**{**args, "now_s": 1.04, "seq": [3], "read_s": [1.04]})
        np.testing.assert_array_equal(h.bias, before)
        self.assertFalse(d["pitch_error_redistributed"][0])

    def test_source_correction_budget_and_slew_unchanged(self):
        h, args = self.blocked()
        last = h.sent.copy()
        for k in range(1, 401):
            t = 1. + .02 * k
            sent, _ = h.update(**{**args, "now_s": t, "read_s": [t], "seq": [1 + k]})
            self.assertTrue(np.all(abs(sent) <= BOUND + 1e-9))
            self.assertTrue(np.all(abs(h.bias) <= .35 + 1e-9))
            self.assertLessEqual(np.max(abs(sent - last)), .006000001)
            last = sent
        self.assertAlmostEqual(h.bias[0, 1], .3)

    def test_disabled_feedback_clears_bias_without_transfer(self):
        h, args = self.blocked()
        _, d = h.update(**{**args, "enabled": [False]})
        self.assertFalse(d["pitch_error_redistributed"][0])
        np.testing.assert_array_equal(h.bias, np.zeros((1, 4)))

    def test_original_phase_matrix_and_warmup_still_govern_transfer(self):
        forbidden = [
            ("NEUTRAL_FINE_REACQUIRE", "NEUTRAL_FINE_STEP", [0, 0, 0], True),
            ("NEUTRAL_FINE_RETURN", "NEUTRAL_FINE_STEP", [0, 0, 0], True),
            ("NEUTRAL_FINE_STEP", "NEUTRAL_FINE_STEP", [.2, 0, 0], True),
            ("NEUTRAL_FINE_SETTLE", "NEUTRAL_FINE_STEP", [0, 0, 0], True),
            ("REACQUIRE_BALL", "HEAD_RETURN", [0, 0, 0], True),
            ("REACQUIRE_BALL", "KICK", [0, 0, 0], True),
            ("REACQUIRE_BALL", "APPROACH", [0, 0, .8], True),
            ("REACQUIRE_BALL", "APPROACH", [0, 0, 0], False),
        ]
        for state, handoff, velocity, permitted in forbidden:
            h = ObservationHeadFeedback()
            for i in range(300):
                _, d = base_tests.HeadFeedbackTests().call(
                    h, i, state=state, handoff_state=handoff, velocity=velocity,
                    head_motion_allowed=permitted)
                self.assertFalse(d["integration_enabled"])
                self.assertFalse(d["pitch_error_redistributed"])
        h = ObservationHeadFeedback()
        for i in range(250):
            _, d = base_tests.HeadFeedbackTests().call(h, i)
            self.assertFalse(d["integration_enabled"])
            self.assertFalse(d["pitch_error_redistributed"])

    def test_bad_clock_fails_before_state_mutation(self):
        h, args = self.blocked()
        before = copy.deepcopy(h.__dict__)
        with self.assertRaises(ValueError):
            h.update(**{**args, "now_s": float("nan")})
        self.assertEqual(h.last_now, before["last_now"])
        np.testing.assert_array_equal(h.bias, before["bias"])


if __name__ == "__main__":
    unittest.main()
