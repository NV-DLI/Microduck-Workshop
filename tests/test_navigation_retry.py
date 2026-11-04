"""A failed observation retry must retain head-command and sensor continuity."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab/onboard"))
sys.path.insert(0, str(ROOT / "lab/runtime/simulation/core/evaluation/football"))
from navigation_retry import restart_head_feedback
from head_feedback import ObservationHeadFeedback
from head_feedback_core import HOME


class RetryTest(unittest.TestCase):
    def test_retry_does_not_jump_from_actual_head_to_zero(self):
        old = ObservationHeadFeedback()
        old.failure = "head_feedback_withdrawal_timeout"
        old.core.sent[0] = [0, 0, .4, 0]
        old.core.last_now = 2.
        old.core.last_seq[0] = 100
        old.core.last_read[0] = 2.
        new = restart_head_feedback(old)
        self.assertIsNone(new.failure)
        self.assertIs(new.core, old.core)
        actual, _ = new.core.update(now_s=2.02, nominal=np.zeros((1, 4)),
            measured_q=HOME[None], seq=[101], read_s=[2.02], valid=[True], enabled=[False])
        self.assertAlmostEqual(actual[0, 2], .394)

    def test_invalid_new_packet_holds_last_sent_command(self):
        old = ObservationHeadFeedback()
        old.core.sent[0] = [.1, -.2, .4, .05]
        expected = old.core.sent.copy()
        new = restart_head_feedback(old)
        actual, _ = new.core.update(now_s=2., nominal=np.zeros((1, 4)),
            measured_q=HOME[None], seq=[-1], read_s=[2.], valid=[False], enabled=[False])
        np.testing.assert_array_equal(actual, expected)

    def test_compensation_and_sensor_sequence_are_retained(self):
        old = ObservationHeadFeedback()
        old.core.bias[0, 2] = .02
        old.core.last_seq[0] = 100
        old.core.last_read[0] = 2.
        new = restart_head_feedback(old)
        self.assertTrue(new.compensated)
        self.assertEqual(new.core.last_seq[0], 100)
        self.assertEqual(new.core.last_read[0], 2.)
        self.assertIsNone(new.withdraw_started)
        self.assertIsNone(new.observe_started)


if __name__ == "__main__":
    unittest.main()
