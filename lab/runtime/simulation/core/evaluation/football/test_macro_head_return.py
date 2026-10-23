import unittest
import numpy as np
from head_feedback import ObservationHeadFeedback
from head_feedback_core import HOME


class MacroHeadReturn(unittest.TestCase):
    def call(self, h, **kw):
        p = dict(
            seq=1,
            read_timestamp_s=0.02,
            timestamp_s=0.02,
            encoder_read_timestamp_s=0.02,
            encoder_q_rad=np.r_[np.zeros(5), HOME, np.zeros(5)].tolist(),
            orientation_valid=True,
        )
        args = dict(
            now_s=0.02,
            nominal=[0, 0, 0, 0],
            packet=p,
            velocity=[0, 0, 0],
            state="NEUTRAL_FINE_RETURN",
            handoff_state="NEUTRAL_FINE_STEP",
            navigation_enabled=True,
            head_motion_allowed=False,
            macro_neutral_return=True,
        )
        args.update(kw)
        return h.update(**args)

    def ready(self):
        h = ObservationHeadFeedback()
        h.core.sent[0] = [-0.7, 0.6, 0, 0]
        return h

    def test_actual_slew_changes_and_has_no_integral(self):
        h = self.ready()
        s, d = self.call(h)
        self.assertAlmostEqual(s[0], -0.687)
        self.assertAlmostEqual(s[1], 0.587)
        self.assertFalse(d["integration_enabled"])
        self.assertTrue(d["macro_neutral_return_active"])

    def test_no_fast_motion_outside_owned_zero_body_return(self):
        for kw in [
            dict(macro_neutral_return=False),
            dict(velocity=[0.3, 0, 0]),
            dict(handoff_state="APPROACH"),
            dict(nominal=[-0.8, 0.6, 0, 0]),
        ]:
            h = self.ready()
            s, d = self.call(h, **kw)
            self.assertFalse(d["macro_neutral_return_active"])
            self.assertLessEqual(
                max(abs(np.asarray(s) - [-0.7, 0.6, 0, 0])), 0.00600001
            )

    def test_invalid_packet_freezes_actual_command(self):
        h = self.ready()
        s, d = self.call(h, packet=None)
        np.testing.assert_array_equal(s, [-0.7, 0.6, 0, 0])
        self.assertFalse(d["macro_neutral_return_active"])
