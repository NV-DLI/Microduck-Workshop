import unittest
from unittest.mock import patch
from final_approach import FinalApproach


class PulseClockTest(unittest.TestCase):
    def setup_turn(self):
        f = FinalApproach()
        f.phase = "TURN"
        f.turn_goal = True
        f.last_t = 1.0
        f.pulse_stop_s = 1.04
        return f

    def test_expiry_holds_zero_until_next_capture(self):
        f = self.setup_turn()
        with patch("neutral_fine_step.NeutralFineStep.packet_valid", return_value=True):
            self.assertEqual(
                f.policy_command(t=1.02, packet={}, velocity=[0, 0, 0.8])[0],
                [0, 0, 0.8],
            )
            self.assertEqual(
                f.policy_command(t=1.04, packet={}, velocity=[0, 0, 0.8])[0], [0, 0, 0]
            )
            self.assertEqual(
                f.policy_command(t=1.06, packet={}, velocity=[0, 0, 0.8])[0], [0, 0, 0]
            )

    def test_stale_packet_stops(self):
        f = self.setup_turn()
        with patch(
            "neutral_fine_step.NeutralFineStep.packet_valid", return_value=False
        ):
            self.assertEqual(
                f.policy_command(t=1.02, packet=None, velocity=[0, 0, 0.8])[0],
                [0, 0, 0],
            )

    def test_other_motion_unchanged(self):
        f = FinalApproach()
        self.assertEqual(
            f.policy_command(t=1.0, packet=None, velocity=[0.3, 0, 0])[0], [0.3, 0, 0]
        )
