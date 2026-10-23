import unittest
from short_primitive import select
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet


class Fixed(unittest.TestCase):
    def test_original_box_predictions_include_heading_and_four_replicas(self):
        for p in ([0.18, -0.05, -0.08], [0.21, -0.075, -0.08]):
            r = select(p, 0.0)
            self.assertIsNotNone(r)
            self.assertEqual(len(r["predicted_endpoints"]), 4)
            self.assertTrue(
                all(
                    0.075 <= x <= 0.105 and -0.05 <= y <= -0.02 and abs(h) <= 0.08
                    for x, y, h in r["predicted_endpoints"]
                )
            )

    def test_old_unmeasured_intermediate_duration_is_rejected(self):
        self.assertIsNone(select([0.196, -0.039, -0.08], 0.035))

    def test_selected_lateral_command_is_constant_despite_sway(self):
        f = NeutralFineStep()
        f.primitive = {"velocity_command": [0.3, -0.1, 0]}
        f.goal_heading = 0.0
        for y in [-0.09, 0, 0.03]:
            self.assertEqual(
                f.macro_command([0.17, y, -0.08], packet(1.0), 0.02)[1], -0.1
            )
