import unittest
from neutral_fine_step import NeutralFineStep
from short_primitive import select
from test_neutral_fine_step import packet


class Decoupled(unittest.TestCase):
    def test_lateral_sign_does_not_shift_goal_heading_equilibrium(self):
        for y in [-0.08, 0.02]:
            f = NeutralFineStep()
            f.primitive = {"velocity_command": [0.3, -0.1, 0]}
            f.goal_heading = 0.0
            v = f.macro_command([0.18, y, -0.08], packet(1.0), 0.02)
            self.assertEqual(v[1], -0.1)
            self.assertAlmostEqual(v[2], 0.0)

    def test_large_heading_requires_alignment_before_macro(self):
        self.assertIsNone(select([0.1867, -0.0321, -0.08], -0.1822))
        self.assertIsNone(select([0.2, -0.05, -0.08], 0.081))
