import unittest, math
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet


class ShotFrame(unittest.TestCase):
    def ready(self, h):
        f = NeutralFineStep()
        f.primitive = {"velocity_command": [0.3, -0.1, 0.0]}
        f.goal_heading = h
        return f

    def test_rotated_same_goal_frame_point_does_not_command_sideways(self):
        h = 0.15
        c, s = math.cos(h), math.sin(h)
        p = [c * 0.2 - s * (-0.035), s * 0.2 + c * (-0.035), -0.08]
        f = self.ready(h)
        v = f.macro_command(p, packet(1.0), 0.02)
        self.assertAlmostEqual(f.macro_shot_error, 0.0)
        self.assertEqual(v[1], -0.1)

    def test_declared_hysteresis_and_measured_command_magnitude(self):
        f = self.ready(0.0)
        self.assertEqual(
            f.macro_command([0.2, -0.06, -0.08], packet(1.0), 0.02)[1], -0.1
        )
        for i in range(30):
            v = f.macro_command([0.15, -0.030, -0.08], packet(1.0), 0.02)
        self.assertEqual(v[1], -0.1)
        self.assertLess(abs(v[2]), 0.5)
