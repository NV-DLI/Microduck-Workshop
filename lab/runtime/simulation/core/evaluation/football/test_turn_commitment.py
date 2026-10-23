import unittest
from final_approach import FinalApproach


class TurnCommitment(unittest.TestCase):
    def step(self, f, t, point=(0.35, 0.0, 0.0), **kwargs):
        p = dict(
            t=t,
            point=point,
            gyro_body=[0.0, 0.0, 0.74],
            gyro_level=0.74,
            goal_heading=0.0,
            authorized=True,
            uncertainty=0.02,
        )
        p.update(kwargs)
        return f.update(**p)

    def turn(self):
        f = FinalApproach()
        f.staging_phase = "GOAL_ALIGN"
        f.phase = "TURN"
        f.sign = 1.0
        f.started = 1.0
        f.turns = 1
        return f

    def test_prediction_before_minimum_is_diagnostic(self):
        f = self.turn()
        u, d = self.step(f, 1.13)
        self.assertTrue(d["turn_stop_prediction"]["stop_hint"])
        self.assertFalse(d["turn_stop_prediction"]["minimum_duration_elapsed"])
        self.assertEqual(u, [0.0, 0.0, 0.8])
        self.step(f, 1.27)
        u, d = self.step(f, 1.4)
        self.assertEqual(u, [0.0, 0.0, 0.0])
        self.assertTrue(d["turn_stop_prediction"]["minimum_duration_elapsed"])

    def test_sign_cross_before_minimum_is_diagnostic(self):
        f = self.turn()
        u, d = self.step(f, 1.13, point=(0.35, -0.08, 0.0))
        self.assertLess(d["bearing_rad"], 0.0)
        self.assertEqual(u, [0.0, 0.0, 0.8])

    def test_upstream_guard_immediately_revokes(self):
        f = self.turn()
        u, d = self.step(f, 1.13, authorized=False)
        self.assertEqual(u, [0.0, 0.0, 0.0])
        self.assertEqual(d["reason"], "final_upstream_guard")
        self.assertEqual(f.turns, 1)

    def test_invalid_geometry_immediately_revokes(self):
        f = self.turn()
        u, d = self.step(f, 1.13, point=(float("nan"), 0.0, 0.0))
        self.assertEqual(u, [0.0, 0.0, 0.0])
        self.assertEqual(f.turns, 1)

    def test_duration_cap_unchanged(self):
        f = self.turn()
        u, d = self.step(f, 3.4, point=(0.35, 0.3, 0.0), gyro_level=0.0)
        self.assertEqual(u, [0.0, 0.0, 0.0])
        self.assertEqual(d["reason"], "final_turn_stop")


if __name__ == "__main__":
    unittest.main()
