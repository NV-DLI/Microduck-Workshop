import unittest
from mechanical_scoring import PhysicalGoalScorerV2


class MechanicalTests(unittest.TestCase):
    def scorer(self):
        return PhysicalGoalScorerV2(["joint" + str(i) for i in range(14)])

    def sample(self, scorer, t, x, servo=0.0, output=0.0, passive=0.02):
        return scorer.update(
            timestamp_s=t,
            ball_world_xyz=[x, 0.0, 0.035],
            paired_foot_forces_world_n=[[0.0, 0.0, 0.1], [0.0, 0.0, 0.0]],
            foot_names=["ankle_right", "ankle_left"],
            active_policy="kick_right",
            kick_requested=True,
            joint_violation_rad=passive,
            servo_coordinate_violation_rad=[servo] * 14,
            physical_output_violation_rad=[output] * 14,
            passive_soft_limit_residual_rad=[passive] * 14,
        )

    def test_soft_residual_does_not_redefine_legacy_or_fail_output(self):
        s = self.scorer()
        self.sample(s, 0.005, 0.9)
        self.sample(s, 0.010, 1.0)
        r = s.summary()
        self.assertTrue(r["physical_episode_success"])
        self.assertEqual(r["joint_limit_violation_rad"], 0.02)
        self.assertEqual(r["legacy_all28_peak_violation_rad"], 0.02)
        self.assertEqual(
            r["passive_soft_limit_per_joint"][0]["over_range_duration_s"], 0.01
        )

    def test_either_real_coordinate_violation_fails(self):
        for which in ["servo", "output"]:
            s = self.scorer()
            self.sample(s, 0.005, 0.9, **{which: 0.002})
            self.sample(s, 0.010, 1.0)
            self.assertFalse(s.summary()["physical_episode_success"])

    def test_nonfinite_and_bad_width_rejected(self):
        with self.assertRaises(ValueError):
            self.sample(self.scorer(), 0.005, 0.9, servo=float("nan"))

    def test_missing_sample_cannot_qualify(self):
        self.assertFalse(self.scorer().summary()["mechanical_output_bounds_passed"])


if __name__ == "__main__":
    unittest.main()
