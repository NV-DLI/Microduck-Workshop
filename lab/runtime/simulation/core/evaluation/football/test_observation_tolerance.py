import unittest
from final_approach import FinalApproach


class ObservationTolerance(unittest.TestCase):
    def ready(self, p, h):
        f = FinalApproach()
        f.staging_phase = "GOAL_ALIGN"
        for t in [0, 0.07, 0.135, 0.2, 0.27, 0.335, 0.4]:
            v, d = f.update(
                t=t,
                point=p,
                gyro_body=[0, 0, 0],
                gyro_level=0,
                goal_heading=h,
                authorized=True,
                uncertainty=0.02,
            )
        return f, v, d

    def test_recorded_far_observation_can_advance(self):
        f, v, d = self.ready([0.309, -0.025, -0.081], 0.025)
        self.assertEqual(v, [0.3, 0, 0])
        self.assertEqual(f.turns, 0)

    def test_close_standoff_never_bypasses_fine_handoff(self):
        f, v, d = self.ready([0.20, -0.030, -0.081], 0.025)
        self.assertEqual(v, [0, 0, 0])
        self.assertEqual(f.advances, 0)
