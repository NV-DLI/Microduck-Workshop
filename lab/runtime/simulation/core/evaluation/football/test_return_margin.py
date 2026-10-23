import unittest, math
from final_approach import FinalApproach


class Margin(unittest.TestCase):
    def ready(self, p, h=0.0, turns=0, authorized=True):
        f = FinalApproach()
        f.turns = turns
        for t in [0, 0.07, 0.135, 0.2, 0.27, 0.335, 0.4]:
            out, d = f.update(
                t=t,
                point=p,
                gyro_body=[0, 0, 0],
                gyro_level=0.0,
                goal_heading=h,
                authorized=authorized,
                uncertainty=0.02,
            )
        return f, out, d

    def test_actual_v55_entry_now_aligns_before_return(self):
        f, out, d = self.ready(
            [0.21117481969586968, -0.03229128878690758, -0.08058550058307669],
            -0.18813148903526247,
        )
        self.assertEqual(out, [0, 0, 0.8])
        self.assertEqual(d["turn_entry_bearing_limit_rad"], 0.06)
        self.assertEqual(f.advances, 0)
        self.assertEqual(f.turns, 1)

    def test_far_bearing_limit_stays15(self):
        p = [0.4, -0.035 + math.tan(0.1) * 0.14, 0]
        f, out, d = self.ready(p)
        self.assertEqual(out, [0.3, 0, 0])
        self.assertEqual(d["turn_entry_bearing_limit_rad"], 0.15)

    def test_final_box_goal_gate_stays08(self):
        f, out, d = self.ready([0.09, -0.035, 0], 0.07)
        self.assertEqual(out, [0, 0, 0])
        self.assertEqual(f.advances, 0)
        self.assertEqual(f.turns, 0)

    def test_original_guard_and_budget_no_refund(self):
        p = [0.2, -0.035 + math.tan(0.1) * 0.11, 0]
        f, out, d = self.ready(p, turns=24)
        self.assertEqual(f.failed, "final_turn_budget_exhausted")
        self.assertEqual(out, [0, 0, 0])
        self.assertEqual(f.turns, 24)
        f, out, d = self.ready(p, authorized=False)
        self.assertEqual(out, [0, 0, 0])
        self.assertEqual(f.turns, 0)


if __name__ == "__main__":
    unittest.main()
