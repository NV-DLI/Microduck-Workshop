import json, unittest
from final_approach import FinalApproach


class TraceTypes(unittest.TestCase):
    def test_all_phase_outputs_are_strict_json(self):
        seen = set()
        for point, heading in [
            ([0.34, -0.035, 0.035], 0.0),
            ([0.3, 0.2, 0.035], 0.0),
            ([0.09, -0.035, 0.035], 0.0),
            ([0.09, -0.02, 0.035], 0.12),
            ([0.1, -0.03, 0.035], 0.0),
        ]:
            f = FinalApproach()
            for k in range(80):
                cmd, d = f.update(
                    t=k / 15.0,
                    point=point,
                    gyro_body=[0.0, 0.0, 0.0],
                    gyro_level=0.0,
                    goal_heading=heading,
                    authorized=True,
                    uncertainty=0.02,
                )
                json.dumps(dict(command=cmd, diagnostic=d), allow_nan=False)
                seen.add(d["phase"])
        self.assertTrue({"OBSERVE", "ADVANCE", "TURN", "PREPARED"}.issubset(seen))


if __name__ == "__main__":
    unittest.main()
