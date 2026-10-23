import unittest, json, copy
from pathlib import Path
from zone_floor_projection import project_floor_zones
from nav2_service.map_contract import snapshot, check_path

R = Path(
    "@COURSE@/configuration/core/football-v107-independent-empty10-v1/episode-01-s976101"
)


class FloorProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.request = json.loads((R / "nav2/exchange/request.json").read_text())
        t = cls.request["case"]["query_s"]
        with (R / "sensor-inputs.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["timestamp_s"] == t:
                    cls.frame = row["tof_frame"]
                    break

    def run_case(self, frame=None, sample=None, accepted=True):
        return project_floor_zones(
            sample or copy.deepcopy(self.request["case"]["tof_samples"][0]),
            frame or copy.deepcopy(self.frame),
            floor_accepted=accepted,
        )

    def test_actual_blocked_goal_restores_original_10cm_envelope(self):
        case = copy.deepcopy(self.request["case"])
        prior = self.request["provenance"]
        old = snapshot(case, prior)
        new, diag = self.run_case()
        case["tof_samples"][0] = new
        snap = snapshot(case, prior)
        self.assertFalse(check_path([old["pose"][:2], old["goal"]], old)["accepted"])
        self.assertTrue(check_path([snap["pose"][:2], snap["goal"]], snap)["accepted"])
        corrected = {x["zone"] for x in diag["corrected_zones"]}
        self.assertTrue({32, 33}.issubset(corrected))
        self.assertFalse({19, 20} & corrected)
        original = self.request["case"]["tof_samples"][0]
        for key in original:
            if key != "unit_rays_sensor":
                self.assertEqual(original[key], new[key])

    def test_no_floor_or_wrong_virtual_model_is_identity(self):
        original = self.request["case"]["tof_samples"][0]
        new, _ = self.run_case(accepted=False)
        self.assertEqual(new, original)
        frame = copy.deepcopy(self.frame)
        frame["model"]["source_width_px"] = 64
        new, _ = self.run_case(frame=frame)
        self.assertEqual(new, original)

    def test_obstacle_returns_are_not_reprojected(self):
        # A closer return at a floor-like zone represents an intervening obstacle.
        frame = copy.deepcopy(self.frame)
        sample = copy.deepcopy(self.request["case"]["tof_samples"][0])
        zone = 33
        sample["ranges_m"][zone] *= 0.6
        frame["cells"][zone // 8][zone % 8]["range_m"] = sample["ranges_m"][zone]
        new, d = self.run_case(frame, sample)
        self.assertEqual(
            new["unit_rays_sensor"][zone], sample["unit_rays_sensor"][zone]
        )
        self.assertNotIn(zone, [x["zone"] for x in d["corrected_zones"]])

    def test_partial_and_tooclose_returns_not_reclassified(self):
        for key, value in [
            ("contains_too_close", True),
            ("no_return_samples", 1),
            ("invalid_samples", 1),
        ]:
            frame = copy.deepcopy(self.frame)
            frame["cells"][4][1][key] = value
            new, d = self.run_case(frame)
            self.assertNotIn(33, [x["zone"] for x in d["corrected_zones"]])

    def test_stale_fk_rejected(self):
        sample = copy.deepcopy(self.request["case"]["tof_samples"][0])
        sample["fk_capture_s"] -= 0.01
        with self.assertRaises(ValueError):
            self.run_case(sample=sample)

    def test_inputs_are_unchanged(self):
        sample = copy.deepcopy(self.request["case"]["tof_samples"][0])
        frame = copy.deepcopy(self.frame)
        a = copy.deepcopy(sample)
        b = copy.deepcopy(frame)
        self.run_case(frame, sample)
        self.assertEqual(a, sample)
        self.assertEqual(b, frame)


if __name__ == "__main__":
    unittest.main()
