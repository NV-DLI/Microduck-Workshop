import ast
import hashlib
import json
from pathlib import Path
import unittest

D = Path(__file__).resolve().parent
V64 = D.parent / "native_football_v64"


class Contract(unittest.TestCase):
    def test_unchanged_sensor_and_kick_authority_modules(self):
        for name in (
            "visibility_gate.py",
            "attention_owner.py",
            "native_head_domain.py",
        ):
            self.assertEqual((D / name).read_bytes(), (V64 / name).read_bytes(), name)

    def test_original_kick_box_and_memory_lifetime_remain(self):
        from final_approach import in_training_box
        from neutral_fine_step import NeutralFineStep

        for point, expected in [
            ([0.09, -0.035, 0], True),
            ([0.106, -0.035, 0], False),
            ([0.09, -0.019, 0], False),
        ]:
            self.assertEqual(in_training_box(point), expected)
        self.assertEqual(NeutralFineStep().status(0, {})["maximum_memory_age_s"], 4.5)

    def test_routing_runs_before_action_selection(self):
        s = (D / "executor.py").read_text()
        ast.parse(s)
        route = s.index("fine_route=route_at_step_end(")
        self.assertLess(s.index("fine_before=fine_step.state"), route)
        self.assertLess(
            route, s.index("kick_requested=handoff_state['kick_runtime_authorized']")
        )
        self.assertEqual(s.count("fine_step.capture("), 1)
        self.assertEqual(s.count("settled_entry("), 1)
        self.assertNotIn("fine_latch.evaluate(", s)

    def test_frozen_sources(self):
        p = D / "release.json"
        if not p.exists():
            self.skipTest("Pre-freeze check; run again after freeze")
        for path, sha in json.loads(p.read_text())["source_sha256"].items():
            self.assertEqual(hashlib.sha256(Path(path).read_bytes()).hexdigest(), sha)


if __name__ == "__main__":
    unittest.main()
