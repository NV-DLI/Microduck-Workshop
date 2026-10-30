import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualify_recovery import qualify


class QualificationCounterexamples(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directories = [self.root / "baseline", self.root / "assisted"]
        for directory in self.directories:
            directory.mkdir()
            plan = dict(episodes=[dict(seed=i) for i in range(3)], seconds=2, style="A")
            (directory / "plan.json").write_text(json.dumps(plan))
            attempts = []
            for seed in range(3):
                ep = directory / f"episode-{seed}"
                (ep / "learning").mkdir(parents=True)
                (ep / "source-snapshot/learning").mkdir(parents=True)
                (ep / "source-snapshot/learning/recovery.py").write_text(
                    "frozen expert"
                )
                schema = dict(
                    policy_identity=dict(
                        checkpoint_sha256="same",
                        native_horizon_contract={"native_prediction_horizon": 16},
                    ),
                    control_contract={"version": 2},
                )
                (ep / "learning/schema.json").write_text(json.dumps(schema))
                (ep / "report.json").write_text(
                    json.dumps(dict(scene_profile={"mat_style": "A"}))
                )
                (ep / "learning/samples.jsonl").write_text(
                    json.dumps(dict(source="learner", correction=False, timestamp_s=0))
                    + "\n"
                )
                (ep / "physical-evaluation.jsonl").write_text(
                    json.dumps(
                        dict(
                            timestamp_s=0.005,
                            robot_position=[0, 0, 0.12],
                            ball_position=[0, 0, 0.05],
                        )
                    )
                    + "\n"
                )
                attempts.append(dict(seed=seed, path=str(ep), exit_code=0))
            (directory / "attempts.json").write_text(json.dumps(attempts))

    def tearDown(self):
        self.temp.cleanup()

    def test_success_without_intervention_does_not_qualify_expert(self):
        with patch(
            "qualify_recovery.score_episode",
            return_value=dict(
                physical_success=True,
                policy_unassisted_success=True,
                control_sources=["learner"],
            ),
        ):
            result = qualify(*self.directories)
        self.assertFalse(result["passed"])
        self.assertEqual(result["triggered_matched_pairs"], 0)

    def test_changed_model_rejects_pair(self):
        path = self.directories[1] / "episode-0/learning/schema.json"
        schema = json.loads(path.read_text())
        schema["policy_identity"]["checkpoint_sha256"] = "different"
        path.write_text(json.dumps(schema))
        with self.assertRaisesRegex(ValueError, "same frozen model"):
            qualify(*self.directories)

    def test_same_weights_different_inference_horizon_rejected(self):
        for directory, horizon in zip(self.directories, [40, 16]):
            path = directory / "episode-0/learning/schema.json"
            schema = json.loads(path.read_text())
            schema["policy_identity"]["native_horizon_contract"] = {
                "native_prediction_horizon": horizon
            }
            path.write_text(json.dumps(schema))
        with self.assertRaisesRegex(ValueError, "same frozen model"):
            qualify(*self.directories)

    def test_same_weights_with_different_request_mapping_rejected(self):
        path = self.directories[1] / "episode-0/learning/schema.json"
        schema = json.loads(path.read_text())
        schema["policy_identity"]["inference_output_mapping"] = "demonstrated forward"
        path.write_text(json.dumps(schema))
        with self.assertRaisesRegex(ValueError, "same frozen model"):
            qualify(*self.directories)

    def test_both_missing_inference_horizons_rejected(self):
        for directory in self.directories:
            path = directory / "episode-0/learning/schema.json"
            schema = json.loads(path.read_text())
            del schema["policy_identity"]["native_horizon_contract"]
            path.write_text(json.dumps(schema))
        with self.assertRaisesRegex(ValueError, "same frozen model"):
            qualify(*self.directories)

    def test_truncated_baseline_cannot_match_later_intervention(self):
        episode = self.directories[1] / "episode-0"
        (episode / "learning/samples.jsonl").write_text(
            json.dumps(
                dict(
                    source="automatic_expert",
                    correction=True,
                    timestamp_s=1,
                    action=[0.0, 0.0, 0.0, 0.0],
                )
            )
            + "\n"
        )
        physical = episode / "physical-evaluation.jsonl"
        row = json.loads(physical.read_text())
        row["timestamp_s"] = 1.005
        with physical.open("a") as f:
            f.write(json.dumps(row) + "\n")
        with patch(
            "qualify_recovery.score_episode",
            return_value=dict(
                physical_success=True,
                policy_unassisted_success=True,
                control_sources=["learner"],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "cover the intervention time"):
                qualify(*self.directories)

    def test_contact_completion_cannot_use_braking_qualification(self):
        path = self.directories[1] / "episode-0/learning/schema.json"
        schema = json.loads(path.read_text())
        schema["configuration"] = {"automatic_contact_completion": True}
        path.write_text(json.dumps(schema))
        with self.assertRaisesRegex(ValueError, "own qualification"):
            qualify(*self.directories)

    def test_changed_scope_or_nonzero_expert_action_rejected(self):
        for scope, action, message in [
            (
                "near_ball_contact_completion_bounded",
                [0.0, 0.0, 0.0, 0.0],
                "another correction scope",
            ),
            ("released_ball_braking_only", [0.6, 0.0, 0.0, 0.0], "zero action"),
        ]:
            with self.subTest(scope=scope, action=action):
                path = self.directories[1] / "episode-0/learning/samples.jsonl"
                path.write_text(
                    json.dumps(
                        dict(
                            source="automatic_expert",
                            correction=True,
                            timestamp_s=0,
                            recovery_scope=scope,
                            action=action,
                        )
                    )
                    + "\n"
                )
                with patch(
                    "qualify_recovery.score_episode",
                    return_value=dict(
                        physical_success=True,
                        policy_unassisted_success=True,
                        control_sources=["learner"],
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        qualify(*self.directories)

    def test_missing_attempt_rejects_qualification(self):
        path = self.directories[1] / "attempts.json"
        path.write_text(json.dumps(json.loads(path.read_text())[:-1]))
        with self.assertRaisesRegex(ValueError, "complete"):
            qualify(*self.directories)

    def test_different_planned_lighting_rejects_pair(self):
        path = self.directories[1] / "plan.json"
        plan = json.loads(path.read_text())
        plan["lighting"] = "warm"
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "matched"):
            qualify(*self.directories)

    def test_actual_lighting_cannot_hide_behind_matching_plans(self):
        path = self.directories[1] / "episode-0/report.json"
        report = json.loads(path.read_text())
        report["scene_profile"]["lighting"] = "warm"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "lighting"):
            qualify(*self.directories)


if __name__ == "__main__":
    unittest.main()
