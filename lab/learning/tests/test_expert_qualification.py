import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qualify_expert import make_plan, qualify


class ProspectiveExpertQualification(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = dict(mode="policy", automatic_release_brake=True)
        config = self.root / "controller.json"
        config.write_text(json.dumps(self.cfg))
        self.schema = dict(
            configuration=self.cfg,
            policy_identity=dict(
                checkpoint_sha256="fixed", native_horizon_contract={"steps": 16}
            ),
            control_contract={"version": 2},
        )
        ref = self.root / "reference"
        (ref / "learning").mkdir(parents=True)
        (ref / "learning/schema.json").write_text(json.dumps(self.schema))
        (ref / "report.json").write_text(
            json.dumps(dict(scene_profile={"mat_style": "A"}))
        )
        self.protocol = self.root / "protocol.json"
        proto = make_plan(config, 80000, 3, ref)
        self.protocol.write_text(json.dumps(proto))
        self.collection = self.root / "collection"
        self.collection.mkdir()
        from qualify_expert import digest

        sha = digest(self.protocol)
        (self.collection / "expert-protocol.json").write_bytes(
            self.protocol.read_bytes()
        )
        (self.collection / "plan.json").write_text(
            json.dumps(
                dict(
                    controller=self.cfg,
                    episodes=proto["fixtures"],
                    seconds=24.0,
                    style="A",
                    position_jitter_m=0.002,
                    expert_protocol_sha256=sha,
                )
            )
        )
        attempts = []
        expert = Path(__file__).resolve().parents[1] / "recovery.py"
        import time

        (self.collection / "launch-records").mkdir()
        for i in range(3):
            started = time.time()
            ep = self.collection / f"episode-{80000+i}"
            (ep / "learning").mkdir(parents=True)
            (ep / "source-snapshot/learning").mkdir(parents=True)
            (ep / "source-snapshot/learning/recovery.py").write_bytes(
                expert.read_bytes()
            )
            (ep / "learning/schema.json").write_text(json.dumps(self.schema))
            fixture = proto["fixtures"][i]
            (ep / "report.json").write_text(
                json.dumps(
                    dict(
                        scene_profile={"mat_style": "A"},
                        fixture=dict(root_xy=[fixture["root_x"], fixture["root_y"]]),
                    )
                )
            )
            rows = [
                dict(
                    frame=0,
                    timestamp_s=0,
                    source="learner",
                    correction=False,
                    state=[i],
                ),
                dict(
                    frame=1,
                    timestamp_s=0.08,
                    source="automatic_expert",
                    correction=True,
                ),
            ]
            (ep / "learning/samples.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            registration = dict(
                expert_protocol_sha256=sha, started_at_unix=started, seed=80000 + i
            )
            (ep / "expert-protocol-registration.json").write_text(
                json.dumps(registration)
            )
            (self.collection / "launch-records" / f"episode-{80000+i}.json").write_text(
                json.dumps(registration)
            )
            attempts.append(dict(**registration, path=str(ep), exit_code=0))
        (self.collection / "attempts.json").write_text(json.dumps(attempts))
        self.gate_patch = patch("qualify_expert.verify_intervention")
        self.gate_patch.start()

    def tearDown(self):
        self.gate_patch.stop()
        self.temp.cleanup()

    def score(self, ep):
        return dict(
            physical_success=True,
            control_sources=["learner", "automatic_expert"],
            seed=int(ep.name.split("-")[1]),
            duration_s=24.0,
            fallen=False,
        )

    def test_explicit_aligned_fixture_parameters(self):
        result = make_plan(
            self.root / "controller.json",
            90000,
            3,
            self.root / "reference",
            seconds=20.0,
            root_x=-0.2,
            root_y=0.055,
            position_jitter=0.0,
        )
        self.assertEqual(result["seconds"], 20.0)
        self.assertEqual(result["position_jitter_m"], 0.0)
        self.assertTrue(
            all(
                r["root_x"] == -0.2 and r["root_y"] == 0.055 for r in result["fixtures"]
            )
        )

    def test_invalid_fixture_parameters_rejected(self):
        for kwargs in [
            {"root_y": float("nan")},
            {"seconds": 0},
            {"position_jitter": -0.01},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_plan(
                    self.root / "controller.json",
                    90000,
                    3,
                    self.root / "reference",
                    **kwargs,
                )

    def test_successful_interventions_qualify_without_counterfactual_claim(self):
        with patch("qualify_expert.score_episode", side_effect=self.score):
            result = qualify(self.collection, self.protocol)
        self.assertTrue(result["passed"])
        self.assertEqual(result["triggered_episodes"], 3)
        self.assertNotIn("rescued", result)

    def test_no_intervention_cannot_qualify_expert(self):
        for ep in self.collection.glob("episode-*"):
            path = ep / "learning/samples.jsonl"
            path.write_text(path.read_text().splitlines()[0] + "\n")
        with patch("qualify_expert.score_episode", side_effect=self.score):
            result = qualify(self.collection, self.protocol)
        self.assertFalse(result["passed"])

    def test_one_frame_failed_intervention_stays_in_denominator(self):
        def score(ep):
            value = self.score(ep)
            value["physical_success"] = not str(ep).endswith("80002")
            return value

        with patch("qualify_expert.score_episode", side_effect=score):
            result = qualify(self.collection, self.protocol)
        self.assertFalse(result["passed"])
        self.assertEqual(result["triggered_episodes"], 3)
        self.assertEqual(result["assisted_completions"], 2)

    def test_missing_attempt_rejected(self):
        p = self.collection / "attempts.json"
        p.write_text(json.dumps(json.loads(p.read_text())[:-1]))
        with self.assertRaisesRegex(ValueError, "Complete every"):
            qualify(self.collection, self.protocol)

    def test_changed_expert_rejected(self):
        (
            self.collection / "episode-80000/source-snapshot/learning/recovery.py"
        ).write_text("changed")
        with self.assertRaisesRegex(ValueError, "Expert changed"):
            qualify(self.collection, self.protocol)

    def test_mixed_policy_rejected(self):
        p = self.collection / "episode-80000/learning/schema.json"
        data = json.loads(p.read_text())
        data["policy_identity"]["checkpoint_sha256"] = "other"
        p.write_text(json.dumps(data))
        with patch("qualify_expert.score_episode", side_effect=self.score):
            with self.assertRaisesRegex(ValueError, "one frozen model"):
                qualify(self.collection, self.protocol)


class InterventionReplayChecks(unittest.TestCase):
    def setUp(self):
        import copy

        self.temp = tempfile.TemporaryDirectory()
        self.ep = Path(self.temp.name)
        (self.ep / "learning").mkdir()
        self.truth = [
            dict(
                robot_position=[0, 0, 0.12],
                ball_position=[0.11, 0, 0.05],
                ball_velocity=[0.5, 0, 0],
                forward_velocity=0.2,
                gravity_z=-1.0,
                angular_speed=0.0,
                timestamp_s=0.0,
                recovery_active=False,
                trigger=None,
            )
        ]
        self.truth.append(copy.deepcopy(self.truth[0]))
        self.truth[1].update(
            timestamp_s=0.08,
            recovery_active=True,
            trigger=dict(
                timestamp_s=0.08,
                projected_ball_y=0.0,
                reason="goal_directed_ball_released_while_policy_still_propels",
            ),
        )
        self.rows = [
            dict(
                timestamp_s=0.0,
                source="learner",
                correction=False,
                learner_proposal=[0.0, 0.0, 0.0, 0.0],
                action=[0.0, 0.0, 0.0, 0.0],
            ),
            dict(
                timestamp_s=0.08,
                source="automatic_expert",
                correction=True,
                learner_proposal=[0.6, 0.0, 0.0, 0.0],
                action=[0.0, 0.0, 0.0, 0.0],
            ),
        ]
        self.physics = [
            dict(
                timestamp_s=(i + 1) * 0.005,
                robot_position=[0, 0, 0.12],
                ball_position=[0.11, 0, 0.05],
            )
            for i in range(16)
        ]

    def tearDown(self):
        self.temp.cleanup()

    def verify(self):
        from qualify_expert import verify_intervention

        for file, rows in [
            ("learning/expert-truth.jsonl", self.truth),
            ("physical-evaluation.jsonl", self.physics),
        ]:
            (self.ep / file).write_text("".join(json.dumps(r) + "\n" for r in rows))
        verify_intervention(self.ep, self.rows)

    def test_legal_intervention_matches_gate(self):
        self.verify()

    def test_early_source_relabel_is_rejected(self):
        self.rows[0].update(source="automatic_expert", correction=True)
        with self.assertRaisesRegex(ValueError, "attribution"):
            self.verify()

    def test_nonzero_expert_action_is_rejected(self):
        self.rows[1]["action"][0] = 0.6
        with self.assertRaisesRegex(ValueError, "zero requests"):
            self.verify()

    def test_nan_expert_action_rejected(self):
        self.rows[1]["action"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.verify()

    def test_false_trigger_rejected(self):
        self.truth[1]["ball_velocity"][0] = 0.1
        with self.assertRaisesRegex(ValueError, "gate replay"):
            self.verify()

    def test_privileged_position_must_match_physics(self):
        self.truth[1]["ball_position"][1] = 0.2
        with self.assertRaisesRegex(ValueError, "physical trajectory"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
