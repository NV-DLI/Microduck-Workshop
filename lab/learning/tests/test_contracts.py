"""Counterexamples for data leakage and full-task scoring; no GPU required."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.data import score_episode, segments, validate_segment_union, summarize
from learning.evaluate import assert_disjoint


class DataContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.episode = self.root / "episode"
        self.episode.mkdir()
        (self.episode / "learning").mkdir()
        report = dict(
            seed=1,
            scene_profile=dict(
                ball_diameter_m=0.1,
                goal_plane_x_m=0.6,
                goal_opening_width_m=0.388,
                goal_opening_height_m=0.2,
            ),
        )
        (self.episode / "report.json").write_text(json.dumps(report))
        self.rows = []
        for i in range(401):
            t = (i + 1) * 0.005
            self.rows.append(
                dict(
                    timestamp_s=t,
                    ball_contact_force_n=0.1 if i == 5 else 0,
                    ball_position=[0.64 if i < 20 else 0.66, 0, 0.05],
                    robot_position=[0.3, 0, 0.12],
                    linear_speed=0.01,
                    angular_speed=0.1,
                    gravity_z=-1.0,
                )
            )
        self.save()
        self.samples = [
            dict(source="learner", frame=i, timestamp_s=i * 0.08) for i in range(26)
        ]
        self.save_samples()

    def save_samples(self):
        (self.episode / "learning/samples.jsonl").write_text(
            "\n".join(json.dumps(r) for r in self.samples)
        )

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.episode / "physical-evaluation.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in self.rows)
        )

    def test_full_success(self):
        result = score_episode(self.episode)
        self.assertTrue(result["passed"])
        self.assertTrue(result["policy_unassisted_success"])

    def test_cancel_before_start_is_neither_physics_nor_failure(self):
        (self.root / "plan.json").write_text(json.dumps({"episodes": [{"seed": 1}]}))
        (self.root / "attempts.json").write_text(
            json.dumps([{"seed": 1, "exit_code": 0, "cancelled_before_start": True}])
        )
        result = summarize(self.root)
        self.assertEqual(result["cancelled_before_start"], 1)
        self.assertEqual(result["completed_physics_episodes"], 0)
        self.assertEqual(result["infrastructure_failures"], 0)
        self.assertIsNone(result["complete_task_success_rate"])

    def test_assistance_not_independent(self):
        self.samples[5]["source"] = "human"
        self.save_samples()
        result = score_episode(self.episode)
        self.assertTrue(result["passed"])
        self.assertFalse(result["policy_unassisted_success"])

    def test_overrun_and_return_rejected(self):
        self.rows[12]["robot_position"][0] = 0.7
        self.save()
        self.assertFalse(score_episode(self.episode)["passed"])

    def test_nan_physics_rejected(self):
        self.rows[10]["gravity_z"] = float("nan")
        self.save()
        with self.assertRaises(ValueError):
            score_episode(self.episode)

    def test_missing_physics_prefix_rejected(self):
        self.rows.pop(0)
        self.save()
        with self.assertRaises(ValueError):
            score_episode(self.episode)

    def test_incomplete_source_provenance_rejected(self):
        self.samples = self.samples[:1]
        self.save_samples()
        with self.assertRaises(ValueError):
            score_episode(self.episode)

    def test_center_crossing_not_whole_ball(self):
        for r in self.rows:
            r["ball_position"][0] = 0.62
        self.save()
        self.assertFalse(score_episode(self.episode)["passed"])

    def test_final_motion_rejects_earlier_stop(self):
        self.rows[-1]["linear_speed"] = 0.04
        self.save()
        self.assertFalse(score_episode(self.episode)["passed"])

    def test_prior_fall_rejected(self):
        self.rows[12]["gravity_z"] = 0.0
        self.save()
        self.assertFalse(score_episode(self.episode)["passed"])

    def test_missing_physics_tick(self):
        self.rows.pop(10)
        self.save()
        with self.assertRaises(ValueError):
            score_episode(self.episode)

    def test_split_same_seed_rejected(self):
        a = self.root / "train"
        b = self.root / "test"
        a.mkdir()
        b.mkdir()
        for p, digest in [(a, "one"), (b, "two")]:
            (p / "provenance.json").write_text(
                json.dumps([dict(seed=7, samples_sha256=digest)])
            )
        with self.assertRaises(ValueError):
            assert_disjoint(a, b)

    def test_split_same_source_rejected(self):
        a = self.root / "train"
        b = self.root / "test"
        a.mkdir()
        b.mkdir()
        for p, seed in [(a, 7), (b, 8)]:
            (p / "provenance.json").write_text(
                json.dumps([dict(seed=seed, samples_sha256="same")])
            )
        with self.assertRaises(ValueError):
            assert_disjoint(a, b)

    def test_correction_boundary_splits_chunks(self):
        p = self.episode / "learning"
        (p / "schema.json").write_text(json.dumps(dict(state_names=["a"])))
        (p / "frame.png").write_bytes(b"only file existence is checked here")
        rows = []
        for i in range(40):
            source = "learner" if 17 <= i < 21 else "human"
            rows.append(
                dict(
                    eligible_for_training=source == "human",
                    source=source,
                    correction=source == "human",
                    segment=0 if i < 17 else 1 if i < 21 else 2,
                    timestamp_s=i * 0.08,
                    state=[0.0],
                    action=[0.0] * 4,
                    image="frame.png",
                )
            )
        (p / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
        _, groups = segments(self.episode, "human", True)
        self.assertEqual([len(g) for g in groups], [17, 19])

    def test_cross_dataset_partial_overlap_rejected(self):
        records = [
            dict(samples_sha256="same", start_frame=0, end_frame=19),
            dict(samples_sha256="same", start_frame=10, end_frame=29),
        ]
        with self.assertRaises(ValueError):
            validate_segment_union(records)

    def test_disjoint_segments_and_visual_variants_allowed(self):
        validate_segment_union(
            [
                dict(samples_sha256="same", start_frame=0, end_frame=19),
                dict(samples_sha256="same", start_frame=20, end_frame=39),
                dict(
                    samples_sha256="same",
                    start_frame=0,
                    end_frame=19,
                    augmentation="reviewed-clip",
                ),
            ]
        )


if __name__ == "__main__":
    unittest.main()
