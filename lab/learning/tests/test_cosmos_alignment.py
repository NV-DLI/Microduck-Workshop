"""Reject visually accepted clips whose action alignment later changes."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.cosmos_data import validate_review, training_source_binding


class ReviewedClipAlignment(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.episode = self.root / "episode"
        (self.episode / "learning").mkdir(parents=True)
        self.clip = self.root / "clip"
        (self.clip / "generated").mkdir(parents=True)
        self.rows = [dict(frame=i, action=[i, 0, 0, 0]) for i in range(20)]
        source = self.episode / "learning/samples.jsonl"
        source.write_text("\n".join(json.dumps(row) for row in self.rows))
        self.dataset = self.root / "training-data"
        self.dataset.mkdir()
        (self.dataset / "split.json").write_text(json.dumps(dict(split="train")))
        (self.dataset / "provenance.json").write_text(
            json.dumps(
                [
                    dict(
                        source=str(self.episode),
                        split="train",
                        start_frame=0,
                        end_frame=19,
                        samples_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    )
                ]
            )
        )
        self.meta = dict(
            episode=str(self.episode),
            start_frame=1,
            frames=17,
            samples=self.rows[1:18],
            source_samples_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            training_source=training_source_binding(self.episode, 1, 17, self.dataset),
        )
        self.meta_path = self.clip / "generation.json"
        self.meta_path.write_text(json.dumps(self.meta))
        hashes = []
        for i in range(17):
            path = self.clip / "generated" / f"{i:04d}.png"
            path.write_bytes(f"frame-{i}".encode())
            hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
        self.review = dict(
            accepted=True,
            reviewer="Test reviewer",
            reason="Test geometry review",
            inspected_frames=list(range(17)),
            generated_frame_sha256=hashes,
            generation_sha256=hashlib.sha256(self.meta_path.read_bytes()).hexdigest(),
        )
        self.save_review()

    def save_review(self):
        (self.clip / "review.json").write_text(json.dumps(self.review))

    def tearDown(self):
        self.temp.cleanup()

    def test_unchanged_review_preserves_alignment(self):
        self.assertEqual(validate_review(self.clip)["samples"], self.rows[1:18])

    def test_truthy_non_boolean_acceptance_is_rejected(self):
        for value in ("false", "true", 1, [True]):
            with self.subTest(value=value):
                self.review["accepted"] = value
                self.save_review()
                with self.assertRaisesRegex(ValueError, "not been accepted"):
                    validate_review(self.clip)

    def test_validation_source_cannot_be_declared_as_training_augmentation(self):
        (self.dataset / "split.json").write_text(json.dumps(dict(split="validation")))
        with self.assertRaisesRegex(ValueError, "requires a training dataset"):
            validate_review(self.clip)

    def test_unadmitted_source_frame_cannot_be_augmented(self):
        path = self.dataset / "provenance.json"
        records = json.loads(path.read_text())
        records[0]["end_frame"] = 16
        path.write_text(json.dumps(records))
        with self.assertRaisesRegex(ValueError, "not admitted"):
            validate_review(self.clip)

    def test_changed_start_frame_rejected(self):
        self.meta["start_frame"] = 2
        self.meta_path.write_text(json.dumps(self.meta))
        with self.assertRaisesRegex(ValueError, "metadata changed"):
            validate_review(self.clip)

    def test_even_new_review_cannot_relabel_different_source_frames(self):
        self.meta["start_frame"] = 2
        self.meta_path.write_text(json.dumps(self.meta))
        self.review["generation_sha256"] = hashlib.sha256(
            self.meta_path.read_bytes()
        ).hexdigest()
        self.save_review()
        with self.assertRaisesRegex(ValueError, "alignment differs"):
            validate_review(self.clip)

    def test_one_changed_generated_frame_rejected(self):
        (self.clip / "generated/0008.png").write_bytes(b"replacement")
        with self.assertRaisesRegex(ValueError, "frames changed"):
            validate_review(self.clip)

    def test_eight_overview_frames_do_not_cover_whole_clip(self):
        self.review["inspected_frames"] = list(range(0, 17, 2))
        self.save_review()
        with self.assertRaisesRegex(ValueError, "complete clip"):
            validate_review(self.clip)


if __name__ == "__main__":
    unittest.main()
