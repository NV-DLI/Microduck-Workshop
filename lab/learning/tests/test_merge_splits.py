"""Prevent held-out episodes from silently becoming training data during aggregation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.data import merge


class MergeSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, name, split, source_split=None):
        root = self.root / name
        root.mkdir()
        (root / "split.json").write_text(json.dumps({"split": split}))
        (root / "control-contract.json").write_text(json.dumps({"actions": 4}))
        (root / "provenance.json").write_text(
            json.dumps(
                [
                    dict(
                        source=name,
                        samples_sha256=name,
                        start_frame=0,
                        end_frame=15,
                        split=source_split or split,
                        frames=16,
                    )
                ]
            )
        )
        return root

    def run_merge(self, inputs, split="train"):
        with patch("lerobot.datasets.lerobot_dataset.LeRobotDataset"), patch(
            "lerobot.datasets.dataset_tools.merge_datasets",
            side_effect=lambda *args, output_dir: output_dir.mkdir(),
        ):
            return merge(inputs, self.root / "combined", split)

    def test_validation_keeps_held_out_split_and_lineage(self):
        inputs = [self.dataset(n, "validation") for n in ["first", "second"]]
        self.run_merge(inputs, "validation")
        root = self.root / "combined"
        self.assertEqual(
            json.loads((root / "split.json").read_text()), {"split": "validation"}
        )
        self.assertEqual(
            [r["split"] for r in json.loads((root / "provenance.json").read_text())],
            ["validation", "validation"],
        )

    def test_validation_cannot_enter_default_training_merge(self):
        inputs = [
            self.dataset("first", "validation"),
            self.dataset("second", "validation"),
        ]
        with self.assertRaises(ValueError):
            self.run_merge(inputs)
        self.assertFalse((self.root / "combined").exists())

    def test_training_or_corrections_cannot_enter_validation(self):
        for source in ["train", "correction"]:
            with self.subTest(source=source):
                inputs = [
                    self.dataset(source, source),
                    self.dataset(source + "-val", "validation"),
                ]
                with self.assertRaises(ValueError):
                    self.run_merge(inputs, "validation")

    def test_renamed_split_does_not_hide_training_lineage(self):
        inputs = [
            self.dataset("first", "validation", "train"),
            self.dataset("second", "validation"),
        ]
        with self.assertRaises(ValueError):
            self.run_merge(inputs, "validation")

    def test_original_training_and_correction_merge_remains_training(self):
        inputs = [self.dataset("first", "train"), self.dataset("second", "correction")]
        self.run_merge(inputs)
        self.assertEqual(
            json.loads((self.root / "combined/split.json").read_text())["split"],
            "train",
        )


if __name__ == "__main__":
    unittest.main()
