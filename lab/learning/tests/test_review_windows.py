"""Bound accepted correction labels without changing expert qualification."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.review import qualified_manifest


class ReviewWindows(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "qualification.json"
        self.evidence = dict(
            passed=True,
            protocol="prospective_assisted_completion_v1",
            episodes=[
                dict(
                    assisted_success=True,
                    intervention_frames=100,
                    assisted_episode="/example",
                    seed=1,
                    samples_sha256="bound",
                )
            ],
        )
        self.path.write_text(json.dumps(self.evidence))
        self.rows = [dict(frame=i, segment=1) for i in range(50, 150)]

    def tearDown(self):
        self.temp.cleanup()

    def test_cap_retains_contiguous_actual_prefix_and_qualification(self):
        with patch(
            "learning.review.validate_qualification", return_value="qualification-sha"
        ), patch("learning.review.segments", return_value=({}, [self.rows])):
            result = qualified_manifest(self.path, 64)
        row = result[0]["accepted_segments"][0]
        self.assertEqual((row["start_frame"], row["end_frame"]), (50, 113))
        self.assertEqual(row["qualification_sha256"], "qualification-sha")
        self.assertEqual(row["source_samples_sha256"], "bound")

    def test_default_preserves_full_segment(self):
        with patch(
            "learning.review.validate_qualification", return_value="qualification-sha"
        ), patch("learning.review.segments", return_value=({}, [self.rows])):
            row = qualified_manifest(self.path)[0]["accepted_segments"][0]
        self.assertEqual((row["start_frame"], row["end_frame"]), (50, 149))

    def test_short_window_and_failed_qualification_rejected(self):
        with self.assertRaises(ValueError):
            qualified_manifest(self.path, 15)
        self.evidence["passed"] = False
        self.path.write_text(json.dumps(self.evidence))
        with self.assertRaises(ValueError):
            qualified_manifest(self.path, 64)


if __name__ == "__main__":
    unittest.main()
