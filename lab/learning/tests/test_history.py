"""Keep the deployed temporal observation causal across control ownership changes."""

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.history import CausalHistory, transform_rows


def trace():
    return [
        dict(
            frame=k,
            timestamp_s=k * 0.08,
            state=[float(k * 100 + j) for j in range(37)],
            action=[float(k + j) for j in range(4)],
            source="learner" if k < 28 else "automatic_expert",
        )
        for k in range(60)
    ]


class HistoryTests(unittest.TestCase):
    def test_full_episode_formula_including_authority_change(self):
        rows = trace()
        actual = transform_rows(rows)
        for k, row in enumerate(actual):
            expected = sum((rows[max(0, k - lag)]["state"] for lag in (0, 12, 24)), [])
            expected += rows[k - 1]["action"] if k else [0.0] * 4
            self.assertEqual(row["state"], expected)
            self.assertEqual(row["action"], rows[k]["action"])
            self.assertEqual(row["source"], rows[k]["source"])

    def test_current_label_and_future_cannot_change_current_observation(self):
        rows = trace()
        expected = transform_rows(rows)[28]["state"]
        changed = copy.deepcopy(rows)
        changed[28]["action"] = [999.0] * 4
        for row in changed[29:]:
            row["state"] = [-999.0] * 37
            row["action"] = [-999.0] * 4
        self.assertEqual(transform_rows(changed)[28]["state"], expected)
        changed[27]["action"] = [123.0] * 4
        actual = transform_rows(changed)[28]["state"]
        self.assertEqual(actual[:-4], expected[:-4])
        self.assertEqual(actual[-4:], [123.0] * 4)

    def test_reset_does_not_reuse_a_previous_episode(self):
        history = CausalHistory()
        history.observe([99.0] * 37, [0.0] * 4, 0)
        other = CausalHistory()
        self.assertEqual(
            other.observe([2.0] * 37, [0.0] * 4, 0), [2.0] * 111 + [0.0] * 4
        )

    def test_rejects_filtered_or_nonfinite_episode(self):
        rows = trace()
        for invalid in (rows[1:], rows[:10] + rows[11:], rows[:10] + rows[9:]):
            with self.assertRaises(ValueError):
                transform_rows(invalid)
        with self.assertRaises(ValueError):
            CausalHistory().observe([0.0] * 37, [1.0] * 4, 0)
        with self.assertRaises(ValueError):
            CausalHistory().observe([float("nan")] * 37, [0.0] * 4, 0)


if __name__ == "__main__":
    unittest.main()
