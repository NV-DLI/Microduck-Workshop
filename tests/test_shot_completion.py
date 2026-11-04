"""Behavioral termination checks: missed kicks and early failures keep time running."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("shot_completion", Path(__file__).resolve().parents[1] / "lab/onboard/shot_completion.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ShotCompletionTest(unittest.TestCase):
    def run_samples(self, shot, start, stop, *, contact=False, complete=False, moving=False):
        result = None
        for i in range(start, stop):
            t = i / 100
            result = shot.update(now_s=t, task_s=t, ball_xyz=[t if moving else 0, 0, .05],
                                 kick_contact=contact, kick_completed=complete)
        return result

    def test_ball_at_rest_without_kick_does_not_finish(self):
        s = module.ShotCompletion(60)
        self.assertIsNone(self.run_samples(s, 0, 6000))
        self.assertEqual(self.run_samples(s, 6000, 6001), "task_duration_limit")

    def test_completed_kick_without_contact_waits_for_timeout(self):
        s = module.ShotCompletion(60)
        self.assertIsNone(self.run_samples(s, 0, 6000, complete=True))
        self.assertEqual(self.run_samples(s, 6000, 6001, complete=True), "task_duration_limit")

    def test_contact_during_incomplete_action_cannot_finish(self):
        s = module.ShotCompletion(60)
        self.assertIsNone(self.run_samples(s, 0, 500, contact=True))

    def test_goal_and_miss_both_finish_only_after_sustained_rest(self):
        # There is deliberately no goal flag in this termination interface.
        s = module.ShotCompletion(60)
        self.assertIsNone(self.run_samples(s, 0, 500, contact=True, complete=True, moving=True))
        self.assertIsNone(self.run_samples(s, 500, 551, contact=True, complete=True))
        self.assertEqual(self.run_samples(s, 551, 552, contact=True, complete=True), "shot_complete_ball_settled")

    def test_motion_resets_rest_timer(self):
        s = module.ShotCompletion(60)
        self.run_samples(s, 0, 50, contact=True, complete=True)
        self.run_samples(s, 50, 51, contact=True, complete=True, moving=True)
        self.assertIsNone(self.run_samples(s, 51, 102, complete=True))
        self.assertEqual(self.run_samples(s, 102, 103, complete=True), "shot_complete_ball_settled")

    def test_gap_does_not_count_as_observed_rest(self):
        s = module.ShotCompletion(60)
        self.run_samples(s, 0, 50, contact=True, complete=True)
        self.assertIsNone(self.run_samples(s, 500, 550, contact=True, complete=True))

    def test_deadline_is_not_extended_by_late_contact(self):
        s = module.ShotCompletion(60)
        self.run_samples(s, 0, 5975)
        self.assertEqual(self.run_samples(s, 5975, 6001, contact=True, complete=True), "task_duration_limit")

    def test_nonfinite_and_backwards_time_rejected(self):
        s = module.ShotCompletion(60)
        self.run_samples(s, 0, 2)
        with self.assertRaises(ValueError):
            self.run_samples(s, 0, 1)
        with self.assertRaises(ValueError):
            s.update(now_s=.02, task_s=.02, ball_xyz=[float('nan'), 0, 0], kick_contact=False, kick_completed=False)


if __name__ == "__main__":
    unittest.main()
