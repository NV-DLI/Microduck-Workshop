import unittest
from ball_reacquisition import BallSearch


class DeepSearch(unittest.TestCase):
    def test_far_view_seed_covers_close_pitch_without_metric_authority(self):
        s = BallSearch()
        s.begin(1.0, [-0.029, 0.538])
        self.assertAlmostEqual(s.targets[1][1], 1.1)
        self.assertAlmostEqual(s.targets[2][1], 1.3)
        self.assertTrue(all(0 <= t[1] <= 1.4 for t in s.targets))
        self.assertEqual(s.started, 1.0)

    def test_existing_near_search_is_retained(self):
        s = BallSearch()
        s.begin(1.0, [-0.4, 1.103])
        self.assertEqual(len(s.domain_checks), 9)
