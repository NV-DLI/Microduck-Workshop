import unittest
import torch
from dual_policy_v12 import DualPolicy


class DualPolicyTest(unittest.TestCase):
    def test_same_current_observation_and_smooth_switch(self):
        seen = []
        def actor(x):
            seen.append(x.clone())
            return x[:, 34:48]
        model = DualPolicy(actor, lambda x: actor(x) + 1, 3, 'cpu')
        obs = torch.zeros(2, 61)
        obs[:, 34:48] = torch.arange(14).float()
        ids = torch.tensor([0, 2])
        first = model(obs, ids)
        self.assertTrue(torch.allclose(first, obs[:, 34:48] + .05))
        self.assertTrue(torch.equal(seen[0], seen[1]))
        self.assertEqual(model.alpha[1].item(), 0)
        for _ in range(20): model(obs, ids)
        self.assertTrue(torch.equal(model.alpha[ids], torch.ones(2)))
        obs[:, 48] = .06
        model(obs, ids)
        self.assertTrue(torch.allclose(model.alpha[ids], torch.full((2,), .95)))

    def test_identical_policies_preserve_raw_action_exactly(self):
        torch.manual_seed(9871)
        actor = lambda x: x[:, 34:48].sin()
        model = DualPolicy(actor, actor, 16, 'cpu')
        for _ in range(100):
            obs = torch.randn(16, 61)
            obs[:8, 48:51] = 0
            self.assertTrue(torch.equal(model(obs, torch.arange(16)), actor(obs)))

    def test_lost_world_never_updates_or_infers_from_zero_sentinel(self):
        calls = []
        def actor(x):
            calls.append(len(x)); return torch.ones(len(x), 14)
        model = DualPolicy(actor, actor, 3, 'cpu')
        model(torch.zeros(1, 61), torch.tensor([2]))
        self.assertEqual(calls, [1, 1])
        self.assertTrue(torch.equal(model.alpha[:2], torch.zeros(2)))


if __name__ == '__main__':
    unittest.main()
