"""CPU training-contract checks; not physical skill acceptance."""
import unittest
from types import SimpleNamespace
import torch
from rsl_rl.modules.distribution import GaussianDistribution
from exploration_reset_v8 import reset_exploration


class ExplorationResetTest(unittest.TestCase):
    def actor(self):
        actor = torch.nn.Module()
        actor.mean = torch.nn.Linear(61, 14)
        actor.distribution = GaussianDistribution(14, init_std=.8, std_range=(.03, .1))
        optimizer = torch.optim.Adam(actor.parameters(), lr=.001)
        loss = actor.mean(torch.ones(2, 61)).sum() + actor.distribution.std_param.sum()
        loss.backward()
        optimizer.step()
        return SimpleNamespace(actor=actor, optimizer=optimizer)

    def test_preserves_policy_and_other_moments(self):
        algorithm = self.actor()
        x = torch.randn(32, 61)
        before = algorithm.actor.mean(x).detach().clone()
        mean_state = algorithm.optimizer.state[algorithm.actor.mean.weight]
        saved_moment = mean_state['exp_avg'].clone()
        report = reset_exploration(algorithm)
        self.assertTrue(torch.equal(before, algorithm.actor.mean(x)))
        self.assertTrue(torch.equal(saved_moment, mean_state['exp_avg']))
        self.assertNotIn(algorithm.actor.distribution.std_param, algorithm.optimizer.state)
        self.assertEqual(report['noise_gradient_channels_nonzero'], 14)
        self.assertTrue(report['all_non_noise_actor_tensors_unchanged'])

    def test_rejects_clamp_boundaries(self):
        for value in (.03, .1, .2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reset_exploration(self.actor(), initial_std=value)

    def test_fixed_distribution_is_not_silently_changed(self):
        algorithm = self.actor()
        algorithm.actor.distribution.std_param.requires_grad_(False)
        with self.assertRaises(ValueError):
            reset_exploration(algorithm)


if __name__ == '__main__':
    unittest.main()
