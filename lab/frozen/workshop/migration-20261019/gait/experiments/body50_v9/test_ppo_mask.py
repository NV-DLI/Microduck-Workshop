"""CPU tests against actual installed RSL PPO, models and optimizer."""
import copy
from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

from ppo_mask import install


def observations(values):
    return TensorDict({'policy': torch.tensor(values).float()[:, None].repeat(1, 61)},
                      batch_size=[len(values)])


def make(n=3, steps=3):
    torch.manual_seed(741)
    obs = observations([0.] * n)
    groups = {'actor': ['policy'], 'critic': ['policy']}
    actor = MLPModel(obs, groups, 'actor', 14, hidden_dims=[16], obs_normalization=True,
        distribution_cfg={'class_name': 'rsl_rl.modules.distribution.GaussianDistribution', 'init_std': .05})
    critic = MLPModel(obs, groups, 'critic', 1, hidden_dims=[16], obs_normalization=True)
    storage = RolloutStorage('rl', n, steps, obs, [14])
    alg = PPO(actor, critic, storage, num_learning_epochs=2, num_mini_batches=4,
              schedule='fixed', gamma=.9, lam=.8, normalize_advantage_per_mini_batch=False)
    env = SimpleNamespace(ready_mask=torch.zeros(n, dtype=torch.bool),
                          task_active_mask=torch.zeros(n, dtype=torch.bool))
    return alg, env


def step(alg, obs, rewards, dones=None):
    with torch.inference_mode():
        action = alg.act(obs)
        alg.process_env_step(obs, torch.tensor(rewards),
            torch.zeros(len(rewards), dtype=torch.bool) if dones is None else torch.tensor(dones), {})
    return action


class TestMaskedPPO(unittest.TestCase):
    def test_install_preserves_parent_parameters_normalizer_and_optimizer(self):
        alg, env = make()
        loss = alg.actor(observations([1., 2., 3.])).sum()
        loss.backward(); alg.optimizer.step()
        actor = copy.deepcopy(alg.actor.state_dict())
        critic = copy.deepcopy(alg.critic.state_dict())
        optimizer = copy.deepcopy(alg.optimizer.state_dict())
        install(alg, env)
        for key, value in actor.items():
            self.assertTrue(torch.equal(value, alg.actor.state_dict()[key]))
        for key, value in critic.items():
            self.assertTrue(torch.equal(value, alg.critic.state_dict()[key]))
        self.assertEqual(optimizer['param_groups'], alg.optimizer.state_dict()['param_groups'])
        for key, value in optimizer['state'].items():
            for name, tensor in value.items():
                self.assertTrue(torch.equal(tensor, alg.optimizer.state_dict()['state'][key][name]))

    def test_all_initialization_never_calls_network_or_optimizer(self):
        alg, env = make()
        audit = install(alg, env)
        calls = []
        handles = [m.register_forward_pre_hook(lambda *x: calls.append('forward')) for m in (alg.actor, alg.critic)]
        initial = copy.deepcopy(alg.actor.state_dict())
        for _ in range(3):
            actions = step(alg, observations([float('nan')] * 3), [float('nan')] * 3)
            self.assertEqual(int(actions.count_nonzero()), 0)
        with torch.inference_mode():
            alg.compute_returns(observations([float('nan')] * 3))
        result = alg.update()
        self.assertEqual(calls, [])
        self.assertEqual(result['masked_optimizer_steps'], 0.)
        self.assertEqual(audit.report()['all_not_ready_rollouts'], 1)
        self.assertEqual(alg.optimizer.state, {})
        for key, value in initial.items():
            self.assertTrue(torch.equal(value, alg.actor.state_dict()[key]))
        for h in handles: h.remove()

    def test_mixed_mask_action_time_snapshot_release_and_normalizer(self):
        alg, env = make()
        audit = install(alg, env)
        network_observations = []
        def inspect(module, args):
            x = args[0]['policy']
            self.assertTrue(torch.isfinite(x).all())
            network_observations.append(x[:, 0].tolist())
        handles = [m.register_forward_pre_hook(inspect) for m in (alg.actor, alg.critic)]
        env.ready_mask[:] = torch.tensor([True, False, True])
        env.task_active_mask[:] = torch.tensor([True, False, False])
        with torch.inference_mode():
            obs = observations([1., float('nan'), 10.])
            action = alg.act(obs)
            self.assertEqual(int(action[1].count_nonzero()), 0)
            self.assertGreater(int(action[2].count_nonzero()), 0)
            # The post-step env says env1 is active; that must NOT retroactively
            # turn its pre-step placeholder into a training transition.
            env.ready_mask[:] = True
            env.task_active_mask[:] = True
            alg.process_env_step(obs, torch.tensor([1., float('nan'), 50.]), torch.zeros(3).bool(), {})
        env.task_active_mask[:] = torch.tensor([True, True, False])
        step(alg, observations([2., 3., 11.]), [2., 3., 60.])
        env.ready_mask[:] = torch.tensor([False, True, True])
        env.task_active_mask[:] = torch.tensor([False, True, True])
        step(alg, observations([float('nan'), 4., 12.]), [float('nan'), 4., 12.])
        self.assertEqual(int(audit.valid.sum()), 5)
        with torch.inference_mode():
            alg.compute_returns(observations([float('nan'), 5., 13.]))
        result = alg.update()
        self.assertEqual(result['masked_optimizer_steps'], 8.)
        self.assertEqual(audit.report()['task_transitions'], 5)
        self.assertEqual(audit.report()['ready_normalizer_rows'], 7)
        self.assertEqual(audit.report()['release_excluded_transitions'], 2)
        self.assertEqual(int(alg.actor.obs_normalizer.count), 7)
        self.assertAlmostEqual(float(alg.actor.obs_normalizer.mean[0]), 43/7, places=5)
        self.assertTrue(all(torch.isfinite(p).all() for p in alg.actor.parameters()))
        self.assertTrue(alg.optimizer.state)
        for h in handles: h.remove()

    def test_no_bootstrap_across_invalid_gap_and_real_done(self):
        alg, env = make(n=2)
        audit = install(alg, env)
        with torch.no_grad():
            for p in alg.critic.parameters(): p.zero_()
            alg.critic.mlp[-1].bias.fill_(2.)
        env.ready_mask[:] = True; env.task_active_mask[:] = True
        step(alg, observations([1., 2.]), [1., 1.], [False, True])
        env.ready_mask[0] = False; env.task_active_mask[0] = False
        step(alg, observations([float('nan'), 3.]), [float('nan'), 2.])
        env.ready_mask[:] = True; env.task_active_mask[:] = True
        step(alg, observations([4., 4.]), [4., 3.])
        with torch.inference_mode(): alg.compute_returns(observations([5., 5.]))
        # env0 gap cuts t0 even without a task done; env1's real done cuts t0.
        self.assertAlmostEqual(float(alg.storage.returns[0, 0]), 1.)
        self.assertAlmostEqual(float(alg.storage.returns[0, 1]), 1.)
        self.assertEqual(float(alg.storage.returns[1, 0]), 0.)
        self.assertAlmostEqual(float(alg.storage.returns[2, 0]), 5.8, places=5)
        self.assertTrue(torch.isfinite(alg.storage.returns).all())

    def test_release_only_updates_normalizer_not_optimizer(self):
        alg, env = make(n=1, steps=1)
        audit = install(alg, env)
        env.ready_mask[:] = True
        step(alg, observations([7.]), [999.])
        with torch.inference_mode(): alg.compute_returns(observations([8.]))
        alg.update()
        self.assertEqual(int(alg.actor.obs_normalizer.count), 1)
        self.assertEqual(float(alg.actor.obs_normalizer.mean[0]), 7.)
        self.assertEqual(audit.report()['optimizer_steps'], 0)
        self.assertEqual(alg.optimizer.state, {})

    def test_single_valid_transition_is_finite_and_optimized(self):
        alg, env = make(n=2, steps=1)
        install(alg, env)
        env.ready_mask[1] = True; env.task_active_mask[1] = True
        step(alg, observations([float('nan'), 2.]), [float('nan'), 1.])
        with torch.inference_mode(): alg.compute_returns(observations([float('nan'), 3.]))
        result = alg.update()
        self.assertEqual(result['masked_optimizer_steps'], 2.)
        self.assertTrue(all(torch.isfinite(p).all() for p in alg.actor.parameters()))


if __name__ == '__main__': unittest.main()
