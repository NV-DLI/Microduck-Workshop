"""Ready-only inference and task-only PPO for nonrecurrent RSL-RL 5.5.1.

Install after constructing/loading the runner, before learn(). No actor,
normalizer or optimizer state is reset. The environment remains responsible
for applying its physical startup fixture when ready_mask is false. Zero
actions returned for those rows are placeholders, never policy inference.

ready_mask allows inference and normalizer observations; task_active_mask
allows reward transitions into PPO. Both masks are cloned before env.step.
Release can therefore run the actor while remaining outside the task loss.
"""

from types import MethodType

import torch
from rsl_rl.storage import RolloutStorage


class MaskedPPO:
    def __init__(self, algorithm, env):
        self.alg, self.env = algorithm, env
        if algorithm.actor.is_recurrent or algorithm.critic.is_recurrent:
            raise ValueError("This patch supports only feed-forward actor and critic")
        if algorithm.rnd or algorithm.symmetry or algorithm.is_multi_gpu:
            raise ValueError(
                "RND, symmetry and distributed PPO need separate reviewed masking"
            )
        if algorithm.normalize_advantage_per_mini_batch:
            raise ValueError(
                "Use rollout-wide masked advantage normalization in this version"
            )
        if algorithm.storage.training_type != "rl":
            raise ValueError("RL rollout storage required")
        if getattr(algorithm, "body50_masking", None) is not None:
            raise ValueError("PPO already has a body50 mask adapter")
        distribution = algorithm.actor.distribution
        if (
            distribution is None
            or type(distribution).__name__ != "GaussianDistribution"
        ):
            raise ValueError("Reviewed diagonal Gaussian distribution required")
        st = algorithm.storage
        self.ready = torch.zeros(
            (st.num_transitions_per_env, st.num_envs),
            device=st.device,
            dtype=torch.bool,
        )
        self.valid = torch.zeros_like(self.ready)
        self.pending = None
        self.original = {
            name: getattr(algorithm, name)
            for name in ("act", "process_env_step", "compute_returns", "update")
        }
        self.counts = {
            "rollouts_seen": 0,
            "rollouts_with_optimizer_updates": 0,
            "all_task_inactive_rollouts": 0,
            "all_not_ready_rollouts": 0,
            "optimizer_steps": 0,
            "ready_inference_rows": 0,
            "startup_placeholder_action_rows": 0,
            "task_transitions": 0,
            "ready_normalizer_rows": 0,
            "release_excluded_transitions": 0,
            "task_active_without_ready_dropped": 0,
        }
        self.last = {}
        self._masks()  # Fail early before patching any method.
        algorithm.act = self.act
        algorithm.process_env_step = self.process_env_step
        algorithm.compute_returns = self.compute_returns
        algorithm.update = self.update
        algorithm.body50_masking = self

    def _masks(self):
        masks = []
        for name in ("ready_mask", "task_active_mask"):
            value = getattr(self.env, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != torch.bool
                or value.shape != (self.alg.storage.num_envs,)
            ):
                raise ValueError(f"{name} must be a bool tensor of shape [num_envs]")
            masks.append(value.detach().to(self.valid.device).clone())
        return masks

    def act(self, obs):
        if self.pending is not None:
            raise RuntimeError("act called twice without process_env_step")
        ready, active = self._masks()
        valid = ready & active
        self.counts["task_active_without_ready_dropped"] += int((active & ~ready).sum())
        self.counts["ready_inference_rows"] += int(ready.sum())
        self.counts["startup_placeholder_action_rows"] += int((~ready).sum())
        st, tr = self.alg.storage, self.alg.transition
        actions = torch.zeros_like(st.actions[0])
        values = torch.zeros_like(st.values[0])
        log_prob = torch.zeros(st.num_envs, device=st.device)
        mean, std = torch.zeros_like(actions), torch.ones_like(actions)
        if ready.any():
            observations = obs[ready]
            selected_actions = self.alg.actor(
                observations, stochastic_output=True
            ).detach()
            actions[ready] = selected_actions
            values[ready] = self.alg.critic(observations).detach()
            log_prob[ready] = self.alg.actor.get_output_log_prob(
                selected_actions
            ).detach()
            parameters = self.alg.actor.output_distribution_params
            if len(parameters) != 2 or any(
                p.shape != selected_actions.shape for p in parameters
            ):
                raise ValueError("Gaussian distribution parameter shape changed")
            mean[ready], std[ready] = (p.detach() for p in parameters)
        tr.actions, tr.values, tr.actions_log_prob = actions, values, log_prob
        tr.distribution_params = (mean, std)
        tr.observations = obs
        tr.hidden_states = (None, None)
        self.pending = (ready, valid)
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        if self.pending is None:
            raise RuntimeError("process_env_step requires an action-time mask")
        ready, valid = self.pending
        st, tr = self.alg.storage, self.alg.transition
        self.ready[st.step].copy_(ready)
        self.valid[st.step].copy_(valid)
        # torch.where rather than multiply: NaN initialization sentinels must
        # not contaminate even the ignored cells of GAE.
        tr.rewards = torch.where(
            valid, rewards.reshape(-1), torch.zeros_like(rewards.reshape(-1))
        )
        tr.dones = torch.where(valid, dones.reshape(-1).bool(), torch.ones_like(valid))
        if "time_outs" in extras:
            timeout = extras["time_outs"].reshape(-1).to(st.device).bool() & valid
            tr.rewards += self.alg.gamma * tr.values.squeeze(-1) * timeout
        st.add_transition(tr)
        tr.clear()
        self.alg.actor.reset(dones)
        self.alg.critic.reset(dones)
        self.pending = None

    def compute_returns(self, obs):
        st = self.alg.storage
        if self.pending is not None or st.step != st.num_transitions_per_env:
            raise RuntimeError("Returns require a complete, committed rollout")
        ready, active = self._masks()
        last_valid = ready & active
        last_values = torch.zeros_like(st.values[0])
        # Only ready rows are evaluated; only task-active rows bootstrap.
        if ready.any():
            last_values[ready] = self.alg.critic(obs[ready]).detach()
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(st.step)):
            next_valid = last_valid if step == st.step - 1 else self.valid[step + 1]
            next_values = last_values if step == st.step - 1 else st.values[step + 1]
            continuation = (~st.dones[step].bool()) & next_valid[:, None]
            delta = (
                st.rewards[step]
                + self.alg.gamma * torch.where(continuation, next_values, 0.0)
                - st.values[step]
            )
            advantage = torch.where(
                self.valid[step, :, None],
                delta
                + self.alg.gamma
                * self.alg.lam
                * torch.where(continuation, advantage, 0.0),
                0.0,
            )
            st.returns[step] = torch.where(
                self.valid[step, :, None], advantage + st.values[step], 0.0
            )
        st.advantages.copy_(
            torch.where(self.valid[..., None], st.returns - st.values, 0.0)
        )
        selected = st.advantages[self.valid]
        if selected.numel():
            # Matches stock unbiased rollout normalization for >=2 samples;
            # singleton has zero centered advantage without an undefined std.
            denominator = (
                selected.std() if selected.numel() > 1 else selected.new_zeros(())
            )
            st.advantages[self.valid] = (selected - selected.mean()) / (
                denominator + 1e-8
            )

    def update(self):
        st = self.alg.storage
        ready_count, valid_count = int(self.ready.sum()), int(self.valid.sum())
        self.counts["rollouts_seen"] += 1
        self.counts["task_transitions"] += valid_count
        self.counts["ready_normalizer_rows"] += ready_count
        self.counts["release_excluded_transitions"] += ready_count - valid_count
        self.last = {
            "ready_rows": ready_count,
            "task_rows": valid_count,
            "optimizer_steps": 0,
        }
        if not valid_count:
            self.counts["all_task_inactive_rollouts"] += 1
            self.counts["all_not_ready_rollouts"] += int(not ready_count)
            if ready_count:
                observations = st.observations[self.ready]
                self.alg.actor.update_normalization(observations)
                self.alg.critic.update_normalization(observations)
            self._clear()
            return {
                "value": 0.0,
                "surrogate": 0.0,
                "entropy": 0.0,
                "masked_optimizer_steps": 0.0,
            }

        adapter = _CompactStorage(self, st)
        original_batches = self.alg.num_mini_batches
        self.alg.num_mini_batches = min(original_batches, valid_count)
        optimizer_step = self.alg.optimizer.step

        def counted_step(*args, **kwargs):
            result = optimizer_step(*args, **kwargs)
            self.counts["optimizer_steps"] += 1
            self.last["optimizer_steps"] += 1
            return result

        self.alg.optimizer.step = counted_step
        self.alg.storage = adapter
        try:
            result = self.original["update"]()
            self.counts["rollouts_with_optimizer_updates"] += int(
                self.last["optimizer_steps"] > 0
            )
            result["masked_optimizer_steps"] = float(self.last["optimizer_steps"])
            return result
        finally:
            self.alg.optimizer.step = optimizer_step
            self.alg.storage = st
            self.alg.num_mini_batches = original_batches

    def _clear(self):
        self.alg.storage.step = 0
        self.ready.zero_()
        self.valid.zero_()

    def report(self):
        return {
            "schema": "microduck.body50-ppo-mask.v1",
            **self.counts,
            "last_rollout": dict(self.last),
            "normalizer_scope": "All ready observations including physical release, no initialization sentinels",
            "loss_scope": "Action-time ready AND task_active transitions only",
            "inference_scope": "Action-time ready rows only; zero placeholder for physical startup fixture",
            "bootstrap_scope": "No cross-invalid bootstrap; real dones truncate GAE",
            "actor_and_optimizer_state_reset": False,
            "rollout_count_is_optimizer_update_count": False,
        }


class _CompactStorage:
    """Use stock PPO loss/optimizer with explicitly selected real transitions."""

    def __init__(self, owner, original):
        self.owner, self.original = owner, original
        self.valid_observations = original.observations[owner.valid]
        # Stock update's post-optimization normalization sees ready rows,
        # including release. Batch generation below sees task rows only.
        self.observations = original.observations[owner.ready].unsqueeze(0)
        self.data = {
            name: getattr(original, name)[owner.valid]
            for name in (
                "actions",
                "values",
                "advantages",
                "returns",
                "actions_log_prob",
            )
        }
        self.parameters = tuple(p[owner.valid] for p in original.distribution_params)

    def mini_batch_generator(self, num_mini_batches, num_epochs):
        count = self.valid_observations.batch_size[0]
        for _ in range(num_epochs):
            indices = torch.randperm(count, device=self.original.device)
            for ids in torch.tensor_split(indices, num_mini_batches):
                yield RolloutStorage.Batch(
                    observations=self.valid_observations[ids],
                    actions=self.data["actions"][ids],
                    values=self.data["values"][ids],
                    advantages=self.data["advantages"][ids],
                    returns=self.data["returns"][ids],
                    old_actions_log_prob=self.data["actions_log_prob"][ids],
                    old_distribution_params=tuple(p[ids] for p in self.parameters),
                )

    def clear(self):
        self.original.clear()
        self.owner.ready.zero_()
        self.owner.valid.zero_()


def install(algorithm, env):
    """Return the audit object; runtime entrypoint owns ledger integration."""
    return MaskedPPO(algorithm, env)
