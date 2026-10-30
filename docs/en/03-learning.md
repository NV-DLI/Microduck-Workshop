---
myst:
  html_meta:
    description: How reinforcement learning learns to walk from trial and error, why PPO limits each policy update, and where these formulas live in the Isaac Lab training code.
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 3.1 Reinforcement learning and PPO

:::{lesson-intro}
:meta: Section 3.1 · Theory · Estimated learning time: 30 minutes

How reinforcement learning learns to walk from trial and error, why PPO limits each policy update, and where these formulas live in the Isaac Lab training code. This section covers the principles; Section 3.2 starts a training and tests a trained policy.
:::

```{symbols}
J:
  tex: J(\theta)
  text: 'Learning objective: the expected discounted return when sampling with the current policy; training adjusts the policy parameters to increase it.'
E_traj:
  tex: \mathbb E_{\tau\sim\pi_\theta}
  text: 'Expectation: an average over many trajectories sampled with the current policy.'
tau:
  tex: \tau
  text: 'One interaction trajectory: the observations, actions and rewards of an episode in order.'
pi_theta:
  tex: \pi_\theta
  text: 'Policy: the neural network, set by its parameters, that maps an observation to a probability distribution over actions.'
theta:
  tex: \theta
  text: Parameters (weights) of the policy network, updated during training.
t_idx:
  tex: t
  text: Index of a control step.
T_horizon:
  tex: T
  text: Number of control steps counted in the return of this trajectory; a different quantity from the steps sampled per iteration below.
gamma:
  tex: \gamma
  text: 'Discount factor: closer to 1 gives more weight to distant rewards; the course training uses 0.99.'
r_t:
  tex: r_t
  text: 'Immediate reward of the current control step: the weighted sum of all reward terms.'
r_v:
  tex: r_t^{v}
  text: Linear-velocity tracking reward of the current control step, between 0 and 1.
exp:
  tex: \exp
  text: 'Exponential function: 1 when the error is 0, approaching 0 as the error grows.'
v_cmd:
  tex: \mathbf v_{xy}^{\mathrm{cmd}}
  text: Horizontal part of the velocity command (m/s).
v_xy:
  tex: \mathbf v_{xy}
  text: Actual horizontal body velocity (m/s).
v_z:
  tex: v_z
  text: Vertical body velocity (m/s); its square is part of the error, which discourages bouncing.
sigma_v:
  tex: \sigma_v
  text: Tolerance scale for the velocity error, `std` in the code.
dt:
  tex: \Delta t
  text: 'Duration of one control step: reward terms are accumulated per time step.'
i:
  tex: i
  text: Index of a reward term.
w_i:
  tex: w_i
  text: Weight of one reward term.
r_it:
  tex: r_{i,t}
  text: Value of one reward term at the current control step.
delta_t:
  tex: \delta_t
  text: 'Temporal-difference error: how much better this step turned out than the value network predicted.'
m_t:
  tex: m_t
  text: 'Continuation flag: 0 when the episode terminates at this step, otherwise 1.'
V:
  tex: V
  text: 'Value network: estimates the discounted return from a state.'
s_t:
  tex: s_t
  text: State at the current control step.
s_next:
  tex: s_{t+1}
  text: State at the next step.
A_hat:
  tex: \hat A_t
  text: 'Advantage estimate at the current step: how much better this action was than expected on average.'
A_next:
  tex: \hat A_{t+1}
  text: Advantage estimate at the next step; the recursion runs backward in time.
lambda:
  tex: \lambda
  text: 'GAE parameter: weights multi-step errors; the course training uses 0.95.'
rho:
  tex: \rho_t
  text: 'Probability ratio: the probability density of the same sampled action under the new and old policies.'
pi_old:
  tex: \pi_{\theta_{\mathrm{old}}}
  text: 'Old policy: the policy that collected this batch; it stays fixed during the update.'
a_t:
  tex: a_t
  text: Action actually sampled at the current control step.
o_t:
  tex: o_t
  text: Observation at the current control step.
L_clip:
  tex: L^{\mathrm{CLIP}}
  text: PPO clipped surrogate objective, which training increases.
E_t:
  tex: \mathbb E_t
  text: Average over the time steps in this batch.
min:
  tex: \min
  text: Take the smaller of the two terms, keeping the more conservative estimate.
clip:
  tex: \operatorname{clip}
  text: 'Clipping: limits the probability ratio to between $1-\epsilon$ and $1+\epsilon$.'
epsilon:
  tex: \epsilon
  text: Clip range; the course training uses 0.2.
L_total:
  tex: \mathcal L
  text: Total loss actually minimized by the optimizer.
c_V:
  tex: c_V
  text: Value-loss coefficient; 1.0 in this configuration.
L_V:
  tex: L_V
  text: 'Value loss: the error of the value network''s return prediction.'
c_H:
  tex: c_H
  text: Entropy coefficient; 0.01 in this configuration.
H:
  tex: H
  text: 'Entropy: how random the action distribution is; this term encourages continued exploration.'
N_batch:
  tex: N_{\mathrm{batch}}
  text: Number of samples collected per training iteration.
N_env:
  tex: N_{\mathrm{env}}
  text: Number of parallel environments; 1024 in the full training of Section 3.2.
T_rollout:
  tex: T
  text: Control steps collected per environment per iteration; 24 in Section 3.2. It is a different quantity from the episode time limit.
N_total:
  tex: N_{\mathrm{total}}
  text: Total samples collected over 2500 iterations.
```

{#loop}
## The task: walk steadily on velocity commands

Sections 2.1 and 2.2 only ran the official, fixed policy. Now the question changes: how can a randomly initialized policy learn to walk through repeated trials in simulation? This section connects the learning objective, rewards, advantage estimation and the PPO update to the actual Isaac Lab and RSL-RL code; Section 3.2 then starts training and tests the course’s pretrained policy.

We want the robot to coordinate its 14 joints when it receives forward, sideways and turning commands: keep balance while stepping, avoid slipping on the support foot, and move smoothly. Hand-writing a periodic trajectory for every joint hardly adapts to different speeds, forces and initial states at once.

Reinforcement learning learns control by “try an action → get feedback → adjust the policy” and needs no examples of “which joint angles for which state”: Newton produces the state transitions and the Isaac Lab reward manager describes the desired behavior.

```{learning-flow}
label: Online interaction and updates in reinforcement learning
nodes:
- [Parallel simulation, 'Builds observations from the current state']
- [Current policy πθ, 'Samples joint actions aₜ']
- [Execute and evaluate, 'Gets reward rₜ and the next state']
- [Store a batch, 'Observations, actions, rewards, terminations']
- [PPO update, 'Estimates advantages, optimizes policy and value']
- [Sample again, 'Collects new data with the updated policy']
details:
- Many environments produce observations at once. A copy that falls or times out is reset on its own while the others continue.
- Actions are sampled from the current observation. Random exploration during training lets the policy see the consequences of different actions; deployment uses the chosen inference mode.
- Actions go into the physics environment, then rewards for velocity tracking, posture, energy and so on are computed. Rewards only judge behavior; the policy discovers the joint motions itself.
- The observations, actions, rewards and terminations of a stretch of interaction are stored in time order to compute returns and advantages.
- The collected samples update the policy and value networks. The clipped objective limits each update and avoids sudden policy changes.
- The updated policy produces new actions and a new state distribution, so data is sampled again. Alternating interaction and updates is reinforcement-learning training.
loop: true
caption: Isaac Lab arranges many environments on the GPU as batched state; observations, actions, rewards and terminations are stored per environment index. Once the policy changes, so does the data distribution, which is why PPO makes a limited number of updates on each freshly collected batch and then interacts with the environment again.
```

```{formula} Learning objective: increase long-term discounted return
\symx{J}{J(\sym{theta})}=\symx{E_traj}{\mathbb E_{\sym{tau}\sim\sym{pi_theta}}}\!\left[\sum_{\sym{t_idx}=0}^{\sym{T_horizon}-1}\sym{gamma}^{\sym{t_idx}} \sym{r_t}\right]
```

$\sym{theta}$ are the policy parameters, $\sym{tau}$ an interaction trajectory, $\sym{r_t}$ the immediate reward and $\sym{gamma}$ the discount factor. Optimizing long-term return means that whether an action is good depends on whether the robot stays upright afterwards, not only on the reward of the current frame.

During training the policy network chooses actions from the 61-dimensional deployable observation; the value network may also read extra simulation information to estimate return. Only the policy network, which reads those 61 values, is exported; the extra simulation information the value network uses (for example base linear velocity and foot contact forces) is used only during training.

{#reward-design}
## Turn “walking well” into rewards

Rewarding only velocity tracking can produce unstable motion; rewarding only upright posture can teach the robot to stand still. The task therefore combines velocity tracking, posture, foot motion and action changes. Below is one actual reward term and how terms are combined; the complete list and weights are in the `env.yaml` saved with each training run.

```{formula} Linear-velocity tracking reward
\sym{r_v}=\sym{exp}\!\left(-\frac{\|\sym{v_cmd}-\sym{v_xy}\|_2^2+\sym{v_z}^2}{\sym{sigma_v}^2}\right),\qquad \sym{r_t}=\sym{dt}\sum_{\sym{i}}\sym{w_i}\sym{r_it}
```

The smaller the velocity error, the closer this exponential reward is to 1; vertical velocity is penalized too, so bouncing does not count as ideal translation. $\sym{sigma_v}$ sets the tolerance. The reward manager multiplies each term by its weight and accumulates over the control time step $\sym{dt}$; when comparing reward curves, check that weights and time step match.

```{code-story}
label: Python · core excerpt
source: https://github.com/isaac-sim/IsaacLab/blob/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/mdp/rewards.py
language: python
code: |
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  return zero_non_finite(torch.exp(-(xy_error + z_error) / std**2))
steps:
- lines: [0, 2]
  title: Compute the velocity error
  body: For each environment, `xy_error` is the squared difference between horizontal target and actual velocity; `z_error` is the squared vertical velocity, whose target is 0.
- lines: [2, 3]
  title: Turn it into a reward between 0 and 1
  body: The summed error is divided by `std` squared and passed through a negative exponential, giving 1 at zero error; `zero_non_finite` records numerically invalid results as 0.
```

The excerpt comes from `microduck/mdp/rewards.py` of the pinned Isaac Lab. `command` is the target velocity, `actual` the actual velocity and `std` corresponds to $\sym{sigma_v}$. At the start of training the linear-velocity tracking weight is 2.0 and the action-rate penalty −0.1. For long runs, also check the curriculum settings: the action-rate penalty grows as training progresses. These values are tuned for this task; another robot needs its own.

When checking a reward design, look at behavior before the total. If reward rises while falls become more frequent, inspect velocity, posture, action-change and termination terms one by one to find which term hides the problem.

{#ppo}
## Why PPO: limit how far each update moves the policy

If a policy update is too large, a balance behavior just learned can disappear in the next sampling round. PPO (Proximal Policy Optimization) clips its surrogate objective so that very large probability changes earn no extra benefit. It suits continuous joint control and collects data easily from many parallel simulations; we use the RSL-RL implementation, so formulas, configuration and training logs line up. PPO makes updates more stable, but a single update can still make behavior worse.

First estimate how much better an action was than the average expectation in that state: the advantage. A positive advantage pushes the probability of that action up, a negative one pushes it down. The value network $\sym{V}$ provides a baseline for the state's return; generalized advantage estimation (GAE) combines multi-step temporal-difference errors to balance bias and variance.

```{formula} Temporal-difference error and generalized advantage estimation
\sym{delta_t}=\sym{r_t}+\sym{gamma}\sym{m_t}\sym{V}(\sym{s_next})-\sym{V}(\sym{s_t}),\qquad \sym{A_hat}=\sym{delta_t}+\sym{gamma}\sym{lambda}\sym{m_t}\sym{A_next}
```

$\sym{m_t}$ is 0 when the episode terminates and 1 otherwise; $\sym{gamma}$ discounts future rewards and $\sym{lambda}$ weights multi-step errors. Truncation at the time limit means something different from a fall: the implementation bootstraps the value separately for truncation. The course training uses $\sym{gamma} = 0.99$ and $\sym{lambda} = 0.95$.

```{formula} PPO clipped surrogate objective
\sym{rho}(\sym{theta})=\frac{\sym{pi_theta}(\sym{a_t}\mid \sym{o_t})}{\sym{pi_old}(\sym{a_t}\mid \sym{o_t})},\qquad \sym{L_clip}=\sym{E_t}\!\left[\sym{min}\!\left(\sym{rho}\sym{A_hat},\sym{clip}(\sym{rho},1-\sym{epsilon},1+\sym{epsilon})\sym{A_hat}\right)\right]
```

$\sym{rho}$ is the ratio of the probability densities of the same sampled action under the new and old policies, and $\sym{epsilon} = 0.2$ sets the clip range. Clipping acts only on the objective: the ratio itself can still leave that range, and joint angles are not constrained by it.

```{concept-diagram}
kind: ppo
label: 'PPO clipped objective: advantage sign and probability ratio'
caption: The dashed grey line is the unclipped objective, the solid green line the PPO objective, and the light-green band is $[1-\sym{epsilon},\,1+\sym{epsilon}]$. With a positive advantage, raising the action probability beyond the upper bound no longer improves this sample's objective; with a negative advantage the plateau lies beyond the lower bound.
```

Switch the advantage between +1 and −1 and move the ratio slider. The objective takes the smaller of the two terms, so the plateau appears on one side only, depending on the sign of the advantage. The figure analyses a single sample's policy objective; a real update also depends on the other samples, the value loss and the entropy term.

```{code-story}
label: Python · core excerpt
source: https://github.com/leggedrobotics/rsl_rl/blob/v5.5.1/rsl_rl/algorithms/ppo.py
language: python
code: |
  ratio = torch.exp(
      actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
  )
  surrogate = -torch.squeeze(batch.advantages) * ratio
  surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
      ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
  )
  surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
steps:
- lines: [0, 3]
  title: Ratio of new to old policy
  body: Subtracting log-probabilities and exponentiating gives the ratio for the same action under the new and old policies.
- lines: [3, 7]
  title: Limit the update
  body: The advantage and the ratio form the surrogate objective; clipping the ratio limits overly large policy changes.
- lines: [7, 8]
  title: Aggregate the loss
  body: The code takes the larger of the negated terms and averages over samples; the optimizer minimizes this loss.
```

In the code, `ratio` is $\sym{rho}$, `batch.advantages` is $\sym{A_hat}$ and `clip_param` is $\sym{epsilon}$. Because the optimizer minimizes, the implementation negates the objective and uses `max`, which matches “maximize the min” in the formula. The old policy's log-probabilities are stored at sampling time and stay fixed during the update.

```{formula} Total loss actually optimized
\sym{L_total}=-\sym{L_clip}+\sym{c_V} \sym{L_V}-\sym{c_H}\sym{E_t}[\sym{H}(\sym{pi_theta}(\cdot\mid \sym{o_t}))]
```

The value loss $\sym{L_V}$ trains return prediction and the entropy $\sym{H}$ encourages exploration. The configuration uses a value-loss coefficient of 1.0, an entropy coefficient of 0.01 and value clipping. The learning rate starts at 0.001 and then adapts to the KL divergence; you can check all of these in `agent.yaml` in the output folder.

Theory: [PPO paper](https://arxiv.org/abs/1707.06347); implementation: [RSL-RL 5.5.1 algorithms](https://github.com/leggedrobotics/rsl_rl/tree/v5.5.1/rsl_rl/algorithms).

{#rollout-data}
## Where training data comes from and what the settings mean

PPO needs no downloaded dataset: its training data is produced by the simulation during training. The full training in Section 3.2 starts from a random initialization and reads no weights from the fixed walking model. The script creates 1024 Newton environments from the Isaac Lab task configuration, and each environment collects 24 interaction steps per iteration. Each record holds the observation, action, reward, old-policy log-probability, value estimate and termination flag; records live in a GPU memory buffer and are replaced by the next iteration's data.

```{formula} Samples per iteration and in total
\sym{N_batch}=\sym{N_env}\times \sym{T_rollout}=1024\times24=24576,\qquad \sym{N_total}=2500\times24576=61{,}440{,}000
```

| Setting | Full training in Section 3.2 | Main effect |
|---|---|---|
| Parallel environments | 1024 | Interaction collected in parallel per iteration |
| Steps per environment | 24 | Length of each trajectory segment, not a full episode (at most 20 s, 1000 control steps) |
| Epochs per update | 5 | Passes over each batch during optimization |
| Mini-batches | 4 | Each pass is split into 4 mini-batches of 6144 samples |
| Training iterations | 2500 | Sampling-and-update loops; not 2500 actions |
| Learning rate | 0.001, adapted to the KL divergence | Step size of each parameter update |

This budget produces about 61.4 million environment interactions and 2500 × 5 × 4 = 50,000 optimizer updates. At the 50 Hz control rate that is about 341 hours of simulated “practice”; parallel environments on the GPU collect it in about 2 hours. A training environment is reset on its own when its episode terminates while the others keep sampling.

```{code-story}
label: Python · core excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/gait/student_train.py
language: python
code: |
  base = gym.make(task, cfg=cfg).unwrapped
  env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
  runner = OnPolicyRunner(
      env, to_dict(agent), log_dir=str(args.output), device="cuda:0"
  )
  runner.learn(
      num_learning_iterations=args.iterations, init_at_random_ep_len=True
  )
steps:
- lines: [0, 2]
  title: Create the training environment
  body: Build the task and wrap it as a vectorized environment so the algorithm reads observations and rewards from all environments at once.
- lines: [2, 5]
  title: Connect the PPO runner
  body: Give the runner the environment, algorithm settings, output folder and device.
- lines: [5, 8]
  title: Learn for a fixed number of iterations
  body: The iteration count sets this run's budget; random initial episode lengths spread out the environments' terminations.
```

The Isaac Lab task provides state transitions, managers and batched resets; the wrapper connects it to RSL-RL, and `OnPolicyRunner` repeats sampling and updates. Newton's Warp/CUDA path advances these environments in parallel on the GPU; the speed comes from batched execution and a single task interface.

{#read-curves}
## Checkpoints, configuration and exported files

Training writes several kinds of files to the output folder. `initial.pt` stores the starting point, `model_*.pt` are training checkpoints saved every few hundred iterations, `env.yaml` and `agent.yaml` record the environment and optimizer settings actually used, and `exported/policy.onnx` is the file for inference. A checkpoint can include optimizer state; an ONNX file usually keeps only the inference graph. They serve different purposes and cannot replace each other.

The reward combines velocity tracking, posture, smoothness and contact terms. Two policies with the same total reward might earn it differently: one by tracking speed more accurately, the other by moving less and staying stable. So check actual behavior and the individual terms together.

::::{quiz}
Can you resume training exactly from an ONNX file alone?

:::{answer}
No. Exact resumption also needs optimizer state, normalization state, random state and the same configuration, so keep the full checkpoint and training configuration.
:::
::::

::::{quiz}
Why not report the highest return seen during training as an independent test result?

:::{answer}
That result already took part in learning or model selection, and a high reward does not necessarily mean task success. An independent test fixes the scenarios and success criteria in advance and counts every result under conditions not used for selection.
:::
::::

{#beyond-flat}
## Next: confront training with variation

You now know how the objective, rewards, advantages and clipped updates work together. One question remains: friction, mass and servo delay are fixed numbers in simulation, but every real robot differs. Can a policy learned in one “perfect” simulation cope with those differences? Section 3.2 introduces domain randomization, starts a training, lets you watch the parallel environments and tests the course’s pretrained policy.

{#references}
## References

{.references}
- [Original PPO paper ↗](https://arxiv.org/abs/1707.06347)
- [Isaac Lab: Microduck task source ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
