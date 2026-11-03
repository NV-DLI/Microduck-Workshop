---
myst:
  html_meta:
    description: 'The general method of reinforcement learning: goals, rewards and returns, how the Actor and Critic divide the work, advantage estimation, and why PPO limits the size of each update.'
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 3.1 Reinforcement learning and PPO

:::{lesson-intro}
:meta: Section 3.1 · Theory · Estimated learning time: 15 minutes

The general method of reinforcement learning: goals, rewards and returns, how the Actor and Critic divide the work, advantage estimation, and why PPO limits the size of each update. Section 3.2 then maps these ideas onto Microduck's walking task.
:::

```{symbols}
G_t:
  tex: G_t
  text: 'Return: the sum of all rewards from step $t$ onward, each discounted.'
gamma:
  tex: \gamma
  text: 'Discount factor between 0 and 1: the closer to 1, the more distant rewards count.'
k_idx:
  tex: k
  text: Number of steps counted forward from the current step.
t_idx:
  tex: t
  text: Time-step index.
r_t:
  tex: r_t
  text: Immediate reward at step $t$.
delta_t:
  tex: \delta_t
  text: 'Temporal-difference error: how much better this step turned out than the Critic expected.'
m_t:
  tex: m_t
  text: 'Continuation flag: 0 if the episode terminates at this step, otherwise 1.'
V:
  tex: V
  text: 'Value function (Critic): the return expected from a state when the current policy continues.'
s_t:
  tex: s_t
  text: State at the current step.
s_next:
  tex: s_{t+1}
  text: State at the next step.
A_hat:
  tex: \hat A_t
  text: 'Advantage estimate: how much better the action actually taken was than the Critic expected.'
A_next:
  tex: \hat A_{t+1}
  text: Advantage estimate of the next step; the recursion accumulates from the end backward.
lambda:
  tex: \lambda
  text: 'GAE parameter: a compromise between looking one step ahead and many steps ahead.'
rho:
  tex: \rho_t
  text: 'Probability ratio: the probability density of the same sampled action under the new and the old policy.'
theta:
  tex: \theta
  text: Policy network parameters (weights), updated during training.
pi_theta:
  tex: \pi_\theta
  text: 'New policy (Actor): a probability distribution over actions given the observation.'
pi_old:
  tex: \pi_{\theta_{\mathrm{old}}}
  text: 'Old policy: the policy that sampled this batch; it stays fixed during the update.'
a_t:
  tex: a_t
  text: Action actually sampled at the current step.
o_t:
  tex: o_t
  text: Observation at the current step.
L_clip:
  tex: L^{\mathrm{CLIP}}
  text: 'PPO clipped surrogate objective: training increases it.'
E_t:
  tex: \mathbb E_t
  text: Average over the time steps of this batch.
min:
  tex: \min
  text: The smaller of the two terms, keeping the more conservative estimate.
clip:
  tex: \operatorname{clip}
  text: Clips the probability ratio to $1-\epsilon$ … $1+\epsilon$ before computing the objective.
epsilon:
  tex: \epsilon
  text: Clipping threshold, commonly 0.2.
```

{#loop}
## Learning from trial and error

Reinforcement learning studies how an agent that acts repeatedly in an environment can learn better behavior from the feedback it receives. It needs no correct answer for “which action in which state”, only a signal that rates how good the behavior was.

```{learning-flow}
label: The interaction and update loop of reinforcement learning
nodes:
- [Environment, Gives the current observation]
- [Policy, Chooses an action from the observation]
- [Execute and evaluate, Gets a reward and the next observation]
- [Collect a batch, 'Observations, actions, rewards, terminations']
- [Update the policy, Makes good actions more likely]
- [Interact again, Collects new data with the new policy]
details:
- The environment can be the real world or a simulator. The agent learns about the environment's state only through observations.
- The policy maps observations to actions. During training it is random so that it can try different actions.
- After the action the environment moves to a new state and returns a reward that says how good the result of this step was.
- A stretch of interaction is stored in time order to compute returns and advantages.
- The batch adjusts the policy parameters: actions that led to good results become more frequent, bad ones less frequent.
- Once the policy changes, so do the states it meets, so new data is collected with the new policy. Alternating interaction and updates is training.
loop: true
caption: The basic loop of reinforcement learning. The data comes from the policy's own attempts, so the data distribution changes whenever the policy does.
```

{#return}
## Goals, rewards and returns

The **goal** is what we ultimately want the agent to achieve. The **reward** is the immediate feedback at each step, our way of turning the goal into numbers. Good actions often pay off only later, so learning looks at the **return**: all rewards from a given step onward, each discounted.

```{formula} Discounted return
\sym{G_t}=\sum_{\sym{k_idx}=0}^{\infty}\sym{gamma}^{\sym{k_idx}}\,\symx{r_t}{r_{\sym{t_idx}+\sym{k_idx}}}
```

The closer the discount factor $\sym{gamma}$ is to 1, the more distant results count. Training adjusts the policy so that the expected return is as large as possible.

A reward is only a training signal, not task success. If the reward is poorly written, the agent may find a way to score high without completing the task, so trained results are checked against a separate success criterion.

{#actor-critic}
## Actor-Critic: during training, one acts and one evaluates

Adjusting the policy directly from returns is very noisy: the same action sometimes earns a high return because of later luck, and sometimes the opposite. Actor-Critic is a training paradigm that splits the work between two networks during training:

| Network | Output | Role |
|---|---|---|
| Actor (policy) | Action | Decides what to do; this is what is deployed after training |
| Critic (value function) | One number: the expected return | Judges “how good is the current situation”; used only during training |

With the Critic's expectation, we can tell whether an action did better than expected. That difference is the **advantage**. With a positive advantage the Actor makes the action more likely; with a negative one, less likely. The Critic is trained too: its prediction should match the return actually obtained. After training only the Actor is kept; the Critic's job is done.

{#ppo}
## PPO: change a little at a time

If one update on a batch changes the policy too much, the policy can suddenly get worse, and the next batch is then collected by that worse policy, so training easily collapses. PPO (Proximal Policy Optimization) compares how likely the new and old policies are to choose the same action. Once the change passes the clipping threshold $\sym{epsilon}$, pushing further in that direction brings no extra benefit, so each update naturally stays small.

```{concept-diagram}
kind: ppo
label: 'PPO clipped objective: advantage sign and probability ratio'
caption: The dashed gray line is the unclipped objective, the solid green line the PPO objective and the light green band $[1-\sym{epsilon},\,1+\sym{epsilon}]$. With a positive advantage the plateau lies right of the upper bound; with a negative advantage, left of the lower bound.
```

Switch the advantage between +1 and −1 and move the ratio slider to see on which side the plateau appears. Clipping acts only on the objective and does not hard-limit the actual probability ratio; PPO makes updates steadier but does not guarantee that every update improves behavior. It is simple and suits data collected in massive parallel simulation, which is why it is widely used for robot locomotion.

::::{enrichment} Extension: the formulas for advantage estimation and the PPO objective
The Critic's temporal-difference error $\sym{delta_t}$ compares “this step's reward plus the expectation of the next state” with “the original expectation of this state”. Generalized Advantage Estimation (GAE) sums multi-step errors decayed by $\sym{gamma}\sym{lambda}$ into a smoother advantage; at termination $\sym{m_t}=0$, and expectations after termination no longer count.

```{formula} Temporal-difference error and generalized advantage estimation
\sym{delta_t}=\sym{r_t}+\sym{gamma}\sym{m_t}\sym{V}(\sym{s_next})-\sym{V}(\sym{s_t}),\qquad \sym{A_hat}=\sym{delta_t}+\sym{gamma}\sym{lambda}\sym{m_t}\sym{A_next}
```

PPO multiplies the probability ratio $\sym{rho}$ by the advantage and takes the smaller of the unclipped and clipped terms:

```{formula} PPO clipped surrogate objective
\sym{rho}(\sym{theta})=\frac{\sym{pi_theta}(\sym{a_t}\mid \sym{o_t})}{\sym{pi_old}(\sym{a_t}\mid \sym{o_t})},\qquad \sym{L_clip}=\sym{E_t}\!\left[\sym{min}\!\left(\sym{rho}\sym{A_hat},\sym{clip}(\sym{rho},1-\sym{epsilon},1+\sym{epsilon})\sym{A_hat}\right)\right]
```
::::

::::{quiz}
Does a deployed policy still need the Critic?

:::{answer}
No. The Critic is used only during training to estimate expected return and compute advantages; at deployment the Actor maps observations to actions.
:::
::::

::::{quiz}
Why can't the run with the highest training return serve as the test result?

:::{answer}
Return only measures how much reward the policy collected, and reward is not task success; that run has also already been part of training or selection. A test fixes its scenarios and success criterion in advance and counts every result.
:::
::::

{#next}
## Next

Section 3.2 maps these ideas onto Microduck's walking task: what the observation and action are, how the reward is written, and what the Actor and Critic each read and how they work together during training.

{#references}
## References

{.references}
- [Original PPO paper ↗](https://arxiv.org/abs/1707.06347)
- [GAE: generalized advantage estimation ↗](https://arxiv.org/abs/1506.02438)
- [Sutton and Barto: Reinforcement Learning, An Introduction ↗](http://incompleteideas.net/book/the-book-2nd.html)
