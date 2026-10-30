---
myst:
  html_meta:
    description: Understand why domain randomization helps a policy cope with real-world differences, then start a walking training in Isaac Lab, watch the parallel environments and test a trained policy.
---

# 3.2 Domain randomization: train and test a walking policy

:::{lesson-intro}
:meta: Section 3.2 · Hands-on · Estimated learning time: 45 minutes (watching training for a few minutes is enough; a full run takes about 2 hours and is optional)

Understand why domain randomization helps a policy cope with real-world differences, then start a walking training in Isaac Lab, watch the parallel environments and test a trained policy.
:::

```{symbols}
J_dr:
  tex: J(\theta)
  text: 'Learning objective under domain randomization: the expected discounted return averaged over many sets of random physical parameters.'
theta:
  tex: \theta
  text: Parameters (weights) of the policy network, updated during training.
E_xi:
  tex: \mathbb E_{\xi\sim p(\xi)}
  text: Average over randomly drawn sets of physical parameters.
xi:
  tex: \xi
  text: One set of simulation parameters, such as foot friction, trunk mass, joint friction and servo delay.
p_xi:
  tex: p(\xi)
  text: 'Parameter distribution: each parameter is sampled within its configured range.'
E_traj:
  tex: \mathbb E_{\tau\sim\pi_\theta,\,\xi}
  text: Average over trajectories sampled with the current policy in a simulation with parameters $\xi$.
pi_theta:
  tex: \pi_\theta
  text: 'Policy: the neural network that maps an observation to an action distribution.'
gamma:
  tex: \gamma
  text: Discount factor; 0.99 in the course training.
t_idx:
  tex: t
  text: Index of a control step.
T_horizon:
  tex: T
  text: Number of control steps counted in an episode's return.
r_t:
  tex: r_t
  text: Immediate reward of the current control step.
```

{#reality-gap}
## Why domain randomization

The reinforcement learning of Section 3.1 happens entirely in simulation, where every parameter is one exact number: foot friction, trunk mass, joint friction, the delay before a servo starts moving… Real Microducks differ: servos from different batches have different friction, battery charge changes servo output, encoder zeros are slightly off, and floors are harder or softer. A policy that practises only with one “perfect” set of parameters can learn to rely on its details and lose balance on another robot or another floor. This gap between simulation and reality is often called the sim-to-real gap.

Domain randomization takes a direct approach: since we do not know the exact real parameters, each parallel environment draws its own set of parameters during training. The policy faces a thousand “slightly different robots” at once and earns a high return overall only if its actions work across those differences.

```{formula} Learning objective under domain randomization
\symx{J_dr}{J(\sym{theta})}=\sym{E_xi}\,\sym{E_traj}\!\left[\sum_{\sym{t_idx}=0}^{\sym{T_horizon}-1}\sym{gamma}^{\sym{t_idx}}\sym{r_t}\right]
```

Compared with the objective in Section 3.1 there is one more average: first draw a set of physical parameters $\sym{xi}$ from the distribution $\sym{p_xi}$, then sample trajectories with the policy $\sym{pi_theta}$ under those parameters and compute the discounted return. PPO updates the same way; what changes is the data, because one batch mixes many physical conditions.

Choosing ranges is a trade-off. Too narrow, and the policy still fits only a few cases; too wide, and the task can become too hard, so the policy learns cautious, sluggish motion or nothing at all. Ranges are best grounded in measurements or hardware specifications, such as measured servo delays or the battery's operating voltage. Domain randomization improves robustness but does not replace testing on the robot: it covers only the differences you thought of and wrote into the configuration.

{#what-varies}
## What the course training actually varies

Isaac Lab implements randomization with its event manager. Each event has a trigger: `startup` samples once when the environment is created and then keeps the value; `reset` samples again whenever that environment resets; `interval` fires at time intervals during an episode. The table comes from the `env.yaml` saved by the full training; these are the values that were in effect.

| Randomized item | Range | Trigger | Real-world difference it imitates |
|---|---|---|---|
| Static and dynamic foot friction | 0.7–1.3 | startup | Different floors and sole materials |
| Encoder zero offset | ±0.015 rad | startup | Assembly and calibration error |
| Trunk mass | nominal × 0.95–1.05 | startup | Battery, cable and shell differences |
| Trunk and head centre of mass | ±3 mm per axis | reset | Uneven mass distribution |
| Joint friction | nominal × 0.9–1.1 | reset | Servo variation and wear |
| Joint armature | nominal × 0.9–1.1 | reset | Gear and motor differences |
| External push | every 3–6 s, horizontal velocity change up to ±0.3 m/s | interval | Bumps, being pushed |
| Servo delay | 3–6 physics steps (15–30 ms) | actuator model | Communication and control delay |
| Servo supply voltage | 6.5–8.2 V, plus load-dependent voltage drop | actuator model | Battery charge |
| Observation noise | on | every step | Sensor noise |
| Start position and heading | ±0.5 m, any heading | reset | Walking off from different places |

```{code-story}
label: Python · core excerpt
source: https://github.com/isaac-sim/IsaacLab/blob/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/flat_env_cfg.py
language: python
code: |
  foot_friction = EventTerm(
      func=mdp.randomize_rigid_body_material,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES),
          "static_friction_range": (0.7, 1.3),
          "dynamic_friction_range": (0.7, 1.3),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 64,
      },
  )
  encoder_bias = EventTerm(func=mdp.randomize_encoder_bias, mode="startup", params={"bias_range": (-0.015, 0.015)})
  randomize_joint_friction = EventTerm(
      func=mdp.randomize_bam_friction, mode="reset", params={"scale_range": (0.9, 1.1)}
  )
  push_robot = EventTerm(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 6.0),
      params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
  )
steps:
- lines: [0, 11]
  title: Foot friction is sampled when environments are created
  body: '`mode="startup"` draws once per environment. Static and dynamic friction lie in 0.7–1.3; `num_buckets` splits the range into 64 levels so thousands of environments share a few materials.'
- lines: [11, 12]
  title: Encoder zero offset
  body: Each joint reading gets a fixed offset within ±0.015 rad, imitating assembly and calibration error; the joint angles the policy sees differ slightly from the true ones.
- lines: [12, 15]
  title: New joint friction at every reset
  body: '`mode="reset"` gives an environment a new friction set every time it restarts, so even one environment experiences different servo behaviour during training.'
- lines: [15, 21]
  title: A random push while walking
  body: The `interval` event changes the body's horizontal velocity by up to 0.3 m/s every 3–6 seconds; the policy has to learn to recover its balance.
```

The excerpt comes from the Microduck task configuration of the pinned Isaac Lab. Note that the tests and demonstrations in this section turn pushes and parameter randomization off to compare policies under nominal conditions; randomization is used only in training.

{#train}
## Start a full training and watch it in parallel

Run the command below in the course container shell from Section 1.3 (to reopen it, see <a href="setup.html#start">Section 1.3</a>). It trains from a random initialization and reads no weights from any existing model; `--show` opens the Isaac Sim window so you see a thousand ducks practising at once, each in its own randomized environment. Use a new folder for `--output`.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py train-walk \
  --num-envs 1024 \
  --iterations 2500 \
  --show \
  --output outputs/walking-training-01
```

| Argument | Meaning |
|---|---|
| `--num-envs 1024` | Environments running on the GPU at the same time; each has its own random parameters |
| `--iterations 2500` | PPO sampling-and-update loops; Section 3.1 computed the matching sample count |
| `--show` | Opens Isaac Sim to display the parallel environments; rendering uses GPU memory and slows every iteration |
| `--output` | Where checkpoints, configuration, TensorBoard logs and the exported model are saved |

:::{enrichment} Key code parameters
- **Start point.** Training starts from a random initialization and reads no existing model.
- **Scene and task.** The robot model is `lab/assets/microduck_contact.usda` (the full-body collision model); the task, rewards and randomization come from the runtime code arranged into `/workspace/state` when the container starts.
- **Output.** Checkpoints (one every 250 iterations), `env.yaml` and `agent.yaml`, TensorBoard logs and the exported model go to `outputs/walking-training-01`, under the host repository's `outputs/`.
:::

**How long does a full training take?** The course's pretrained result was trained on a DGX Spark without a window: 1024 environments × 2500 iterations, a training phase of 7246 seconds (about 2 hours), roughly 4.8 minutes per 100 iterations, with other jobs on the GPU at the time. That run used the original official collision model, while the command above uses the full-body collision model, and `--show` adds rendering on top, so your time will differ.

**You do not need to wait for it to finish.** Once the window appears and the terminal prints reward and episode length for each iteration, watch for a few minutes: in the first tens of iterations most ducks fall quickly and are reset one by one, and as iterations go by more and more stay up.

When you have seen enough, press Ctrl+C; the container keeps running. Checkpoints already saved (one every 250 iterations) remain, but `exported/policy.onnx` is exported only when training finishes. The test below uses the course's pretrained policy by default; to let your own training finish, you can drop `--show` to speed it up, then select your own model explicitly.

:::{course-image} media/isaac-ui-ppo-training.png
:alt: The complete Isaac Sim window of a training run with --show: many Microduck copies on a white floor at different points of their gait, some standing, some stepping, some tipping over; RTX real-time rendering and Newton MJWarp physics in the toolbar.

The training window opened by `--show`, from a real training run; it only shows what training looks like at the start. Your command uses 1024 environments, so the window shows more, denser ducks.
:::

{#watch}
## Training time and how to judge the result

The course repository includes the checkpoint, exported model, configuration and TensorBoard log of this full training (`datasets/walking-ppo` about 12 MB). The TensorBoard curves below use that log; the green numbers match the explanations that follow.

:::{course-image} media/tensorboard-walk-full.png
:alt: The TensorBoard scalars page with four charts outlined in green: 1 mean episode length, 2 mean episode reward, 3 linear-velocity tracking error, 4 yaw-rate tracking error. The horizontal axis is the training iteration.
:no-zoom-label:

Real TensorBoard record of the full training (1024 environments × 2500 iterations). The curves use TensorBoard's default 0.6 smoothing; the faint lines are the raw values.
:::

Read them by number:

1. **Mean episode length** rises from about 23 to about 870 steps (an episode lasts at most 1000 steps, 20 seconds): the robot goes from falling almost immediately to staying up for most of the episode. This is the first sign that balance has been learned.
2. **Mean episode reward** rises quickly over the first 800 iterations and then fluctuates around 80 without clear further gains. A plateau does not mean the skill is perfect, only that more training with the same settings helps less.
3. **Linear-velocity tracking error** falls from about 0.38 to about 0.22 m/s: forward and sideways tracking improves, but the error is still sizeable.
4. **Yaw-rate error** falls from about 1.9 to about 0.7 rad/s and stays large: a warning that turning may not have been learned and needs its own test.

```{training-chart}
data: walk-full-training-scalars.json
title: 1024 environments · 2500-iteration training
badge: Training phase 7246 s
metrics:
- key: Train/mean_episode_length
  label: Mean episode length / control steps
- key: Train/mean_reward
  label: Mean episode reward
- key: Metrics/base_velocity/error_vel_xy
  label: Linear-velocity error / (m/s)
- key: Metrics/base_velocity/error_vel_yaw
  label: Yaw-rate error / (rad/s)
ticks: [1, 500, 1000, 1500, 2000, 2500]
x_label: Training iteration
iteration_label: Iteration
caption: One raw (unsmoothed) logged value every 25 iterations from the TensorBoard event file of the same run; hover to read the values.
```

Better curves show that the training objective is improving; whether the robot walks on command is checked in an independent test. The course tested the policy exported by this run with fixed commands: 4 environments per command, 20 seconds each, randomization off.

| Command | No fall | What happened |
|---|---|---|
| Forward 0.1 m/s | 4/4 | Follows the command; mean velocity error about 0.056 m/s |
| Sideways 0.06 m/s | 4/4 | Stays up but moves far too little sideways |
| Turn left 0.4 rad/s | 4/4 | Hardly turns; yaw-rate error about 0.39 rad/s |

None of the 12 tests fell; forward followed the command, while sideways and turning did not. This is exactly what the fourth curve warned about: 2500 iterations from a random start learned balance and walking forward, but not turning on command. The official model comes from a separate training pipeline by Pollen Robotics, and the following sections keep it as the fixed reference.

| What you see | Check first | Verify next |
|---|---|---|
| Reward rises but falls are still frequent | Whether reward terms mask each other | Run fixed commands and record fall times |
| Episodes get longer but the robot barely moves | Velocity-tracking and posture rewards | Compare actual and commanded velocity |
| Forward works, turning does not | Turning reward weight and training length | Test turning commands on their own |
| NaN in losses or actions | Numerical state, configuration and logs | Keep the failed output; fix the problem before training again |

A very short run makes the gap even clearer: after only a few tens of iterations the reward is already rising, yet the exported policy usually falls within a second. A workflow that runs is not a policy that has learned the skill.

:::{enrichment} Appendix: open TensorBoard yourself
TensorBoard reads the event files in a training output folder. In a second container shell (`docker exec -it microduck-workshop bash` on the host), run `python3 scripts/run.py tensorboard --logdir datasets/walking-ppo --port 6006`, then open the printed address in a browser on the host; the container uses the host network. Change `--logdir` to `outputs/walking-training-01` to see the curves your own run has recorded so far, including an interrupted one. Typing `Train/mean_(reward|episode_length)$|error_vel_(xy|yaw)` in the search box filters the four charts above.
:::

{#test}
## Test the trained policy

The command below puts the course's pretrained policy (the ONNX exported by the full training above) into the same scene as Section 2.2. `--model` names the file to test explicitly; course commands without it still use the official model.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py walk \
  --model datasets/walking-ppo/exported/policy.onnx \
  --velocity 0.25 0 0 \
  --seconds 12 \
  --seed 202 \
  --output outputs/ppo-walk-test-01
```

:::{enrichment} Key code parameters
- **`--model`.** Reads `/workspace/Microduck-Workshop/datasets/walking-ppo/exported/policy.onnx` in the container, the course's pretrained policy in the host repository's `datasets/walking-ppo/` (read-only). The command reads only the file given by `--model`.
- **`--velocity 0.25 0 0`.** Forward, sideways and turning velocity command; the program first sends about 2 seconds of zero velocity, then keeps sending this command.
- **`--seed`.** The same value as the recorded run in Section 2.2, so the two results compare directly.
- **Start.** The joints are first locked in the default standing pose and the robot settles on the ground (nothing supports the body); the lock is released after the policy computes its first action and before that action is applied, so no drop from the air adds extra wobble.
- **Output.** `report.json` and `trajectory.json`, with one position every 0.1 s, go to the folder given by `--output`.
:::

:::{course-image} media/isaac-ui-ppo-walk-test.png
:alt: The complete Isaac Sim window while the command above tests the pretrained PPO policy: one Microduck stepping on flat ground, RTX real-time rendering and Newton MJWarp physics in the toolbar.

The Isaac Sim window when the course ran the command above in the container.
:::

The settings match the recorded run in Section 2.2, so the results compare directly: first check for falls, then compare distance and direction in `trajectory.json`. In the course's run the robot did not fall in 12 seconds and covered about 1.11 m during the forward command (about 0.11 m/s on average), but drifted right, about 26° overall and about 37° in the last 2 seconds; the official model in Section 2.2 moved at a similar speed but drifted left, about 8° (about 15° in the last 2 seconds). Both walk, but neither tracks the command closely. Change `--velocity` to `0 0 0.4` to test turning and see whether it walks but does not turn.

**Testing your own policy.** Only when your training has run to completion and `outputs/walking-training-01/exported/policy.onnx` exists, change `--model` in the command to that path by hand and use a new `--output` folder. One successful run is one sample; to draw conclusions, repeat with several commands and from different starts, and keep the failures.

::::{quiz}
Why turn off the random pushes and parameter randomization of training when testing?

:::{answer}
This exercise first turns off pushes and parameter randomization to compare basic responses under nominal conditions. To evaluate robustness, define the distributions and test seeds in advance, expose each policy to the same parameter draws and disturbances, and report every result. Randomized evaluation can still be reproducible.
:::
::::

{#next}
## Next: from flat ground to slopes

Domain randomization exposes the policy to varying parameters, but the ground is still flat. Even a policy that walks well on flat ground may not lift its feet high enough, place them right or shift its weight correctly on a slope. Section 3.3 first discusses how rewards should be designed, then extends randomization to terrain so the robot walks steadily up and down slopes.

{#references}
## References

{.references}
- [Isaac Lab: Microduck task configuration ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
- [Tobin et al.: domain randomization for sim-to-real transfer ↗](https://arxiv.org/abs/1703.06907)
- [Original PPO paper ↗](https://arxiv.org/abs/1707.06347)
