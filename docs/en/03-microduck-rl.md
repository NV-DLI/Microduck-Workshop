---
myst:
  html_meta:
    description: 'Map reinforcement learning onto Microduck''s walking task: what the observation, velocity command, action and reward are, what the Actor and Critic each read and do, and why deployment needs only the Actor.'
---

# 3.2 Reinforcement learning for Microduck

:::{lesson-intro}
:meta: Section 3.2 · Theory · Estimated learning time: 20 minutes

Map the reinforcement learning of Section 3.1 onto Microduck's walking task: what the observation, velocity command, action and reward are, what the Actor and Critic each read and do, and why deployment needs only the Actor.
:::

{#task}
## Walking as a reinforcement-learning task

Sections 2.1 and 2.2 ran the official, already trained walking policy. What it has to learn: receive forward, sideways and turning velocity commands, coordinate 14 joints, and follow the command without falling. Here is how the concepts of Section 3.1 map onto this task:

| Reinforcement-learning concept | In Microduck's walking task |
|---|---|
| Observation | 61 values: body angular velocity, gravity direction, angles and velocities of the 14 joints, the previous action, and velocity, head and body-pose commands (order in <a href="02-anatomy.html#policy-interface">Section 2.2</a>) |
| Command | Forward −0.4 to 0.4 m/s, sideways −0.3 to 0.3 m/s, turning −1.0 to 1.0 rad/s, resampled every 3–8 s; the share of environments practicing standing still starts at 2% and rises step by step to 25% |
| Action | 14 values: target joint-angle offsets from the default stance, output every 20 ms (50 Hz) |
| Reward | A weighted sum of velocity tracking, staying upright, posture, foot lift and placement, smooth actions and more |
| Episode end | A body tilt beyond 70° counts as a fall; otherwise at most 20 s (1000 control steps) |

During training, Newton advances thousands of environments on the GPU at once, Isaac Lab computes each environment's observations, rewards and terminations, and RSL-RL's PPO updates the networks from that data. PPO needs no downloaded dataset: the training data is the interaction these environments produce in real time.

{#actor-critic}
## How the Actor and Critic divide the work

```{learning-flow}
label: The Actor that runs at deployment
nodes:
- [61-D observation, State and commands the robot can measure]
- [Actor, 512-256-128 fully connected network]
- [14-D action, Joint target offsets]
details:
- The observation contains only quantities also available on the real robot: IMU, joint encoders, the previous action and the commands. Noise is added during training to imitate sensor errors.
- The Actor has three hidden layers (512, 256, 128). During training it outputs an action distribution and samples from it to explore; the exported deployment keeps only the deterministic output.
- The 14 values are added to the default stance to give each joint's target angle, which goes to the servo model and the physics simulation.
caption: This is the part that runs at deployment. The Critic is not on this path; it works alongside the Actor only during training.
```

The Actor and Critic read their own observations and work in parallel:

| | Actor (policy) | Critic (value function) |
|---|---|---|
| Input | The 61-D observation, with noise | The Actor's observation plus base linear velocity, foot height, air time, contact state and contact forces, without noise |
| Output | 14 joint actions | One number: the return expected from the current state |
| Role in training | Interacts with the environment and produces actions; updated by the advantages | Provides the expected return used to compute each action's advantage; itself updated from actual returns |
| Deployment | Exported to ONNX and run on the robot | Not needed |

The Critic can use simulation information the real robot cannot measure because it is never deployed; the better its estimate, the more reliable the advantages and the steadier the Actor learns. The Actor must rely only on observations available on the real robot, or it could not run after export.

{#reward-design}
## What the reward judges

Rewarding only velocity tracking can produce unstable motion; rewarding only upright posture can teach the robot to stand still. The reward therefore has several terms, each answering one question:

| Reward term | Weight | What it asks |
|---|---|---|
| Linear-velocity tracking | 2.0 | Is the horizontal velocity close to the command, without bouncing up and down? |
| Angular-velocity tracking | 2.0 | Does the angular velocity about the body z axis follow the turning command, while roll and pitch rates stay small? |
| Upright | 2.0 | Does the body stay vertical? |
| Posture | 1.0 | Do the joints stay near the default stance? |
| Air time | 3.0 | Does it take full steps instead of shuffling? |
| Swing height and clearance | −1.1, −2.0 | Does the swing foot lift to a suitable height? |
| Foot slip | −0.1 | Does a foot in contact slide? |
| Action rate | −0.1 at first, increased during training | Does the action change abruptly between steps? |

The weights come from the `env.yaml` saved with the full training run. Tracking rewards are largest at zero error and shrink as the error grows; terms with negative weights are penalties. Added together, only “following the command while walking steadily and smoothly” scores high. Section 3.3 shows the actual code of the linear-velocity term, and Sections 3.4 and 3.5 continue with reward design.

::::{quiz}
The Critic's input includes base linear velocity and foot contact forces, the Actor's does not. Why not give these to the Actor too?

:::{answer}
The Actor is deployed on the robot, so it can use only quantities the real robot measures reliably; base linear velocity and contact forces are hard to measure accurately on hardware. The Critic is used only in training and can read this simulation information to estimate return more accurately.
:::
::::

{#next}
## Next

Section 3.3 starts this training in Isaac Lab, watches thousands of parallel environments, reads the core reward and PPO code, and introduces domain randomization, which the training uses to cope with differences between simulation and the real robot.

{#references}
## References

{.references}
- [Isaac Lab: Microduck task source ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
- [RSL-RL 5.5.1 algorithm source ↗](https://github.com/leggedrobotics/rsl_rl/tree/v5.5.1/rsl_rl/algorithms)
