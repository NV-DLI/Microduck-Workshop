---
myst:
  html_meta:
    description: "Experience Microduck's motion, train walking and slopes with reinforcement learning, see how an onboard policy scores with sensors and hand-designed stages, then learn how GR00T learns the walking shot from demonstrations and consider how to carry it to more environments."
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 1.1 Teach Microduck to score a goal

:::{lesson-intro}
:meta: 6 parts · 17 sections · Simulation and frontier methods
:hero: media/isaac-football-hero-still.webp
:hero-animation: media/isaac-football-hero.webp
:hero-size: 1600x872
:hero-alt: A recording of the full Isaac window, played on the simulation clock: the onboard policy controls Microduck as it scores on the green pitch, alongside synchronized head RGB and ToF sensor views.

Experience Microduck's motion, train walking and slopes with reinforcement learning, see how an onboard policy scores with sensors and hand-designed stages, then learn how GR00T learns the walking shot from demonstrations and consider how to carry it to more environments.
:::

```{toctree}
:caption: 1 Overview
:hidden:

self
setup
```

```{toctree}
:caption: 2 Meet Microduck
:hidden:

01-play
02-anatomy
```

```{toctree}
:caption: 3 Teach Microduck skills with reinforcement learning
:hidden:

03-learning
03-microduck-rl
03-walk-training
04-terrain
05-skills
```

```{toctree}
:caption: 4 Complete the shooting task with an onboard policy
:hidden:

06-perception
06-onboard
```

```{toctree}
:caption: 5 (Extension) Score autonomously with GR00T
:hidden:

07-groot
08-groot-interface
09-worlds
```

```{toctree}
:caption: 6 Reference and help
:hidden:

models
troubleshooting
references
```

{#outcome}
## What you will build

The whole course follows one Microduck, one green football pitch, one ball and one goal: meet the robot and try its motion skills, train walking and slopes with reinforcement learning and domain randomization, run an onboard policy and see how it completes the shot with sensors and hand-designed stages, then learn how GR00T learns the walking shot from demonstrations and consider how to adapt to more environments. Every lab runs in the same course container.

{.pipeline}
1. Overview and container setup
2. Meet Microduck
3. Train skills with reinforcement learning
4. Complete the shooting task with an onboard policy
5. (Extension) Score autonomously with GR00T
6. Reference and help

{#learn}
## By the end, you can

- Explain mechanisms, actuators, observations, actions and rewards.
- Run fixed policies, PPO training and domain-randomization experiments with Isaac Lab, Newton and Isaac Sim in the course container.
- Explain how an onboard policy completes the shot with RGB detection, ToF ranging and hand-designed stages, and what it and action prediction by a vision-language-action model (VLA) are each responsible for.
- Explain how GR00T's 8-D high-level requests divide the work with low-level skills, and how demonstrations, action-head training and closed-loop validation work.
- Design environment variation and held-out evaluation; tell apart a working pipeline, better model metrics and complete task success.

{#route}
## Course route

| Part | Sections |
|---|---|
| 1 Overview | 1.1 Teach Microduck to score a goal · 1.2 Set up the environment and get the code |
| 2 Meet Microduck | 2.1 Meet Microduck: play football first · 2.2 Mechanism and control timing |
| 3 Teach Microduck skills with reinforcement learning | 3.1 Reinforcement learning and PPO · 3.2 Reinforcement learning for Microduck · 3.3 Train and test a walking policy · 3.4 Extension: walking steadily up and down slopes · 3.5 (Extension) Community skills: reward design |
| 4 Complete the shooting task with an onboard policy | 4.1 Find the ball and goal: RGB detection and ToF ranging · 4.2 Designing the shooting stages |
| 5 (Extension) Score autonomously with GR00T | 5.1 Vision-language-action (VLA) models and GR00T · 5.2 Score autonomously with GR00T · 5.3 From the pitch to more environments |
| 6 Reference and help | 6.1 Models and data · 6.2 Troubleshooting · 6.3 Further learning and references |
