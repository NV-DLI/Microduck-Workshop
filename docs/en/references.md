---
myst:
  html_meta:
    description: Find official source code, documentation, and papers by topic.
---

# 6.3 Further learning and references

:::{lesson-intro}
:meta: Reading resources

Find official source code, documentation, and papers by topic.
:::

{#robotics}
## Robotics and simulation

Begin with task observations, actions, and rewards, then explore physics and sensors. Course commands target fixed versions; current documentation may describe newer versions. Check the version before comparing parameters.

{#next}
## Suggested extensions

- Vary the Section 3.3 training budget and compare on the same held-out seeds.
- Pick a new task, such as dribbling the ball to the center circle and stopping, and compare it with the Section 4.2 shooting stages: which perception can be reused, and which stages and transitions must be redesigned?
- Analyse the failed rounds of the Section 4.3 onboard shot: which stage did they stop in, and was the cause observation, the path or the kick conditions?
- Read the [GR00T chapter of the SO-101 Sim-to-Real course ↗](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/10-groot.html) and, with the data design in Section 5.2, identify the shared data-workflow steps and the different action interfaces of Microduck and SO-101.
- Use the Section 5.3 counter to plan held-out room evaluation and separate training from test conditions.

{#references}
## References

{.references}
- [Pollen Robotics: Microduck hardware ↗](https://github.com/pollen-robotics/microduck)
- [Isaac Lab: Microduck task source ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
- [Newton physics engine ↗](https://github.com/newton-physics/newton)
- [Proximal Policy Optimization paper ↗](https://arxiv.org/abs/1707.06347)
- [RSL-RL: reinforcement-learning implementation ↗](https://github.com/leggedrobotics/rsl_rl)
- [Ultralytics documentation ↗](https://docs.ultralytics.com/)
- [VL53L8CX sensor documentation ↗](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8cx.html)
