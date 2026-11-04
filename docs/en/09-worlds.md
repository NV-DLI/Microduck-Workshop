---
myst:
  html_meta:
    description: 'After a task has been made reliable on the green pitch, how environment domain randomization and Cosmos 3 data augmentation can prepare a model for the different floors, lighting and layouts of real homes; which changes keep the original action labels and which require new simulation.'
---

# 5.3 From the pitch to more environments

:::{lesson-intro}
:meta: Section 5.3 · Estimated learning time: 35 minutes

After a task has been made reliable on the green pitch, how environment domain randomization and Cosmos 3 data augmentation can prepare a model for the different floors, lighting and layouts of real homes; which changes keep the original action labels and which require new simulation.
:::

{#first}
## Make one scene reliable first

The shots in this course (keyboard football in Section 2.1, the onboard policy in Part 4 and the GR00T walking shot in Section 5.2) all use the same green football pitch: the same field, goal, ball and lighting. That is deliberate. Only when a task is stable in one fixed scene are failures easy to explain; if the original scene cannot be completed reliably, adding ten rooms only makes problems harder to locate.

A real robot will not stay on a green pitch, though. Floors at home may be wooden boards, tiles or carpet; light differs between day and night; the background holds sofas, table legs and clutter. Once shooting is reliable on the fixed pitch, the next question is how to expand the environment step by step. This section covers ideas and principles only.

{#why-variation}
## Why a real robot needs environment variation

A model trained only on the green pitch may have learned cues we do not want it to rely on, such as “a white spot on a green background is the ball” or “the goal always appears within this brightness range”. In simulation those cues always hold, so the model has no reason not to use them; on a light wooden floor they vanish and the model may lose the ball.

There are other gaps between simulation and reality: real cameras have exposure changes, motion blur and noise; real lights cast shadows and reflections; real floors have friction that differs from the simulated setting. Together these are called the sim-to-real gap. Letting a model see sufficiently varied conditions during training is one of the main ways to narrow it.

:::{scene-cards}
---
cards: |
  media/environment-pitch-striped.webp | Striped pitch
  media/environment-pitch-plain.webp | Plain pitch
  media/environment-marble.webp | Light stone floor
  media/environment-wood.webp | Wooden floor
  media/environment-furnished-room.webp | Furnished room
---

The same Microduck in different visual and spatial conditions. Compare floor texture, reflections, shadows and furniture in the camera background. The furnished room uses the official Isaac Sim Simple Room asset. These static illustrations explain environmental variation; they are not shooting-validation results.
:::

{#domain-randomization}
## Environment domain randomization

Sections 3.3 and 3.4 already used domain randomization to vary servo characteristics, friction and terrain: rather than matching one real environment exactly, selected conditions vary randomly within a range, ideally with the real environment inside it. GR00T, which reads images, also needs visual variation such as lighting, materials and backgrounds. Choose ranges for the intended setting and evaluate unseen combinations and boundary conditions.

| Category | Conditions that can vary | Effect on the shooting task |
|---|---|---|
| Appearance | Floor material and pattern, walls and background, unrelated furniture and clutter | Changes the image; the actions need not change |
| Lighting | Brightness, color temperature, light direction, shadows | Changes the image; the actions need not change |
| Camera | Exposure, small mounting-angle offsets, noise | Changes the image; larger mounting changes also alter the visible geometry |
| Starts | Robot and ball positions, headings and distances | Changes both the image and the required actions |
| Physics | Floor friction, ball mass and bounce, joint characteristics | Changes the motion outcome; the original actions may no longer work |

A useful rule of thumb: **think about what you expect to change in the real scene, and randomize that**. Too narrow a range misses real conditions; too wide a range gives the model too much to learn and can make it conservative everywhere.

Randomization must also keep the task itself unchanged. The goal size, the ball size, the robot’s body and the criterion “a goal counts as success” should not drift along with the appearance. Changing appearance is meant to teach the model to ignore irrelevant variation, not to turn the task into a different one.

{#initial-state-variation}
## Randomize starts first, then expand rooms

Start variation usually comes first: on the same green pitch, change the robot and ball positions and test whether the model completes the task from unseen combinations; then widen the range of headings and distances; only then introduce indoor backgrounds, lighting and floor appearance.

When starts or physics change, the robot needs different actions, so the physics simulation must be rerun and the teacher must provide new actions under the new conditions; the action labels of the original trajectory no longer apply. Only pure appearance changes (same positions, same motion, just a different-looking image) can possibly keep the original action labels.

{#diagnose}
## “Can’t see” or “can’t do”?

After moving to new environments, failures have different causes and need different help:

| Observed failure | Likely help | What to verify next |
|---|---|---|
| Loses the ball under new lighting | Visual data coverage and appearance randomization | Whether finding the ball and full shots both recover under lighting not seen in training |
| Sees the ball but drifts off to one side | Corrective demonstrations for drifted states | Whether corrective data covers the drifted states the model actually reaches |
| Unsteady or falls after kicking on a new floor | Physics randomization and adaptation of the low-level skills | Whether measured friction and contact fall within the simulated range |
| Furniture blocks the original path | Layout variation and an executable detour skill | Whether an executable route exists in the new layout and the shot still succeeds after detouring |

Appearance randomization changes “what it looks like”; physics randomization changes “how it actually moves”. Design and verify them separately.

{#navigation}
## Navigate complex environments with Isaac Lab 3 and Nav2

Furniture, narrow passages and slopes separate navigation into two questions: which route to take, and how to walk it reliably. One design is to train a terrain-appropriate locomotion policy in Isaac Lab 3 and connect the ROS 2 Nav2 stack through Isaac Sim's ROS 2 interfaces.

```{learning-flow}
label: From a map to actual steps
nodes:
- ["Map and localization", "Pose, obstacles and traversable space"]
- ["Nav2 navigation", "Plan a path and request local velocity"]
- ["Locomotion policy", "Use body feedback to generate joint actions"]
- ["Simulation and feedback", "Update pose, contacts and obstacle observations"]
details:
- Supply consistent transforms, odometry, a map and obstacle observations. Include robot dimensions and clearance margins when identifying traversable space.
- Nav2 belongs to ROS 2. A global planner finds a route and a local controller tracks it while responding to obstacles. Adapt and limit commands to the robot's actual velocity and turning capabilities.
- Isaac Lab 3 can train the walking policy. It reads velocity requests plus joint and IMU feedback, then generates joint targets to handle balance and ground contact.
- Isaac Sim provides scenes, sensors and ROS 2 interfaces. New pose and obstacle information feeds navigation; body state feeds the walking policy.
layout: learning
loop: true
caption: This is a design for more complex environments. The simple pitch in section 4.2 retains its direct straight approach and does not require this navigation stack.
```

A geometric route does not establish that the robot can walk it. Check slopes, steps, low obstacles and narrow passages against traversability, body clearance and policy capabilities. A conventional 2D costmap does not teach bipedal foot placement. Validate walking and recovery on the target terrain before integration, then evaluate localization, obstacle avoidance and whole-task outcomes separately.

The official [Isaac Sim ROS 2 navigation tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/robot_control/tutorial_ros2_navigation.html) and [RL controller tutorial](https://docs.isaacsim.omniverse.nvidia.com/latest/ros2_tutorials/robot_control/tutorial_ros2_rl_controller.html) demonstrate the two interfaces separately and provide a starting point for this composition.

{#direct-joints}
## Another option: GR00T outputs the 14 servo actions directly

In the previous section, we designed GR00T to output 8D high-level requests, and the skill execution layer turns them into targets for the 14 servos (Section 5.2). One could also remove that middle layer and let GR00T output joint targets directly:

```{learning-flow}
label: "Idea: GR00T outputs the 14 joint targets directly"
nodes:
- ["Observation and instruction", "RGB images, task instruction and body state"]
- ["GR00T", "Predicts a chunk of actions"]
- ["Joint target sequence", "Ordered in time, 14 target angles per step"]
- ["Joint controller", "Servo position control, limits and safety protection"]
details:
- The input types remain camera images, the task instruction and body state; the state encoding and action labels must be redesigned for the 14-joint interface.
- The model predicts the actions for a short stretch of time ahead.
- Every step gives the 14 servo target angles directly, without the walking policy.
- The servos still need position control, and targets still need clamping and rate limits; when inference times out or the output is abnormal, the robot should enter a controlled stop.
layout: learning
loop: true
caption: This is only a design idea; the course has not trained or validated this kind of control.
```

The potential benefit is flexibility: movements are no longer limited by the low-level skills, and one model might learn a wider range of tasks. The price is much stricter requirements:

- **Data**: many more and more varied demonstrations, including recovery after losing balance, because no low-level policy keeps the robot balanced any more.
- **Latency and balance**: action chunks can run step by step, but the model still needs timely corrections from fresh observations. Continuous execution, expired actions and recovery from imbalance must be handled; remote inference also adds network delay.

{#cosmos}
## Cosmos 3: add visual variation to the same actions

Changing materials and lights in simulation needs assets for every condition and a fresh render. A complementary route is to change the appearance of existing video directly with a generative model.

NVIDIA Cosmos is a platform of world foundation models for physical AI. Cosmos 3 brings scene understanding (reasoning), video generation (world generation) and action generation into one model family. For this task, a use worth studying is video transfer: starting from an existing simulated demonstration video, structural conditions such as edges, depth or segmentation constrain where objects are and what shape they have, while a text prompt changes the lighting, floor or room appearance to produce new candidate videos.

```{learning-flow}
label: Change visual appearance; check whether action labels still hold
nodes:
- ["Reliable source trajectory", "Images, state, executed actions"]
- ["Cosmos 3 transfer", "Generate constrained appearance"]
- ["Check data consistency", "Geometry, contact, timing"]
- ["Train and compare", "Autonomous success in unfamiliar scenes"]
details:
- The source data first needs correct actions and timing. Behavior the original demonstration never taught cannot be added by changing its appearance.
- Structural conditions help keep the layout while the prompt changes lighting or room appearance. Generated results are only candidates; even with constraints, some frames can be wrong.
- Check the size and position of the ball and goal, the robot’s shape, the moment of contact and consistency across frames. If the ball is generated on the other side of the foot, the original action label no longer holds and the clip should be discarded.
- Compare against a model trained without generated data, with the same demonstrations and training budget; keep new rooms or new combinations for testing to judge whether the augmentation helps.
layout: learning
loop: false
```

When augmenting data with generated video, several constraints must hold:

- **Geometry must be preserved.** The relative positions of the ball, goal, camera view and robot must not change. An action label means “what to do given this geometry”; if the geometry changes, the label is invalid.
- **The meaning of the action labels must be preserved.** Generation changes only the images; the state and the 8-D requests are reused unchanged. That is valid only when the appearance change does not affect what should be done.
- **It must be consistent over time.** The walking-shot GR00T reads four history images spanning about 5.8 seconds at once. If earlier and later frames of the same clip are generated as different rooms or lighting, the model sees a scene that could never happen.
- **Physics does not happen again.** A generated video changes only the picture; how the ball rolls and how the feet touch the floor are not recomputed. Swapping the carpet for a more slippery floor may break the original gait and kick timing; changes involving friction, mass or contact must be collected again in physics simulation.

So the role Cosmos 3 suits is a **supplement of appearance diversity**: letting the model see more lighting and room appearances while geometry, actions and physics stay the same. It cannot replace new action demonstrations, nor prove that the actions can be executed in a new environment; reliability still has to be tested in closed-loop simulation and then on the real robot.

::::{quiz}
If a video of the same actions is given ten different backgrounds, is that ten times as much action experience?

:::{answer}
No. It broadens the visual conditions, but the coverage of actions and states stays the same. Learning to recover from drift, take another route or adapt to a new floor needs corresponding new trajectories or corrective data.
:::
::::

::::{quiz}
Why can the original action labels be kept after a lighting change, but not after switching to a more slippery floor?

:::{answer}
Lighting changes only the images; the motion of the ball and robot is unchanged, so the original actions remain correct. A slippery floor changes the gait and contact, so the original actions may not be executable; the physics simulation must be rerun and new actions recorded.
:::
::::

{#references}
## References

{.references}
- [NVIDIA technical blog: develop physical AI reasoning, world and action models with Cosmos 3 ↗](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/)
- [NVIDIA Cosmos examples and workflows ↗](https://github.com/NVIDIA/cosmos)
- [Cosmos 3 video transfer cookbook: new appearance from edge, blur, depth and segmentation controls ↗](https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/generator/transfer/README.md)
- [NVIDIA Cosmos ↗](https://www.nvidia.com/en-us/ai/cosmos/)
- [Domain randomization for transferring from simulation to the real world ↗](https://arxiv.org/abs/1703.06907)
- [Isaac Lab-Arena ↗](https://github.com/isaac-sim/IsaacLab-Arena)

- [Isaac Sim environment assets ↗](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_environments.html)
