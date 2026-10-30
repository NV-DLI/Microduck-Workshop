---
myst:
  html_meta:
    description: Drive Microduck with the keyboard to walk, turn and kick, switch to roller skates in the same window, then see how Isaac Sim connects the scene, the robot and pretrained policies.
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 2.1 Meet Microduck: play football first

:::{lesson-intro}
:meta: Section 2.1 · Estimated learning time: 20 minutes

Drive Microduck with the keyboard to walk, turn and kick, switch to roller skates in the same window, then see how Isaac Sim connects the scene, the robot and pretrained policies.
:::

{#meet}
## A small robot that learns new skills

Microduck is an open-source biped from Pollen Robotics and Hugging Face. It has 14 actuated joints and can be trained to walk, kick and learn other skills. We use NVIDIA Isaac Sim to build repeatable scenes and Isaac Lab to organize training and evaluation, step by step, until the robot finds the ball and scores on its own. No training yet: load the official pretrained walking and right-foot kick ONNX policies and take a manual shot.

{#play}
## Open the pitch and try the keyboard

Run the command below in the course container shell from Section 1.3 (to reopen it, see <a href="setup.html#start">Section 1.3</a>). It opens Isaac Sim with the same green pitch, ball and goal as the autonomous shot in Part 4.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py manual \
  --seconds 600 \
  --output outputs/manual-football-01
```

The window opens in **walking mode**. Switch your input method to English, click the simulation window, and use the keys in the table; if keys do nothing, an input method is usually capturing them. Press **M** for roller mode: the same window rebuilds the pitch with roller feet, which takes some tens of seconds. The key help is also in the “Microduck • Keyboard football” tab on the right of the window.

:::{enrichment} Key code parameters
- **Models.** `run.py manual` needs no model argument: it reads `/workspace/Microduck-Workshop/models` in the container, which is the host repository's `models/` (read-only). Walking mode uses `walk.onnx` and `kick-teacher.onnx`; roller mode uses `official-browser/rollers.onnx`.
- **Scene assets.** The pitch, ball, goal and robot come from the repository's `lab/assets/`: `pitch_A.usda`, `official_ball.usda`, `steel_goal.usda`, `microduck_contact.usda`, and `rollers/microduck_rollers_analytic.usdc` for the roller feet.
- **Start.** The duck first settles on the pitch with its joints locked, and the lock is released when the policy takes control; M works only after that.
- **Switching modes.** M closes the current scene and rebuilds the pitch in the same window with the other robot configuration and its policy.
- **Output.** `outputs/manual-football-01`, given by `--output`, is under the host repository's `outputs/`. Each switch starts a numbered subfolder such as `00-walking` and `01-rollers`, so the two modes keep separate records. `--seconds` is the longest simulated time the window runs.
:::

:::{course-image} media/isaac-ui-keyboard-walking.png
:alt: The complete Isaac Sim window of keyboard football in walking mode: Microduck, ball and goal on the green pitch, and the walking keys listed in a tab on the right.

Walking mode. The screenshot comes from a real session of the command above in the course container.
:::

| Key | Walking mode | Roller mode |
|---|---|---|
| W / S | Walk forward / backward | Push / brake (holding S can start rolling backward) |
| A / D | Step left / right | Not used |
| Q / E | Turn left / right | Turn left / right |
| Release keys | Zero-velocity command; return to standing | Keep coasting; it may not stop at once |
| Space | One right-foot kick, then standing | One time-limited gliding-shot assist; no kick |
| M | Switch to roller mode | Switch to walking mode |
| R / Esc | Reset the current scene / quit | Reset the current scene / quit |

**Walking mode:** walk next to the ball and stop with the ball in front of the right foot and the body facing the goal, then press Space. If you miss, adjust your stance; if the robot loses balance, press R. You decide where the ball is and where to go; the policy only turns your commands into joint motion. When a shot misses, ask whether the stance, the heading or the timing was off: those are exactly the decisions Part 4 hands to an autonomous shooting system. Standing after you release the keys is also done by the walking policy: it keeps running and holds a stance under a zero-velocity command.

<span id="rollers"></span>**Roller mode:** the four wheels are passive; the same 14 servos shape the legs to push, steer and brake, and the robot never switches to walking or kicking. Notice that it keeps coasting when you release the keys, and practise slowing down early; the goal is to roll the ball into the goal and stop in front of it. W gives a push amount, not a speed in metres per second. Compare the two styles: a walking shot stops first and then kicks; a roller shot transfers momentum through contact while moving, then brakes.

:::{course-image} media/isaac-ui-keyboard-rollers.png
:alt: Roller mode in the same window after pressing M: the Microduck on the same pitch now has wheeled feet, and the tab on the right lists the roller keys.

Roller mode after pressing M, from the same session and window as the previous screenshot: the scene was rebuilt, the feet swapped for roller feet, and the key help changed with them.
:::

:::{admonition} The two kinds of feet are two different robot configurations
:class: note

Plain feet and the official roller feet differ in structure, mass and collision shape, so switching modes rebuilds the scene with the other robot configuration and its official policy. It is like changing the robot's feet on the bench; it does not mean the real robot can transform itself while moving. The four tyres keep the official appearance, mass and inertia; to compute rolling contact stably, their collision shapes are cylinders 30 mm in diameter and 7.6 mm wide. Real tyre friction and deformation have not been calibrated.
:::

{#scene}
## How Isaac Sim builds the experiment

```{learning-flow}
label: Scene to motion
nodes:
- [USD scene, 'Robot, pitch, ball and goal']
- [Physics, 'Mass, contact and motors']
- [Policy and input, Keys → ONNX → joint goals]
details:
- USD places the robot, ball and goal in a shared space. Geometry and materials also affect camera images.
- Mass, inertia, friction and colliders determine contact and motion. Identical appearances can have different physics; realism alone does not validate dynamics.
- Keys request a direction and speed; the ONNX policy converts observations and commands into joint targets.
caption: Rendering, physics and policy have separate roles. A new skill reuses the same scene, and moving the ball needs no retraining of the walking policy.
```

Beyond the visible mesh, the robot USD holds joints, collision shapes and mass properties: joints define connections, collision shapes decide where contact happens, mass and inertia shape acceleration and turning, and the servo model decides how a target angle becomes motion. The code below sets the official ball's 5 cm radius and 30 g mass. Isaac Sim provides the scene and rendering; the Isaac Lab task configuration connects the Newton physics backend.

```{code-story}
label: Python · Core excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/manual.py
language: python
code: |
  cfg.scene.ball = RigidObjectCfg(
      prim_path="{ENV_REGEX_NS}/Ball",
      spawn=sim_utils.SphereCfg(
          radius=0.05,
          mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
          collision_props=sim_utils.CollisionPropertiesCfg(),
          rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
      ),
      init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
  )
  # Apply the shared official-ball asset, goal and floor materials.
  scene_profile = configure_football(cfg, style="A")
steps:
- lines: [0, 2]
  title: Register the ball in the scene
  body: '`RigidObjectCfg` declares a rigid body; `{ENV_REGEX_NS}` gives every parallel environment its own ball.'
- lines: [2, 8]
  title: Set shape and physical properties
  body: A 0.05 m radius and 0.03 kg mass, with collision and rigid-body dynamics enabled so Newton computes how the ball moves when kicked.
- lines: [8, 10]
  title: Place it on the floor
  body: The initial height of 0.05 m equals the radius, so the ball just touches the floor.
- lines: [10, 12]
  title: Apply the shared football scene
  body: '`configure_football` swaps in the official ball asset and adds the goal and floor materials; `style="A"` selects the green striped pitch used throughout the course.'
```

{#scene-contract}
## Fix the scene and hardware contract

Keyboard football and the GR00T shot in Part 4 use the same 1 × 1.6 m green pitch, the official black-and-white ball (10 cm diameter, 30 g) and a fixed steel-mesh goal. Scoring uses the goal's clear opening, 38.8 cm wide and 24.4 cm high. Part 3 uses the training surfaces and test terrain described in each section; Section 4.4 discusses how other pitches, lighting or indoor backgrounds affect a learned policy.

:::{enrichment} Scene parameters
- **Goal.** Outer size 40 cm wide × 20 cm deep × 25 cm high; with a 6 mm frame diameter, the post centres are 39.4 cm apart. Localization uses the centre spacing and scoring uses the clear opening, so keep them apart.
- **Floor.** A rigid plane without foam compression or raised edges; the friction of the mat against a real floor has not been calibrated.
:::

The robot uses official Microduck CAD, base mass and joint parameters, and a full-body collision model with backlash. The course pins the XL330 BAM actuator fit, 200 Hz physics and a 50 Hz policy period. The walking policy controls 14 of the 15 physical servos; it does not drive the mouth. Full-body collisions let the head, legs and battery rest on the ground after a fall instead of sinking through it.

This simulation configuration still differs from the real robot. The camera corresponds to GC2093, but simulation samples at 960 × 540 and 15 Hz with an uncalibrated 70° horizontal field of view; ToF and body IMU models are also approximate. It therefore cannot establish parity with the latest firmware, every hardware error or full deployment on RK3566. Before transfer, check the real hardware revision, voltage, joint offsets, lens parameters and end-to-end delay.

{#policies}
## How walking mode switches between gait and kick policies

| File | Size | Role |
|---|---|---|
| walk.onnx | 793,705 bytes, about 0.79 MB | Walking and standing |
| kick-teacher.onnx | 793,685 bytes, about 0.79 MB | Right-foot kick |

Walking and kicking use pinned official weights (see the version link under References), which may differ from the latest walking weights in the official browser simulator. The roller lab uses the official roller model.

:::{network-diagram}
The policy network reads a 61-dimensional observation, passes it through hidden layers of 512, 256 and 128 units, and outputs 14 joint actions. The official walking and kick policies share this structure with different weights, which is why they behave as different skills. The value network used during training is not shown.
:::

```{code-story}
label: Python · Core excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/manual.py
language: python
code: |
  # Read the state once per policy step, then select the active skill.
  obs = env.observation_manager.compute(update_history=True)["policy"]
  action = walk(obs)

  if kick_start is not None:
      phase = (step - kick_start) * 0.02
      weight = 1.0 if phase < 0.14 else max(0.0, 1.0 - (phase - 0.14) / 0.12)
      if weight:
          action = torch.lerp(action, kick(obs), weight)

  # The action manager maps network outputs to joint targets.
  env.action_manager.process_action(action)
steps:
- lines: [0, 4]
  title: Read the observation once per policy step
  body: The observation manager updates its history and returns the policy observation; the walking policy computes the default action first.
- lines: [4, 7]
  title: Compute the kick weight
  body: The policy period is 0.02 s (50 Hz), so `phase` is the time in seconds since the kick started. The weight is 1 for the first 0.14 s, then falls linearly to 0 over 0.12 s.
- lines: [7, 10]
  title: Blend the two policies' actions
  body: '`torch.lerp` interpolates between the walking and kick actions by that weight; at weight 0 the walking action is used unchanged.'
- lines: [10, 12]
  title: Turn the action into joint targets
  body: The action manager maps network outputs to joint targets for the actuators and physics simulation.
```

The lab plays the kick fully for 0.14 s, then blends back to the walking policy's stance over 0.12 s. These times only shape the handoff between skills; the official weights stay unchanged.

A policy file is under 1 MB, but the full simulation also needs GPU memory for the scene, physics and rendering, so do not size memory by file size. The `inspect-model` command from Section 1.3 prints a policy file's SHA-256, input and output tensor shapes, and the operators it uses.

{#reflect}
## Who decided the next step?

::::{quiz}
Which decisions did you just make that the robot cannot yet make on its own?

:::{answer}
You identified the ball and goal, estimated distance, selected a route and stance, and decided when to kick. Part 4 trains GR00T to learn these decisions from demonstrations; Section 4.4 then discusses environment variation and data expansion.
:::
::::

Section 2.2 looks at the mechanism behind these motions and the rhythm at which the policy and the physics engine work together.

{#references}
## References

{.references}
- [Pollen Robotics: pinned official policies ↗](https://huggingface.co/pollen-robotics/microduck-policies/tree/665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448)
- [NVIDIA: Isaac platform and Microduck ↗](https://nvidianews.nvidia.com/news/nvidia-advances-isaac-platform-for-general-purpose-robotics-with-new-reference-humanoid-frameworks-simulation-pipelines-and-data/)
