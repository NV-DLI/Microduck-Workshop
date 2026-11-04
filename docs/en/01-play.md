---
myst:
  html_meta:
    description: Drive Microduck with the keyboard to walk, turn and kick, then see how Isaac Sim connects the scene, the robot and pretrained policies.
---

# 2.1 Meet Microduck: play football first

:::{lesson-intro}
:meta: Section 2.1 · Estimated learning time: 20 minutes

Drive Microduck with the keyboard to walk, turn and kick, then see how Isaac Sim connects the scene, the robot and pretrained policies.
:::

{#meet}
## A small robot that learns new skills

Microduck is an open-source biped from Pollen Robotics and Hugging Face. It has 14 actuated joints and can be trained to walk, kick and learn other skills. We use NVIDIA Isaac Sim to build repeatable scenes and Isaac Lab to organize training and evaluation, step by step, until the robot finds the ball and scores on its own. No training yet: load the official pretrained walking and right-foot kick ONNX policies and take a manual shot.

{#play}
## Open the pitch and try the keyboard

Run the command below in the course container shell from Section 1.2 (to reopen it, see <a href="setup.html#start">Section 1.2</a>). It opens Isaac Sim with the same green pitch, ball and goal as the autonomous shots in Parts 4 and 5.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py manual \
  --seconds 600 \
  --output outputs/manual-football-01
```

:::{enrichment} Parameter details

The command needs no model argument: it reads the official walking model `walk.onnx` and right-foot kick model `kick-teacher.onnx` from `models/`. `--seconds` is the longest simulated time the window runs, and records go to the folder given by `--output`.
:::

Click the simulation window and use the keys in the table. The key help is also in the “Microduck • Keyboard football” tab on the right of the window.

:::{admonition} Opening Isaac Sim for the first time
:class: note

The first launch may take a few minutes to load assets and compile shaders. If the desktop reports that Isaac Sim is not responding, keep the window open and choose to wait while initialization continues.
:::

:::{course-image} media/isaac-ui-keyboard-walking.png
:alt: The complete Isaac Sim window of keyboard football: Microduck, ball and goal on the green pitch, and the keys listed in a tab on the right.

The keyboard football window: the pitch on the left, the key help in the tab on the right. Look at where the robot, ball and goal are before deciding where to walk.
:::

| Key | Action |
|---|---|
| W / S | Walk forward / backward |
| A / D | Step left / right |
| Q / E | Turn left / right |
| Release keys | Zero-velocity command, return to standing |
| Space | One right-foot kick, then return to standing |
| R / Esc | Reset the current scene / quit |

Walk next to the ball and stop with the ball in front of the right foot and the body facing the goal, then press Space. If you miss, adjust your stance; if the robot loses balance, press R. You decide where the ball is and where to go; the policy only turns your commands into joint motion. When a shot misses, ask whether the stance, the heading or the timing was off: those are exactly the decisions Part 4 hands to an autonomous shooting system. Standing after you release the keys is also done by the walking policy: it keeps running and holds a stance under a zero-velocity command.

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
          rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
          mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
          collision_props=sim_utils.CollisionPropertiesCfg(),
          visual_material=sim_utils.PreviewSurfaceCfg(
              diffuse_color=(0.9, 0.91, 0.88), roughness=0.75
          ),
      ),
      init_state=RigidObjectCfg.InitialStateCfg(pos=(a.ball_x, a.ball_y, 0.05)),
  )

  from scene_style import configure_football

  scene_profile = configure_football(cfg, style=a.style, lighting=a.lighting)
steps:
- lines: [0, 2]
  title: Register the ball in the scene
  body: '`RigidObjectCfg` declares a rigid body; `{ENV_REGEX_NS}` gives every parallel environment its own ball.'
- lines: [2, 11]
  title: Set shape, physical properties and appearance
  body: A 0.05 m radius and 0.03 kg mass, with rigid-body dynamics and collision enabled so Newton computes how the ball moves when kicked; `visual_material` affects only its look.
- lines: [11, 13]
  title: Place it on the floor
  body: The initial height of 0.05 m equals the radius, so the ball just touches the floor.
- lines: [13, 17]
  title: Apply the shared football scene
  body: '`configure_football` swaps in the official ball asset and adds the goal and floor materials; the default `style` is the green striped pitch used throughout the course.'
```

{#scene-contract}
## Fix the scene and hardware contract

Keyboard football and the autonomous shots in Parts 4 and 5 use the same 1 × 1.6 m green pitch, the official black-and-white ball (10 cm diameter, 30 g) and a fixed steel-mesh goal. Scoring uses the goal's clear opening, 38.8 cm wide and 24.4 cm high. Part 3 uses the training surfaces and test terrain described in each section; Section 5.3 discusses how other pitches, lighting or indoor backgrounds affect a learned policy.

:::{enrichment} Scene parameters
- **Goal.** Outer size 40 cm wide × 20 cm deep × 25 cm high; with a 6 mm frame diameter, the post centres are 39.4 cm apart. Localization uses the centre spacing and scoring uses the clear opening, so keep them apart.
- **Floor.** A rigid plane without foam compression or raised edges; the friction of the mat against a real floor has not been calibrated.
:::

The robot uses official Microduck CAD, base mass and joint parameters, and a full-body collision model with backlash. This keyboard lab uses the pinned XL330 BAM actuator fit, 200 Hz physics and a 50 Hz policy period. Section 4.3 uses 100 Hz physics with the same 50 Hz policy period. The walking policy controls 14 of the 15 physical servos; it does not drive the mouth. Full-body collisions let the head, legs and battery rest on the ground after a fall instead of sinking through it.

This simulation configuration still differs from the real robot. The camera corresponds to GC2093, but simulation samples at 960 × 540 and 15 Hz with an uncalibrated 70° horizontal field of view; ToF and body IMU models are also approximate. It therefore cannot establish parity with the latest firmware, every hardware error or full deployment on RK3566. Before transfer, check the real hardware revision, voltage, joint offsets, lens parameters and end-to-end delay.

{#policies}
## How the walking and kick policies switch

| File | Size | Role |
|---|---|---|
| walk.onnx | 793,705 bytes, about 0.79 MB | Walking and standing |
| kick-teacher.onnx | 793,685 bytes, about 0.79 MB | Right-foot kick |

Walking and kicking use pinned official weights (see the version link under References), which may differ from the latest walking weights in the official browser simulator.

:::{network-diagram}
The policy network reads a 61-dimensional observation, passes it through hidden layers of 512, 256 and 128 units, and outputs 14 joint actions. The official walking and kick policies share this structure with different weights, which is why they behave as different skills. The value network used during training is not shown.
:::

```{code-story}
label: Python · Core excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/manual.py
language: python
code: |
  obs = env.observation_manager.compute(update_history=True)["policy"]
  action = walk(obs)
  if kick_start is not None:
      phase = (step - kick_start) * 0.02
      # A bounded official kick, then a smooth handoff to standing.
      weight = (
          1.0 if phase < 0.14 else max(0.0, 1.0 - (phase - 0.14) / 0.12)
      )
steps:
- lines: [0, 2]
  title: Read the observation once per policy step
  body: The observation manager updates its history and returns the 61-D policy observation; the walking policy computes the default action first.
- lines: [2, 8]
  title: Compute the kick weight
  body: The policy period is 0.02 s (50 Hz), so `phase` is the time in seconds since the kick started. The weight is 1 for the first 0.14 s, then falls linearly to 0 over 0.12 s.
```

With the weight computed, the program uses `torch.lerp` to interpolate between the walking and kick actions and hands the result to the action manager, which maps it to joint targets. The lab plays the kick fully for 0.14 s, then blends back to the walking policy's stance over 0.12 s. These times only shape the handoff between skills; the official weights stay unchanged.

A policy file is under 1 MB, but the full simulation also needs GPU memory for the scene, physics and rendering, so do not size memory by file size. The `inspect-model` command from Section 1.2 prints a policy file's SHA-256, input and output tensor shapes, and the operators it uses.

{#reflect}
## Who decided the next step?

::::{quiz}
Which decisions did you just make that the robot cannot yet make on its own?

:::{answer}
You identified the ball and goal, estimated distance, selected a route and stance, and decided when to kick. Part 4 has people write these decisions into an onboard policy; Part 5 shows how GR00T learns them from demonstrations, and Section 5.3 discusses environment variation and data expansion.
:::
::::

Section 2.2 looks at the mechanism behind these motions and the rhythm at which the policy and the physics engine work together.

{#references}
## References

{.references}
- [Pollen Robotics: pinned official policies ↗](https://huggingface.co/pollen-robotics/microduck-policies/tree/665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448)
- [NVIDIA: Isaac platform and Microduck ↗](https://nvidianews.nvidia.com/news/nvidia-advances-isaac-platform-for-general-purpose-robotics-with-new-reference-humanoid-frameworks-simulation-pipelines-and-data/)
