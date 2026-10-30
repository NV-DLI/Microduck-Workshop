---
myst:
  html_meta:
    description: Separate shells, joints, actuators, and sensors, then see how they form a controllable robot.
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 2.2 Mechanism and control timing

:::{lesson-intro}
:meta: Section 2.2 · Estimated learning time: 20 minutes

Separate shells, joints, actuators, and sensors, then see how they form a controllable robot.
:::

```{symbols}
T_hom:
  tex: T
  text: Homogeneous transformation matrix (4×4) that holds a rotation and a translation together, so transforms can be multiplied along a mechanical chain.
R:
  tex: R
  text: '3×3 rotation matrix: the orientation of the child frame relative to the parent frame.'
t_vec:
  tex: \mathbf t
  text: 'Translation vector: the position of the child frame''s origin in the parent frame.'
zero_row:
  tex: \mathbf 0^{\mathsf T}
  text: 1×3 row of zeros; with the 1 in the corner it makes the matrix a 4×4 that can be chained by multiplication.
T_BF:
  tex: '{}^{B}T_F'
  text: 'Transform from the foot frame to the trunk frame: it expresses a point on the foot in body coordinates.'
q:
  tex: q
  text: Vector of all joint angles along this mechanical chain.
q_joint:
  tex: q
  text: Angle of the joint selected in the 3D duck, as the joint coordinate of the simulation model; stored in radians and shown in degrees.
T_joint:
  tex: '{}^{i-1}T_i'
  text: Transform between two adjacent links, set by the assembly position and the angle of this joint.
q_i:
  tex: q_i
  text: Angle of one joint on the chain (rad); the subscript is that joint's place in the chain.
frame_B:
  tex: B
  text: Trunk (body) frame.
frame_F:
  tex: F
  text: Foot frame.
a_t:
  tex: \mathbf a_t
  text: The 14-dimensional action (normalized values) output by the policy in the current control period.
pi_theta:
  tex: \pi_\theta
  text: The walking policy network; the subscript stands for its trained parameters, fixed during inference.
o_t:
  tex: \mathbf o_t
  text: The 61-dimensional observation given to the policy in the current control period.
q_target:
  tex: \mathbf q_t^{*}
  text: Joint target angles sent to the actuators in the current control period (rad).
q_default:
  tex: \mathbf q_{\mathrm{default}}
  text: Joint angles of the default standing pose.
s:
  tex: s
  text: Action scale; 1 for this policy.
rmse:
  tex: \mathrm{RMSE}_v
  text: Root-mean-square error of horizontal velocity tracking (m/s); smaller means the actual velocity follows the command more closely.
N:
  tex: N
  text: Number of sampled instants in the statistic.
t_idx:
  tex: t
  text: Index of a sampled instant.
sqnorm:
  tex: \|\cdot\|_2^2
  text: Squared length of the velocity error vector.
v_xy:
  tex: \mathbf v_{xy,t}
  text: Actual horizontal body velocity at that instant (m/s).
v_cmd:
  tex: \mathbf v_{xy,t}^{\mathrm{cmd}}
  text: Commanded horizontal velocity at the same instant (m/s).
```

{#assembly}
## From assembly to components

Microduck is a small biped: its legs support and move the body, the neck directs its view, and the trunk houses computing, power and connections. In Isaac Sim, the robot model holds joints, collision geometry and mass properties alongside the visual meshes; they define which motions and contacts can happen. Learning this structure first lets you connect joint names and action vectors to motion in simulation.

(duck-explorer)=
```{robot-explorer}
Colours are simplified to show what each part does; they are not the real paint. Part shapes, proportions, joint positions and axes come from the original CAD and the simulation model. The exploded spacing is only there to reveal the structure and is not an assembly sequence.
```

Once the 3D duck has loaded, go through the three views at the top:

- **The whole duck**: find the six parts: head, neck, body, hip, knee and foot. Click a name to see which servos move it, click a part to look closer, and drag to turn the duck.
- **Exploded view**: parts that belong together move apart as a group. The numbered labels match the parts in How motors drive joints and tell how many servos drive each one; a leader starts at each servo housing and they merge before the name. Leg labels count per side. The eye is the camera. The ToF range sensor sits on the same face board, right beside the eye. The CAD has only the board's window for it, not a part for the sensor itself, so a crossed circle marks its position.
- **How motors drive joints**: choose a part, then a motion, or click a servo in the model. The thin white line is the servo's output shaft, and the white paper-ribbon arrow beyond its end shows the drive direction: drag the ribbon, or click the dot in its middle and fine-tune with the arrow keys. Coral is the rotating servo, violet the bracket that turns with it, and parts that stay still are drawn only as light outlines. The leg controls show the left leg; the right leg mirrors it.

All three views use the same parts: head 3 servos (head pitch, yaw and roll), neck 1, beak 1 (fixed in the simulation model), and per side hip 3, knee 1 and foot 1, 15 in total. Servos of one part are not always mounted together: each hip's yaw, roll and pitch servos sit in the body, the hip bracket and the thigh, so in the exploded view three leaders merge from three places into one Hip label.

The panel's $\sym{q_joint}$ is the joint's angle in the simulation model. Code stores it in radians; the panel shows degrees ($1\ \mathrm{rad}\approx 57.3^\circ$), and the sign follows the right-hand rule about the joint axis. When standing, $\sym{q_joint}$ is usually not 0 (the left hip pitch is about $-26.2^\circ$), so the panel also shows the change from the stand pose. A walking policy's actions are offsets added to these stand angles. Real servos also carry their own calibration offsets, so these readings are model joint angles, not raw servo encoder values.

:::{admonition} Hardware revision and simulation joints
:class: note

Of the 15 servos, the simulated policy drives 14 joints; the beak is fixed in the simulation model, so it cannot turn in the 3D duck. The controller board keeps the Raspberry Pi Zero 2 W shape from the CAD, while the current Beta electronics are based on RK3566. The camera, ToF sensor and IMU observe the task and become data sources for learned policies in Section 4.1.
:::

::::{quiz}
In How motors drive joints, compare the left leg's Hip roll and Hip pitch. When the joint turns, does the housing of the servo that drives it move?

:::{answer}
For hip roll, no: that servo is fixed to the hip-yaw bracket above it, and its output shaft turns the bracket below together with the whole leg. For hip pitch, yes: that servo sits in the thigh and its output shaft is fixed to the hip bracket above, so the servo housing turns with the thigh, shank and foot around the shaft. Both mountings drive the joint; they differ only in which side of the joint moves. Neck pitch, head yaw and head roll use the second mounting too.
:::
::::

{#kinematic-chain}
## From mechanism to joint control

This part answers a basic question: how do the 14 numbers a policy outputs become visible robot motion? First find the trunk, both legs and the head in the model, then follow the connection “trunk → hip → knee → ankle → foot”. Later, when you meet joint order, coordinate transforms or foot contact problems, you can locate them in the physical structure.

```{learning-flow}
label: From joint targets to foot position
nodes:
- [Joint target q*, where the policy wants each joint to go]
- [Servos and joints, the actual angle q changes under load]
- [Links and foot, the links determine where the foot is]
details:
- The policy gives target joint angles, which express an intention; a joint needs time to track them.
- Actuators are limited by torque, speed and response, and contact also changes the actual angle, so look at both the joint targets and the actual state.
- The foot position follows from the links and joint angles together. Changing one joint moves the end of the chain and can change the support.
caption: Target and actual angles are not always equal. Inertia, contact and actuator response all shape the final motion.
```

Joint angles set the relative pose of the links; combining these poses along a mechanical chain gives the position of the foot or camera relative to the body. With this relation in mind you can tell “the head turned” from “the body turned”.

:::{enrichment} Further reading: computing positions along a mechanical chain
A part's position only makes sense relative to some frame. A homogeneous transform holds a rotation $\sym{R}$ and a translation $\sym{t_vec}$ together; multiplying them along a mechanical chain converts a point in the foot frame into the trunk frame. This is forward kinematics: computing where the end of the chain is from known joint angles.

```{formula} Homogeneous transform and mechanical chain
\sym{T_hom}=\begin{bmatrix}\sym{R}&\sym{t_vec}\\\sym{zero_row}&1\end{bmatrix},\qquad \sym{T_BF}(\sym{q})= \symx{T_joint}{{}^{B}T_1}(\symx{q_i}{q_1})\,\symx{T_joint}{{}^{1}T_2}(\symx{q_i}{q_2})\cdots\symx{T_joint}{{}^{k}T_F}(\symx{q_i}{q_k})
```

The superscript $\sym{frame_B}$ denotes the trunk frame and the subscript $\sym{frame_F}$ the foot frame; each transform comes from a part's assembly position and its joint angle. A camera mounted on the head likewise needs the neck and head chain to compute where it points, so a head turn is not mistaken for a turn of the whole body.
:::

In the exploded model, find the knee servo, the battery and the foot, and say what each does for actuation, power and support.

{#joint-map}
## Parts, joints, and degrees of freedom

A pen held at its tip can rotate in several directions, while a hinge normally rotates around one fixed axis. Robot joints impose similar constraints. Degrees of freedom count the independent variables needed to describe permitted motion; a component count describes physical pieces. The viewer shows 70 components, yet the policy outputs actions for only the 14 active joints.

| Region | Active joints | Main role |
|---|---|---|
| Each leg | Hip yaw, roll, pitch; knee; ankle: 5 joints | Support the body and place the foot |
| Neck and head | Neck pitch; head pitch, yaw, roll: 4 joints | Orient the camera and head |
| Beak | Physical servo; not controlled by this policy | Shown in 3D model, excluded from the 14-action interface |

In the 3D duck's How motors drive joints view, choose Hip and turn the left leg's hip yaw, hip roll and hip pitch in turn, inspecting the three servo outputs and their nearby bearings. The servo drives the joint, the bearing supports the rotating part, and brackets connect adjacent joints. For each joint, notice which parts turn and which remain only as outlines.

{#control}
## How software moves the mechanism

{.pipeline}
1. Policy action
2. Joint targets
3. Actuator torque
4. Physics update
5. New observation

The policy outputs actions, and the action interface turns them into joint targets. Actuators produce torque from the difference between target and measured state; the resulting motion also depends on contact, gravity, inertia and friction.

The policy runs at 50 Hz and physics at 200 Hz, so each action spans four physics steps. The backlash model represents transmission play with passive joints; the physics engine computes them, and they are not policy outputs.

{#frames}
## Describe motion with frames

| Frame | What is expressed here |
|---|---|
| World | Pitch, goal, and overall robot position |
| Body | Forward, lateral, and turn commands; ball relative to the robot |
| Camera | Image rays and detected target directions |
| Joint-local | A joint’s axis and angle |

::::{quiz}
If the head joints stay fixed, is the camera image necessarily stable?

:::{answer}
No. Body pitch and roll move the neck and camera. Image stability depends on camera orientation relative to the world, not only on joint angles.
:::
::::

{#read-assembly}
## Exercise: trace one load path

Start with the assembled model and trace upward from the ankle through the shin, knee, thigh and hip. When the foot is on the ground, support reaches the body along this path. Then separate the model and click the servos near a joint to see how adjacent links connect.

Besides a motor, a servo contains reduction gears, position feedback and control electronics. After receiving a target angle it needs time to track it, limited by available torque, speed, friction and backlash. This section observes that gap between target and actual angle; Section 3.2 shows how training with randomized actuator properties helps a policy cope with it.

{#policy-interface}
## What the policy receives and produces

Section 2.1 runs inference with a trained policy: it maps the current observation to 14 joint actions while its neural-network parameters stay fixed. Changing the velocity command tests how the same model responds; changing the policy itself requires training or loading different weights.

| Observation, in interface order | Width | What it tells the policy |
|---|---|---|
| Angular velocity and projected gravity | 6 | Body rotation and tilt |
| Relative joint positions | 14 | Pose relative to the nominal posture |
| Joint velocities | 14 | Motion at each joint |
| Previous action | 14 | Previous control output and temporal context |
| Velocity, head, and body commands | 3 + 4 + 6 | Requested behavior |

Remember <a href="01-play.html#policies">the policy network in Section 2.1</a>? These observations are concatenated in a fixed order and normalized to form its 61 inputs. Even with the same input shape, reordering joints breaks their meaning: the policy might read a knee as a hip. The course entry points therefore keep joint order, nominal posture and control timing fixed. Experiment with commands, duration and output folders first; change the observations only once you understand the interface.

With the policy inputs and outputs defined, we can now follow how the policy exchanges data with the physics engine.

{#control-loop}
## Close the command, action and feedback loop

Isaac Lab organizes observations and actions, Newton computes motion and contact, and Isaac Sim displays the scene. Follow the control loop to check three things: are the inputs right, are actions updated on time, and does the actual motion follow the command?

:::{platform-diagram}
Isaac Lab organizes the learning task and Newton advances the physics; Isaac Sim rendering and sensors are added when camera images or an interactive window are needed. Separating these roles helps you choose how to run training and validation.
:::

```{learning-flow}
label: Walking loop
nodes:
- [Observation oₜ, 'Pose, joint state, previous action and command']
- [Policy π, '61 inputs → 14 actions']
- [Newton simulation, 'Applies actions, yields the next observation']
details:
- The observation packs robot pose, joint state, the previous action and the current velocity command into a fixed order. Order and normalization must match training.
- The policy maps the 61-dimensional observation to a 14-dimensional action. The output passes through the action conversion and actuator model before it reaches the joints.
- Newton advances dynamics and contact and produces a new pose and joint state. The policy reads a fresh observation on the next tick, so actions always sit inside a feedback loop.
loop: true
caption: The velocity command says where and how fast to go; the policy turns it into coordinated joint motion.
```

```{formula} Position-control interface
\sym{a_t}=\sym{pi_theta}(\sym{o_t}),\qquad \sym{q_target}=\sym{q_default}+\sym{s}\sym{a_t},\quad \sym{s}=1
```

$\sym{q_target}$ is the joint target, $\sym{q_default}$ the default standing pose, and the action scale $\sym{s}$ is 1. An action is therefore an offset from the standing pose; the actuators and physics engine compute the actual motion from it. Even with matching input and output sizes, a different joint order, offset or observation normalization makes the actions wrong.

```{concept-diagram}
kind: timing
label: 50 Hz policy updates and 200 Hz physics updates
caption: 'One policy period contains four physics updates. The display frame rate is a separate quantity: a slower picture does not mean the simulation time step changed.'
```

```{code-story}
label: Python · core excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/walk.py
language: python
code: |
  action = policy(
      env.observation_manager.compute(update_history=True)["policy"]
  )
  env.action_manager.process_action(action)
  for tick in range(4):
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      env.sim.step(render=not a.headless and tick == 3)
      env.scene.update(0.005)
steps:
- lines: [0, 3]
  title: Build the observation, then run the policy
  body: Update the observation history and feed the policy observation to the locomotion policy.
- lines: [3, 4]
  title: Process the policy action
  body: The action manager receives this policy output and prepares it for simulation.
- lines: [4, 9]
  title: Advance four physics steps
  body: One policy action spans 4 × 5 ms; each step writes the action and updates the scene, giving 50 Hz control over 200 Hz physics.
```

This excerpt is the loop in the figure: the Isaac Lab observation manager builds the policy input, the action manager processes the action and drives the actuators, and Newton then runs four 5 ms physics steps. The full program also advances counters and checks for falls.

Below is a run of this program recorded in the course container; you do not need to run it. The robot first settles on the ground with its joints locked (Section 3.2 explains this start-up); the program then sends a zero velocity command for about 2 seconds, followed by a constant 0.25 m/s forward command, for a total of 12 seconds of simulated time. The stillness at the start is not an unresponsive policy.

```{recorded-run}
where: Course container · Isaac Sim window
recorded: DGX Spark · official walking model `models/walk.onnx`
command: |
  python3 scripts/run.py walk \
    --model models/walk.onnx --velocity 0.25 0 0 \
    --seconds 12 --seed 202 \
    --output outputs/final-live-livew
image: media/isaac-ui-walk-container.png
image_alt: 'The complete Isaac Sim window during this run: a Microduck stepping on flat ground in the viewport, RTX real-time rendering and Newton MJWarp physics in the toolbar, the stage tree and Isaac Lab panel on the right.'
output_label: report.json printed at the end of the terminal output (the startup section recording the settling is omitted)
output: |
  {
    "status": "completed",
    "seed": 202,
    "model_sha256": "2fc7f8ee6acd7d0d3f66a99927b5d5a56589aeed02ec714de329b252d5ccf492",
    "command": [
      0.25,
      0.0,
      0.0
    ],
    "physics_hz": 200,
    "policy_hz": 50,
    "scope": "nominal walking observation; not full-task evaluation",
    "viewport": {
      "status": "ok",
      "requested": [
        960,
        540
      ],
      "render_texture": [
        960,
        540
      ],
      "resolution_scale": 1.0,
      "render_every_control_steps": 2
    },
    "real_time_factor": 0.9490529826355415,
    "fell": false,
    "simulated_seconds": 12.0,
    "automatic_resets": false
  }
result: |
  Start with three fields: `status` is completed, `fell` is false and `simulated_seconds` is 12.0, so the run lasted the full time without falling; `automatic_resets` false means no automatic reset hid a failure. `model_sha256` matches `models/manifest.json`, confirming that the fixed official model was used. `viewport` records the window's display settings (drawn at 960×540, redrawn every 2nd control step), and a `real_time_factor` of about 0.95 means the 12 simulated seconds took only slightly longer in real time; both describe the window only and do not affect the simulation result.

  Then look at `trajectory.json` in the same folder, one position every 0.1 s. In the roughly 10 seconds after the forward command starts, the body moves from (0.00, 0.00) m to (1.07, 0.15) m: about 1.08 m in a straight line, roughly 0.11 m/s on average, less than half of the 0.25 m/s command, and the direction of travel drifts to the left, about 8° over the whole displacement and about 15° in the last 2 seconds. A velocity command is a request to the policy; the actual speed and direction come from the policy, the actuators and contact together. “It did not fall” is not enough: compare the actual motion with the command.
caption: Recorded run. The command, the actual window and the terminal output all come from the same run.
```

:::{enrichment} Key code parameters
`--model models/walk.onnx` reads `/workspace/Microduck-Workshop/models/walk.onnx` in the container, the official walking model in the host repository's `models/` (read-only); the robot model is `lab/assets/microduck_contact.usda`; `report.json` and `trajectory.json` go to `outputs/final-live-livew`, given by `--output`.
:::

```{formula} One statistic for velocity tracking
\sym{rmse}=\sqrt{\frac{1}{\sym{N}}\sum_{\sym{t_idx}=1}^{\sym{N}}\symx{sqnorm}{\|\sym{v_xy}-\sym{v_cmd}\|_2^2}}
```

The root-mean-square error summarizes velocity tracking over a period; this section sets no universal pass threshold: first confirm there is no fall, then compare directions and speeds. Foot contact, actuator delay and backlash make joints lag behind their targets, which is why the policy reads a new observation every tick.

You now know how the policy reads its state and outputs actions, and how the servos and the physics simulation carry them out. Section 3.1 moves from loading a ready-made model to training one: Isaac Lab lets a policy learn to walk through repeated trials.

{#references}
## References

{.references}
- [Pollen Robotics: Microduck hardware ↗](https://github.com/pollen-robotics/microduck)
- [Isaac Lab: Microduck task source ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
- [Newton physics engine ↗](https://github.com/newton-physics/newton)
