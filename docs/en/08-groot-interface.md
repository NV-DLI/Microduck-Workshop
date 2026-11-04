---
myst:
  html_meta:
    description: 'How GR00T completes the walking shot: what it reads, which 8 high-level requests it outputs and who executes them, how often each module updates, and how demonstrations, action-head training and closed-loop validation work.'
---

# 5.2 Score autonomously with GR00T

:::{lesson-intro}
:meta: Section 5.2 · Estimated learning time: 30 minutes

How GR00T completes the walking shot: what it reads, which 8 high-level requests it outputs and who executes them, how often each module updates, and how demonstrations, action-head training and closed-loop validation work. GR00T training and inference were done in advance by the course; nothing needs to be run in this section.
:::

```{symbols}
H:
  tex: H
  text: High-level request steps per prediction; 16 here.
K:
  tex: K
  text: Request steps actually executed after each prediction; 4 here.
T_exec:
  tex: T_{\mathrm{execute}}
  text: Simulation time covered by one executed segment; 4 ÷ 12.5 Hz = 0.32 s here, excluding inference and communication time.
f_action:
  tex: f_{\mathrm{action}}
  text: Rate at which high-level requests are recorded and executed; 12.5 Hz here.
k_round:
  tex: k
  text: 'Round index: for GR00T a round is one new prediction and its execution, for the onboard policy one sensor capture and its decision; $k-1$ is the previous round and $k+1$ the next.'
```

{#task}
## One pitch, one gait, one instruction

GR00T here does one thing: on the course's green football pitch, walk up to the ball, kick it into the goal with the right foot, then recover to standing. The pitch, goal, ball and lighting are exactly those used by the onboard policy. The model always receives the same instruction:

<q>Walk to the football, kick it into the goal with the right foot, and recover to standing.</q>

The instruction does not say where the ball is or where to stand before kicking; the robot has to work that out from camera images and body state. Every demonstration uses this one instruction, so the model learns this one task, and nothing can be inferred about other instructions.

:::{course-video} media/groot-walking-success-recorded
:alt: Simulation recording of a walking shot controlled by GR00T: on the green pitch the robot walks to the ball, settles, kicks it into the goal with the right foot and then recovers to standing.

A walking shot controlled by GR00T: watch how the robot walks up to the ball and settles, requests the kick, and recovers to standing after the ball enters the goal. GR00T only issues the 8 high-level requests for motion, gaze and the kick; stepping, balance and the kick's joint motions come from trained skills.
:::

{#reuse-skills}
## Layered control: GR00T decides what, skills decide how

```{learning-flow}
label: The model learns to use skills; the skills carry out the motion
nodes:
- [Observation and goal, '4 images, 127-D state, task instruction']
- [GR00T, Predicts 8-D high-level requests]
- [Skill execution layer, Walking / gaze / kick and recovery]
- [Robot and feedback, 'Joints move, observe again']
details:
- Every 80 ms an observation is assembled: 4 history frames from the same onboard camera, current and past body state, and the fixed task instruction. It contains no simulation coordinates of the ball or goal and none of the onboard policy's internal task stages.
- GR00T outputs a 16 × 8 request segment: each step has 3 motion requests (forward, sideways, turn), 4 gaze requests for the neck and head, and 1 right-foot kick request.
- The official walking policy turns motion requests into joint actions while keeping balance; head requests are clamped and rate-limited; when the kick request crosses its threshold the execution layer checks that the robot has settled and is posed correctly, then runs the local kick-and-recovery skill.
- The low level keeps controlling the joints at 50 Hz, and the new images and body state feed the next prediction.
layout: learning
loop: true
```

| Index | Request | Range | Executed by |
|---|---|---|---|
| 1–3 | Forward, sideways, turn | ±0.35, ±0.20, ±0.80, native command values of the walking policy | The official walking policy, which turns them into 14 joint actions using body feedback |
| 4–7 | Neck pitch, head pitch, head yaw, head roll | Radians from neutral | The head adapter clamps and rate-limits them before the low level |
| 8 | Right-foot kick | 0 to 1 | Above 0.5 the execution layer confirms that the robot has settled and the head is centered, then runs the roughly 4-second local kick-and-recovery skill |

The first three numbers are the walking policy's native command values, not measured velocities; the kick value is only a “kick now” request, not a joint angle or a success probability. GR00T learns “when to walk where, where to look and when to kick”; balance, stepping and ball contact are left to skills already trained with reinforcement learning and running at 50 Hz. In this execution layer the kick and recovery combine the reference kick and standing policies, a course-trained residual correction and recovery logic, which differs from the official-only configuration of the onboard policy in Section 4.2.

**Why not output joint angles directly?** One GR00T inference takes about 0.33 s of server compute, while the walking policy updates its joint actions from body feedback every 20 ms. Controlling joints directly, the model would also have to learn balance and contact, which needs richer demonstrations and strict timing validation. The price is that GR00T is limited by the skills below it: a motion the walking policy cannot perform cannot be requested either.

GR00T predicts $\sym{H}$ = 16 request steps at a time; the execution side executes only the first $\sym{K}$ = 4 and then predicts again from a new observation:

```{formula} Simulation time covered by one executed segment
\sym{T_exec} = \sym{K} / \sym{f_action}
```

At $\sym{f_action}$ = 12.5 Hz, 4 steps cover 0.32 s of simulation time; each request holds for 80 ms, during which the walking policy updates 4 times.

{#inputs}
## What GR00T reads

| Input | Structure | Notes |
|---|---|---|
| Camera history | 4 RGB images, each 270 × 480 | The current frame of the same head camera plus frames from about 1.9, 3.8 and 5.8 s earlier |
| Task instruction | Text | The fixed English instruction above |
| Numeric state | 127 values | Body angular velocity and gravity direction, joint positions and velocities, the previous motor action and high-level request, kick-skill progress, and state summaries matching the three history images |

The onboard policy of Section 4.1 uses the RGB camera, ToF, joint encoders and IMU together; GR00T reads only part of them:

```{concept-diagram}
kind: onboard-sensors
label: The RGB view and body state that GR00T reads, and the ToF distances shown for observation only
description: The first row is a line drawing of the camera view; GR00T reads the downscaled raw image, with no detector boxes. The second row is the 64 ToF zone distances from the same instant, greyed as observation only, with no connection to GR00T. The third row is the joint encoders and IMU, which enter the 127-value state. Images, state and the task instruction go into GR00T, which predicts 8-D high-level requests.
rgb:
  title: RGB camera · 960 × 540 pixels
  size: [960, 540]
  detections: false
  boxes:
  - [ball, 0.954, [443, 312, 576, 448]]
  - [goal, 0.946, [362, 119, 607, 273]]
tof:
  title: ToF ranging · 8 × 8 zones · observation only
  used: false
  grid:
  - [null, null, null, null, null, null, null, null]
  - [null, null, 1.15, 1.15, 1.15, null, null, null]
  - [2.34, 2.32, 1.30, 1.30, 1.23, 1.23, 2.51, 2.61]
  - [1.27, 1.26, 1.25, 1.25, 1.26, 1.20, 1.31, 1.35]
  - [0.80, 0.79, 0.78, 0.86, 0.50, 0.50, 0.81, 0.83]
  - [0.63, 0.62, 0.61, 0.61, 0.48, 0.48, 0.63, 0.65]
  - [0.50, 0.49, 0.48, 0.51, 0.49, 0.49, 0.50, 0.51]
  - [0.42, 0.41, 0.43, 0.42, 0.42, 0.43, 0.42, 0.43]
proprio: [Joint encoders + IMU, "Joint angles, gravity direction and angular velocity"]
steps:
- [Camera history, "Current frame and 3 past frames\nDownscaled to 270 × 480\nNo detector boxes"]
- [Observation only, "The simulation still produces ToF\nGR00T does not read it"]
- [127-value state, "Joints, body angular velocity and gravity, previous request and kick progress"]
result: [GR00T, "Images, state and task instruction\nPredicts 8-D high-level requests"]
caption: The same capture as the figure in Section 4.1, used to compare what the two methods read. GR00T has no detector boxes and must recognize the ball and goal in the image itself; there is no connection between ToF and GR00T.
```

The onboard policy uses ToF to judge the floor and the distance ahead of its feet, while GR00T can learn the corresponding behavior only from images, body state and demonstrations. Keep this in mind when comparing the two approaches.

{#timing}
## Control timing: how often each module updates

The figure below comes from a completed walking-shot simulation record. It shows the previous, current and next rounds in a row and compares the order and update rates of the modules in the onboard policy and in GR00T.

```{control-timing}
title: 'Control cycle: from observation to action'
subtitle: Walking shot · three consecutive rounds from a recorded simulation
modes: {onboard: Onboard policy, groot: GR00T}
play_label: Play sequence
replay_ms: {groot: 15000, onboard: 9000}
rounds:
- {label: '$\sym{k_round}-1$ previous', name: Previous round, plain: k−1, command: 'Execute round $\sym{k_round}-1$ command'}
- {label: '$\sym{k_round}$ current', name: Current round, plain: k, command: 'Execute round $\sym{k_round}$ command'}
- {label: '$\sym{k_round}+1$ next', name: Next round, plain: k+1, command: 'Execute round $\sym{k_round}+1$ command'}
groot:
  round_start_ms: [0, 320, 640, 960]
  inference_ms: [326.5, 323.5, 333.9]
  cached_ms:
  - [34.3, 15.7, 15.4]
  - [15.5, 15.2, 15.7]
  - [16.1, 14.7, 16.0]
  command_ms: 80
  actor_step_ms: 20
  capture_ms: [30, 95, 160, 230, 295, 360, 430, 495, 560, 630, 695, 760, 830, 895]
  latest_frame_ms: [-40, 295, 630]
  history_frame_ms:
  - [-5770, -3840, -1970, -40]
  - [-5440, -3570, -1640, 295]
  - [-5170, -3240, -1305, 630]
  actor_ms: [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940]
  infer_share: 0.36
  cache_share: 0.025
  phase_name: Phase
  infer_label: Inference<br>{ms} ms
  exec_label: 4 commands<br>320 ms simulation
  paused_label: paused
  actor_paused_label: Actions paused
  predict_label: 16 new<br>commands
  lanes:
    camera: {name: Camera capture, rate: 'RGB · 15 Hz, about every 67 ms'}
    commands: {name: High-level motion commands, rate: 80 ms each · new inference after 4}
    actor: {name: Low-level motion policy, rate: 50 Hz · every 20 ms}
  say_infer: '{round} · new inference: the simulation pauses while the server computes 16 new commands in {ms} ms. Its images are the latest frame (captured {age} ms before the pause) and three history frames from about {history} s earlier; only the first 4 commands of this prediction are used; the other 12 are not executed.'
  say_first: '{round} · command 1: taken directly from the new inference and executed for 80 ms of simulation; the low-level motion policy updates 4 times, computing a new joint action each time.'
  say_cached: '{round} · command {n}: fetched from this round’s cache (the server took {ms} ms, with no new inference), now executed for 80 ms of simulation with 4 low-level updates.'
  say_wait: '{round} · fetch cached command {n}: simulation pauses for the response; this server call took {ms} ms. It uses this round’s existing prediction without running a new inference.'
  cache_title: 'Fetch the next cached command: server {ms} ms; the simulation does not advance while waiting'
  used_title: Latest image read by the {round} new inference
  lead_title: Latest frame, captured {age} ms before this excerpt, read by the previous round's new inference
  capture_title: Camera capture; a newer frame replaces it before any new inference reads it
  legend: ['Vertical line: one camera capture (bright = latest frame read by a new inference)', 'Short bar: one low-level motion-policy update', 'Hatching: simulation paused (new inference or cached command)', 'Color: owning round; grey: no longer used by the current step; dashed: not yet happened']
  note: The inference column and the narrow cached-command columns only show order; their widths are not to scale. Inside each command, positions follow simulation time. Simulation intervals exclude capture, communication and server waiting.
onboard:
  capture_ms: [0, 70, 140, 200]
  actor_ms: [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
  actor_round: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
  decision_ms: [42.631, 39.377, 45.633]
  detector_ms: [7.927, 8.047, 10.893]
  navigation_ms: [34.703, 31.328, 34.739]
  lanes:
    sensors: {name: Sensor capture, rate: 'RGB + ToF · 15 Hz, about every 67 ms'}
    decision: {name: Onboard decision, rate: once after each capture ◆}
    commands: {name: Perception reference, rate: feedback continues between captures}
    actor: {name: Low-level motion policy, rate: 50 Hz · every 20 ms}
  command_labels: ['Use round $\sym{k_round}-1$ perception', 'Use round $\sym{k_round}$ perception', 'Use round $\sym{k_round}+1$ perception']
  capture_label: RGB + ToF
  decision_label: Detect · decide
  decision_note: measured {ms} ms
  capture_title: RGB and ToF update in the same simulated capture with the same timestamp
  decision_title: 'The onboard policy processes this capture: detector call {det} ms (including raw pixel transfer and local communication), geometry, straight-approach decisions, gaze and safety guards {nav} ms, total {ms} ms (host-measured from sensor data ready to capture-decision completion, excluding sensor capture and 50 Hz feedback; simulation time does not advance meanwhile)'
  next_title: Sensor capture for the following round
  say: '{round}: the sensors capture RGB and ToF together → the simulation pauses for detection, geometry, straight-approach and gaze decisions, measured at {ms} ms → feedback combines this perception reference with body state to keep correcting velocity; the low-level motion policy updates {n} times.'
  legend: ['Vertical line: one sensor capture', '◆ processing event: done while the simulation is paused; measured time below, not to axis scale', 'Short bar: one low-level motion-policy update', 'Color: owning round; grey: replaced by a newer round; dashed: not yet happened']
  note: The horizontal axis is to scale in simulation time; processing time takes no axis width. The 15 Hz capture is rounded to the 10 ms physics step, so captures are 60 or 70 ms apart (70, 70, 60 ms here); the low-level motion policy updates 4, 3 and 3 times under the three perception references, while velocity requests continue changing between captures.
facts:
- modes: [onboard]
  text: In this course's simulation RGB and ToF update in the same capture with the same timestamp; this is a simulation setting, not a claim of hardware sensor synchronization. After each capture the onboard policy uses the images, depth and body state for object detection, geometry, straight-approach and gaze decisions, then applies safety guards, and outputs gaze, velocity and skill requests.
- modes: [onboard]
  text: 'Captures run at 15 Hz, about 67 ms of simulation time apart, yet the three decisions here took 42.6, 39.4 and 45.6 ms on the host. These do not conflict: evaluation is synchronous and simulation time does not advance until a decision finishes, so the times are written under ◆ instead of stretching the axis. Timing starts after sensor capture and archive submission and ends when the capture decision is ready; it excludes rendering, archive submission, 50 Hz feedback and low-level motion-policy inference. This is a DGX Spark workstation measurement, not a measurement on the robot''s controller board.'
- modes: [onboard]
  text: Colors identify the latest perception round, not a constant velocity request. Between captures, joint encoders and the IMU continuously update position memory. Velocity feedback, safety guards and head limits run at 50 Hz before the low-level motion policy produces joint actions.
- modes: [groot]
  text: Each new inference outputs 16 eight-dimensional high-level motion commands (velocity, gaze and skill requests); only the first 4 are executed. Each command is held for 80 ms of simulation while the low-level motion policy computes a new joint action every 20 ms, so one command spans 4 joint-action computations. The other 12 are not executed; after the first 4, fresh observations trigger a new prediction.
- modes: [groot]
  text: 'In synchronous evaluation every call, once per 80 ms, waits for the server before the simulation continues: 1 call in 4 is a new inference, the other 3 only take the next command from the cache. The three new inferences here took 326.5, 323.5 and 333.9 ms of server computation. Across the evaluation batch, 899 new inferences had a median of 332 ms (P95 351 ms) and 2,680 cached calls a median of 15 ms. These are server computation times only, excluding network transfer and image capture, and must not be added to simulation time as one cycle.'
- modes: [groot]
  text: The camera captures at 15 Hz while the simulation advances and not while it is paused. A new inference reads only the latest frame plus three history frames from about 1.9, 3.8 and 5.8 s earlier; the frames in between are not fed to that inference directly.
caption: 'Data from the course''s completed simulation logs: GR00T, 12 consecutive calls in one walking-shot evaluation (simulation 8.24–9.20 s, start-up warm-up skipped); onboard policy, three consecutive decisions during continuous walking in a straight-approach run (simulation 4.600–4.800 s). The excerpts show update order, not task success rates or end-to-end real-time performance. [View timing data](_static/data/control-timing-evidence.json)'
```

The simulation evaluation is synchronous: simulation stops while waiting for the server, so live viewing shows pauses, while a video exported on simulation time hides these waits. A real robot cannot pause physical time: the low-level motion policy needs a control loop that keeps running, high-level inference must run asynchronously with a command buffer, and stale commands and observations must be handled.

{#data-train}
## Demonstrations, training and validation

The course pairs camera images, body state and 8-D requests from successful onboard-policy demonstrations, then divides the data by whole episode, with no repeated episode across training and validation. Training freezes the vision-language backbone and updates only the action head, teaching the model when to move, look and kick.

After training, GR00T controls the simulated robot itself in closed-loop tests, judged by a full goal, no fall and recovery to standing. A lower training loss only shows a better fit to the demonstrations; the video above shows one successful episode, while reliability needs repeated independent tests. This section requires observation and understanding, with no training or inference to run.

{#deployment}
## On a real robot: infer on a workstation, connect over the network

Microduck's onboard controller board is not suited to a model of about three billion parameters such as GR00T. For real deployment, the robot can send images and state to a GPU workstation that runs GR00T and returns requests; small networks such as the walking policy keep running locally on the robot. Network transfer, image encoding and model compute all add to end-to-end latency; on a timeout or lost connection the robot should enter a controlled stop or stable standing and must not keep executing a stale forward request.

::::{quiz}
GR00T outputs a forward request of 0.3. Is the robot moving forward at 0.3 m/s?

:::{answer}
No. The forward request is a native command value for the walking policy; the actual speed depends on gait, ground and body state and must be read from the measured motion.
:::
::::

{#references}
## References

{.references}
- [GR00T N1.7 model card ↗](https://huggingface.co/nvidia/GR00T-N1.7-3B)
- [Isaac GR00T: post-training and inference ↗](https://github.com/NVIDIA/Isaac-GR00T)
