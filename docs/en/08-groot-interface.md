---
myst:
  html_meta:
    description: 'What GR00T reads in the walking shot, which eight high-level requests it outputs and who executes them; why only the action head is post-trained instead of predicting joint angles; how the demonstrations are designed; and why a real robot would run inference on a workstation.'
---

# 5.2 Walking shot: GR00T inputs, outputs and data

:::{lesson-intro}
:meta: Section 5.2 · Estimated learning time: 60 minutes

What GR00T reads in the walking shot, which eight high-level requests it outputs and who executes them; why only the action head is post-trained instead of predicting joint angles; how the demonstrations are designed; and why a real robot would run inference on a workstation.
:::

```{symbols}
H:
  tex: H
  text: Number of high-level request steps predicted at once; 16 in this section.
K:
  tex: K
  text: Number of predicted request steps executed before predicting again; 4 in this section.
T_exec:
  tex: T_{\mathrm{execute}}
  text: 'Simulation time covered by the executed requests: 4 ÷ 12.5 Hz = 0.32 s, excluding inference and communication.'
f_action:
  tex: f_{\mathrm{action}}
  text: Rate at which high-level requests are recorded and executed; 12.5 Hz in this section.
k_round:
  tex: k
  text: 'Round index: for GR00T one round is a new prediction and its execution; for the onboard policy one round is one sensor capture and its decision. $k-1$ is the previous round and $k+1$ the next.'
```

{#task}
## One pitch, one gait, one instruction

The GR00T in this chapter does one thing: on the course’s green football pitch, walk on its normal feet up to the ball, kick it into the goal with the right foot, then recover to standing. The pitch, goal, ball and lighting are exactly those used by the onboard policy, and the model only controls walking; it never switches to rollers.

The model always receives the same instruction:

<q>Walk to the football, kick it into the goal with the right foot, and recover to standing.</q>

The instruction does not say where the ball is or where to stand before kicking; the robot has to work that out from camera images and body state. Because every demonstration uses this one instruction, the model learns this one task, and nothing can be inferred about other instructions.

{#reuse-skills}
## Hierarchical control: GR00T decides what, existing skills decide how

```{learning-flow}
label: Learn how to use skills; skills execute movement
nodes:
- [Observe and set a goal, '4 images, 127-D state, instruction']
- [GR00T, Predict 8-D high-level requests]
- [Skill executor, Walk / gaze / kick and recover]
- [Robot and feedback, 'Move joints, observe again']
details:
- Every 80 ms an observation is assembled from four history images from the same onboard camera, current and past body state, and the fixed instruction. It contains no simulated world coordinates of the ball or goal and no internal task stage of the onboard policy.
- GR00T outputs a 16 × 8 block of requests. Each step has three movement requests (forward, lateral, turn), four gaze requests for the neck and head, and one right-foot kick request.
- The official walking policy turns movement requests into joint actions and keeps balance; head requests pass through limits and rate limiting; once the kick request crosses its threshold, the executor checks that the robot has stopped and is stable, then runs the course’s local kick-and-recovery skill.
- Low-level control keeps driving the joints at 50 Hz, and fresh images and body state feed the next prediction.
layout: learning
loop: true
```

With this split, skills from the earlier chapters remain the foundation of the task: the official walking policy is the same kind of reinforcement-learning policy you train in Section 3.2, and kicking and recovery are handled by the course’s local skill, which combines reference kick and standing policies, a course-trained residual correction and recovery logic. GR00T learns “when to walk where, where to look and when to kick”, not how to stay upright.

{#inputs}
## What GR00T reads each time

| Input | Structure | Description |
|---|---|---|
| Camera history | 4 RGB images, 270 × 480 each | The current frame from the onboard camera and frames from about 1.92, 3.84 and 5.76 seconds earlier (24, 48 and 72 samples apart at 12.5 Hz). |
| Task instruction | Text | The fixed sentence above. |
| Current state | 58 values | Body angular velocity and gravity direction 6; positions of the 14 joints relative to the default pose 14; joint velocities 14; previous motor action 14; previous 8-D high-level request 8; kick-skill busy flag and progress 2. |
| Past-state summaries | 3 × 23 = 69 values | One group per past image: angular velocity and gravity 6, positions of the four head and neck joints 4, the previous high-level request at that time 8, age 1 and integral summaries 4. |
| Full numeric state | 58 + 69 = 127 values | Encoded separately from images and text. |

A few points are easy to misread:

- The integral summaries accumulate the measured vertical angular velocity and the request values. A request integral is “how far the robot was asked to go”, not a measured displacement or position.
- The previous motor action and the previous high-level request differ. The first is the joint action actually sent by the low level; the second is what GR00T requested at the preceding 80 ms sample. A request may be delayed by stability conditions or head rate limits, so requesting something is not the same as completing it. Before the first decision the previous request is zero.
- At the start of an episode, before enough history exists, past images and states are filled with the first observation of the episode.
- The 127 values contain no simulated coordinates of the ball or goal. The kick-skill busy flag and progress come from the executor itself and can be maintained on a real robot.

{#sensors}
## Which sensors GR00T uses

The onboard policy in Section 4.1 uses the RGB camera, ToF, joint encoders and IMU together; GR00T reads only some of them.

:::{course-video} media/sensors-groot-recorded
:alt: One simulated GR00T validation episode: the third-person view on the left, the head RGB image that GR00T reads at the top right, the observation-only 8×8 ToF distances at the bottom right and GR00T's requests and stages at the bottom.

One simulated GR00T validation episode, with all views synchronized to the same capture time. The RGB image at the top right is GR00T's input source; ToF is shown for observation only and does not reach GR00T. In this episode the robot touches the ball but does not score, and it does not fall.
:::

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

The two methods do not see the same information: the onboard policy relies on ToF for the floor and the distance in front of its feet, while GR00T can only learn the corresponding behavior from images, body state and demonstrations. Keep this in mind when comparing them.

{#outputs}
## The eight high-level requests: what, which units, who executes

Each predicted step outputs eight values in a fixed order. The executor first clips each value to the range below, then passes it to the corresponding module.

| # | Request | Range | Unit | Executed by |
|---|---|---|---|---|
| 1 | Forward | −0.35 to 0.35 | Native walking-policy command value | The official walking policy, which combines it with body feedback to produce 14 joint actions |
| 2 | Lateral | −0.20 to 0.20 | Native walking-policy command value | Same |
| 3 | Turn | −0.80 to 0.80 | Native walking-policy command value | Same |
| 4 | Neck pitch | −1.10 to 1.10 | Radians from neutral | The head adapter limits it and its rate of change, then passes it to the low level |
| 5 | Head pitch | −1.10 to 1.10 | Radians from neutral | Same |
| 6 | Head yaw | −1.40 to 1.40 | Radians from neutral | Same |
| 7 | Head roll | −0.31 to 0.31 | Radians from neutral | Same |
| 8 | Right-foot kick | 0 to 1 | Trigger request | Above 0.5 the executor first confirms the robot has stopped and the head is neutral, then runs the roughly 4-second local kick-and-recovery skill |

The first three values go straight into the official walking policy’s command input. They are the policy’s own native values, not measured speeds: how fast the robot actually moves also depends on the gait, the floor and the body state, so do not read them as m/s or rad/s. The kick value is not a foot joint angle or a success probability either; it is only a “kick now” signal. Whether the kick skill is running, and how far along it is, is fed back to the model through the busy flag and progress in the state.

{#why-high-level}
## Why not let GR00T output joint angles directly?

Microduck’s walking policy updates its 14 joint actions from body feedback every 20 ms (50 Hz). GR00T inference took a median of about 0.33 s of server computation. A predicted action chunk can run at the control frequency, but corrections based on new observations are still limited by inference latency. Direct joint control must also learn balance and contact, requiring more varied demonstrations and careful timing validation.

With a hierarchy, each part does what it is good at:

- the walking and kick skills, already trained with reinforcement learning and running at 50 Hz, keep handling balance, stepping and contact;
- GR00T handles what requires understanding the scene: which direction the ball is in, how far it is, whether to look down to check, and when a kick fits;
- high-level requests have intuitive meanings and few dimensions, so demonstrations are easier to record and easier to check.

The cost is that GR00T is limited by the skills beneath it: a request cannot ask for a motion the walking policy cannot perform.

Action chunks connect this division of labor in time. Each GR00T prediction covers $\sym{H}$ = 16 request steps, the executor runs only the first $\sym{K}$ = 4, then predicts again from a fresh observation:

```{formula} Simulation time covered by one executed chunk
\sym{T_exec} = \sym{K} / \sym{f_action}
```

At $\sym{f_action}$ = 12.5 Hz, four steps cover 0.32 s of simulation time. Each request is held for 80 ms, during which the low-level walking policy updates 4 times. The 0.32 s is simulation time only and excludes waiting for inference and communication.

{#timing}
## Control cycle: how often each module updates

The figure below comes from the course's completed walking-shot simulation logs. It shows three consecutive rounds (previous, current and next) and compares in what order, and how often, each module updates with the onboard policy and with GR00T. While it plays, follow the colors to see where each round's inputs come from and which round produced each command.

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
  capture_ms: [0, 70, 135, 200]
  actor_ms: [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]
  actor_round: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2]
  decision_ms: [110.418338, 266.223269, 148.434121]
  detector_ms: [40.214468, 36.874673, 58.136762]
  navigation_ms: [70.1371660688892, 229.20526703819633, 90.22252692375332]
  lanes:
    sensors: {name: Sensor capture, rate: 'RGB + ToF · 15 Hz, about every 67 ms'}
    decision: {name: Onboard decision, rate: once after each capture ◆}
    commands: {name: High-level command, rate: kept until the next decision}
    actor: {name: Low-level motion policy, rate: 50 Hz · every 20 ms}
  capture_label: RGB + ToF
  decision_label: Detect · decide
  decision_note: measured {ms} ms
  capture_title: RGB and ToF update in the same simulated capture with the same timestamp
  decision_title: 'The onboard policy processes this capture: detector call {det} ms (including image encoding and local communication), geometry, navigation, gaze and safety guards {nav} ms, total {ms} ms (host-measured from sensor data ready to command ready, excluding sensor capture; simulation time does not advance meanwhile)'
  next_title: Sensor capture for the following round
  say: '{round}: the sensors capture RGB and ToF together → the simulation pauses while the onboard policy runs detection, geometry, navigation and gaze decisions, measured at {ms} ms → this round''s high-level command is kept until the next decision, while the low-level motion policy updates {n} times.'
  legend: ['Vertical line: one sensor capture', '◆ processing event: done while the simulation is paused; measured time below, not to axis scale', 'Short bar: one low-level motion-policy update', 'Color: owning round; grey: replaced by a newer round; dashed: not yet happened']
  note: The horizontal axis is to scale in simulation time; processing time takes no axis width. The 15 Hz capture is rounded to the 5 ms physics step, so captures are 65 or 70 ms apart (70, 65, 65 ms here); the recorded run shows the low-level motion policy using the three rounds' commands 4, 3 and 3 times.
facts:
- modes: [onboard]
  text: In this course's simulation RGB and ToF update in the same capture with the same timestamp; this is a simulation setting, not a claim of hardware sensor synchronization. After each capture the onboard policy uses the images, depth and body state for object detection, geometry, navigation and gaze decisions, then applies safety guards, and outputs gaze, velocity and skill requests.
- modes: [onboard]
  text: 'Captures run at 15 Hz, about 67 ms of simulation time apart, yet the three decisions here took 110.4, 266.2 and 148.4 ms on the host. These do not conflict: evaluation is synchronous and simulation time does not advance until a decision finishes, so the times are written under ◆ instead of stretching the axis. Timing starts when the sensor data is ready and ends when the command is ready; it excludes sensor capture and low-level motion-policy inference. During the measurement DGX Spark was also computing a GR00T feature cache, so these numbers do not represent a robot''s onboard computer or dedicated hardware.'
- modes: [onboard]
  text: The high-level command is kept until the next decision. While simulation advances, the low-level motion policy updates at 50 Hz; local state estimation, safety guards and head limits also run on this rhythm and may modify the command before it becomes joint actions.
- modes: [groot]
  text: Each new inference outputs 16 eight-dimensional high-level motion commands (velocity, gaze and skill requests); only the first 4 are executed. Each command is held for 80 ms of simulation while the low-level motion policy computes a new joint action every 20 ms, so one command spans 4 joint-action computations. The other 12 are not executed; after the first 4, fresh observations trigger a new prediction.
- modes: [groot]
  text: 'In synchronous evaluation every call, once per 80 ms, waits for the server before the simulation continues: 1 call in 4 is a new inference, the other 3 only take the next command from the cache. The three new inferences here took 326.5, 323.5 and 333.9 ms of server computation. Across the evaluation batch, 899 new inferences had a median of 332 ms (P95 351 ms) and 2,680 cached calls a median of 15 ms. These are server computation times only, excluding network transfer and image capture, and must not be added to simulation time as one cycle.'
- modes: [groot]
  text: The camera captures at 15 Hz while the simulation advances and not while it is paused. A new inference reads only the latest frame plus three history frames from about 1.9, 3.8 and 5.8 s earlier; the frames in between are not fed to that inference directly.
caption: 'Data from the course''s completed simulation logs: GR00T, 12 consecutive calls in one walking-shot evaluation (simulation 8.24–9.20 s, start-up warm-up skipped); onboard policy, three consecutive decisions after entering the approach-the-ball stage in one timing run (simulation 2.200–2.400 s). The excerpts show update order, not task success rates or end-to-end real-time performance. [View timing data](_static/data/control-timing-evidence.json)'
```

Synchronous evaluation stops the simulation whenever it waits for the server, so a live view shows pauses; a video exported at a fixed simulation frame rate hides those waits, so smooth video does not show that the robot can move continuously in real time. A real robot cannot pause physical time: the low-level motion policy needs its own continuously running control loop, and high-level inference has to run asynchronously with a command buffer that also handles expired commands and stale observations. This course's simulation evaluation is synchronous.

{#action-head}
## Why post-train only the action head?

Do not assume that the GR00T N1.7 base weights understand this course’s 127-D state and 8-D high-level requests: the course defines this interface itself, and post-training teaches the interface and the shooting task. The vision-language backbone is already good at recognizing objects and directions in images, but the action head still has to learn what this robot’s state looks like and what it should output. So the course freezes the vision-language backbone and trains only the action head:

| Part | Trained? | Why |
|---|---|---|
| Vision encoder and language model (Cosmos-Reason2-2B) | Frozen | Keeps pretrained visual and language knowledge; with only dozens of demonstration episodes, unfreezing risks overfitting and needs far more memory |
| State encoder, action encoder and action decoder | Trained | Connect the 127-D state and 8-D requests, the interface unique to Microduck |
| Action-sequence generator (the flow-matching transformer) | Trained | Learns to generate sensible request sequences from noise for this task |
| Layers in the action head that process visual-language features | Trained | Let visual features enter action generation in a way suited to this task |

That is more than replacing the last output layer, and it differs from LoRA, which trains only small added matrices. GR00T’s internal action representation is wider than eight values; the extra dimensions are masked so the training loss covers only the real eight.

Freezing the backbone has a practical benefit too: the features computed from the same image and instruction never change, so they can be computed once and cached before training, after which each update runs only the action head. Section 5.3 uses the supplied cache by default: the log first shows the cache checks, then the optimizer updates; full feature-generation progress appears only when you choose to recompute the features. The cache is valid only while the backbone stays frozen and image preprocessing is fixed; it cannot be reused after changing image augmentation or unfreezing the backbone.

{#data-design}
## How the demonstrations are designed

The demonstrations come from the onboard policy, which decides using only the onboard camera, IMU, joint encoders and ToF ranging, without simulated ground truth for the ball or goal. In simulation it completes the shot from different starts while we record the inputs GR00T will later see and the 8-D requests the onboard policy actually issued.

```{learning-flow}
label: Reliable demonstrations before scale
nodes:
- [Fix the task, 'One pitch, walking, one instruction']
- [Record demonstrations, Observations paired with issued requests]
- [Filter and split, 'Keep full successes, group by episode']
- [Train and test, 'Fit requests, then test in closed loop']
details:
- The pitch, goal, ball and lighting stay fixed; only the robot and ball starts change. Teach one situation well before widening the scope.
- The onboard policy decides at its own pace; the recording is resampled at 12.5 Hz into GR00T inputs and outputs, pairing each sample with the images and state available at that moment and the requests issued afterwards. Future frames and the exact positions used for scoring are excluded from inputs.
- Only episodes that score a goal, never fall and end standing quietly become labels; failed episodes are kept separately for analysis. Training and validation sets are split by whole episode and start seed, so neighboring frames of one episode never land on both sides.
- Falling training loss only shows the demonstrations are fitted. When the model controls the robot, check full-task success from starts that were not used in training.
layout: learning
loop: false
```

A few design principles:

- **Inputs must be things the robot can really obtain.** Putting the ball’s simulated coordinates or the onboard policy’s internal stage number into the inputs would make training easy, but a real robot could not supply them.
- **A reliable teacher, clean demonstrations.** If the onboard policy hesitates and readjusts, the model learns that too; only fully successful episodes enter training.
- **Coverage matters more than count.** Dozens of nearly identical smooth runs teach less than demonstrations that cover approaching, adjusting heading, looking down to check, preparing to kick and recovering.
- **Watch for imbalance.** Most of an episode is walking, and the frames just before a kick are few. Training can adjust sampling probabilities by stage: this course’s training set is divided into 7 stage groups based on the onboard policy’s recorded behavior, so standing still, fine adjustment at the ball and the moments just before a kick are drawn more often, and frames during the kick skill less often. Stage labels are used only for offline sampling and never enter the model’s inputs.

The course has already prepared training and validation sets by these principles; Section 5.3 uses them directly, so you do not need to collect data.

{#improve-data}
## When the model drifts: the idea of corrective data

A model trained only on successful runs has a common weakness: once it drifts, the images may no longer resemble any demonstration, and it does not know how to get back to the task. The remedy is to let the learning model drive first and then add correct actions in the states it actually reaches:

- **Human-gated takeover (HG-DAgger):** an observer takes over when the model is about to go wrong and provides corrective actions, and those segments are added to the training data.
- **Automated expert correction (DAgger-style):** a controller takes over automatically and supplies the actions. This requires the expert to be just as reliable in the drifted states the learner creates: an onboard policy that shoots well from standard starts may not recover from a skewed posture, so it must first be checked on those states.

Both require recording who is in control and when a takeover starts, using only verified correct actions as labels, and finally turning takeover off to measure the model’s own success rate.

{#deployment}
## On a real robot: infer on a workstation, connect over the network

Microduck’s onboard computing hardware is not suited to running a model of about three billion parameters such as GR00T. For real deployment, Microduck can share a network with a workstation that has a suitable GPU: the robot sends images and state to the workstation, the workstation runs GR00T, and the requests return to the robot, while small networks such as the walking policy keep running locally on the robot. Choose the workstation by model, precision and measured latency.

```{learning-flow}
label: Remote model inference; local continuous control and feedback
nodes:
- [Microduck sensors, Images + state + timestamps]
- [GPU workstation, 'GR00T inference, return requests']
- [Microduck executor, 'Check freshness, keep balance']
details:
- Send camera and robot state with sample times and sequence numbers, and pair each state causally with the latest image that is not older than the allowed age; keep each intentional history frame with its own timestamp, and never mix in stale or mislabelled data.
- The workstation generates a request chunk from the latest valid observation. Network transfer, image encoding and model computation all count toward end-to-end latency; even when the server replies quickly, the image it used may already be stale.
- The robot accepts only requests that match the agreed format and are still valid. The walking policy keeps running locally; on a timeout or lost connection it should stop under control or stand steady instead of continuing an expired forward request.
layout: learning
loop: true
```

In this course’s simulation the GR00T server and the simulator run on the same DGX Spark, and the simulator waits for every inference before continuing, so the network delays and expired commands of a real robot do not occur. Completing the task in simulation does not show that the same model would run in real time over a real network; that needs asynchronous inference, a command buffer, timeout protection and measurement on the real robot.

::::{quiz}
GR00T outputs a forward request of 0.3. Can you say the robot is moving forward at 0.3 m/s?

:::{answer}
No. The forward request is a native command value written into the walking policy; the actual speed depends on the gait, the floor and the body state and must be read from the measured motion.
:::
::::

::::{quiz}
Why does the input contain the request actually issued last time instead of the request to be predicted now?

:::{answer}
The previous request already happened at decision time and is available on a real robot; the current request is the answer being predicted, and putting it in the input would leak the label.
:::
::::

{#references}
## References

{.references}
- [GR00T N1.7 model card ↗](https://huggingface.co/nvidia/GR00T-N1.7-3B)
- [Isaac GR00T: post-training and inference ↗](https://github.com/NVIDIA/Isaac-GR00T)
- [HG-DAgger: human takeover and corrective data ↗](https://arxiv.org/abs/1810.02890)
- [DAgger: expert labels on policy-visited states ↗](https://arxiv.org/abs/1011.0686)
