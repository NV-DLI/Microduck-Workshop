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
```

{#task}
## Design the learning task around the goal

Start with the desired outcome: approach the ball, adjust the stance, kick into the goal with the right foot, and recover to standing. Then define what the model can observe, which actions it can request, and how completion will be evaluated. The Microduck example connects these choices into a method that also applies to other robot tasks.

The demonstrations use this instruction:

<q>Walk to the football, kick it into the goal with the right foot, and recover to standing.</q>

Images describe the ball and goal relative to the robot; body state describes its motion; text specifies the objective. A fixed task and pitch provide an inspectable baseline. This experiment's coverage is not a limit on GR00T's capabilities: broader instructions, starting poses and environments require corresponding demonstrations and evaluation on held-out conditions.

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

The first three numbers are the walking policy's native command values, not measured velocities; the kick value is only a “kick now” request, not a joint angle or a success probability. GR00T learns “when to walk where, where to look and when to kick”; balance, stepping and ball contact are left to skills already trained with reinforcement learning and running at 50 Hz. In this execution layer the kick and recovery combine the reference kick and standing policies, a course-trained residual correction and recovery logic, which differs from the official-only configuration of the onboard policy in Section 4.3.

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

{#data-train}
## Prepare demonstrations that teach the whole task

Begin with a capable teacher and record the request it makes given each observation. The teacher may be a human teleoperator or a validated automated policy. This course uses onboard-policy demonstrations covering approach, observation, adjustment, kicking and recovery. Recording only the kick would leave the model without examples of how to reach a suitable stance.

| Data decision | Course design | Reason |
|---|---|---|
| Observations and labels | Align RGB, body state, task text and the 8-D request | Labels must correspond to information available before the request, without future-state leakage |
| Sequence | One request every 80 ms; 16-step targets at each training anchor | Learn how requests develop over time |
| History | Current image plus three past images and matching state summaries | Distinguish approach, turning and recovery; all history must precede the decision |
| Split | Separate complete episodes into training and validation sets | Adjacent frames from the same episode must not leak across the split |
| Checks | Fixed dimensions, joint order, units, timestamps and valid-frame flags | Prevent swapped fields or padded samples from becoming misleading training targets |

A new task also needs corrective examples: approaching from the side, briefly losing sight of the ball, or recovering an unsuitable stance. Adding similar successful episodes alone may not cover the states the model reaches after making a mistake. Mark valid labels at episode boundaries so padding is not treated as real action data.

{#training}
## Freeze the vision-language backbone and train the action head

The pretrained vision-language backbone converts images and instructions into features. This course freezes those parameters and trains the action head, including its state and action projections, for Microduck's 127-D state and 8-D request interface. It updates the full action head, rather than only a final linear layer or LoRA adapters. The low-level walking and kicking skills retain their existing weights.

```{learning-flow}
label: Learn Microduck's action interface from demonstrations
nodes:
- [Images and text, Frozen vision-language backbone]
- [State and targets, 127-D state and 16 × 8 requests]
- [Train the action head, Recover demonstration requests from noise]
- [Closed-loop validation, Complete the task from new starts]
details:
- Reuse pretrained visual and language representations while reducing the parameters updated for this task.
- Align state and images; calculate normalization statistics from the training set only and preserve field order and units at deployment.
- Add noise to demonstration actions and learn the denoising direction conditioned on features and state. Backpropagation updates only the enabled action-head and projection parameters.
- Evaluate the model using feedback from its own actions. Check goals, falls, recovery, timeouts and errors; training loss and offline action error cannot establish task success.
layout: learning
```

Comparison with demonstrations can expose interface and alignment errors. Closed-loop simulation reveals whether small errors accumulate. Retain successes, failures and technical errors, distinguishing a completed attempt that failed the task from a program that did not finish. The video shows one successful episode, not a reliability estimate.

{#timing}
## Control cycle

GR00T requests a sequence of high-level actions while the low-level policy handles faster body feedback. The diagram uses readable integer values: each high-level request covers 80 ms, four requests cover 320 ms, and the theoretical 50 Hz control loop updates joint actions every 20 ms.

```{control-timing}
kind: comparison-theory
```


**Connect action chunks with RTC.** Real-Time Chunking can accompany asynchronous inference: predict the next chunk while executing the current one, constrain the prefix that will be too late to change, and smooth the remaining transition. This can reduce the gap between chunks; inference still takes time, and buffer exhaustion, network jitter and timeouts remain design concerns. For this course's high-level requests, retain discrete kick-trigger and skill-state guards instead of smoothing every field indiscriminately.

The [NVIDIA GR00T deployment guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/real_world_deployment.md) and [LeRobot RTC documentation](https://huggingface.co/docs/lerobot/rtc) explain the approach. This is an extention design: the recorded course evaluation is synchronous and does not implement RTC. Adapting the model interface, chunk length and asynchronous executor requires separate continuity and timeout validation.

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
