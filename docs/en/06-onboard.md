---
myst:
  html_meta:
    description: 'How the hand-designed onboard policy reuses the sensor estimates to shoot: why an onboard policy, how hand-designed stages connect Nav2 path planning with trained motion skills, the core code of each stage, and how to repeat shots in Isaac Sim.'
---

# 4.2 Hand-designed shooting stages

:::{lesson-intro}
:meta: Section 4.2 · Estimated learning time: 35 minutes

[Section 4.1](06-perception.md) turned sensor readings into ball and goal positions. This section shows how the hand-designed onboard policy uses those positions to connect navigation, walking and kicking, reads the core code of each stage, and runs it in Isaac Sim for repeated shots.
:::

{#why}
## Why an onboard policy

Microduck uses an onboard controller board based on a Rockchip RK3566, whose small neural accelerator (NPU) delivers 0.8 TOPS on one INT8 core; the motion policy runs in a 50 Hz loop ([official README ↗](https://github.com/pollen-robotics/microduck), [NPU notes ↗](https://github.com/pollen-robotics/microduck/blob/main/docs/project/npu-bringup.md)). That compute can hardly support low-latency inference of a vision-language-action model such as GR00T, with about three billion parameters. The onboard policy therefore uses a small detector, combined with geometry, path planning and the trained walking and kicking skills.

{#pipeline}
## From sensors to motion: Isaac Lab + Nav2

Two separate tools work together. **Isaac Lab** (on Isaac Sim, with Newton physics) simulates the robot, its sensors and the pitch, and runs the learned policies. **Nav2** is the navigation framework of ROS 2; the course uses only its path planner, as a CPU process inside the same container. Course code passes Nav2 the robot's position, estimated from its sensors on a known map of the pitch, and a target, and gets path points back. The pitch map is surveyed in advance and the current ToF only adds nearby observations.

| Part | What it does |
|---|---|
| YOLO26n detector | Marks the ball and goal with detection boxes in the RGB image |
| Geometry | Turns detection boxes, the ToF floor and joint angles into ball and goal positions |
| Nav2 planner (NavFn, A*) | Finds a low-cost path on a 2D cost map to a stance behind the ball |
| Path following and stages | Turn the next waypoint into a walking velocity; switch stages on conditions |
| Walking policy | Turns a velocity request into 14 joint targets at 50 Hz |
| Kick and recovery | Strikes the ball, then returns to standing |

Nav2 returns points on the floor, never joint angles. A short hand-written follower turns the next waypoint into a velocity request, and the walking policy does the balancing and stepping.

{#stages}
## Six stages of the shot

Stages can be designed backward from the success conditions. The kick skill works only in a small region: the ball must be in front of the right foot, the body must face the goal, and the robot must stand still first. To meet these preconditions the robot first walks to a stance behind the ball and then adjusts position and heading precisely; all of this presupposes knowing where the ball and goal are.

Then consider sensor uncertainty. The camera shakes while walking and close-range positions are imprecise, so the robot walks a little, stops, and looks again; a kick must start from a still stance and be confirmed with the latest observation, so it gets its own handover stage.

```{learning-flow}
label: Stages of the onboard walking shot
nodes:
- [Find targets, Turn the head to see goal and ball]
- [Plan the approach, 'Pick a stance behind the ball, approach in segments']
- [Close-range refinement, 'Small steps, look again after settling']
- [Align for the kick, 'Ball before the right foot, body facing the goal']
- [Hand over to the kick, 'Stop, recenter the head, hold still']
- [Kick and recover, 'The kick skill strikes, then returns to standing']
details:
- Turns the head in place to locate the robot relative to the goal and find the ball.
- Nav2 plans a path to the stance behind the ball, and the follower walks along it.
- Close to the ball, it chooses small steps, short continuous movements or turns in place, observing the ball again after settling.
- Checks that the ball is in front of the right foot and the body faces the goal.
- Stops, recenters the head and, once still with a fresh observation, hands control to the kick skill.
- The kick skill strikes and then returns to standing; the evaluator decides whether it scored.
layout: learning
loop: false
```

The stages are described below in order, each with the core code that actually runs. Localization and paths use pitch-map coordinates. Close-range ball positions use a gravity-aligned frame centered on the robot, with x forward and y left. The code converts between frames where needed.

{#stage-find}
### Find targets

The robot turns its head to scan, finds the bases of the two goalposts with the method of Section 4.1, computes its own position and heading relative to the goal from them, and places the observed ball on the same map. A single estimate can be off, so several consecutive results must agree before they are accepted. If the goal has not been seen for too long, or the ball information is too old, the robot returns here to observe again.

```{code-story}
label: Python · Core code excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/football/controller.py
language: python
code: |
  measured = np.asarray(fix["pose_map_xyyaw"])
  self.goal_window.append((t, measured))
  if len(self.goal_window) == 3 and t - self.goal_window[0][0] < 0.5:
      centre = np.mean([p[:2] for _, p in self.goal_window], axis=0)
      a = np.array([p[2] for _, p in self.goal_window])
      yaw = math.atan2(np.sin(a).mean(), np.cos(a).mean())
      if (
          max(np.linalg.norm(p[:2] - centre) for _, p in self.goal_window)
          < 0.04
      ):
          if (
              self.pose is None
              or np.linalg.norm(self.pose[:2] - centre) < 0.20
          ):
              self.pose = np.array([*centre, yaw])
              self.goal_time = t
steps:
- lines: [0, 2]
  title: Record one localization
  body: The position and heading fitted from the goalpost bases go into a window of the last three results.
- lines: [2, 6]
  title: Average three results
  body: Once the window holds three results within 0.5 s, the mean position and mean heading are computed.
- lines: [6, 16]
  title: Accept only if consistent
  body: Each of the three positions must be within 4 cm of their mean. When a previous position estimate exists, the new mean must also be within 20 cm of it. The position and the time of this goal sighting are then updated.
```

{#stage-approach}
### Plan the approach

The aim is to walk from afar to a preparation stance behind the ball. The stance lies about 0.22 m behind the ball along the ball-to-goal direction, shifted sideways so that the ball ends up in front of the right foot. From afar the robot only needs to get roughly closer and position errors matter less, so it can walk continuously and replan often; precise alignment is left to the close-range stage.

Nav2 only plans the path: the start is the estimated robot position, the goal is this stance. The map comes from the surveyed pitch and the current ToF: the pitch interior is free, its border is an obstacle, valid ToF returns add free cells and objects ahead, and cells outside the known-free pitch without a valid observation remain unknown. NavFn searches this map with A*, never crosses unknown cells, keeps a robot-radius margin from obstacles and returns a series of path points on the floor.

```{code-story}
label: Python · Core code excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/football/controller.py
language: python
code: |
  path = self.last_plan["path_xy"]
  if not self.last_plan["envelope"]["accepted"] or not path:
      return self._output(t, reason="Nav2_path_not_admitted", gaze=gaze)
  near = min(
      range(len(path)), key=lambda i: math.dist(path[i], self.pose[:2])
  )
  way = next(
      (p for p in path[near:] if math.dist(p, self.pose[:2]) > 0.10), path[-1]
  )
  vec = np.asarray(rotate(-self.pose[2], np.asarray(way) - self.pose[:2]))
steps:
- lines: [0, 3]
  title: Use only an accepted path
  body: The path returned by Nav2 must pass its check; with an empty or rejected path the robot does not move on this tick.
- lines: [3, 9]
  title: Pick the next waypoint
  body: Find the path point nearest the robot, then take the first point beyond it that is more than 10 cm away as the target.
- lines: [9, 10]
  title: Find a reference direction in the body frame
  body: The map waypoint is transformed into the body frame to give the planning layer a reference direction. Later scheduling uses the observations and settling state to choose the actual walking request or a stop; the trained walking policy executes that request.
```

While walking, the robot keeps replanning and periodically confirms the goal. Close enough to the ball, it enters close-range refinement; if a path is rejected or information is stale, it stops and waits for a new observation. Avoidance of moving obstacles has not been verified.

{#stage-refine}
### Close-range refinement

Close to the ball, the error caused by camera shake while walking is as large as the precision needed. This stage no longer calls Nav2. It chooses small steps, short continuous movements or turns in place according to the ball position. Each segment has distance and time limits; the robot then settles and measures the ball in a new image. If the ball is not fully in view, the body stays still and only the head turns to find it.

```{code-story}
label: Python · Core code excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/evaluation/football/neutral_fine_step.py
language: python
code: |
  if self.state == "SETTLE":
      if self.quiet(packet):
          if self.quiet_start is None:
              self.quiet_start = t
      else:
          self.quiet_start = None
      if self.quiet_start is not None and t - self.quiet_start >= 0.5 - 1e-9:
          self.state = "REACQUIRE"
      return self.status(t, memory)
steps:
- lines: [0, 6]
  title: After stopping, check for stillness
  body: '`quiet` reads the IMU and joint encoders: body angular velocity at most 0.1 rad/s, RMS joint velocity at most 0.25 rad/s, tilt at most 0.35 rad. Any wobble restarts the timer.'
- lines: [6, 9]
  title: Observe again after 0.5 s of stillness
  body: Only after 0.5 s of continuous stillness does the robot observe again; the next small step is chosen from this new image, not from a shaky one taken while walking.
```

{#stage-align}
### Align for the kick

Close-range refinement aims to bring the ball into the small region the kick skill can handle while the body faces the goal. Before kicking, the latest observation taken at rest gets one final check:

```{code-story}
label: Python · Core code excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/onboard/relaxed_gate.py
language: python
code: |
  MARGIN_M=.005
  HEADING_MAX_RAD=math.radians(10.0)
  def exit_box(point):
      p=np.asarray(point,float)
      return bool(p.shape==(3,) and np.isfinite(p).all() and .070<=p[0]<=.110 and -.055<=p[1]<=-.015)
steps:
- lines: [0, 2]
  title: Margin and heading limit
  body: A 5 mm margin around the original kick region; the goal direction may differ from the body heading by at most 10°.
- lines: [2, 5]
  title: Ball in a small region before the right foot
  body: The ball position must be valid and lie 7–11 cm ahead of the body and 1.5–5.5 cm to its right. Otherwise the kick is refused.
```

{#stage-handover}
### Hand over to the kick

Walking stops and the head returns to the kick pose. After the body has been still for 0.5 s, the robot confirms again that the ball is inside the kick region above, the goal observation is at most 0.5 s old and the heading is within limits; only then does the kick skill take control. If a condition fails the kick is refused; where recovery is allowed the robot backs up and observes again, otherwise this attempt ends. The head return must finish within 4.5 s.

```{code-story}
label: Python · Core code excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/evaluation/football/head_handoff.py
language: python
code: |
  self.head += np.clip(-self.head, -0.3 * dt, 0.3 * dt)
  if packet is None:
      self.fail("headreturn_public_sensor_unavailable")
      return self.status(now_s)
  gravity = np.asarray(packet["gravity_body"], float)
  gyro = np.asarray(packet["gyro_body_rad_s"], float)
  qd = np.asarray(packet["encoder_qd_rad_s"], float)
  tilt = math.atan2(float(np.linalg.norm(gravity[:2])), -float(gravity[2]))
  quiet = (
      np.max(np.abs(self.head)) <= 1e-9
      and np.linalg.norm(gyro) <= 0.1
      and np.sqrt(np.mean(qd**2)) <= 0.25
      and tilt <= 0.35
  )
  if quiet:
      if self.quiet_start is None:
          self.quiet_start = now_s
  else:
      self.quiet_start = None
steps:
- lines: [0, 4]
  title: Recenter the head smoothly
  body: The offsets of the four head joints shrink by at most 0.3 rad per second; without sensor data this kick is abandoned.
- lines: [4, 14]
  title: Check for complete stillness
  body: Only with the head recentered and small body angular velocity, joint velocity and tilt does the robot count as still.
- lines: [14, 19]
  title: Note when stillness began
  body: Any wobble restarts the timer; after 0.5 s of continuous stillness the pre-kick checks above are made.
```

{#stage-kick}
### Kick and recover

The kick skill runs once for 4 seconds: the official right-foot kick policy strikes, then the robot smoothly switches to standing. The standing motion here is the output of the official walking policy at zero velocity command; no additional kick-correction network is loaded. The onboard policy does not judge whether it scored; a separate evaluator decides that from simulator truth.

```{code-story}
label: Python · Core code excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/onboard/official_kick_runtime.py
language: python
code: |
  standing = self._infer(0, obs)
  if active:
      kicking = self._infer(1, obs)
      u = np.clip((phase - np.float32(HOLD_S)) / np.float32(BLEND_S), np.float32(0), np.float32(1))
      weight = np.float32(1) - u * u * (np.float32(3) - np.float32(2) * u)
      teacher = weight * kicking + (np.float32(1) - weight) * standing
steps:
- lines: [0, 3]
  title: Run both policies
  body: Every 20 ms the same 61-D observation goes through both official policies, standing and kicking.
- lines: [3, 6]
  title: Blend smoothly from kick to standing
  body: For the first 0.14 s the kick action is used alone; over the next 0.12 s a smoothstep moves the weight from kicking to standing.
```

Stages switch mainly on observations and body state: if a condition fails, the robot stops, observes again or returns to an earlier stage, so refinement often repeats several times. Settling waits and per-stage time limits keep each attempt from running forever.

{#training}
## What was trained and what was not

- **Detector.** Its goal is to find two classes, ball and goal, in 960 × 540 head-camera images. The course started from a YOLO26n detector and trained it further on pitch images rendered in Isaac Sim, covering several floors (including the course's green pitch), different camera poses and ball positions. Detection-box labels were generated automatically by projecting the known scene geometry into the camera image, with rendered depth removing occluded parts. 512 images were used for training, 128 to select the checkpoint and another 128 held out for testing, with an mAP50 of 0.978 on the test set. This is an image-detection metric, not a shooting success rate.
- **Walking.** The official walking policy was trained by Pollen Robotics with reinforcement learning, the method of Part 3; the course uses it with unchanged weights.
- **Kick and recovery.** The official right-foot kick policy is also a local skill trained with reinforcement learning, with unchanged weights; the switch timing between it and standing is programmed.
- **Not trained.** The Nav2 planner, geometry, path following and stage logic were configured or programmed.

{#run}
## Repeat shots in the same window

Prepare the course container as described in [Section 1.2](setup.md), then return to its terminal. If it is already running, enter it from the host with `docker exec -it microduck-workshop bash`. Run:

```{code-block} bash
:caption: Terminal · Inside the container
:class: terminal

python3 scripts/run.py football --seconds 60 --output outputs/onboard-football
```

The robot automatically observes, approaches the ball and attempts a shot. Switch to an English keyboard layout, click the Isaac Sim window and press **R** to restart. The window stays open. Each round starts from the standing initialization, with at most 1.5 cm of forward/backward offset and 1 cm of lateral offset. The ball and goal stay in place. Allow a moment for the scene to initialize; the task then starts automatically.

Let each round finish before pressing R to compare complete attempts. Pressing R during a task records an interruption separately, outside the complete-task success rate. Close the Isaac window to check the completed episodes and print the results.

The simulation renders several views at once, the third-person viewport, the head RGB camera and ToF, so its speed depends on your hardware and the motion in the window may look jerky; the task advances on simulation time. To check only that the pipeline runs, you can skip the window:

```{code-block} bash
:caption: Terminal · Inside the container
:class: terminal

python3 scripts/run.py football --headless --seconds 60 --output outputs/onboard-football-headless
```

Without a window it runs one round, then evaluates it and prints the result.

| Parameter | Meaning |
|---|---|
| `--seconds 60` | Per-round task limit in simulation seconds, excluding standing initialization; completion or failure ends it earlier |
| `--output` | A new folder; in window mode rounds are kept in separate `episode-001`, `episode-002`, etc. folders |
| `--headless` | Run and evaluate one round without opening a window |
| `--single-run` | Open the window but run one round, then close it and evaluate |

:::{enrichment} Key code parameters
- **Reset.** `lab/onboard/session.py` handles R and rebuilds each environment inside the same Isaac Sim process. Sensors, localization, action history and scoring start fresh; previous results are retained.
- **Initialization.** Temporary joint limits hold HOME while the free body settles on the ground. Once sensors and the first real policy output are ready, the limits are released before applying that output.
- **Rates.** `lab/onboard/executor.py` uses Newton physics at 100 Hz, joint policies at 50 Hz and RGB/ToF at 15 Hz. Each action spans two 10 ms physics steps.
- **Scene.** `lab/onboard_scene.py` creates the green pitch, the goal and the initialization above.
- **Walking and kicking.** Walking and recovery use `velstand.onnx`, the runtime copy of official `models/walk.onnx`; kicking uses `ball_kick_right.onnx`, copied from `models/kick-teacher.onnx`.
- **Detection and localization.** `run.py` checks the hashes in `models/scene-detector/deployment.json` before passing the detector to the onboard policy; `lab/scene_goal_points.py` finds the goalpost bases inside the detection box.
- **Nav2.** `lab/scene_nav2/nav2.yaml` configures NavFn (A*, no traversal of unknown space) and a 2.5 cm cost map; `map_contract.py` builds the map from the surveyed pitch and current ToF and checks the path.
:::

:::{course-image} media/isaac-ui-onboard-sensors.png
:alt: The complete Isaac Sim window of the onboard shot, showing the green pitch, the head RGB image and the 8×8 ToF ranges.

Watch the robot on the left and compare what it sees on the right: do the detection boxes stay on the ball and goal, and do the ToF readings ahead of the feet shrink as it approaches? NR means no return and does not mean that nothing is ahead.
:::

{#results}
## Measured results

With the window configuration above, 20 consecutive rounds were run, pressing R between rounds, with the start varying by ±1.5 cm forward/backward and ±1 cm sideways. A full success requires foot contact, the whole ball inside the goal, no fall and recovery to standing; complete rounds without a goal remain in the denominator.

| Rounds | Full successes | Falls |
|---|---|---|
| 1–5 | 5 / 5 | 0 / 5 |
| 6–10 | 4 / 5 | 0 / 5 |
| 11–15 | 4 / 5 | 0 / 5 |
| 16–20 | 4 / 5 | 0 / 5 |

This cohort scored **17/20 (85%)** with **0/20** falls and no technical errors; no failed round was removed. <a href="_static/data/onboard-reset-evaluation.json">View the per-round record</a>. All rounds used the same pitch and configuration; this is a sample under small start perturbations, not a general success rate. Keep recording your own results.

When a round fails, first see which stage it stopped in. `summary.json` summarizes the rounds, and each round's `manifest.json` gives `terminal_reason` and failure details. A visible detection box does not mean the geometry has passed its checks; if the close-range position is not fresh enough, the policy may still stop to observe again or end the attempt. Also check whether the path was feasible, whether the kick conditions were met and whether the robot recovered its balance after contact.

{#design}
## Limits of hand design

Every stage, transition and threshold of this onboard policy was written by people around one task: shooting at this goal. It is clear and easy to debug, but it fits only this task. For "dribble the ball to the center circle and stop" or "walk up to the ball without touching it", most of the perception can be reused, but the stages, conditions and thresholds must be redesigned and retuned.

If model inference can move off the robot to a GPU workstation, or to a cloud server at the cost of more network delay, a much larger vision-language-action model becomes possible: one that learns from a language instruction and demonstrations instead of hand-written stages, while trained motion skills still handle fast control. Part 5 shows how GR00T takes this route.

::::{quiz}
Which parts of the onboard shot were learned from data, and which were written or configured by people?

:::{answer}
The detector and the official walking and kicking policies were trained. The geometry, Nav2 path planning, the path follower and the stage logic were configured or programmed.
:::
::::

::::{quiz}
To make Microduck dribble the ball to the center circle and stop, can the existing onboard policy be reused directly?

:::{answer}
Most of the perception can be reused, but the stance, stage order, transitions and handover were designed for shooting and must be redesigned and retuned. Rewriting the pipeline by hand for every new task is why Part 5 turns to learning from demonstrations.
:::
::::

{#references}
## References

{.references}
- [Isaac Lab ↗](https://developer.nvidia.com/isaac/lab)
- [Nav2: configuring the NavFn planner ↗](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/planners_plugins/configuring_navfn/)
- [Nav2 NavFn planner source ↗](https://github.com/ros-navigation/navigation2/blob/main/nav2_navfn_planner/README.md)
- [Microduck open-source project ↗](https://github.com/pollen-robotics/microduck)
- [Microduck controller-board system image (microduck_yocto) ↗](https://github.com/pollen-robotics/microduck_yocto)
