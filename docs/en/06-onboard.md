---
myst:
  html_meta:
    description: 'How the hand-designed onboard policy reuses the sensor estimates to shoot: why an onboard policy, how a straight approach and feedback connect to trained motion skills, the core code of each stage, and how to repeat shots in Isaac Sim.'
---

# 4.2 Designing the shooting stages

:::{lesson-intro}
:meta: Section 4.2 · Estimated learning time: 35 minutes

[Section 4.1](06-perception.md) turned sensor readings into ball and goal positions. This section shows how the hand-designed onboard policy uses those positions to connect the straight approach, position adjustments and kicking, reads the core code of each stage, and runs it in Isaac Sim for repeated shots.
:::

{#why}
## Why an onboard policy

Microduck uses an onboard controller board based on a Rockchip RK3566, whose small neural accelerator (NPU) delivers 0.8 TOPS on one INT8 core; the motion policy runs in a 50 Hz loop ([official README ↗](https://github.com/pollen-robotics/microduck), [NPU notes ↗](https://github.com/pollen-robotics/microduck/blob/main/docs/project/npu-bringup.md)). That compute can hardly support low-latency inference of a vision-language-action model (VLA) such as GR00T, with about three billion parameters. The onboard policy therefore uses a small detector, combined with geometry, a straight approach and the trained walking and kicking skills.

{#pipeline}
## From sensors to motion: a straight approach

**Isaac Lab** (on Isaac Sim, with Newton physics) simulates the robot, its sensors and the pitch, and runs the learned policies. This pitch has no obstacles. RGB, ToF and joint data provide the estimated ball position; the code chooses a stance behind the ball facing the goal, then connects the current position to that stance with a straight line. The robot approaches in this direction, continually adjusting its velocity from observations and position memory, then slows down and refines its position near the ball.

| Part | What it does |
|---|---|
| YOLO26n detector | Marks the ball and goal with detection boxes in the RGB image |
| Geometry | Turns detection boxes, the ToF floor and joint angles into ball and goal positions |
| Straight approach | Connects the current position to a stance behind the ball and checks that segment for clearance |
| Feedback and stages | Update walking velocity from the ball position and body heading; switch stages on conditions |
| Walking policy | Turns a velocity request into 14 joint targets at 50 Hz |
| Kick and recovery | Strikes the ball, then returns to standing |

The straight line supplies a reference direction; feedback corrects errors while walking. The high-level controller requests forward, lateral and turning velocities, while the walking policy handles balancing and stepping.

{#stages}
## Six stages of the shot

Stages can be designed backward from the success conditions. The kick skill works only in a small region: the ball must be in front of the right foot, the body must face the goal, and the robot must stand still first. To meet these preconditions the robot first walks to a stance behind the ball and then adjusts position and heading precisely; all of this presupposes knowing where the ball and goal are.

Then consider sensor uncertainty. The camera shakes while walking, so the last reliable ball observation is propagated using body motion estimated from joint encoders and the IMU. With a reliable estimate and suitable position and heading, the robot can continue approaching; it stops to observe when confirmation is needed. A kick must start from a still stance, so it gets its own handover stage.

```{learning-flow}
label: Stages of the onboard walking shot
nodes:
- [Find targets, Turn the head to see goal and ball]
- [Straight approach, 'Aim behind the ball, correct while walking']
- [Close-range refinement, 'Slow down, stop or reverse when needed']
- [Align for the kick, 'Ball before the right foot, body facing the goal']
- [Hand over to the kick, 'Stop, recenter the head, hold still']
- [Kick and recover, 'The kick skill strikes, then returns to standing']
details:
- Turns the head in place to locate the robot relative to the goal and find the ball.
- Connects the current position to a stance behind the ball and continually updates velocity requests to approach in that direction.
- Slows near the ball; with a large error, it settles and observes, or reverses while turning if the rear path is clear, then approaches again.
- Checks that the ball is in front of the right foot and the body faces the goal; near the ball, continuous position memory can establish arrival even when the ball is no longer visible.
- Stops, recenters the head and, once still with the arrival conditions satisfied, hands control to the kick skill.
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
### Straight approach

The aim is to walk from afar to a preparation stance behind the ball. The stance lies behind the ball along the ball-to-goal direction, shifted sideways so that the ball ends up in front of the right foot. Connecting the estimated current position to this stance gives a straight approach direction on the empty pitch.

The code samples points along the line and checks clearance against the known pitch boundary and current ToF. Once accepted, walking begins. As the body moves, the ball position relative to the robot changes; the controller continually updates forward, lateral and turning requests to maintain a suitable approach direction.

```{code-story}
label: Python · Core code excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/onboard/direct_path.py
language: python
code: |
  snap=snapshot(case,provenance)
  a,b=snap['pose'][:2],snap['goal']
  distance=math.dist(a,b)
  n=max(1,math.ceil(distance/.02))
  points=[[a[0]+(b[0]-a[0])*k/n,a[1]+(b[1]-a[1])*k/n] for k in range(n+1)]
  gate=check_path(points,snap)
steps:
- lines: [0, 3]
  title: Set the start and target
  body: Take the current position and preparation stance from the estimated state, then compute their straight-line distance.
- lines: [3, 5]
  title: Sample the line
  body: Divide the line into segments no longer than 2 cm so the entire approach can be checked.
- lines: [5, 6]
  title: Check clearance
  body: Check the points and the space needed by the robot against the pitch boundary and current ToF; only then allow the approach.
```

The line is a reference for walking, rather than a one-time distance estimate executed without further observations. The sensors keep capturing and position memory keeps updating. Near the ball, suitable position and heading allow a continuous transition to refinement. If the clearance check fails or information is stale, the robot stops to observe again.

{#stage-refine}
### Close-range refinement

Close to the ball, distance, lateral error and body heading all matter. The controller estimates closing speed from recent changes in ball position and starts slowing earlier when approaching faster. A large error triggers a settled observation. When a larger turn is needed and the rear path is clear, it can reverse while turning before approaching again, giving the walking policy room to adjust.

Stopping to observe happens only when needed. The code below waits for 0.5 s of continuous stillness before using a new image to choose the next move.

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
  title: Enter reacquisition after 0.5 s of stillness
  body: Sensors continue capturing. When a settled observation is needed, 0.5 s of continuous stillness precedes the reacquisition stage, which uses a new image to choose the next small step.
```

{#stage-align}
### Align for the kick

Refinement brings the ball in front of the right foot while the body faces the goal. While the ball remains visible, a settled observation checks its position. If the ball is already close and still cannot be seen fully with the head lowered, the controller checks its previous reliable position and the continuously updated position memory. With arrival, heading, body and floor checks satisfied, it proceeds to handover rather than repeatedly searching downward.

The code below defines the position region for this near-ball arrival check. Observation freshness, continuous motion estimation and body state must also pass; losing sight of the ball alone does not authorize a kick.

```{code-story}
label: Python · Core code excerpt
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/onboard/blind_arrival.py
language: python
code: |
  def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
  def vec(v,n):return isinstance(v,(list,tuple)) and len(v)==n and all(finite(x) for x in v)
  def inside(point):return vec(point,3) and .07<=point[0]<=.15 and -.10<=point[1]<=.01
steps:
- lines: [0, 2]
  title: Require a valid position estimate
  body: The three coordinates must be finite numbers; missing observations cannot be treated as zero distance.
- lines: [2, 3]
  title: Check arrival near the feet
  body: The continuously updated ball estimate must lie 7–15 cm ahead, from 10 cm right to 1 cm left of the body. Heading and the other checks must also pass before settling and handover.
```

{#stage-handover}
### Hand over to the kick

Walking stops and the head returns to the kick pose. After 0.5 s of continuous stillness, the robot confirms that the estimated ball position, goal direction and body state still satisfy the applicable arrival conditions; only then does the kick skill take control. Otherwise it pauses, observes and realigns. A reverse recovery also requires clearance behind the robot.

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

The kick skill runs once: the official right-foot kick policy strikes, then the robot smoothly switches to standing while the ball is observed until it settles. The standing motion here is the output of the official walking policy at zero velocity command; no additional kick-correction network is loaded. The onboard policy does not judge whether it scored; a separate evaluator decides that from simulator truth.

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
  title: Run both policies during the kick
  body: During the authorized kick window, every 20 ms the same 61-D observation goes through both official policies, standing and kicking. After that window only the standing policy runs.
- lines: [3, 6]
  title: Blend smoothly from kick to standing
  body: For the first 0.14 s the kick action is used alone; over the next 0.12 s a smoothstep moves the weight from kicking to standing.
```

Stages switch mainly on observations and body state: if a condition fails, the robot stops, observes again or returns to an earlier stage, so refinement often repeats several times. Individual movements have time limits; press R to reset the overall attempt.

{#training}
## What was trained and what was not

- **Detector.** Its goal is to find two classes, ball and goal, in 960 × 540 head-camera images. The course started from a YOLO26n detector and trained it further on pitch images rendered in Isaac Sim, covering several floors (including the course's green pitch), different camera poses and ball positions. Detection-box labels were generated automatically by projecting the known scene geometry into the camera image, with rendered depth removing occluded parts. 512 images were used for training, 128 to select the checkpoint and another 128 held out for testing.
- **Walking.** The official walking policy was trained by Pollen Robotics with reinforcement learning, the method of Part 3; the course uses it with unchanged weights.
- **Kick and recovery.** The official right-foot kick policy is also a local skill trained with reinforcement learning, with unchanged weights; the timing logic for switching between it and standing is implemented here.
- **Not trained.** Geometry, straight-line generation, velocity feedback, braking and stage logic were programmed.

{#run}
## Shoot!

Prepare the course container as described in [Section 1.2](setup.md), then return to its terminal. If it is already running, enter it from the host with `docker exec -it microduck-workshop bash`. Run:

```{code-block} bash
:caption: Terminal · Inside the container
:class: terminal

python3 scripts/run.py football --output outputs/onboard-football
```

| Parameter | Meaning |
|---|---|
| `--output` | A new folder; rounds are kept in separate `episode-001`, `episode-002`, etc. folders |

The robot automatically observes, approaches the ball and attempts a shot. The first round starts from the nominal position. Click the Isaac Sim window and press **R** to restart: the window stays open and each round starts from the standing initialization, with the start randomly offset by up to 5 cm forward/backward and up to 5 cm sideways, and the heading turned by up to 15° clockwise or counterclockwise. The ball and goal stay in place. Allow a moment for the scene to initialize; the task then starts automatically.

The student experiment has no overall time limit. Scoring, missing or being unable to kick does not automatically pause the simulation; sensors keep updating. Press **R** to reset the attempt. Close Isaac or press **Ctrl+C** in its terminal to stop the program. Perception, alignment or handover failures pause the current maneuver and trigger another observation attempt.

The simulation renders several views at once, the third-person viewport, the head RGB camera and ToF, so its speed depends on your hardware and the motion in the window may look jerky; the task advances on simulation time.

:::{course-image} media/isaac-ui-onboard-sensors.png
:alt: The complete Isaac Sim window of the onboard shot, showing the green pitch, the head RGB image and the 8×8 ToF ranges.

Watch the robot on the left and compare what it sees on the right: do the detection boxes stay on the ball and goal, and do the ToF readings ahead of the feet shrink as it approaches? NR means no return and does not mean that nothing is ahead.
:::

`summary.json` summarizes saved rounds. Each round's `manifest.json` includes `navigation_retries`, which records why the robot reacquired observations. A visible detection box does not mean the geometry has passed its checks; if the close-range position is not fresh enough, the policy pauses to observe again. Also check lateral error and heading during the approach, whether the kick conditions were met and whether the robot recovered its balance after contact.

When you press R, the program saves the current record before starting the next round.

{#design}
## Limits of hand design

Every stage, transition and threshold of this onboard policy was written by people around one task: shooting at this goal. It is clear and easy to debug, but it fits only this task. For "dribble the ball to the center circle and stop" or "walk up to the ball without touching it", most of the perception can be reused, but the stages, conditions and thresholds must be redesigned and retuned.

If model inference can move off the robot to a GPU workstation, or to a cloud server at the cost of more network delay, a much larger VLA becomes possible: one that learns from a language instruction and demonstrations instead of hand-written stages, while trained motion skills still handle fast control. Part 5 shows how GR00T takes this route.

::::{quiz}
Which parts of the onboard shot were learned from data, and which were written or configured by people?

:::{answer}
The detector and the official walking and kicking policies were trained. Geometry, straight-line generation, velocity feedback, braking and stage logic were programmed.
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
- [Microduck open-source project ↗](https://github.com/pollen-robotics/microduck)
- [Microduck controller-board system image (microduck_yocto) ↗](https://github.com/pollen-robotics/microduck_yocto)
