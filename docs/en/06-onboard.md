---
myst:
  html_meta:
    description: "How a hand-designed onboard policy reuses sensor estimates to shoot: why the shot runs onboard, how Nav2 path planning and trained motion skills are connected by hand-designed stages, and how to press R in Isaac Sim to reset and shoot again."
---

# 4.2 Hand-designed shooting stages

:::{lesson-intro}
:meta: Section 4.2 · Estimated learning time: 25 minutes

[Section 4.1](06-perception.md) turned sensor readings into ball and goal positions. This section shows how a hand-designed onboard policy uses those positions to connect navigation, walking and kicking, and lets you watch it shoot repeatedly in Isaac Sim.
:::

{#why}
## Why an onboard policy

Microduck's public hardware uses a Rockchip RK3566, whose small neural accelerator (NPU) delivers 0.8 TOPS on one INT8 core, and its motion policy runs in a 50 Hz loop ([official README ↗](https://github.com/pollen-robotics/microduck), [NPU notes ↗](https://github.com/pollen-robotics/microduck/blob/main/docs/project/npu-bringup.md)). Low-latency inference with a vision-language-action model such as GR00T, with about three billion parameters, is a demanding workload for this hardware. This design therefore chooses a small detector, combined with geometry, path planning and the trained walking and kicking skills.

This course runs that design in simulation on a workstation. It shows the logic of the onboard approach; it is not a measurement of the robot's own computer.

{#pipeline}
## From sensors to motion: Isaac Lab + Nav2

Two separate tools work together. **Isaac Lab** (on Isaac Sim, with Newton physics) simulates the robot, its sensors and the pitch, and runs the learned policies. **Nav2** is the navigation framework of ROS 2; the course uses only its path planner, as a CPU process inside the same container. Course code passes Nav2 the robot's position, estimated from its sensors on a known map of the pitch, and a target, and gets path points back. The pitch map is surveyed in advance and the current ToF only adds nearby observations; the robot does not map an unknown environment from its 8 × 8 ToF.

| Part | What it does |
|---|---|
| YOLO26n detector | Boxes the ball and goal in the RGB image |
| Geometry | Turns boxes, the ToF floor and joint angles into ball and goal positions |
| Nav2 planner (NavFn, A*) | Finds a low-cost path on a 2D cost map to a stance behind the ball |
| Path following and stages | Turn the next waypoint into a walking velocity; switch stages on conditions |
| Walking policy | Turns a velocity request into 14 joint targets at 50 Hz |
| Kick and recovery | Strikes the ball, then returns to standing |

Nav2 returns points on the floor, never joint angles. A short hand-written follower turns the next waypoint into a velocity request, and the walking policy does the balancing and stepping.

{#stages}
## Six stages of the shot

To design the stages, work backward from success. The kick skill works only in a small range: the ball must be in front of the right foot, the body must face the goal, and the robot must be standing still. Meeting these preconditions requires walking to a stance behind the ball and then adjusting position and heading precisely; all of that first requires knowing where the ball and goal are.

Then account for sensor uncertainty. While walking the camera sways, so positions measured up close are inaccurate; the robot therefore walks a little, settles and looks again. The kick must start from a still pose and be confirmed with the latest observation, so it gets its own handover stage.

```{learning-flow}
label: Walking-shot stages of the onboard policy
nodes:
- [Find the targets, Turn the head to see goal and ball]
- [Plan the approach, Pick a stance behind the ball and close in]
- [Refine up close, Short steps; settle and look again]
- [Align for the kick, Ball in front of the right foot; facing the goal]
- [Hand over to the kick, Stop; head back to center; body still]
- [Kick and recover, Kick skill strikes; back to standing]
details:
- Turn the head in place, locate the robot relative to the goal and find the ball.
- Nav2 plans a path to the stance behind the ball; the follower walks it.
- Up close, short steps and turns on the spot, looking again after each settle.
- Check that the ball is in front of the right foot and the body faces the goal.
- Stop, center the head, confirm stillness and a fresh observation, then hand over.
- The kick skill strikes and the robot returns to standing; the evaluator judges the goal.
layout: learning
loop: false
```

{#stage-find}
### Find the targets

The robot turns its head in place, finds the bases of the two goalposts with the method of Section 4.1, works out its own position and heading relative to the goal, and places the ball it sees on the same map. The position is accepted only after several consistent observations. If the goal goes unseen for too long later, or the ball information becomes too old, the policy comes back here to look again.

{#stage-approach}
### Plan the approach

The aim is to walk from afar to a preparation stance behind the ball. The stance lies behind the ball along the ball-to-goal direction, offset to one side so that the ball ends up in front of the right foot. From afar the robot only needs to get roughly close and position errors matter less, so it can walk continuously and replan often; precise alignment is left to the close-range stages.

Nav2 only plans the path: the start is the estimated robot position and the goal is this stance. The map comes from the surveyed pitch and the current ToF. Inside the pitch is free and its boundary is an obstacle; valid ToF returns add free cells along each ray and objects ahead, while zones without a valid return add no new evidence of free space; cells that are neither inside the known pitch nor validly observed stay unknown. NavFn searches this map with A*, never through unknown cells, keeps a margin of the robot radius from obstacles, and returns a list of XY points on the floor.

A hand-written follower takes a point a little further along the path and converts it into forward, sideways and turning velocities, and the trained walking policy turns those into joint motion. While walking, the robot replans repeatedly and periodically confirms the goal. Close enough to the ball, it moves on to refining up close; if the path is rejected or the information is too old, it stops and waits for new observations. The course has not validated dynamic obstacle avoidance.

{#stage-refine}
### Refine up close

Very close to the ball, the error from camera sway while walking is as large as the precision needed. This stage no longer calls Nav2: hand-written close-range logic decides when to step, stop and look again, while the trained walking policy still performs every step. After each brief forward step or turn the robot settles and measures the ball from a new image; if the ball is not completely in view, the body stays still and only the head turns to find it.

{#stage-align}
### Align for the kick

Using the latest observation after settling, the policy checks two things: whether the ball lies in the small range in front of the right foot that the kick skill was trained for, and whether the body faces the goal. Only when both hold does it move on to the handover; otherwise it returns to refining.

{#stage-handover}
### Hand over to the kick

Walking stops and the head returns to its kicking pose. Once the body is still, the policy confirms that the ball is still in kicking range and the goal information is still fresh, and only then passes control to the kick skill. If the conditions are not met, the kick is refused; where recovery is allowed, the robot steps back and looks again, otherwise this attempt stops.

{#stage-kick}
### Kick and recover

The skill runs once: the official right-foot kick policy strikes the ball, then smoothly hands over to the official standing policy. This onboard lab does not load a kick residual network. The onboard policy does not judge whether it scored; a separate evaluator decides that with simulator truth.

Stages switch mainly on observations and body state: when a condition fails, the policy stops, looks again or goes back a stage, so refining often repeats several times. Waiting for stillness and each stage's time limit keep every attempt bounded.

{#training}
## What was trained, and what was not

- **Detector.** The YOLO26n model was fine-tuned on images rendered in Isaac Sim. The box labels were generated automatically from the known scene geometry; 512 images trained it, 128 chose the checkpoint and 128 held-out images tested it (mAP50 0.978). That is an image detection score, not a shooting success rate.
- **Walking.** The walking policy was trained with PPO, the method of Part 3.
- **Kick and recovery.** The pretrained Pollen Robotics right-foot kick and standing policies remain frozen. Their transition timing is programmed.
- **Not trained.** The Nav2 planner, the geometry, the path follower and the stage logic are configured or programmed. The whole shot is never trained end to end.

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

| Parameter | Meaning |
|---|---|
| `--seconds 60` | Per-round task limit in simulation seconds, excluding standing initialization; completion or failure ends it earlier |
| `--output` | A new folder; rounds are kept in separate `episode-001`, `episode-002`, etc. folders |
| `--single-run` | Run one round, close the window and evaluate it |
| `--headless` | Run and evaluate one round without opening a window |

:::{enrichment} Key code parameters
- **Reset.** `lab/onboard/session.py` handles R and rebuilds each environment inside the same Isaac Sim process. Sensors, localization, action history and scoring start fresh; previous results are retained.
- **Initialization.** Temporary joint limits hold HOME while the free body settles on the ground. Once sensors and the first real policy output are ready, the limits are released before applying that output.
- **Rates.** `lab/onboard/executor.py` uses Newton physics at 100 Hz, joint policies at 50 Hz and RGB/ToF at 15 Hz. Each action spans two 10 ms physics steps; actuator delay remains 20 ms.
- **Display.** Each sensor capture uses one render subframe, and the panel displays the same RGB/ToF data supplied to the policy. No additional 1080p archive camera is created by default; `--record-hd` enables it.
- **Scene.** `lab/onboard_scene.py` creates the green pitch, reduced net mesh and standing initialization. `--scene indoor` and `--style A` are the defaults.
- **Walking and kicking.** Walking and recovery use `velstand.onnx`, the runtime copy of official `models/walk.onnx`; kicking uses `ball_kick_right.onnx`, copied from `models/kick-teacher.onnx`. `lab/onboard/official_kick_runtime.py` uses the kick policy exclusively for 0.14 seconds, updating its output every 20 ms, then blends to standing over 0.12 seconds, within a 4-second kick-and-recovery observation window.
- **Detection and localization.** `run.py` verifies `models/scene-detector/deployment.json` before launching the detector. `lab/scene_goal_points.py` finds post endpoints inside the box, accounting for perspective and metal reflections. A detection box is not yet an accepted position: consistency across frames and observation freshness must also pass.
- **Nav2.** `lab/scene_nav2/nav2.yaml` configures NavFn (A*, no unknown-space traversal) and a 2.5 cm costmap. `map_contract.py` builds the map from the measured pitch and current ToF, and checks the returned path. The follower converts path points into velocity requests.
- **Handover.** The near-field stage aligns the right foot, stops and returns the head to neutral. The final point estimate must be 7–11 cm ahead and 1.5–5.5 cm to the right, with heading error at most 10° and goal observations at most 0.5 seconds old. These limits belong to this local course skill, not a guarantee for physical hardware.
:::

Compare the motion with the RGB and ToF panels. Physics frequency describes simulation time; wall-clock display speed also depends on rendering and computation. A 100 Hz setting does not by itself guarantee real-time playback.

:::{course-image} media/isaac-ui-onboard-sensors.png
:alt: The complete Isaac Sim window for onboard shooting, with the green pitch, head RGB image and 8×8 ToF distances.

Watch the robot on the left alongside its current image and distance readings on the right. NR means no return; it does not establish that the space ahead is clear.
:::

{#results}
## Recorded results

On DGX Spark, the window configuration above ran for 20 consecutive rounds, pressing R between rounds and varying the start by at most ±1.5 cm forward/backward and ±1 cm sideways. Full success requires foot contact, the whole ball inside the goal, no fall and a return to standing. Completed attempts that do not score remain in the denominator.

```{benchmark-chart}
title: Twenty shots in one window
caption: Rounds are grouped in sets of five in execution order. All use one pitch and the same configuration. This is a small sample with very small start-position variation, not a general success rate.
```

| Rounds | Full success | Fall |
|---|---|---|
| 1–5 | 5 / 5 | 0 / 5 |
| 6–10 | 4 / 5 | 0 / 5 |
| 11–15 | 4 / 5 | 0 / 5 |
| 16–20 | 4 / 5 | 0 / 5 |

The measured result was **17/20 (85%)**, with **0/20** falls and no technical errors. Failed rounds were retained. <a href="_static/data/onboard-reset-evaluation.json">View every round</a>. Keep your own counts: a small sample around one start cannot guarantee every shot.

The main loop ran at a median of about **0.48 times real time**, ranging from 0.42 to 0.52. A 20-second simulated action would take about 41 seconds of wall time. This excludes loading, resets and post-run evaluation; website demonstrations play on the simulation clock.

For a failed attempt, first identify the stage where it stopped. `summary.json` lists the outcomes; each round’s `manifest.json` records its `terminal_reason` and failure details. A visible detection box does not guarantee an accepted geometric position. Without a sufficiently fresh position up close, the policy may stop to observe again or end the attempt. Also check path feasibility, kick conditions and balance after contact.

{#design}
## Limits of hand design

Every stage, transition and threshold of this onboard policy was written by people around one task: shooting at this goal. It is clear and easy to debug, but it fits only this task. For "dribble the ball to the center circle and stop" or "walk up to the ball without touching it", most of the perception can be reused, but the stages, conditions and thresholds must be redesigned and retuned.

If model inference can move off the robot to a GPU workstation, or to a cloud server at the cost of more network delay, a much larger vision-language-action model becomes possible: one that learns from a language instruction and demonstrations instead of hand-written stages, while trained motion skills still handle fast control. Part 5 takes this route with GR00T on a local workstation, using the supplied shooting demonstrations generated by hand-designed stages.

::::{quiz}
Which parts of the onboard shot were learned from data, and which were written or configured by people?

:::{answer}
The detector and the official walking and kicking policies were trained. The geometry, Nav2 path planning, the path follower and the stage logic were configured or programmed, and the shot as a whole was never trained end to end.
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
