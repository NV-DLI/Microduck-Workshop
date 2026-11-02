---
myst:
  html_meta:
    description: "How a hand-designed onboard policy reuses sensor estimates to shoot: why the shot runs onboard, how Nav2 path planning and trained motion skills are connected by hand-designed stages, and how to repeat the shot automatically in Isaac Sim."
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

The kick skill runs once: the frozen kick teacher plus the trained residual correction strikes the ball, and the robot returns to standing. The onboard policy does not judge whether it scored; a separate evaluator decides that with simulator truth.

Stages switch mainly on observations and body state: when a condition fails, the policy stops, looks again or goes back a stage, so refining often repeats several times. Waiting for stillness and each stage's time limit keep every attempt bounded.

{#training}
## What was trained, and what was not

- **Detector.** The YOLO26n model was fine-tuned on images rendered in Isaac Sim. The box labels were generated automatically from the known scene geometry; 512 images trained it, 128 chose the checkpoint and 128 held-out images tested it (mAP50 0.978). That is an image detection score, not a shooting success rate.
- **Walking.** The walking policy was trained with PPO, the method of Part 3.
- **Kick.** The supplied Pollen Robotics kick and walking policies stay frozen as the teacher; a small residual network, trained with PPO in Isaac Lab, adds a bounded correction on top.
- **Not trained.** The Nav2 planner, the geometry, the path follower and the stage logic are configured or programmed. The whole shot is never trained end to end.

{#run}
## Repeat onboard shots automatically

Prepare the course container as described in [Section 1.2](setup.md), then return to the container terminal; if the container is already running, enter it from the host with `docker exec -it microduck-workshop bash`. Then run:

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py football --repeat 5 --seconds 45 --no-record-hd --output outputs/onboard-football-repeat
```

| Parameter | Meaning |
|---|---|
| `--repeat 5` | Run five shots one after another; `--repeat 0` keeps going until you press **Ctrl+C** |
| `--seconds 45` | Time limit of each shot, in simulation seconds |
| `--no-record-hd` | Skip the extra 1080p third-person archive video; the sensor panels, task logs and scoring are kept |
| `--output` | A new folder; each shot writes its own `episode-001`, `episode-002`, ... inside it |

:::{enrichment} Key code parameters
This explains the implementation behind the `python3 scripts/run.py football` command just above. The table lists the parameters you can change; loading parameters such as `--model` and `--detector-deployment` are passed to the scene automatically by `run.py` after it checks the files, so you do not supply them.

- **Repeating.** `scripts/repeat_episodes.py` launches each shot in its own process group and writes `summary.json` in the `--output` folder. The first shot uses the default `--seed` and each later shot adds one, which changes the simulated sensor noise; the start pose stays the same. `--root-y` moves the robot's sideways start (default 0.04 m). Without `--repeat`, `run.py football` runs one shot.
- **Start-up.** Every shot holds the joints at HOME with the body free on the ground, waits for the sensors and the walking policy's first output, then releases the joints before that output is applied.
- **Scene.** `lab/onboard_scene.py` creates the green pitch with this start-up; `--scene indoor` and `--style A` are the defaults of `run.py football`.
- **Walking policy.** The repository's `models/walk.onnx` is copied to `/workspace/state/student-course/simulation/core/models/reference/velstand.onnx` in the container, and `run.py` loads it automatically through `--model`.
- **Detector.** `run.py` first checks the SHA-256 of the weights listed in `models/scene-detector/deployment.json`, then writes `/workspace/state/scene-detector-deployment.json` and loads it automatically through `--detector-deployment`.
- **Nav2.** `lab/scene_nav2/nav2.yaml` configures only the NavFn planner (A*, no planning through unknown space) and a cost map with 2.5 cm resolution, with a margin based on the robot radius; `lab/scene_nav2/map_contract.py` builds the map from the surveyed pitch and current ToF and checks the path; `lab/runtime/simulation/core/football/controller.py` turns the returned path into velocity requests.
- **Close-range stages.** `ball_reacquisition.py`, `final_approach.py` and `head_handoff.py` in `lab/runtime/simulation/core/evaluation/football/` implement finding the ball again, stop-and-look refining and the handover up close.
- **Kick and recovery.** The executor loads them from the fixed path `/workspace/state/student-course/configuration/core/kick_policy/export/deployment.json`, which binds the two networks copied from `models/kick-teacher.onnx` and `models/kick-residual.onnx`.
:::

Isaac Sim closes after each shot and opens again for the next one, so every shot restarts from the same start pose. Watch the RGB and ToF panels while the robot walks, stops and looks.

After each shot the terminal prints success or the reason for failure, and the final line counts successes and failures; failed shots are counted, never skipped. **Ctrl+C** stops the current shot and starts no new one. A technical error, such as a crash or a failed independent check, also stops the loop; that shot's files are kept, but it is not counted as a success or failure. The window usually runs slower than real time because the workstation renders the scene and sensors; this speed says nothing about the robot's own computer.

:::{course-image} media/isaac-ui-onboard-sensors.png
:alt: The complete Isaac Sim window for onboard shooting, with the green pitch, head RGB image and 8×8 ToF distances.

Watch the robot on the left alongside its current image and distance readings on the right. NR means no return; it does not establish that the space ahead is clear.
:::

{#results}
## Recorded results

To see how the pipeline responds to the start position, the course ran a check fixed in advance: five sideways starts of the robot relative to the ball, one attempt each, without a window and with `--no-record-hd`. Your `--repeat` run keeps one start, so expect your own counts rather than these numbers.

```{benchmark-chart}
title: Five sideways starts, one onboard shot each
caption: Each row is one start position relative to the ball, with one attempt. Full success means foot contact, the whole ball in the goal and a return to standing. Five attempts on one pitch are a small check, not a general success rate.
```

| Sideways start | Full success | Fall | How it ended |
|---|---|---|---|
| −4 cm | 1 / 1 | 0 / 1 | Scored at 31.5 s, back to standing |
| 0 cm | 1 / 1 | 0 / 1 | Scored at 14.5 s, back to standing |
| +4 cm (lab default) | 0 / 1 | 0 / 1 | Stayed re-observing up close until that stage's time limit |
| +8 cm | 0 / 1 | 0 / 1 | Kick refused; stopped after a failed safety check |
| +12 cm | 1 / 1 | 0 / 1 | Scored at 28.1 s, back to standing |

The success rate in this check was **3/5 (60%)**: all three scored and returned to standing, and none fell. Both failures ended before the foot touched the ball, one in each category below. The records show the sequence of events; they do not establish the root cause:

- **Position not confirmed up close.** After approaching, the robot could not keep getting a position estimate reliable enough for alignment, so it stayed in re-observation until that stage's time limit. The detector still boxed the ball during this time; the position built from the box was what kept being rejected.
- **Kick conditions not met.** The ball ended outside the range the kick skill was trained for, so the kick was refused. While stepping back to adjust, the perception safety check did not pass, and the robot stopped trying.

If one of your shots fails, also check for **no admissible path** (Nav2 finds no safe path to the stance) and **balance during the kick**. `summary.json` lists each shot's outcome; the `terminal_reason` in a shot's `manifest.json` names the stage that ended it.

{#design}
## Limits of hand design

Every stage, transition and threshold of this onboard policy was written by people around one task: shooting at this goal. It is clear and easy to debug, but it fits only this task. For "dribble the ball to the center circle and stop" or "walk up to the ball without touching it", most of the perception can be reused, but the stages, conditions and thresholds must be redesigned and retuned.

If model inference can move off the robot to a GPU workstation, or to a cloud server at the cost of more network delay, a much larger vision-language-action model becomes possible: one that learns from a language instruction and demonstrations instead of hand-written stages, while trained motion skills still handle fast control. Part 5 takes this route with GR00T on a local workstation, using demonstrations recorded with this onboard policy.

::::{quiz}
Which parts of the onboard shot were learned from data, and which were written or configured by people?

:::{answer}
The detector, the walking policy and the kick residual were trained. The geometry, Nav2 path planning, the path follower and the stage logic were configured or programmed, and the shot as a whole was never trained end to end.
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
