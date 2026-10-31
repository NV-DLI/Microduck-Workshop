---
myst:
  html_meta:
    description: "The onboard policy splits the walking shot into human-designed stages and moves on only when a stage's conditions hold. The stages come from our understanding of the shooting task, and a new task needs a new design."
---

# 4.2 Hand-designed shooting stages

:::{lesson-intro}
:meta: Section 4.2 · Estimated learning time: 15 minutes

The onboard policy splits the walking shot into human-designed stages and moves on only when a stage's conditions hold.
:::

{#stages}
## Six stages of the shot

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
- The robot turns its head to find the goal and ball and works out its position relative to the goal. If the information is too old or inconsistent, it stops and looks again.
- The ball and goal positions give a shooting direction; a stance is chosen behind the ball along it, and the ROS 2 navigation framework Nav2 plans a path that the robot follows in segments, confirming the goal again along the way.
- Closer to the ball the robot switches to short steps or turns on the spot. After each segment it settles and looks at the ball again, because while walking the camera sways and measured positions are less accurate.
- It checks that the ball is in front of the right foot and that the body faces the goal. If not, it returns to refining.
- Walking stops, the head returns to its kicking pose, and only when the body is still and the goal information is fresh does control pass to the kick skill.
- The trained kick and recovery skills run, then the robot returns to standing. Whether it scored is judged by the evaluator with simulator truth.
layout: learning
loop: false
```

Stages do not switch on a timer but on conditions: when a condition fails, the policy stops, looks again or goes back a stage, so refining often repeats several times in one shot. The onboard policy does not control the joints itself: movement requests go to the walking policy you tried in Section 2.1, and the kick goes to the kick skill.

{#run}
## Observe an autonomous shot

Prepare the course container as described in [Section 1.2](setup.md), then return to the container terminal; if the container is already running, enter it from the host with `docker exec -it microduck-workshop bash`. Then run:

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py football --seconds 45 --output outputs/onboard-football
```

When the Isaac Sim window opens, it automatically shows the head RGB image and the 8×8 ToF panel. Watch how the robot alternates walking, stopping and looking, and how the task stages change. A run is not guaranteed to score; the outcome is in the terminal output and the records in the output folder. Press **Ctrl+C** to stop at any time, and use a new `--output` folder when you run again.

:::{course-image} media/isaac-ui-onboard-sensors.png
:alt: The complete Isaac Sim window for onboard shooting, with the green pitch, head RGB image and 8×8 ToF distances.

Watch the robot on the left alongside its current image and distance readings on the right. NR means no return; it does not establish that the space ahead is clear.
:::

:::{enrichment} Key code parameters
- **Scene.** `lab/onboard_scene.py` creates the green pitch; `--scene indoor` and `--style A` are the defaults of `run.py football`.
- **Walking policy.** The repository's `models/walk.onnx` is copied to `/workspace/state/student-course/simulation/core/models/reference/velstand.onnx` in the container and loaded with `--model`.
- **Detector.** `run.py` first checks the SHA-256 of the weights listed in `models/scene-detector/deployment.json`, then writes `/workspace/state/scene-detector-deployment.json` and loads it with `--detector-deployment`.
- **Kick and recovery.** The executor loads them from the fixed path `/workspace/state/student-course/configuration/core/kick_policy/export/deployment.json`, which binds the two networks copied from `models/kick-teacher.onnx` and `models/kick-residual.onnx`.
:::

{#design}
## Where these stages come from

Every stage and transition comes from our understanding of the shooting task: the kick skill needs the ball in front of the right foot and the body facing the goal, so there is a stance and an alignment stage; positions measured while walking are inaccurate, so the robot alternates walking, stopping and looking; the kick must start from a still pose, so there is a separate handover.

Such a pipeline is clear and easy to debug, but it fits only this task. For "dribble the ball to the center circle and stop" or "walk up to the ball without touching it", most stages, conditions and thresholds must be redesigned. Could a single instruction describe the task, with a model learning from demonstrations what to do? Part 5 tries this route with GR00T, using demonstrations recorded with the onboard policy.

::::{quiz}
To make Microduck dribble the ball to the center circle and stop, can the existing onboard policy be reused directly?

:::{answer}
Most of the perception can be reused, but the stance, stage order, transitions and handover were designed for shooting and must be redesigned and retuned. Rewriting the pipeline by hand for every new task is why Part 5 turns to learning from demonstrations.
:::
::::

{#references}
## References

{.references}
- [Nav2 documentation ↗](https://docs.nav2.org/)
