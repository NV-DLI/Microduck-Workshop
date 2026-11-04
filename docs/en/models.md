---
myst:
  html_meta:
    description: Give your own agent a concrete robot-task brief, a staged implementation process, and evidence-based acceptance criteria.
---

# 6.1 Let Agent do the heavy lift

:::{lesson-intro}
:meta: Section 6.1 · From this course to your own robot task

The course connects robot models, basic motion, perception and autonomous shooting. You can now bring your own robot and objective to your preferred agent for help with design, implementation and validation. Supply the goal and constraints, and use actual evidence to judge the result.
:::

{#start}
## Bring this task guide

[ROBOT_TASK_GUIDE.en.md](https://github.com/NV-DLI/Microduck-Workshop/blob/main/ROBOT_TASK_GUIDE.en.md), the single English guide in the repository, turns the course's workflow and lessons into a project guide that you can give directly to an agent. It does not depend on a particular tool or require a shooting task. Microduck can reuse the course assets and skills; another robot requires its own joint, action and physics checks.

Provide the English guide with your project; you can discuss the task with your agent in your preferred language. Ask it to inspect the model and sensors, ask for necessary missing information, recommend models and whether training is needed, plan data collection, and then propose staged implementation. It does not need to be named AGENTS.md. If your tool reads that project convention, merge the relevant instructions into your own AGENTS.md.

{#brief}
## Supply the essential context

| Information | What to specify | Microduck shooting example |
|---|---|---|
| Robot | USD, URDF or MJCF, meshes, units, joint names/order, limits and default pose | 14 controlled joints with a fixed action mapping |
| Simulation | Isaac Sim / Isaac Lab versions, physics backend, ground, obstacles and contacts | Newton, flat pitch, ball and goal |
| Task and acceptance | Starts, goal, allowed actions, completion, timeout and failure definitions | Approach, actual foot-ball contact, whole ball across the goal and standing recovery |
| Observations | Cameras, ToF, IMU and joints, with rates, frames and latency | Visual target estimates and body feedback |
| Existing capabilities | Reusable models, input/output contracts, data and runnable entry points | Walking, kicking and standing recovery |
| Resources | Hardware, memory, acceptable runtime, training budget and output location | Small initial checks and separate output directories |

Write “unknown; inspect or propose a measurement” when information is missing. Do not let the agent guess joint order, units or success criteria: a program may run while producing meaningless results.

The guide asks the agent to distinguish the sensors actually available from the information the task needs: what can be read directly, what must be estimated, and what is missing. Given existing skills, resources and data, it should explain whether to reuse rules or models, or train perception, reinforcement-learning policies or a VLA. If training is needed, it should specify data sources, collection, annotation and independent validation splits.

{#workflow}
## Advance through verifiable steps

```{learning-flow}
label: Organize a robot project using the course experience
nodes:
- ["Verify the interface", "Joints, frames, observations and actions"]
- ["Build a minimal loop", "Stable initialization and one basic behavior"]
- ["Compose the task", "Perception, approach, execution and recovery"]
- ["Validate and deliver", "New conditions, failure records and reproduction"]
details:
- Read the actual model and configuration and write an input/output contract. Loading an asset does not establish correct joints or physics.
- Test standing, walking or a basic manipulation behavior in a simple environment. Release initialization aids before policy execution; support is not learned capability.
- Separate inspectable stages. Reuse existing skills and simple control first, then use failure evidence to decide whether perception, locomotion or VLA training is needed.
- Freeze a candidate and test predefined new starts. Retain every attempt and deliver configurations, model provenance, outcomes, limits and reproduction steps.
layout: learning
```

The shooting project illustrates how to locate an error before choosing a change. Incorrect ball position calls for sensor, coordinate and timing checks. Drifting despite seeing the ball calls for command, feedback and braking checks. A missed kick from a suitable stance calls for contact-geometry and skill-handoff checks. Falling loss, smooth video and one success do not replace task acceptance.

{#handoff}
## Give the agent a concrete starting point

Adapt this brief and attach the model, scene and guide:

> Use ROBOT_TASK_GUIDE.en.md to help implement a simulated robot task. The robot and model paths are [fill in], simulation versions and physics backend are [fill in], and available sensors and policies are [fill in]. In [scene and starting range], it should complete [task]. Success requires [measurable conditions]; failures include [fill in]. Resources and budget are [fill in]. First inspect the files and interfaces, ask about available and task-required sensor inputs, and recommend models, whether training is needed and a data-collection plan, explaining the rationale and unresolved decisions. Then begin with the smallest verifiable closed loop, provide actual evidence at each stage, preserve failures and original models, and advance only when the stage is ready.

On resuming work, ask the agent to read the saved configuration, results and unresolved questions. This gives the next session a concrete starting point and makes the remaining work visible.

{#inventory}
## Reference: course models and data

| File (host `models/`, container `/workspace/Microduck-Workshop/models/`) | Purpose | Course commands that read it |
|---|---|---|
| `walk.onnx` | Fixed walking policy | Section 2.1 keyboard football (read automatically); Section 2.2 `walk --model models/walk.onnx`; the low-level walking policy of the Section 4.3 onboard shot and of the Section 5.2 GR00T execution layer uses a copy of the same model |
| `terrain-walk.onnx` | Section 3.4 fixed terrain walking actor, about 0.79 MB | Section 3.4 `terrain --model models/terrain-walk.onnx` |
| `kick-teacher.onnx` | Base kicking action | Section 2.1 keyboard football (Space, read automatically); the kick skill of the Section 4.3 onboard shot uses a copy of the same model |
| `kick-residual.onnx` | Residual kick adjustment | Used by the Section 5.2 GR00T execution layer; not loaded by the Section 4.3 onboard shot |
| `kick-checkpoint.pt` | Retained kick training checkpoint | Retained; not read directly by this course's hands-on commands |
| `detector.pt` | Fixed visual detector | Retained; not read directly by this course's hands-on commands |
| `detector.onnx` | ONNX detector export | Retained; not read directly by this course's hands-on commands |
| `indoor-detector/detector.pt` / `indoor-detector/detector.onnx` | Another indoor detector, for a scene with a 7 cm ball; not used on the course pitch | Retained; not read directly by this course's hands-on commands |
| `scene-detector/detector.pt` / `scene-detector/detector.onnx` | Official 10 cm ball and four-surface detector; current default | Checked and loaded automatically by the Section 4.3 `football` command as the onboard policy's visual detector |
| `official-browser/walking.onnx` / `sitstand.onnx` | Browser-version checks and standing tests; the complete walking task uses `walk.onnx` | Retained; not read directly by this course's hands-on commands |

Run the command below inside the container to verify fixed-model sizes and hashes.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py verify-models
```

These files ship with the Git repository, so cloning it is enough; no Git LFS or separate model account is needed. `models/manifest.json` records sizes, sources, and SHA-256 identities, and each lab checks model identity before running to guard against same-name replacements. The relative paths and SHA-256 of the current default detector are recorded in `models/scene-detector/deployment.json`.

**In the repository.** The weights for the labs of Parts 2–4; the pretrained PPO walking result of Section 3.3 and the slope-training logs of Section 3.4 are in `datasets/walking-ppo` and `datasets/terrain-ppo` (`/workspace/Microduck-Workshop/datasets/` in the container, read-only).

**GR00T.** The GR00T demonstrations and checkpoint described in Section 5.2 are kept separately by the course; you do not need to download them for this course.

{#practice-data}
## Practice data and your outputs

`datasets/scene-sample/` is a small set of sample images and labels that the course labs do not use directly; it is too small to test real-world generalization. Keep your own checkpoints and logs in `outputs/` and leave the course files in `models/` unchanged.

{#licenses}
## Sources and usage terms

Robot geometry, community policies, and dependencies each follow their own source and license terms; for third-party assets those terms apply, and the course-code license does not replace them. Before redistributing changes, read the repository’s [attribution and license notices](https://github.com/NV-DLI/Microduck-Workshop/blob/main/THIRD_PARTY_NOTICES.md).
