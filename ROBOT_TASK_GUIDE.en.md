# Build a simulated robot task with your agent

Give this guide to your preferred agent with your robot, scene and project files. It applies to Microduck and can be adapted to other robots; it does not require a particular agent product. Work in the intended project and preserve original course weights as references.

## 1. Complete the task brief

- Robot name/version and model paths:
- USD / URDF / MJCF, meshes and materials, units and world axes:
- Controlled joint names/order, limits, default pose and actuator parameters:
- Isaac Sim / Isaac Lab versions, physics backend and launch method:
- Scene, ground, obstacles, contact and friction parameters:
- Task, starting range, goal and allowed actions:
- Measurable success, failure, interruption, timeout and technical-error definitions:
- Cameras, ToF/depth, IMU and encoders, with rates, frames and latency:
- Existing policies, provenance, licenses, input/output contracts and entry points:
- Hardware, memory, time budget and output directory:
- Files that must remain unchanged, experiments to preserve and decisions requiring the owner:

Mark unknowns explicitly. Inspect files or propose measurements instead of guessing dimensions, inertia, joint order, units or outcomes. Proceed with ordinary implementation choices using available context; ask together for necessary information that actually blocks progress.

## 2. Inspect the model before choosing the architecture

Read the real files and write an interface table: each observation/action field, dimension, order, unit, coordinate frame, normalization, update rate and age. Check active versus passive joints, collision geometry, mass/inertia, axes, zero positions and encoder offsets. Correct appearance does not establish correct dynamics.

Separate physics steps, policy updates and sensor sampling. Record configured simulation time and measured wall-clock throughput. A 50 Hz configuration does not prove 50 completed control cycles per real second. Simulation-time video playback is not real-time performance evidence.

Choose the smallest architecture that meets the task. Microduck shooting combines perception, hand-designed stages and existing walking/kicking skills; a VLA can learn requests above these skills. Other robots may need different layers. Decide who maintains balance, approaches targets and triggers discrete skills before choosing what to train. A simple pitch need not use a full navigation stack; complex layouts can separate Nav2 route planning from terrain locomotion.

## 3. Implement stages and retain evidence

| Stage | Work | Acceptance and deliverable |
|---|---|---|
| Assets and startup | Load, inspect joints/contacts, establish reproducible initialization | Provenance, interface contract, initial contact and pose checks |
| Minimal motion loop | Standing, one walking direction or one basic manipulation | Requests versus feedback, falls/contacts, release of initialization aids |
| Perception loop | Detect/localize targets and align frames/timestamps | Raw observations and estimates, occlusion/boundary/stale-data behavior |
| Task composition | Approach, brake, align, act, recover and handle exceptions | State entry/exit conditions, timeouts and small actual runs |
| Training if needed | Prepare data/rewards/action contract and select trainable parameters | Data checks, episode split, logs and separately saved checkpoints |
| Full evaluation | Freeze configuration and test predefined new conditions | All attempts and denominators, technical errors and raw logs |
| Delivery | Documentation, entry points, dependencies, provenance and recovery | Reproduction steps, verified scope, limits and next steps |

For Microduck, reuse the validated grounded HOME initialization: keep the root free, temporarily lock joints, wait for sensors and the first actual policy output, then release constraints and restore parameters before applying that output. Revalidate ground contact on new terrain. Design appropriate initialization for another robot instead of copying Microduck's height. A temporary lock is neither learned balance nor evidence of unpowered stability.

## 4. Make completion observable

For shooting, distinguish a kick request, actual foot-ball contact, the whole ball crossing the goal plane, and recovery to standing. Apply ball geometry when testing a complete goal; proximity of its center is insufficient. Let the owner select the task's acceptance criteria and do not weaken them after seeing results.

State transitions need explicit observations, tolerances, waits, timeouts and recovery paths. When a target becomes briefly invisible, retain the age and confidence bounds of its last reliable estimate. Do not use stale positions indefinitely or silently substitute simulator ground truth for sensors. Ground truth may support diagnosis and scoring, separately from policy input.

Reuse stable skills first. Low-level control handles joints, body feedback and contacts; high-level logic handles objectives and requests. Define one-shot triggers, requests while a skill is busy, completion/recovery and controlled stopping on exceptions.

## 5. Add training when evidence supports it

- Reinforcement learning: map observations, actions, rewards, termination and randomization to the task. Increasing reward is not task completion. Keep training and independent evaluation configurations separately.
- Imitation learning / VLA: define input/action meaning first, then collect full-task and corrective demonstrations. Align images, states and labels causally. Split complete episodes, use training-only normalization statistics, and define padding and valid targets.
- Record frozen and trainable modules. Microduck's 8-D high-level design freezes the vision-language backbone, trains the action head and projections, and preserves underlying skills. Direct joint output needs a redesigned dataset and balance validation.
- Action chunks reduce high-level calls but need execution horizons, replanning, expiry and buffering rules. Asynchronous inference and RTC can connect chunks; they do not remove computation or network delay. Protect discrete skill requests separately.
- Separate visual variation from physical variation. Pure appearance changes may retain labels; changes to friction, geometry, starts or dynamics generally need new demonstrations. Generated imagery does not establish executable behavior in the new environment.

## 6. Let failures guide the next change

| Symptom | Inspect first |
|---|---|
| Model loads but pose is wrong | Joint mapping, units, zeros, inertia, collisions and startup |
| Target estimate jumps or disappears | Images/depth, exposure, transforms, timestamps and visibility |
| Target is visible but motion drifts or repeatedly stops | Request meaning, body feedback, braking, tolerances and transitions |
| Suitable stance but missing contact | Contact geometry, trigger conditions, handoff and delay |
| Low offline error but closed-loop failure | Distribution shift, corrective data, accumulated errors and timing |
| Smooth video but jerky live execution | Wall-clock timing of rendering, transfer and inference |

Keep each change bounded and preserve its before/after configuration and reason. Retain failed candidates rather than replacing their results. Do not repeat the same training indefinitely. Set stopping conditions and resource budgets beforehand and use stage conclusions to choose the next action.

## 7. Leave a usable handoff

Maintain a short progress record in the target project: current revision, launch commands, input/model identities, completed checks, running processes, failures, unresolved questions, next steps and stopping conditions. Distinguish planned, started, completed and validated.

Deliver actual changes, validation methods, all outcomes and limits. State when a test was not run or lacks dependencies; “should work” is not execution evidence. Keep student instructions focused on the current usable workflow, with internal failures, old media, training logs and raw data preserved separately.
