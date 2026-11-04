# Build a simulated robot task with your agent

Give this guide to your preferred agent with your robot, scene and project files. It applies to Microduck and can be adapted to other robots; it does not require a particular agent product. Discuss the task in the learner’s preferred language; this English file is the single shared guide. Work in the intended project and preserve original course weights as references.

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

## 3. Ask which sensors exist and which information the task needs

Inspect the robot model, sensor configuration and available logs before asking the learner to supply facts that are already present. Create two inventories: sensors actually available to the controller, and information required at each task stage. A sensor in a product description is not proof that the supplied robot revision or simulator exposes it.

Ask the learner to resolve gaps with questions such as:

- Which cameras, ToF/depth sensors, IMUs, joint encoders, contact/force sensors or localization sources are actually available? Which exist only in simulation, and which are intended for later hardware deployment?
- What are their mounting poses, fields of view, ranges, resolutions, units, calibration, update rates and latency? Can sample frames and synchronized logs be inspected?
- Must the robot identify an object, measure its distance, estimate its own pose, detect contact, or understand a language instruction? Which information is essential for success, and which is optional?
- What can become occluded or unavailable? Are maps or known object dimensions allowed? Can missing information be estimated, or must the task or sensor configuration change?

Produce a task-to-input table before selecting a model:

| Task stage | Information needed | Candidate source | Check or missing information |
|---|---|---|---|
| Balance and locomotion | Joint configuration, body motion and orientation estimate | Encoders and IMU/body-state estimator | Joint mapping, axes, calibration and feedback rate |
| Find and approach a target | Identity, bearing and usable distance estimate | RGB detection plus aligned ToF/depth; another method only with stated assumptions | Visibility, camera/depth calibration, frame transforms and usable range |
| Navigate around obstacles | Robot pose, traversable space and obstacles | Localization, map and depth/LiDAR if available | Map frame, footprint, clearance and whether the locomotion policy can traverse the route |
| Execute a contact skill | Relative geometry, skill state and evidence of contact | Available perception and contact feedback, or an explicitly validated estimate | Which signals exist at deployment; scoring ground truth stays separate |
| Follow an instruction | Task meaning and scene context | Text plus compatible visual/state inputs | Whether language variation is required or a fixed goal is enough |

These are design examples, not a claim that every robot has every sensor. For the course Microduck, the onboard shooting path uses RGB and ToF with body feedback. The GR00T example uses RGB history and body state; ToF is displayed but is not a model input. Select inputs per controller instead of forwarding every sensor indiscriminately. If hardware is unavailable, label simulator-only measurements and future calibration work explicitly.

## 4. Recommend an architecture, model and training decision

Ask what pretrained skills, detectors, datasets and control software already exist, what they have actually been validated on, and where inference must run. Establish memory/compute limits, acceptable latency, required update rates, network availability and data-collection effort. Ask whether the task needs new language instructions or visual generalization, or simply a reliable fixed behavior.

Recommend a primary approach and, when useful, one simpler alternative. Explain the required inputs, action semantics, deployment fit, expected data effort and first validation step. Do not recommend a large model just because it is available.

| Situation | Starting recommendation | Is training needed? |
|---|---|---|
| Known task with reusable motion skills and reliable target estimates | Geometric control, explicit stages and existing policies | Often no new training; first validate interfaces and closed-loop behavior |
| Target perception is the missing capability | Compatible pretrained detector, segmenter or pose estimator | Reuse if held-out tests suffice; fine-tune with labeled task images if they do not |
| Balance, locomotion or contact behavior is missing | Compatible existing skill, otherwise a task-specific RL policy trained in simulation, such as with PPO | Train only the missing skill; requires a credible simulator, observations, rewards and termination rules |
| A demonstrated behavior must be learned from observations | Imitation policy; consider a VLA when language and visual context are useful | Requires aligned demonstrations and a model whose embodiment/action interface can be adapted |
| A complex layout requires routing while a walking skill already exists | Navigation/localization plus a velocity-to-skill adapter | Planning itself need not be learned; new terrain may require skill training and separate validation |

Before naming a concrete model or checkpoint, check its current primary documentation, license, input modalities, action interface, robot compatibility and resource requirements. State what has been verified and what is only a proposed candidate. GR00T is one course example, not a mandatory choice. For fine-tuning, name the modules to freeze/update and the reason; for reuse, state that weights stay unchanged. Do not promise suitability from a model name or benchmark alone.

Return an explicit decision: **reuse without training**, **fine-tune a specified component**, or **train a new specified skill**, with evidence and a stopping criterion. If essential evidence is missing, recommend a small feasibility check first. Confirm the learner's task priorities and resource/data constraints when they affect the choice; ordinary implementation work can continue within the agreed scope.

## 5. Propose data collection and ask for the missing inputs

Ask whether existing data or an executable teacher is available, how it was produced, which robot/scene/version it belongs to, whether its use is permitted, and what successful and failed examples it contains. Ask who or what can provide demonstrations: teleoperation, a hand-designed controller, a validated policy, or another reproducible teacher. Establish collection time, storage and labeling budgets; do not invent access to hardware, labels or expert demonstrations.

Match the dataset to the training decision:

- **No new training:** collect a compact evaluation set and complete execution logs to verify the reused system. A training dataset is not automatically required.
- **Perception:** collect images across target poses, occlusion, distance, lighting and backgrounds. Define boxes, masks, keypoints or poses as required by the selected model. Simulator-generated labels must match the visible image and chosen coordinate frame; inspect samples before scaling collection.
- **Reinforcement learning:** collect transitions by interaction with the simulator rather than treating a demonstration corpus as mandatory. Specify observations, actions, rewards, termination, randomized conditions, rollout budget and evaluation starts. Record simulator and policy versions.
- **Imitation/VLA:** collect complete tasks plus recovery and corrective examples, using a teacher that can perform the intended behavior. Record synchronized images, state, task text and the actual command targets with their timestamps. Include enough variation for the desired tasks and instructions; repeated copies of one success do not establish coverage.

Propose a concrete collection plan: source/teacher, initial pilot size and duration, sampling rates, required fields and labels, conditions to vary, quality checks, train/validation/test split, estimated storage and a criterion for collecting more. Justify any proposed count as a planning estimate, not a guaranteed sample requirement. Review pilot data and failure coverage before committing to a large collection or training run.

Use a per-episode manifest containing robot/model identity, scene and task, initial conditions, sensor calibration, timestamps, observations, actions and units, termination/outcome, data provenance and schema version. Align inputs causally: future frames or privileged simulator positions must not leak into deployment observations. Ground truth can supply labels, rewards or evaluation, and a privileged training critic where explicitly designed; the deployed actor still needs only available inputs.

Split complete episodes and, where generalization is claimed, hold out scenes or starting conditions. Do not randomly scatter neighboring frames from the same trajectory across training and evaluation. Fit normalization on training data only; audit missing frames, stale readings, action bounds, mislabeled outcomes and padded action chunks. Retain failed attempts and technical errors with their labels rather than silently discarding them. Keep the final test set independent of model selection.

Deliver a short design recommendation before large-scale work: the task-to-sensor table, proposed model and action interface, reuse/training decision, dataset plan, unresolved questions and the first small experiment. Update it when evidence changes.

## 6. Implement stages and retain evidence

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

## 7. Make completion observable

For shooting, distinguish a kick request, actual foot-ball contact, the whole ball crossing the goal plane, and recovery to standing. Apply ball geometry when testing a complete goal; proximity of its center is insufficient. Let the owner select the task's acceptance criteria and do not weaken them after seeing results.

State transitions need explicit observations, tolerances, waits, timeouts and recovery paths. When a target becomes briefly invisible, retain the age and confidence bounds of its last reliable estimate. Do not use stale positions indefinitely or silently substitute simulator ground truth for sensors. Ground truth may support diagnosis and scoring, separately from policy input.

Reuse stable skills first. Low-level control handles joints, body feedback and contacts; high-level logic handles objectives and requests. Define one-shot triggers, requests while a skill is busy, completion/recovery and controlled stopping on exceptions.

## 8. Add training when evidence supports it

- Reinforcement learning: map observations, actions, rewards, termination and randomization to the task. Increasing reward is not task completion. Keep training and independent evaluation configurations separately.
- Imitation learning / VLA: define input/action meaning first, then collect full-task and corrective demonstrations. Align images, states and labels causally. Split complete episodes, use training-only normalization statistics, and define padding and valid targets.
- Record frozen and trainable modules. Microduck's 8-D high-level design freezes the vision-language backbone, trains the action head and projections, and preserves underlying skills. Direct joint output needs a redesigned dataset and balance validation.
- Action chunks reduce high-level calls but need execution horizons, replanning, expiry and buffering rules. Asynchronous inference and RTC can connect chunks; they do not remove computation or network delay. Protect discrete skill requests separately.
- Separate visual variation from physical variation. Pure appearance changes may retain labels; changes to friction, geometry, starts or dynamics generally need new demonstrations. Generated imagery does not establish executable behavior in the new environment.

## 9. Let failures guide the next change

| Symptom | Inspect first |
|---|---|
| Model loads but pose is wrong | Joint mapping, units, zeros, inertia, collisions and startup |
| Target estimate jumps or disappears | Images/depth, exposure, transforms, timestamps and visibility |
| Target is visible but motion drifts or repeatedly stops | Request meaning, body feedback, braking, tolerances and transitions |
| Suitable stance but missing contact | Contact geometry, trigger conditions, handoff and delay |
| Low offline error but closed-loop failure | Distribution shift, corrective data, accumulated errors and timing |
| Smooth video but jerky live execution | Wall-clock timing of rendering, transfer and inference |

Keep each change bounded and preserve its before/after configuration and reason. Retain failed candidates rather than replacing their results. Do not repeat the same training indefinitely. Set stopping conditions and resource budgets beforehand and use stage conclusions to choose the next action.

## 10. Leave a usable handoff

Maintain a short progress record in the target project: current revision, launch commands, input/model identities, completed checks, running processes, failures, unresolved questions, next steps and stopping conditions. Distinguish planned, started, completed and validated.

Deliver actual changes, validation methods, all outcomes and limits. State when a test was not run or lacks dependencies; “should work” is not execution evidence. Keep student instructions focused on the current usable workflow, with internal failures, old media, training logs and raw data preserved separately.
