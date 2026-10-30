---
myst:
  html_meta:
    description: 'In the course container, post-train the GR00T action head on the prepared walking-shot demonstrations and understand the training budget, sampling and time; then validate the provided checkpoint on the full shot in Isaac Lab and Newton.'
---

# 4.3 Train and validate the GR00T walking shot

:::{lesson-intro}
:meta: Section 4.3 · Estimated learning time: 45 minutes (excluding about 1 hour for a full training run and the validation run)

In the course container, post-train the GR00T action head on the prepared walking-shot demonstrations and understand the training budget, sampling and time; then validate the provided checkpoint on the full shot in Isaac Lab and Newton.
:::

```{symbols}
S_steps:
  tex: S
  text: Number of optimizer updates (training steps); 2000 in this section.
B_batch:
  tex: B
  text: Number of training samples drawn per update (batch size); 16 in this section.
N_train:
  tex: N
  text: Total number of samples (frames) in the training set; 13,431 in this section.
E_pass:
  tex: E
  text: 'Equivalent passes over the data: how many times the total number of draws would cover the training set.'
```

{#before}
## Before you start

This section has two hands-on steps: first try the training, then validate a model that has already been trained. Both commands run inside the course container.

- If you have not started the container yet, build and start it as described in [Section 1.3](setup.md), then open a terminal inside it.
- If the container is already running, return to the terminal you used to enter it; or open a new terminal on the host and run `docker exec -it microduck-workshop bash` to get back into the same container.
- The training data, feature cache and course checkpoint come from the GR00T course data package (`course-data/groot-walking` in the container); get it from your instructor before this section and place it as described in Section 1.3. The package is mounted read-only when the container starts. Your training and validation results go to new folders under `outputs/` and never modify the provided data or models.

- The GR00T base model and the Cosmos-Reason2-2B processor are downloaded from Hugging Face. Request access on the [nvidia/Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) page first, then log in once inside the container with `/workspace/groot-env/bin/hf auth login`; the download cache is kept in `~/.cache/huggingface` on the host.

GR00T training and simulation with a window both use a lot of GPU memory; run only one at a time.

```{learning-flow}
label: The two hands-on steps of this section
nodes:
- [Course data package, 'Demonstrations, start point and feature cache']
- [Action-head training, '2000 optimizer updates (stop any time)']
- [Closed-loop validation, GR00T server + Isaac Lab simulation]
details:
- The training and validation sets, the start-point checkpoint and the frozen-backbone feature cache come from the GR00T course data package, mounted read-only; training never changes them.
- After checking the cache, the training command updates only the action head and writes to the new folder you name. You can stop it with Ctrl+C after a few minutes.
- The validation command starts the GR00T inference server, then runs the walking shot in Isaac Lab with Newton; it uses the course checkpoint by default, and you switch to your own only after a full training.
caption: Training and validation are separate commands, so you can validate the course checkpoint without finishing a training run.
```

{#data}
## The prepared data

The data follows the principles from Section 4.2: the onboard policy completed walking shots from different starts on the green pitch, only fully successful episodes were kept, and the training and validation sets were split by whole episode and start with no overlap.

{.table-center}
| Dataset | Episodes | Samples (frames) | Rate | Size |
|---|---|---|---|---|
| Training | 58 | 13,431 | 12.5 Hz | about 3.9 GB |
| Validation | 14 | 3,098 | 12.5 Hz | about 0.9 GB |

Each sample contains four 270 × 480 RGB images, the 127-D state, the fixed task instruction, and the 8-D request labels for the next 16 steps from that moment. An episode averages about 230 frames, roughly 18 seconds of simulation.

{#recipe}
## The training recipe

Training starts from the course’s initialization checkpoint: it has already been post-trained for the walking-shot task, and its input and output normalization statistics are computed from this training set only. This run continues training the action head from there with a newly created optimizer state; it does not start from the official base weights.

| Setting | Value | Notes |
|---|---|---|
| Trained part | Action head | The vision-language backbone is frozen; the state and action encoders, action decoder, action-sequence generator and visual-language feature layers are trained, for the reasons in <a href="08-groot-interface.html#action-head">Section 4.2</a> |
| Training steps | 2000 | 2000 optimizer updates |
| Batch size | 16 | 16 samples drawn per update |
| Learning rate | 5 × 10⁻⁵ | Rises linearly from a very small value over the first 100 steps, then decays gradually to 0 |
| Optimizer | AdamW, fresh state | The initialization checkpoint’s optimizer state is not reused |
| Image augmentation and dropout | Off | First checks whether the model can fit this fixed scene; cross-scene generalization is not addressed |

{#budget}
## Roughly how much data do 2000 steps see?

Each update draws $\sym{B_batch}$ samples, so $\sym{S_steps}$ updates draw $\sym{S_steps}\sym{B_batch}$ in total. Comparing that with the training-set size $\sym{N_train}$ gives a simple estimate:

```{formula} Equivalent passes over the data
\sym{E_pass} = \frac{\sym{S_steps}\,\sym{B_batch}}{\sym{N_train}} = \frac{2000 \times 16}{13{,}431} \approx 2.38
```

So the total number of draws is about 2.38 times the training set. That does not mean every sample is seen exactly 2.38 times: training uses the stage-weighted sampling described in <a href="08-groot-interface.html#data-design">Section 4.2</a>, so rare moments such as standing still, fine adjustment at the ball and the run-up to a kick are drawn more often and the many walking frames less often; sampling is also random, so a few samples may never be drawn.

{#time}
## How long does it take?

{.table-labels}
| Stage | Time on DGX Spark | What happens |
|---|---|---|
| Checks | A few minutes | Checks the data package files and spot-checks the feature cache on 4 samples |
| Optimizer updates | About 56 minutes | 2000 updates, about 1.6 s each |
| Saving the checkpoint | Extra | The model weights are about 12.6 GB |

On DGX Spark a full training takes about 56 minutes of updates plus the checks at the start and saving the checkpoint at the end. On a 32 GB RTX 5090, a rough estimate from the update interval measured in a short run is 100–170 minutes for 2000 updates; the course has not run a full training on that GPU. You do not have to wait for it to finish: the validation below uses the course checkpoint by default.

**Hardware.** The training command reads the GPU memory and chooses the way of computing automatically; the recipe stays the same:

- **About 56 GB of GPU memory or more (for example DGX Spark):** trains with the full batch; the update phase peaks at about 47 GB.
- **About 30–56 GB (for example a 32 GB RTX 5090):** switches to the low-memory way of computing. The order of computation differs, so results are not bit-identical to the course checkpoint.
- **Less than about 30 GB:** the command stops and explains why.
- **Memory and disk:** the training process peaks at about 8 GB of memory; reserve about 40 GB for the output folder, plus about 32 GB if you recompute the feature cache on your own GPU.

:::{enrichment} Resources and the feature cache
- **Feature cache.** The features computed by the frozen backbone come with the data package (about 32 GB). At the start of training the command checks the SHA-256 of the cache, the starting checkpoint and the training set, then recomputes the features of 4 samples on the GPU and compares them with the cache; only if everything matches does it skip feature computation, otherwise it stops and writes a field-by-field difference record next to the output folder.
- **Other GPU architectures.** This cache was computed on DGX Spark; on another GPU architecture, floating-point differences can make the comparison fail. A cache prepared for the RTX 5090 (compute capability 12.0) is available from your instructor: put its `groot-walking-feature-caches/` folder next to `groot-walking/`, and the command picks the matching one by compute capability.
- **Recomputing features.** Add `--recompute-features` to recompute the cache on your own GPU: about 88 minutes on DGX Spark and about 54 minutes on an RTX 5090.
- **Way of computing.** `--profile full` or `--profile low-memory` chooses it explicitly. The low-memory way splits each batch of 16 samples into 16 computations and accumulates gradients before one parameter update, updates the optimizer state parameter by parameter, and reads the feature cache from disk as needed.
- **WSL 2.** Files are written into a virtual disk file on the Windows drive, which grows with writes and does not shrink by itself, so make sure that Windows drive has enough free space; saving and reloading checkpoints also uses a few GB of swap.
:::

{#lab-train}
## Try post-training

Run the command below in the container terminal. It uses the recipe above and the prepared training set, and writes results to a new folder.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/groot_walking.py train --output outputs/groot-walking-train
```

::::{enrichment} Key code parameters
- **Data package.** The host's `course-data/groot-walking/` (or the folder `$MICRODUCK_COURSE_DATA` points to), `/workspace/Microduck-Workshop/course-data/groot-walking/` in the container (read-only).
- **Files `train` reads.** Always the package's `initialization/` (the starting checkpoint), `datasets/train-history127/` and `datasets/validation-history127/`, `sampling.json` and the feature cache in `backbone-cache/`; you do not write these paths. If the package also provides a feature cache for your GPU architecture, the command picks it automatically.
- **Base model.** The GR00T base model `nvidia/GR00T-N1.7-3B` (pinned revision) recorded in the checkpoint configuration and the `nvidia/Cosmos-Reason2-2B` processor are read from the Hugging Face cache: `~/.cache/huggingface` on the host, `/root/.cache/huggingface` in the container.
- **Optional arguments.** `--recompute-features` recomputes the feature cache on your GPU; `--profile full` or `--profile low-memory` chooses the way of computing.
- **Output.** Training results go to the new folder given by `--output`.
::::

The log first reports the cache check, then reports loss and learning rate per step; those are optimizer updates. In the course record the loss was about 0.095 at step 1 and about 0.016 at step 2000; single-step losses fluctuate, so look at the trend. Falling training loss only shows the demonstrations are fitted, not that the model can score.

After watching the first few dozen steps, press **Ctrl+C** to stop. Ctrl+C ends only this training command; the container and the terminal stay open, so you can go straight to the next step. Stopping early does not produce a usable checkpoint; the validation below uses the course checkpoint by default and is unaffected. If you later have time for a full run, start again with a new output folder and validate after all 2000 steps have finished and been saved.

{#lab-validate}
## Validate the provided checkpoint

The command below loads, by default, the trained 8-D walking-shot checkpoint from the course data package: the model produced by a full 2000-step run of the recipe above, not the run you just interrupted. It starts the GR00T inference server, opens an Isaac Sim window, and runs the full shot in Newton physics from five fixed starts in turn.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/groot_walking.py validate --output outputs/groot-walking-validate
```

::::{enrichment} Key code parameters
- **Checkpoint.** Without `--checkpoint`, the GR00T server loads the package's `course-data/groot-walking/checkpoint/` (`/workspace/Microduck-Workshop/course-data/groot-walking/checkpoint/` in the container), with the base model again from the Hugging Face cache; with `--checkpoint` it loads only the checkpoint you name.
- **Validation scene.** Started by `lab/groot_walking/student/walking_student_scene.py`, which receives the GR00T server address through its `--walking-policy-server` argument.
- **Low-level walking policy.** Loaded through the scene's `--model` argument from `/workspace/state/student-course/simulation/core/models/reference/velstand.onnx`, copied from the repository's `models/walk.onnx` when the container starts (byte-identical).
- **Robot model.** The `--robot-usd` default `/workspace/state/student-course/simulation/core/visuals/assets/microduck_white_orange.usda`, from the repository's `lab/runtime/simulation/core/visuals/assets/`.
- **Output.** `groot_walking.py validate` fills in all of these scene arguments; each episode's results go to the new folder given by `--output`.
::::

:::{course-image} media/isaac-ui-groot-validate.png
:alt: The complete Isaac Sim window during GR00T walking-shot validation: Microduck at one end of the green pitch, the ball in the centre circle and the goal at the other end, RTX real-time rendering in the toolbar.

The window when the course ran the command above in the container, captured shortly after GR00T took over in the first episode. It shows what the validation scene looks like and does not mean this episode scored; each episode's result is in the records in the output folder.
:::

The simulation pauses while waiting for each GR00T request, so brief pauses in the window are expected (see <a href="08-groot-interface.html#timing">Section 4.2</a> for the full timing); these pauses do not establish real-robot timing. Together with the head camera's 15 frames per second, each rendered with RTX and saved, validation on DGX Spark runs at about 0.24 times real speed.

After the run, first read the `Complete walking shots` summary and the one-line result per episode printed in the terminal. Per-episode records are in `attempts.json` in the output folder (for example `outputs/groot-walking-validate/attempts.json`): in each record’s `score`, `passed` means full success, `contact` records ball contact, `goal` a goal, `fall` a fall, and `terminal` the reason the episode ended. Distinguish the outcomes:

| Outcome | Meaning |
|---|---|
| Full success | The whole ball enters the goal, no fall, standing quietly at the end |
| Contact, no goal | The ball was kicked but did not enter the goal |
| No contact | The ball was not touched within the 60-second simulation limit |
| Fall | A fall at any time is never a success |

**Validation results of the course checkpoint.** The checkpoint's record from these five starts: 0 goals, 2 contacts without a goal, 3 episodes without touching the ball within 60 s, and no falls. Episodes 4 and 5 ended normally when the kick observation window closed (task times of about 16 s and 27 s). This record rendered each head-camera frame 4 times; the default command renders once, with slightly different image details, so results may differ. To run under the same condition, put `MICRODUCK_GROOT_RENDER_SUBFRAMES=4` before the command.

The pipeline runs and the robot walks up to the ball and kicks, but none of these five trials completed the shot. Five episodes are too few, and these starts were used during training and tuning, so they cannot establish generalization. Analyse the failures first, then evaluate from more starts excluded from training and tuning, keeping every failure.

If you have fully trained your own model, add `--checkpoint outputs/groot-walking-train/checkpoints/002000/pretrained_model` to the command above, with a new `--output` folder, to compare the two models from the same starts.

::::{quiz}
Microduck keeps pausing during validation. Does that mean GR00T is too slow to be useful?

:::{answer}
It only shows that evaluation is synchronous: the simulation pauses while waiting for inference. It illustrates that inference takes time, but real-robot behavior has to be measured with asynchronous inference, a request buffer and the actual perception and network delays; neither the pauses nor the smoothness in simulation tell you directly how the real robot will behave.
:::
::::

{#references}
## References

{.references}
- [GR00T N1.7 model card ↗](https://huggingface.co/nvidia/GR00T-N1.7-3B)
- [Isaac GR00T: post-training and inference ↗](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot: GR00T policy implementation ↗](https://github.com/huggingface/lerobot)
