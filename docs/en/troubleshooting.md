---
myst:
  html_meta:
    description: Check environment, files, and arguments first. Preserve failed outputs for comparison.
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 5.2 Troubleshooting

:::{lesson-intro}
:meta: Quick reference

Check environment, files, and arguments first. Preserve failed outputs for comparison.
:::

{#doctor}
## Start with the environment report

Run these two read-only checks **inside the container**: `doctor` reports the runtime environment (GPU, key software versions, graphical display), and `verify-models` checks fixed-model sizes and hashes.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py doctor
python3 scripts/run.py verify-models
```

When something fails, note the first error; it explains more than repeatedly launching a full training run. Then confirm two things: the command ran in the course container shell (prompt `(microduck)`, current folder `/workspace/Microduck-Workshop`), and all model checks pass.

{#common}
## Common issues

| Symptom | What to check |
|---|---|
| No GUI window | Open the terminal in the GPU machine's desktop session and check DISPLAY and the graphics driver; a text-only remote shell cannot show windows. |
| The first window is slow, or the desktop says “Isaac Lab” is not responding | The first window compiles RTX shaders and loads assets, so the program cannot answer the desktop for a while (about 1 minute on DGX Spark). Choose **Wait**, not Force Quit, and do not start another instance while the log is progressing. The host cache is `~/.cache/microduck/kit-cache`, so later windows usually open in a few seconds; deleting the cache, updating drivers or software, or opening a scene with new content can trigger compilation again. |
| The terminal prints `Actuator groups cover 14 joints; expected 28` (32 with roller feet) | Expected; nothing to fix. Microduck has 14 servo-driven joints. The model also has 14 passive hinges that represent gear backlash, each free within about ±1°, and the roller feet add 4 freely spinning wheels. None of these has a motor, and Isaac Lab's check counts them as joints without an actuator. |
| The status bar shows `RenderDelegate : Warning /rtx/hydra/readTransformsFromFabricInRenderDelegate and geometry streaming are enabled together …` | This warns about rendered pose updates, not joint configuration. Continue if objects display and move with the simulation. If objects are missing, frozen or clearly misplaced, stop, note the Isaac Sim / Isaac Lab versions and keep the full log. GR00T's head camera also reads the rendered images. |
| Video decoder warning | If a TorchCodec loading error is followed by Falling back to 'pyav' and reading or training continues to progress, an alternative video decoder is active; keep waiting. If the process exits or videos still cannot be read, retain the full error and check the Section 4.3 GR00T learning environment. |
| Output directory exists | Choose a new directory name; the previous result stays intact. |
| Out of memory | Reduce the number of parallel training environments and run one simulation or training at a time; GR00T training can use `--profile low-memory` (Section 4.3). |
| Model hash mismatch | Restore the repository model and check for accidental overwrites. |
| Practice policy walks poorly | A short run teaches the workflow; the policy is usually not yet well trained. Compare reward, episode length and held-out evaluation; later chapters continue with the fixed models. |
| Robot stops or does not score | Inspect the task report and observation validity. When observations are insufficient, stopping can be a sensible safeguard, but only a goal recorded in the task result counts as a successful shot. |

{#report}
## Inspect a run

Replace `outputs/ppo-walk-test-01` (the Section 3.2 test) with the output directory of a run you have actually performed. The command reads the result files in that directory and prints key fields such as status, seed and task result; if the run has not finished writing results yet, it asks you to check the run log first.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py report --run outputs/ppo-walk-test-01
```

When reporting an issue, include the command, seed, environment report, and first error; a success/failure label alone cannot locate the problem. Do not include passwords or access tokens.
