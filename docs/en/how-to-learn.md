---
myst:
  html_meta:
    description: Choose observation, fixed-model labs, or training practice. All three follow the same learning sequence.
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 1.2 How to take this course

:::{lesson-intro}
:meta: Estimated learning time: 5 minutes

Choose observation, fixed-model labs, or training practice. All three follow the same learning sequence.
:::

{#paths}
## Choose a learning path

| Path | What you do | What you need |
|---|---|---|
| Observe | Read, inspect the 3D assembly, watch demonstrations, and answer checks. | A modern browser |
| Use fixed models (recommended) | Copy commands, run the simulation in the course container terminal, and record what you observe. | A prepared GPU environment and the course container |
| Practice training | Add your own training, export, and evaluation. | Additional runtime and disk space |

The course has five parts: Part 1 is the overview and setup; Part 2 introduces Microduck; Part 3 trains skills with reinforcement learning; Part 4 adapts GR00T with demonstrations and evaluates the walking shot in simulation; Part 5 collects models, troubleshooting and references for use at any time. The <a href="index.html#route">course route</a> in Section 1.1 lists every section.

Every hands-on lab runs in the course container prepared in [Section 1.3](setup.md). For long trainings you can watch for a few minutes and continue with the course's pretrained results; all reading and browser interactions work without a GPU. Hardware and system requirements are in Section 1.3, and the GPU memory needed for GR00T training is in Section 4.3.

{#prerequisites}
## Prerequisites

- Open a Linux terminal, change directories, and copy multiline commands.
- Read basic Python functions, arrays, and parameters. Prior PPO knowledge is not required.
- Understand positions, angles, and velocities; robot coordinate frames are introduced in the course.

:::{admonition} About your training results
:class: note

Your trained checkpoint is a practice result. Later commands use the course model specified in each section and never switch automatically to your latest run. These models may be in `models/`, `datasets/` or the GR00T course data package. To use your own policy, select its file or checkpoint explicitly and evaluate it again.
:::

{#routine}
## A useful lab routine

- Read the expected result before running a command.
- Run commands in the course container terminal (it opens at the repository root). Use a new output directory for every run.
- Keep the command, seed and result, including failures.
- Answer the section check before moving on.

Section durations help you plan; actual training time depends on your machine. The first launch may download assets, compile kernels, and populate rendering caches.
