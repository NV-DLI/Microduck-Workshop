---
myst:
  html_meta:
    description: Every main-path lab uses explicitly named fixed files. Download once and reuse them across chapters.
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 6.1 Models and data

:::{lesson-intro}
:meta: Quick reference · No retraining required

Every main-path lab uses explicitly named fixed files. Download once and reuse them across chapters.
:::

{#inventory}
## Course model inventory

| File (host `models/`, container `/workspace/Microduck-Workshop/models/`) | Purpose | Course commands that read it |
|---|---|---|
| `walk.onnx` | Fixed walking policy | Section 2.1 keyboard football (read automatically); Section 2.2 `walk --model models/walk.onnx`; the low-level walking policy of the Section 4.2 onboard shot and of the Section 5.2 GR00T execution layer uses a copy of the same model |
| `terrain-walk.onnx` | Section 3.4 fixed terrain walking actor, about 0.79 MB | Section 3.4 `terrain --model models/terrain-walk.onnx` |
| `kick-teacher.onnx` | Base kicking action | Section 2.1 keyboard football (Space, read automatically); the kick skill of the Section 4.2 onboard shot uses a copy of the same model |
| `kick-residual.onnx` | Residual kick adjustment | Used by the Section 5.2 GR00T execution layer; not loaded by the Section 4.2 onboard shot |
| `kick-checkpoint.pt` | Retained kick training checkpoint | Retained; not read directly by this course's hands-on commands |
| `detector.pt` | Fixed visual detector | Retained; not read directly by this course's hands-on commands |
| `detector.onnx` | ONNX detector export | Retained; not read directly by this course's hands-on commands |
| `indoor-detector/detector.pt` / `indoor-detector/detector.onnx` | Another indoor detector, for a scene with a 7 cm ball; not used on the course pitch | Retained; not read directly by this course's hands-on commands |
| `scene-detector/detector.pt` / `scene-detector/detector.onnx` | Official 10 cm ball and four-surface detector; current default | Checked and loaded automatically by the Section 4.2 `football` command as the onboard policy's visual detector |
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
