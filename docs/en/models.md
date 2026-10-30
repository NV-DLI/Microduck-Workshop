---
myst:
  html_meta:
    description: Every main-path lab uses explicitly named fixed files. Download once and reuse them across chapters.
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 5.1 Models and data

:::{lesson-intro}
:meta: Quick reference · No retraining required

Every main-path lab uses explicitly named fixed files. Download once and reuse them across chapters.
:::

{#inventory}
## Course model inventory

| File (host `models/`, container `/workspace/Microduck-Workshop/models/`) | Purpose | Course commands that read it |
|---|---|---|
| `walk.onnx` | Fixed walking policy | Section 2.1 keyboard football (walking mode, read automatically); Section 2.2 `walk --model models/walk.onnx`; the low-level walking policy in the Section 4.3 validation scene uses a copy of the same model |
| `terrain-walk.onnx` | Section 3.3 fixed terrain walking actor, about 0.79 MB | Section 3.3 `terrain --model models/terrain-walk.onnx` |
| `kick-teacher.onnx` | Base kicking action | Section 2.1 keyboard football (Space, read automatically) |
| `kick-residual.onnx` | Residual kick adjustment | Retained; not read directly by this course's hands-on commands |
| `kick-checkpoint.pt` | Retained kick training checkpoint | Retained; not read directly by this course's hands-on commands |
| `detector.pt` | Fixed visual detector | Retained; not read directly by this course's hands-on commands |
| `detector.onnx` | ONNX detector export | Retained; not read directly by this course's hands-on commands |
| `indoor-detector/detector.pt` / `indoor-detector/detector.onnx` | Another indoor detector, for a scene with a 7 cm ball; not used on the course pitch | Retained; not read directly by this course's hands-on commands |
| `scene-detector/detector.pt` / `scene-detector/detector.onnx` | Official 10 cm ball and four-surface detector; current default | The onboard policy's visual detector; not read directly by this course's hands-on commands |
| `official-browser/rollers.onnx` | Official four-passive-wheel actor, 14 servo actions | Section 2.1 roller mode after M (read automatically); `--model` of Section 3.3 `terrain --locomotion rollers` and `train-terrain --locomotion rollers` |
| `official-browser/walking.onnx` / `sitstand.onnx` | Browser-version checks and standing tests; the complete walking task uses `walk.onnx` | Retained; not read directly by this course's hands-on commands |

Run the command below inside the container to verify fixed-model sizes and hashes.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py verify-models
```

These files ship with the Git repository, so cloning it is enough; no Git LFS or separate model account is needed. `models/manifest.json` records sizes, sources, and SHA-256 identities, and each lab checks model identity before running to guard against same-name replacements. The relative paths and SHA-256 of the current default detector are recorded in `models/scene-detector/deployment.json`.

**In the repository.** The weights for Parts 2–3 and the onboard policy; the pretrained PPO walking result of Section 3.2 and the slope-training logs are in `datasets/walking-ppo` and `datasets/terrain-ppo` (`/workspace/Microduck-Workshop/datasets/` in the container, read-only). `datasets/terrain-rollers-ppo-experiment` is the record of a roller slope training run that did not learn to climb; it is kept as a counterexample for reward design, not as a usable slope model.

**Separate data package.** The GR00T walking-shot data, feature cache and trained checkpoint of Section 4.3 come from the separate GR00T course data package (`course-data/groot-walking` in the container; check it with `python3 scripts/groot_walking.py verify`). The GR00T base model and the Cosmos-Reason2-2B processor are downloaded from Hugging Face under their access and license terms.

{#practice-data}
## Practice data and your outputs

`datasets/scene-sample/` is a small set of sample images and labels that the course labs do not use directly; it is too small to test real-world generalization. Keep your own checkpoints and logs in `outputs/` and leave the course files in `models/` unchanged.

{#licenses}
## Sources and usage terms

Robot geometry, community policies, and dependencies each follow their own source and license terms; for third-party assets those terms apply, and the course-code license does not replace them. Before redistributing changes, read the repository’s [attribution and license notices](https://github.com/NV-DLI/Microduck-Workshop/blob/main/THIRD_PARTY_NOTICES.md).
