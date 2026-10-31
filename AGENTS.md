# Microduck course

The course is **Train Microduck in Simulation with NVIDIA Isaac** / **使用 NVIDIA Isaac 在仿真环境训练 Microduck**. The repository name stays Microduck-Workshop.

Read `.agents/skills/microduck-course/SKILL.md` before changing lessons. MyST Markdown in `docs/zh/` and `docs/en/` is authoritative; `content/course.json` is a read-only archive. Never regenerate lessons from that archive. Both languages share page names, section anchors, commands and model references.

## Learner-facing content

Write for students: core concepts, motivation, steps, observations and interpretation. Keep paragraphs concise. Show only the current version; development history, old experiments, production notes and internal machine paths belong outside the public lessons. Preserve historical evidence separately rather than overwriting it. Avoid irrelevant seeds and recording settings in prose. Do not use U+2014; use normal punctuation or an ASCII hyphen.

The course has six parts and seventeen pages:

1. Overview: `index`, `setup` (the removed `how-to-learn` URL redirects to `setup`).
2. Meet Microduck: `01-play`, `02-anatomy`.
3. Reinforcement learning: `03-learning`, `03-walk-training`, `04-terrain`, `05-skills`.
4. Onboard policy (板载策略完成射门任务): `06-perception` (4.1 RGB detection and ToF ranging), `06-onboard` (4.2 hand-designed stages).
5. GR00T: `07-groot`, `08-groot-interface`, `08-groot-lab`, `09-worlds`.
6. Help: `models`, `troubleshooting`, `references`.

Use ordinary descriptive headings, without “动手” or “Hands-on” prefixes. Chinese lessons say 板载策略 for the onboard policy. Part 4 covers walking onboard shooting only (no roller shooting); it explains the actual detector, ToF floor/near-range logic and stages concisely, without heavy equations. The old `06-autonomy` URL redirects to `06-onboard`; the ASPIRE page remains a redirect only. The community section currently has three examples: one-leg standing, basketball balance and backward rolling.

## Labs and evidence

All student commands run in the course container, or explicitly start/attach to it from the Ubuntu host. Use `scripts/run.py`; `scripts/workshop.py` is a compatibility wrapper. Collapse detailed paths/loading parameters under “代码关键参数 / Key code parameters”. Every command opening Isaac Sim has a representative screenshot of its complete UI. Explain the model, scene or dataset path and which parameter loads it.

Keep the fixed models unchanged. Training writes a new output directory and never replaces course defaults. Students can observe training and stop it; validation uses provided checkpoints unless they explicitly select their completed export. Keep Newton as the physics backend. CPU architecture support covers aarch64 and x86_64; distinguish DGX Spark's complete desktop workflow from the RTX 5090 WSL short headless training checks. Do not imply WSL RTX windows/cameras were validated.

Use one green football pitch. The course GR00T task uses walking and an 8-D high-level interface; direct 14-joint output and Cosmos environment variation are theoretical extensions. Distinguish onboard control, low-level skills and workstation GR00T inference. Do not claim an onboard hardware test from a simulation result.

Slope training uses randomized terrain; evaluation uses a separate small wooden wedge on a plain white floor. Reward is a training signal, success is an evaluation criterion. The current walking tests pass 5/5 each at +5 and -5 degrees; the roller example passes 0/5 despite increased return. These are limited conditions, not a general success rate or proof of one reward term's contribution. Use the current evaluation summary, not older files with different geometry. Do not turn failure examples into claims that a task cannot be learned.

Use recorded logs and actual source code for formulas, timing and metrics. Never fabricate results, screenshots or trajectories. TensorBoard figures mark the relevant metrics and explain them. Simulation videos retain their actual source; offline trajectory replay is not a fresh physics result. Current video quality matches interactive rendering, with more frames when needed. Never start a new training or GPU job merely to refresh the course site.

All new policy simulations use the established grounded HOME initialization: temporarily lock joints with the root free, wait for sensors and the first actual policy output, then restore the original parameters and release constraints before applying it. Revalidate ground contact for new terrain. Holding during initialization is not learned balance. Coordinate GPU use with other work before running simulations; Blender work belongs on the designated RTX 5090 host, not DGX Spark.

## Presentation

The normal site uses the native NVIDIA/PyData Sphinx shell, grouped left navigation, right page contents and theme controls. LMS exports instead use a fixed white canvas without site chrome. Keep captions left-aligned, tables readable with adaptive widths and short cells unwrapped; use thousands separators for large numbers. Links use the NVIDIA green family in both themes.

Code explanations and code chunks share one grid row: explanation left, code right (1:1.25 on desktop), equal height, vertically centered content. Use inline explanation titles when helpful. Soft-wrap code, with no clipping or internal scrolling, and preserve copied source bytes. Source excerpts and recorded runs have no copy button; executable commands do. Recorded output is collapsed by default.

Diagrams appear above their explanations/controls. Flow explanations initially take no space and have no placeholder prompt; hovering or focusing a module reveals its explanation, clicking pins it, and Close or Escape collapses the panel. Keep explanations readable without JavaScript. Four flow modules fit one desktop row; three modules spread evenly across the same width. Match fonts and card styling. Mark mathematical variables with the course symbol directives and preserve accessible tooltips and collapsed definitions.

Use the transparent NPR robot explorer with actual CAD and joint mappings. Before loading it, show no stale image or visible placeholder; preserve accessible status and failure retry. The homepage uses the current transparent antialiased animated WebP of the football replay: autoplay and loop, no pause control, caption or divider. Do not restore superseded decorative animation variants.

The control-cycle figure uses recorded event arrays, three consecutive rounds, onboard mode first, playback from zero and collapsed “详解 / Details”. Keep measured host computation separate from simulation time; do not invent asynchronous execution or infer timing from configuration alone.

## Build and publication

Build with `python3 scripts/build_site.py`; Sphinx warnings are errors. Check with `python3 scripts/check_course.py` (or `--site-only` for a content-only pass). Inspect desktop browsers at 100% zoom, including links, copy controls, formulas, media, language switching and 3D interactions. No mobile test matrix is requested.

Create LMS packages with `scripts/export_lms.py` and `docs/lms/microduck.json`. Export Chinese-default bilingual and English-only editions; include only reachable current assets, deduplicate shared resources and compare package size with the reference archive. Test normal static paths and Open edX opaque asset paths, not only a local folder. Keep reference archives untouched.

The user authorized publishing the current course and README. Do not infer authorization for unrelated changes. Verify the deployed revision after pushing. Keep large GR00T data/checkpoints in separately provisioned `course-data/`, not Git or the LMS archive. Retain third-party asset attribution and licensing in `THIRD_PARTY_NOTICES.md`.
