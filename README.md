# Microduck Workshop

A self-paced NVIDIA Isaac robotics workshop: use Isaac Sim for scenes and sensors, Isaac Lab for GPU-parallel learning, and Newton for robot dynamics and contacts. Connect perception, navigation and learned skills to complete an autonomous football task.

**[中文课程](https://nv-dli.github.io/Microduck-Workshop/zh/index.html) · [English course](https://nv-dli.github.io/Microduck-Workshop/en/index.html)**

## Start learning

```bash
git clone https://github.com/NV-DLI/Microduck-Workshop.git
cd Microduck-Workshop
python3 scripts/workshop.py verify-models
python3 scripts/serve_site.py --port 8080
```

Open http://127.0.0.1:8080/zh/index.html or `/en/index.html`. Reading, demonstrations, search, and the interactive mechanical assembly work without installing Isaac Sim. Serve the folder over HTTP; browser module loading may not work through `file://`.

The nine chapters connect keyboard football, robot structure, walking training, terrain adaptation and community skills with a concise traditional-task overview, then GR00T, ASPIRE and Cosmos/Isaac methodology. Chapters 1–5 retain practical labs; Chapter 6 is a watch-only overview and Chapters 7–9 explain integration requirements and future experiments. Frontier model training and closed-loop control have not run in this course. The Chinese and English pages include interactive diagrams, real recorded examples, source links and clear limits on the reported results.

## Run the labs

Reference platform: Ubuntu 24.04 on DGX Spark (aarch64), a working NVIDIA graphical desktop, Docker, Python 3.12, Isaac Sim 6, and the pinned Isaac Lab / Newton environment. Read the course setup page before installation. Runtime installation downloads substantial third-party dependencies and requires evaluation of their licenses.

```bash
python3 scripts/workshop.py setup
python3 scripts/workshop.py doctor
python3 scripts/workshop.py manual --seconds 600 --output outputs/manual-football-01
python3 scripts/workshop.py football --scene indoor --style A --record-hd --seed 983102 --seconds 180 --output outputs/football-01
python3 scripts/workshop.py report --run outputs/football-01
```

Run graphical commands from the GPU machine's desktop terminal. The website is a textbook; simulation opens in a native window. New runs require new output directories. Use `--help` on any subcommand for parameters.

The original seven fixed model files total about 23 MB and are included directly in Git. Their identities are checked against `models/manifest.json`. The added indoor detector (PT + ONNX, about 14.5 MB) has its own `models/indoor-detector/deployment.json` and is also checked. Student training is optional practice in `outputs/`; it never replaces downstream course defaults. An explicit `walk --model ...` command can evaluate a student's own ONNX export.

The separate `terrain-walk.onnx` actor adapts walking to flat and sloped ground; see `datasets/terrain/contact-evaluation.json` for the current full-contact comparison and limits (`evaluation.json` preserves the historical configuration). It does not replace the original walking actor in football.

The original simple-field configuration scored **9/10** in its recorded independent empty-field trials. That score does not apply to the new indoor scene. The current A-pattern demonstration completed contact, whole-ball crossing and stable recovery in 17.485 task seconds. This is a result for a specific environment and trial distribution, not a guarantee for every seed or device. Four predeclared new-seed trials, one per mat pattern at the standard start, completed **1/4** full tasks; all four stayed upright. The failures expose observation and planning-handoff limits. Pattern and seed changed together, so this is not a causal comparison of mat patterns or a broad reliability estimate; see [trial results](datasets/indoor-sample/task-evaluation.json). The default seed is a teaching example, with robot start (−0.55, 0.04) m and ball (0.02, 0) m. `--scene legacy` retains the original scene. Four mat patterns are selected with `--style A|B|C|D`; changes require new evaluation. Obstacle courses and physical hardware have not been qualified by that result.

The current interactive labs share official full-body collision geometry, white/orange materials and RTX daylight. A 1 × 1.6 m mat and secured 40 × 20 × 25 cm steel-mesh goal follow the supplied product dimensions. Friction, compliance and frame/wire sizes remain unmeasured assumptions; see [scene provenance](lab/assets/README.md). Physics-only practice training uses the same full-contact robot; historical checkpoints and training records remain unchanged.

## Repository

| Directory | Contents |
|---|---|
| `content/` | Bilingual base lessons and Chinese teaching-depth sections |
| `web/` | Static HTML, local JavaScript, images, seekable videos and 3D assets |
| `scripts/` | Course builder and terminal lab entry points |
| `lab/` | Lab entry points and environment locks |
| `lab/runtime/simulation/core/` | Simulation, training, sensors, navigation and task evaluation |
| `lab/runtime/configuration/core/` | Model deployment and calibrated task parameters |
| `models/` | Fixed policies and detector, with SHA-256 manifest |
| `datasets/sample/` | 40 labeled simulation images for short exercises |
| `artwork/` | Editable Blender assembly, source geometry and attribution |

Rebuild and check the site with `python3 scripts/build_site.py` and `python3 scripts/check_course.py`. The repository-local authoring skill is `.agents/skills/microduck-course/SKILL.md`.

## 来源与许可 / Attribution and licenses

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Software, community policies, robot 3D models, and the Ultralytics detector have distinct terms. In particular, the robot geometry is under Pollen Robotics' noncommercial share-alike terms; a source-code license does not remove those restrictions.

中文学习主线：先观看和理解，再加载课程固定模型运行；训练只是独立练习。不要覆盖 `models/`，不要把某一次成功演示当作独立成功率。HTML、实验代码和固定模型都在本仓库；课程不依赖网页内嵌 Isaac 窗口。
