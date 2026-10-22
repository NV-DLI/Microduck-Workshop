# Microduck Workshop

A self-paced NVIDIA Isaac robotics workshop: use Isaac Sim for scenes and sensors, Isaac Lab for GPU-parallel learning, and Newton for robot dynamics and contacts. Connect perception, navigation and learned skills to complete an autonomous football task.

**[中文课程](https://nv-dli.github.io/Microduck-Workshop/zh/index.html) · [English course](https://nv-dli.github.io/Microduck-Workshop/en/index.html)**

## Start learning

```bash
git clone https://github.com/NV-DLI/Microduck-Workshop.git
cd Microduck-Workshop
python3 scripts/workshop.py verify-models
python3 -m http.server 8080 --directory web --bind 127.0.0.1
```

Open http://127.0.0.1:8080/zh/index.html or `/en/index.html`. Reading, demonstrations, search, and the interactive mechanical assembly work without installing Isaac Sim. Serve the folder over HTTP; browser module loading may not work through `file://`.

The seven chapters cover anatomy, simulation, reinforcement learning, skill composition, perception, navigation, and autonomous football. Each lesson explains concepts, provides terminal commands and parameters, describes expected outputs, and includes self-checks. The expanded Chinese edition adds locally rendered equations, interactive theory diagrams, real training curves, seekable demonstration videos, and clickable part labels. Illustrative code excerpts link directly to the full source; terminal commands include copy controls.

## Run the labs

Reference platform: Ubuntu 24.04 on DGX Spark (aarch64), a working NVIDIA graphical desktop, Docker, Python 3.12, Isaac Sim 6.2, and the pinned Isaac Lab / Newton environment. Read the course setup page before installation. Runtime installation downloads substantial third-party dependencies and requires evaluation of their licenses.

```bash
python3 scripts/workshop.py setup
python3 scripts/workshop.py doctor
python3 scripts/workshop.py football --seed 983102 --seconds 180 --output outputs/football-01
python3 scripts/workshop.py report --run outputs/football-01
```

Run graphical commands from the GPU machine's desktop terminal. The website is a textbook; simulation opens in a native window. New runs require new output directories. Use `--help` on any subcommand for parameters.

The six fixed model files total about 22 MB and are included directly in Git. Their identities are checked against `models/manifest.json`. Student training is optional practice in `outputs/`; it never replaces downstream course defaults. An explicit `walk --model ...` command can evaluate a student's own ONNX export.

The retained full-task combination scored **9/10** in its recorded independent empty-field trials. This is a result for a specific environment and trial distribution, not a guarantee for every seed or device. The default seed is a teaching example. Obstacle courses and physical hardware have not been qualified by that result.

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

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Software, community policies, robot CAD, and the Ultralytics detector have distinct terms. In particular, the robot geometry is under Pollen Robotics' noncommercial share-alike terms; a source-code license does not remove those restrictions.

中文学习主线：先观看和理解，再加载课程固定模型运行；训练只是独立练习。不要覆盖 `models/`，不要把某一次成功演示当作独立成功率。HTML、实验代码和固定模型都在本仓库；课程不依赖网页内嵌 Isaac 窗口。
