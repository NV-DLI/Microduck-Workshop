# Simulate and train Microduck with NVIDIA Isaac

使用 NVIDIA Isaac 仿真并训练 Microduck.

A self-paced course: drive the open-source Microduck biped in Isaac Sim, train walking and slope skills with reinforcement learning in Isaac Lab and Newton, then post-train GR00T on walking-shot demonstrations and plan how to carry it to more environments.

**[中文课程](https://nv-dli.github.io/Microduck-Workshop/zh/index.html) · [English course](https://nv-dli.github.io/Microduck-Workshop/en/index.html)**

This repository holds the course site, the lab code, the container and the fixed course models. The rest of this README gets the container running; the course pages explain every lab.

## Course structure

1. **Overview**: how to take the course, set up the course container.
2. **Meet Microduck**: keyboard football, mechanism and the control loop.
3. **Teach Microduck skills with reinforcement learning**: PPO, domain randomization, reward design on a slope, community skills.
4. **Train GR00T to complete tasks autonomously**: vision-language-action models, the 8-D walking-shot interface and data, training and validation, environment variation and Cosmos 3.
5. **Reference and help**: models and data, troubleshooting, references.

Reading, figures, search and the interactive 3D duck work in a browser without Isaac Sim.

## Requirements

| Device | System | What you can use |
|---|---|---|
| DGX Spark (aarch64) | Ubuntu 24.04 desktop | The complete course, including Isaac Sim windows, cameras and recordings |
| RTX 5090 | Ubuntu in Windows 11 + WSL 2 | Commands without a window only (for example PPO and GR00T training); Isaac Sim windows, cameras and recordings did not work with the tested driver |

The RTX 5090 checks covered two PPO iterations and three GR00T updates, including checkpoint saving and reloading; a complete training run was not measured on that machine.

The host needs an NVIDIA driver, Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Git and a desktop session that can show windows. The image pins Isaac Sim 6.2, Isaac Lab at a fixed commit, the Newton physics backend, RSL-RL 5.5.1 and LeRobot + GR00T N1.7. Details are in Section 1.3 of the course.

## Quick start

All commands below run on the **host** from the repository root unless marked as inside the container.

1. **Get the repository.**

   ```bash
   cd ~
   git clone https://github.com/NV-DLI/Microduck-Workshop.git
   cd Microduck-Workshop
   ```

2. **Build the course container** (the first build downloads Isaac Sim and the learning environments; leave enough disk space).

   ```bash
   docker compose -f docker/compose.yaml build
   ```

3. **Start it and open a shell** from a terminal opened on the desktop, so `DISPLAY` and `XAUTHORITY` are set.

   ```bash
   ./docker/prepare_host.sh
   xhost +local:
   docker compose -f docker/compose.yaml up -d
   docker exec -it microduck-workshop bash
   ```

   The prompt changes to `(microduck)` in `/workspace/Microduck-Workshop`. Under WSL 2, start with `docker compose -f docker/compose.yaml -f docker/compose.wslg.yaml up -d` instead and use only commands without a window.

4. **Return to the same container** at any time with `docker exec -it microduck-workshop bash`. Ctrl+C stops only the running command. After a reboot, repeat the same compose start command, including both compose files on WSL 2. `docker compose -f docker/compose.yaml down` removes the container when you are done. Outputs and caches stay on the host.

5. **Check the setup** inside the container:

   ```bash
   python3 scripts/run.py doctor
   python3 scripts/run.py inspect-model --model models/walk.onnx
   ```

   On DGX Spark with a desktop session, start the keyboard football lab with `python3 scripts/run.py manual --seconds 600 --output outputs/manual-football-01`.

   `scripts/run.py` is the lab entry point (`python3 scripts/run.py --help` lists the commands, `python3 scripts/run.py <command> --help` their parameters); the GR00T lab in Section 4.3 uses `scripts/groot_walking.py`.

## Data and models

| What | Where | How you get it |
|---|---|---|
| Fixed course models | `models/` (under 60 MB), checked against `models/manifest.json` | In the repository |
| Pretrained PPO results and training logs for Part 3 | `datasets/` (about 50 MB) | In the repository |
| GR00T walking-shot data package (demonstrations, training start, trained checkpoint, feature cache) | `course-data/groot-walking/` (about 62 GB) | From your course instructor, needed only for Section 4.3 |
| Optional RTX 5090 feature cache | `course-data/groot-walking-feature-caches/` (about 32 GB) | From your course instructor; DGX Spark does not need it |

Put the GR00T package under `course-data/` in the repository root, or set `MICRODUCK_COURSE_DATA` to the absolute path of the folder that contains `groot-walking/` before `docker compose ... up -d`. The container starts without it, and `prepare_host.sh` reports whether its `manifest.json` is present. GR00T also downloads its base model and the `nvidia/Cosmos-Reason2-2B` processor from Hugging Face: request access on the [model page](https://huggingface.co/nvidia/Cosmos-Reason2-2B), then run `/workspace/groot-env/bin/hf auth login` once inside the container.

`models/`, `datasets/` and `course-data/` are mounted read-only; `lab/` and `scripts/` are mounted live, so host edits apply without a rebuild. Every run writes to a new folder under `outputs/`, and training never replaces the course models: evaluate your own export explicitly with `--model`.

## Repository

| Directory | Contents |
|---|---|
| `docs/zh/`, `docs/en/` | Lesson source: MyST Markdown, one file per page, same file names in both languages |
| `docs/_ext/`, `docs/_static/`, `docs/_templates/` | Sphinx directives for interactive figures, their styles, and small theme additions |
| `web/` | Built site (`web/zh`, `web/en`) plus shared assets in `web/assets`: images, seekable videos, 3D model, fonts and figure scripts |
| `content/` | Read-only archive of the pre-Sphinx JSON lessons; not used by the build |
| `docker/` | Course image, compose files and in-container helpers |
| `scripts/` | Lab entry points (`run.py`, `groot_walking.py`) and site build/check tools |
| `lab/` | Simulation, training and evaluation code, scene assets and environment locks |
| `models/` | Fixed course policies and detectors, with SHA-256 manifest |
| `datasets/` | Pretrained PPO results and training logs used in Part 3, small sample data |
| `artwork/` | Editable Blender assembly, source geometry and attribution |

## Build the course site

Lessons are MyST Markdown under `docs/zh/` and `docs/en/`, built with Sphinx and `nvidia-sphinx-theme` on the host (the course container is not needed). Edit the Markdown directly; large media stay in `web/assets/`.

```bash
python3 -m venv .local/docs-venv
.local/docs-venv/bin/pip install -r docs/requirements.txt
python3 scripts/build_site.py                      # writes web/zh and web/en
python3 scripts/check_course.py --site-only        # source, links, anchors, commands
python3 scripts/serve_site.py --port 8080          # preview at http://localhost:8080
```

`build_site.py` treats Sphinx warnings as errors (`--out DIR` makes a staging build); `check_course.py` without `--site-only` also checks model and runtime-bundle identities. Serve the site over HTTP: browser module loading does not work through `file://`. The repository-local authoring skill is `.agents/skills/microduck-course/SKILL.md`.

## Export for an LMS

After building the site, export both Open edX editions with the same docs environment:

```bash
.local/docs-venv/bin/python scripts/export_lms.py export \
  --site web --manifest docs/lms/microduck.json --out dist/lms \
  --strict --report dist/lms/export-report.json
```

`microduck_zh.tar.gz` opens in Chinese with an English switch. `microduck_en.tar.gz` contains English only. Both use white pages without site navigation and preserve the lesson interactions. Shared media are deduplicated; unused assets, model weights and training datasets stay outside these packages. The manifest controls course identity, page order and language editions. The exporter validates OLX bindings and local assets; browser review is still needed for interactions and the target LMS. See `scripts/export_lms.py --help` for validation and preview commands.

## 来源与许可 / Attribution and licenses

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Software, community policies, robot 3D models and detectors have distinct terms. In particular, the robot geometry is under Pollen Robotics' noncommercial share-alike terms; a source-code license does not remove those restrictions.
