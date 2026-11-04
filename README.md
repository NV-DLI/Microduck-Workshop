# Train Microduck in Simulation with NVIDIA Isaac

使用 NVIDIA Isaac 在仿真环境训练 Microduck.

A self-paced course: drive the open-source Microduck biped in Isaac Sim, train walking and slope skills with reinforcement learning in Isaac Lab and Newton, run an onboard policy that scores a walking shot with its own sensors and hand-designed stages, then see how GR00T learns the same shot from demonstrations and plan how to carry it to more environments.

**[中文课程](https://nv-dli.github.io/Microduck-Workshop/zh/index.html) · [English course](https://nv-dli.github.io/Microduck-Workshop/en/index.html)**

This repository holds the course site, the lab code, the container and the fixed course models. The rest of this README gets the container running; the course pages explain every lab.

## Course structure

1. **Overview**: the course goal and setting up the course container.
2. **Meet Microduck**: keyboard football, mechanism and the control loop.
3. **Teach Microduck skills with reinforcement learning**: PPO basics, reinforcement learning for Microduck, walking training with domain randomization, an extension to slopes, (extension) community skills and their reward design.
4. **Complete the shooting task with an onboard policy**: RGB and ToF sensors, compact YOLO detection, direct path planning and hand-designed shooting stages; repeat the task and interpret successes and failures.
5. **(Extention) Score autonomously with GR00T**: vision-language-action (VLA) models, how GR00T scores the walking shot with 8-D high-level requests and trained skills (training and inference are done in advance; no student commands), environment variation and Cosmos 3.
6. **Reference and help**: design your own robot task with an agent, and continue with references.

Reading, figures, search and the interactive 3D duck work in a browser without Isaac Sim.

## Requirements

The experiments require a computer with an NVIDIA DGX-system GPU or RTX GPU that meets Isaac Sim requirements. The container supports Intel / AMD x86_64 CPUs and DGX Spark ARM CPUs (aarch64) on Ubuntu 24.04. All experiments have been tested only on DGX Spark; Spark is not mandatory, and other hardware needs local validation.

The host needs an NVIDIA driver, Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Git and a desktop session that can show windows. The image pins Isaac Sim 6.2, the Isaac Lab development snapshot (`e527032b`, including the Microduck tasks and BAM motor model), Newton 1.8.0.dev0 (`71eeba16`), and RSL-RL 5.5.1. This Isaac Lab snapshot is not the 3.0.0 GA release. The course lock separately pins Newton for the supplied policies; a narrowly scoped USD-import adapter supports the course Newton/BAM route. Details are in Section 1.2 of the course.

## Quick start

All commands below run on the **host** from the repository root unless marked as inside the container.

1. **Get the repository.**

   ```bash
   cd ~
   git clone https://github.com/NV-DLI/Microduck-Workshop.git
   cd Microduck-Workshop
   ```

2. **Build the course container** (expect about 10–12 GB of downloads on a fresh install; allow at least 50 GB of free disk space).

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

   The prompt changes to `(microduck)` in `/workspace/Microduck-Workshop`.

4. **Return to the same container** at any time with `docker exec -it microduck-workshop bash`. Ctrl+C stops only the running command. After a reboot, repeat the same compose start command. `docker compose -f docker/compose.yaml down` removes the container when you are done. Outputs and caches stay on the host.

5. **Check the setup** inside the container:

   ```bash
   python3 scripts/run.py doctor
   python3 scripts/run.py inspect-model --model models/walk.onnx
   ```

   With a desktop session, start the keyboard football lab with `python3 scripts/run.py manual --seconds 600 --output outputs/manual-football-01`.

   `scripts/run.py` is the lab entry point (`python3 scripts/run.py --help` lists the commands, `python3 scripts/run.py <command> --help` their parameters).

## Data and models

| What | Where | How you get it |
|---|---|---|
| Fixed course models | `models/` (under 60 MB), checked against `models/manifest.json` | In the repository |
| Pretrained PPO results and training logs for Part 3 | `datasets/` (about 50 MB) | In the repository |

The GR00T part of the course is an extension that students read; its demonstrations and checkpoints are kept separately by the course team and are not a student prerequisite.

`models/` and `datasets/` are mounted read-only; `lab/` and `scripts/` are mounted live, so host edits apply without a rebuild. Every run writes to a new folder under `outputs/`, and training never replaces the course models: evaluate your own export explicitly with `--model`.

## Repository

| Directory | Contents |
|---|---|
| `web/` | Built site (`web/zh`, `web/en`) plus shared assets in `web/assets`: images, seekable videos, 3D model, fonts and figure scripts |
| `docker/` | Course image, compose files and in-container helpers |
| `scripts/` | Lab entry point `run.py` and environment setup tools |
| `lab/` | Simulation, training and evaluation code, scene assets and environment locks |
| `models/` | Fixed course policies and detectors, with SHA-256 manifest |
| `datasets/` | Pretrained PPO results and training logs used in Part 3, small sample data |

## 来源与许可 / Attribution and licenses

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Software, community policies, robot 3D models and detectors have distinct terms. In particular, the robot geometry is under Pollen Robotics' noncommercial share-alike terms; a source-code license does not remove those restrictions.
