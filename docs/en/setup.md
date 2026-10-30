---
myst:
  html_meta:
    description: Build the course container, mount data and models, open a shell in the same container, and confirm that Isaac Sim and the GPU work.
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 1.3 Set up the environment and get the code

:::{lesson-intro}
:meta: Hands-on · Ubuntu 24.04 / DGX Spark · Docker

Build the course container, mount data and models, open a shell in the same container, and confirm that Isaac Sim and the GPU work.
:::

{#requirements}
## What you need

Every lab in this course runs in one course container. The image holds Isaac Sim 6.2, the pinned Isaac Lab and Newton, and the GR00T learning environment. The host only builds and starts the container, shows the simulation windows, and stores data and results.

| Device | System | What you can use |
|---|---|---|
| DGX Spark (aarch64) | Ubuntu 24.04 desktop | The complete course, including Isaac Sim windows, cameras and recordings |
| RTX 5090 | Ubuntu in Windows 11 + WSL 2 | Commands without a window, such as PPO and GR00T training; the current driver does not support Isaac Sim windows, cameras or recordings |

The host needs an NVIDIA driver, Docker with the NVIDIA Container Toolkit, Git, and a desktop session that can show windows. The image pins these versions:

| Component | Course configuration |
|---|---|
| Container base | CUDA 13, Ubuntu 24.04, Python 3.12 |
| Simulation | Isaac Sim 6.2, Isaac Lab at a fixed commit, Newton physics backend |
| Learning | RSL-RL 5.5.1 (PPO), LeRobot + GR00T N1.7 (Part 4) |

{#get-code}
## Get the course code

Run these commands in a desktop terminal on the **host**.

```{code-block} bash
:caption: Terminal · host
:class: terminal

cd ~
git clone https://github.com/NV-DLI/Microduck-Workshop.git
cd Microduck-Workshop
```

The fixed course models (`models/` under 60 MB) and the pretrained results and training logs used in Part 3 (`datasets/` about 50 MB) come with the repository; your own runs go to `outputs/`, kept separate from them.

**GR00T course data package (Section 4.3 only).** The GR00T walking shot in Section 4.3 also needs a data package of about 62 GB: demonstrations, the training start point, the trained checkpoint and a precomputed feature cache. It is distributed separately by the instructors: before Section 4.3, get it from your instructor; until you have it, you can complete Parts 1–3 and read Part 4. Put its `groot-walking/` folder under `course-data/` in the repository root; if you keep it elsewhere, set `MICRODUCK_COURSE_DATA` to the absolute path of the folder that contains `groot-walking/` before starting the container. The container starts without the package.

**Optional: precomputed feature cache for the RTX 5090.** To train GR00T on an RTX 5090 (compute capability 12.0), you can also get from your instructor a feature cache of about 32 GB computed on that GPU. Put its `groot-walking-feature-caches/` folder next to `groot-walking/`, and training does not need to recompute the features (see Section 4.3). DGX Spark does not need it.

```{code-block} text
:caption: Workspace
:class: workspace-tree

Microduck-Workshop/
├── docker/        # image, start-up configuration and in-container helpers
├── lab/           # simulation and learning code (mounted live into the container)
├── scripts/       # lab entry points (same)
├── models/        # fixed course models (read-only in the container)
├── datasets/      # sample data and pretrained PPO results (read-only)
├── course-data/   # GR00T course data package (read-only, Section 4.3)
└── outputs/       # your runs (writable in the container)
```

{#build}
## Build the course container

Run this from the repository root. The first build downloads Isaac Sim, the Isaac Lab dependencies and the GR00T learning environment; how long it takes depends on your network, and the image is large, so leave enough disk space.

```{code-block} bash
:caption: Terminal · host
:class: terminal

docker compose -f docker/compose.yaml build
```

The image installs exactly the versions in `lab/environment/uv.lock` and checks the SHA-256 of the Isaac Lab source archive. Later changes to `lab/` or `scripts/` do not need a rebuild: both folders are mounted into the container when it starts.

{#start}
## Start the container and open a shell

Prepare the output and cache folders on the host once, allow the container to open windows on your desktop, start the container in the background, and open a shell in it.

```{code-block} bash
:caption: Terminal · host
:class: terminal

./docker/prepare_host.sh
xhost +local:
docker compose -f docker/compose.yaml up -d
docker exec -it microduck-workshop bash
```

`prepare_host.sh` checks that `DISPLAY` and `XAUTHORITY` point to your desktop session; if it reports them empty, run it from a terminal opened on the desktop. `up -d` keeps a container named `microduck-workshop` running in the background; `docker exec -it` opens a shell inside it. The prompt changes to `(microduck)` and the working directory is `/workspace/Microduck-Workshop`.

**Every later lab command runs in this container shell**, and the pages label those blocks “inside the container”. Ctrl+C stops only the command that is running; the container keeps running and you can type the next command.

:::{admonition} Return to the same container
:class: note

If you closed the terminal, or need a second shell (for example to watch logs while training in Section 4.3), run `docker exec -it microduck-workshop bash` on the host. You are back in the same container and earlier outputs are still there. If the container has stopped (for example after a reboot), run `docker compose -f docker/compose.yaml up -d` again from the repository root; data, outputs and caches live on the host. When you have finished all labs, `docker compose -f docker/compose.yaml down` stops and removes the container.
:::

A relative path in a command (for example `models/walk.onnx`) is the file of the same name in the host repository. Download, shader and compilation caches stay on the host under `~/.cache/huggingface` and `~/.cache/microduck/`, so the second start is much faster.

:::{admonition} Starting in WSL 2
:class: note

In Ubuntu on Windows 11 WSL 2, `prepare_host.sh` tells you to start with the command below, which adds the WSLg display settings: `docker compose -f docker/compose.yaml -f docker/compose.wslg.yaml up -d`. Use this environment only for commands without a window; do the labs that need an Isaac Sim window or a camera on DGX Spark.
:::

{#check}
## Check that you can start

Run these commands **inside the container**.

```{code-block} bash
:caption: Terminal · inside the container
:class: terminal

python3 scripts/run.py doctor
python3 scripts/run.py inspect-model --model models/walk.onnx
```

`doctor` checks the SHA-256 of every model file, CUDA, the display variable and package versions; continue when every item passes. `inspect-model` shows the walking model's input and output names, shapes and SHA-256. If CUDA is reported unavailable, run `nvidia-smi` on the host to confirm the driver works, then see [Troubleshooting](troubleshooting.md).

:::{admonition} Opening Isaac Sim for the first time
:class: note

The first window compiles shaders and loads assets, which can take a few minutes; the caches stay on the host, so later starts are much faster. The window appears on the host desktop; click it before using the keyboard.
:::

{#references}
## References

{.references}
- [Newton physics engine ↗](https://github.com/newton-physics/newton)
- [Isaac Sim system requirements ↗](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [NVIDIA Container Toolkit installation ↗](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
