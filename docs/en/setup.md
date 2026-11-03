---
myst:
  html_meta:
    description: Build the course container, mount data and models, open a shell in the same container, and confirm that Isaac Sim and the GPU work.
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 1.2 Set up the environment and get the code

:::{lesson-intro}
:meta: Hands-on · Ubuntu 24.04 / DGX Spark · Docker

Build the course container, mount data and models, open a shell in the same container, and confirm that Isaac Sim and the GPU work.
:::

{#requirements}
## What you need

Every lab in this course runs in one course container. The image holds Isaac Sim 6.2 and the pinned Isaac Lab and Newton. The host only builds and starts the container, shows the simulation windows, and stores data and results.

| Device | System | What you can use |
|---|---|---|
| DGX Spark (aarch64) | Ubuntu 24.04 desktop | The complete course, including Isaac Sim windows, cameras and recordings |

The host needs an NVIDIA driver, Docker with the NVIDIA Container Toolkit, Git, and a desktop session that can show windows. The image pins these versions:

| Component | Course configuration |
|---|---|
| Container base | CUDA 13, Ubuntu 24.04, Python 3.12 |
| Simulation | Isaac Sim 6.2, Isaac Lab at a fixed commit, Newton physics backend |
| Learning | RSL-RL 5.5.1 (PPO) |

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

```{code-block} text
:caption: Workspace
:class: workspace-tree

Microduck-Workshop/
├── docker/        # image, start-up configuration and in-container helpers
├── lab/           # simulation and learning code (mounted live into the container)
├── scripts/       # lab entry points (same)
├── models/        # fixed course models (read-only in the container)
├── datasets/      # sample data and pretrained PPO results (read-only)
└── outputs/       # your runs (writable in the container)
```

{#build}
## Build the course container

Run this from the repository root. The first build downloads Isaac Sim, the Isaac Lab dependencies and more; how long it takes depends on your network, and the image is large, so leave enough disk space.

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

If you closed the terminal, or need a second shell (for example to watch TensorBoard while training in Section 3.3), run `docker exec -it microduck-workshop bash` on the host. You are back in the same container and earlier outputs are still there. If the container has stopped (for example after a reboot), run `docker compose -f docker/compose.yaml up -d` again from the repository root; data, outputs and caches live on the host. When you have finished all labs, `docker compose -f docker/compose.yaml down` stops and removes the container.
:::

A relative path in a command (for example `models/walk.onnx`) is the file of the same name in the host repository. Shader and compilation caches stay on the host under `~/.cache/microduck/`, so the second start is much faster.

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
