---
myst:
  html_meta:
    description: 构建课程容器、挂载数据与模型、进入同一个容器终端，并确认 Isaac Sim 与 GPU 可用。
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 1.3 准备环境与获取代码

:::{lesson-intro}
:meta: 操作实验 · Ubuntu 24.04 / DGX Spark · Docker

构建课程容器、挂载数据与模型、进入同一个容器终端，并确认 Isaac Sim 与 GPU 可用。
:::

{#requirements}
## 需要什么

课程的全部实验都在一个课程容器里运行：Isaac Sim 6.2、锁定版本的 Isaac Lab 与 Newton，以及 GR00T 学习环境都装在同一个镜像里。宿主机只负责构建和启动容器、显示仿真窗口，并保存数据和结果。

| 设备 | 系统 | 可用的课程内容 |
|---|---|---|
| DGX Spark（aarch64） | Ubuntu 24.04 桌面 | 完整课程，包括 Isaac Sim 窗口、相机与录像 |
| RTX 5090 | Windows 11 + WSL 2 中的 Ubuntu | 不开窗口的命令，例如 PPO 与 GR00T 训练；当前驱动不支持 Isaac Sim 窗口、相机和录像 |

宿主机需要 NVIDIA 驱动、Docker 与 NVIDIA Container Toolkit、Git，以及一个能显示窗口的桌面会话。镜像中的软件版本固定如下：

| 组件 | 课程配置 |
|---|---|
| 容器基础 | CUDA 13、Ubuntu 24.04、Python 3.12 |
| 仿真 | Isaac Sim 6.2，Isaac Lab 固定提交，Newton 物理后端 |
| 学习 | RSL-RL 5.5.1（PPO），LeRobot + GR00T N1.7（第 4 部分） |

{#get-code}
## 下载课程代码

在**宿主机**的桌面终端中运行下面的命令。

```{code-block} bash
:caption: Terminal · 宿主机
:class: terminal

cd ~
git clone https://github.com/NV-DLI/Microduck-Workshop.git
cd Microduck-Workshop
```

固定的课程模型（`models/` 不到 60 MB）和第 3 部分用到的预训练结果与训练日志（`datasets/` 约 50 MB）都随仓库下载；你自己运行得到的结果写在 `outputs/`，与它们分开保存。

**GR00T 课程资料包（只有 4.3 节需要）。** 4.3 节的 GR00T 步行射门另外需要一个约 62 GB 的资料包：示范数据、训练起点、训练好的检查点和预先算好的特征缓存。它由授课方单独发放：开始 4.3 节前，请向授课教师领取；还没有拿到时，可以先完成第 1–3 部分和第 4 部分的阅读。把其中的 `groot-walking/` 文件夹放到仓库根目录的 `course-data/` 下；放在别处时，启动容器前把 `MICRODUCK_COURSE_DATA` 设为包含 `groot-walking/` 的文件夹的绝对路径。没有资料包也可以启动容器。

**可选：RTX 5090 的预计算特征缓存。** 在 RTX 5090（计算能力 12.0）上训练 GR00T 时，可以向授课教师另外领取一份约 32 GB、在这种 GPU 上算好的特征缓存，把其中的 `groot-walking-feature-caches/` 文件夹放在 `groot-walking/` 旁边，训练时就不必重算（见 4.3 节）。DGX Spark 不需要它。

```{code-block} text
:caption: Workspace
:class: workspace-tree

Microduck-Workshop/
├── docker/        # 镜像、启动配置与容器内辅助命令
├── lab/           # 仿真与学习代码（启动时实时挂载进容器）
├── scripts/       # 实验入口（同上）
├── models/        # 固定课程模型（容器内只读）
├── datasets/      # 示例数据与预训练 PPO 结果（容器内只读）
├── course-data/   # GR00T 课程资料包（容器内只读，第 4.3 节）
└── outputs/       # 你的运行结果（容器内可写）
```

{#build}
## 构建课程容器

在仓库根目录运行。首次构建要下载 Isaac Sim、Isaac Lab 依赖和 GR00T 学习环境，耗时取决于网络，镜像也较大，请预留足够磁盘空间。

```{code-block} bash
:caption: Terminal · 宿主机
:class: terminal

docker compose -f docker/compose.yaml build
```

镜像按 `lab/environment/uv.lock` 安装与课程完全相同的软件版本，并校验 Isaac Lab 源码包的 SHA-256。之后修改 `lab/` 或 `scripts/` 不需要重新构建：启动时这两个目录会直接挂载进容器。

{#start}
## 启动容器并进入终端

先准备一次宿主机上的输出与缓存目录，再允许容器在你的桌面上打开窗口，最后在后台启动容器。

```{code-block} bash
:caption: Terminal · 宿主机
:class: terminal

./docker/prepare_host.sh
xhost +local:
docker compose -f docker/compose.yaml up -d
docker exec -it microduck-workshop bash
```

`prepare_host.sh` 会检查 `DISPLAY` 和 `XAUTHORITY` 是否指向你的桌面会话；如果提示为空，请在桌面里打开的终端中运行。`up -d` 让名为 `microduck-workshop` 的容器在后台持续运行；`docker exec -it` 在这个容器里打开一个终端，提示符会变成 `(microduck)`，当前目录是 `/workspace/Microduck-Workshop`。

**之后所有实验命令都在这个容器终端里运行**，页面上会标注“容器内”。用 Ctrl+C 停止某个实验，只会结束这一条命令，容器仍在运行，可以接着输入下一条。

:::{admonition} 回到同一个容器
:class: note

关闭了终端，或者需要第二个终端（例如 4.3 节一边训练一边看日志）时，在宿主机运行 `docker exec -it microduck-workshop bash`，就会回到同一个容器，之前的输出都还在。容器停止后（例如重启了电脑），在仓库根目录再运行一次 `docker compose -f docker/compose.yaml up -d` 即可；数据、输出和缓存都保存在宿主机上。全部练习结束后，用 `docker compose -f docker/compose.yaml down` 停止并移除容器。
:::

命令里的相对路径（例如 `models/walk.onnx`）就是宿主机仓库里的同名文件。下载、着色器和编译缓存保存在宿主机的 `~/.cache/huggingface` 与 `~/.cache/microduck/` 下，第二次启动会快很多。

:::{admonition} 在 WSL 2 中启动
:class: note

在 Windows 11 的 WSL 2 Ubuntu 中，`prepare_host.sh` 会提示改用下面的启动命令，它额外加载 WSLg 的显示设置：`docker compose -f docker/compose.yaml -f docker/compose.wslg.yaml up -d`。这种环境只用于不开窗口的命令，需要 Isaac Sim 窗口或相机的实验请在 DGX Spark 上进行。
:::

{#check}
## 确认可以开始

在**容器内**运行下面的命令。

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py doctor
python3 scripts/run.py inspect-model --model models/walk.onnx
```

`doctor` 逐一核对模型文件的 SHA-256，并检查 CUDA、显示变量和各软件包版本；所有项通过后再继续。`inspect-model` 显示行走模型的输入输出名称、形状和 SHA-256。如果 CUDA 显示为不可用，先在宿主机运行 `nvidia-smi` 确认驱动正常，再查阅[故障排查](troubleshooting.md)。

:::{admonition} 第一次打开 Isaac Sim
:class: note

第一次打开窗口时需要编译着色器、加载资源，可能要等几分钟；缓存保存在宿主机后，之后会快很多。窗口出现在宿主机桌面上，点击窗口后键盘操作才会生效。
:::

{#references}
## 参考资料

{.references}
- [Newton 物理引擎 ↗](https://github.com/newton-physics/newton)
- [Isaac Sim 系统要求 ↗](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
- [NVIDIA Container Toolkit 安装说明 ↗](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
