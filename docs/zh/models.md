---
myst:
  html_meta:
    description: 所有主线实验使用明确命名的固定文件。下载一次，即可在各章节重复使用。
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 5.1 模型与数据

:::{lesson-intro}
:meta: 快速参考 · 无需重新训练

所有主线实验使用明确命名的固定文件。下载一次，即可在各章节重复使用。
:::

{#inventory}
## 课程模型清单

| 文件（宿主机 `models/`，容器内 `/workspace/Microduck-Workshop/models/`） | 用途 | 哪些课程命令读取它 |
|---|---|---|
| `walk.onnx` | 固定行走策略 | 2.1 节键盘踢球（步行模式，自动读取）；2.2 节 `walk --model models/walk.onnx`；4.3 节验证场景里的底层行走策略使用同一模型的副本 |
| `terrain-walk.onnx` | 3.3 节固定坡地行走策略，约 0.79 MB | 3.3 节 `terrain --model models/terrain-walk.onnx` |
| `kick-teacher.onnx` | 踢球基础动作 | 2.1 节键盘踢球（空格键，自动读取） |
| `kick-residual.onnx` | 踢球残差修正 | 保留；本课程的动手命令不直接读取 |
| `kick-checkpoint.pt` | 保留的踢球训练检查点 | 保留；本课程的动手命令不直接读取 |
| `detector.pt` | 固定视觉检测器 | 保留；本课程的动手命令不直接读取 |
| `detector.onnx` | 检测器 ONNX 导出 | 保留；本课程的动手命令不直接读取 |
| `indoor-detector/detector.pt` / `indoor-detector/detector.onnx` | 另一套室内检测器，对应 7 cm 足球的场景；本课的球场不使用 | 保留；本课程的动手命令不直接读取 |
| `scene-detector/detector.pt` / `scene-detector/detector.onnx` | 官网 10 cm 球与四种地面检测器；当前默认 | 机载策略（Onboard）的视觉检测器；本课程的动手命令不直接读取 |
| `official-browser/rollers.onnx` | 官方四被动轮策略；14 个舵机动作 | 2.1 节按 M 后的轮滑模式（自动读取）；3.3 节 `terrain --locomotion rollers` 与 `train-terrain --locomotion rollers` 的 `--model` |
| `official-browser/walking.onnx` / `sitstand.onnx` | 用于官网版本核对与站立试验；完整步行任务使用 `walk.onnx` | 保留；本课程的动手命令不直接读取 |

在**容器内**运行下面的命令，核对固定模型的文件大小与哈希。

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py verify-models
```

这些文件随 Git 仓库一同提供，克隆仓库即可使用，无需 Git LFS 或额外的模型账户。`models/manifest.json` 记录了文件大小、来源和 SHA-256；实验入口在运行前会核对模型身份，避免误用同名文件。当前默认检测器的相对路径与 SHA-256 记录在 `models/scene-detector/deployment.json`。

**仓库自带。** 第 2–3 部分实验和机载策略（Onboard）所用的权重；3.2 节预先训练好的 PPO 行走结果与坡地训练日志在 `datasets/walking-ppo` 和 `datasets/terrain-ppo`（容器内 `/workspace/Microduck-Workshop/datasets/`，只读）。`datasets/terrain-rollers-ppo-experiment` 是一次没有学会上坡的轮滑坡地训练记录，作为奖励设计的反例保留，不是可用的坡地模型。

**单独的资料包。** 4.3 节的 GR00T 步行射门数据、特征缓存与训练好的检查点来自单独的 GR00T 课程资料包（容器内 `course-data/groot-walking`，用 `python3 scripts/groot_walking.py verify` 核对）。GR00T 基础模型与 Cosmos-Reason2-2B 处理器按其授权与访问条件从 Hugging Face 下载。

{#practice-data}
## 练习数据与自己的输出

`datasets/scene-sample/` 是一套小型图像与标签样例，本课的动手实验不直接使用；它的规模不足以检验模型在真实环境中的泛化能力。自行训练的模型和日志请保存在 `outputs/`，`models/` 中的课程文件保持原样。

{#licenses}
## 来源与使用条款

机器人几何、社区策略和依赖组件分别适用各自的来源与许可条款；第三方资产以其自身条款为准，课程代码的许可不能替代这些条款。分发修改版本前，请先查阅仓库中的 [来源与许可说明](https://github.com/NV-DLI/Microduck-Workshop/blob/main/THIRD_PARTY_NOTICES.md)。
