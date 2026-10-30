---
myst:
  html_meta:
    description: 从环境、文件和参数开始检查，保留失败输出，以便比较下一次运行。
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 5.2 故障排查

:::{lesson-intro}
:meta: 快速参考

从环境、文件和参数开始检查，保留失败输出，以便比较下一次运行。
:::

{#doctor}
## 先收集基本信息

在**容器内**运行下面两条只读检查：`doctor` 报告 GPU、关键软件版本和图形显示等运行环境，`verify-models` 核对固定模型的文件大小与哈希。

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py doctor
python3 scripts/run.py verify-models
```

出现问题时，先记下第一条失败信息，它比反复启动完整训练更能说明原因。然后确认两件事：命令是在课程容器的终端里运行的（提示符为 `(microduck)`，当前目录 `/workspace/Microduck-Workshop`），并且所有模型校验都已通过。

{#common}
## 常见问题

| 现象 | 检查与处理 |
|---|---|
| 窗口没有打开 | 在 GPU 机器的桌面会话中打开终端，检查 DISPLAY 和显卡驱动；纯命令行的远程会话无法显示图形窗口。 |
| 第一次打开窗口很慢，或桌面提示 “Isaac Lab” is not responding | 首次打开窗口要编译 RTX 着色器、加载资源，程序暂时无法响应桌面（DGX Spark 上约 1 分钟）。选择 **Wait**，不要强制退出；日志仍有进度时不要重复启动。缓存保存在宿主机的 `~/.cache/microduck/kit-cache`，之后通常只需几秒；删除缓存、更新驱动或软件、打开含新内容的场景时可能再编译一次。 |
| 终端提示 `Actuator groups cover 14 joints; expected 28`（轮滑脚为 32） | 这是预期的提示，不需要处理。Microduck 有 14 个由舵机驱动的关节；模型里另有 14 个表示齿轮间隙的被动铰链，每个只能在约 ±1° 内自由转动，轮滑脚还多 4 个自由转动的轮子。它们本来就没有电机，Isaac Lab 的检查把它们计为“没有执行器”的关节。 |
| 窗口底部状态栏显示 `RenderDelegate : Warning /rtx/hydra/readTransformsFromFabricInRenderDelegate and geometry streaming are enabled together …` | 这是渲染位姿更新的提示，不是关节配置错误。画面中物体正常显示并随仿真移动时可以继续；如果物体缺失、停在原处或位置明显不对，请停止运行，记录 Isaac Sim / Isaac Lab 版本并保留完整日志。GR00T 的头部相机也读取渲染画面。 |
| 视频解码提示 | 若 TorchCodec 加载失败后显示 Falling back to 'pyav'，且读取或训练进度仍在增长，程序已切换到另一种视频解码方式，可以继续等待。若进程退出或视频仍无法读取，请保留完整报错并检查 4.3 节的 GR00T 学习环境。 |
| 提示输出目录已存在 | 为新实验换一个目录名，上一次运行的结果会保留下来。 |
| 显存或内存不足 | 先减少训练的并行环境数，并且一次只运行一个仿真或训练；GR00T 训练可以改用 `--profile low-memory`（4.3 节）。 |
| 模型校验失败 | 重新获取仓库中的原模型，确认没有覆盖固定文件。 |
| 自训练模型行走效果不佳 | 短训练用于体验流程，策略通常尚未练好。比较奖励、回合长度和独立测试表现；后续各节继续使用固定模型。 |
| 机器人停止或没有进球 | 查看任务结果和观测有效性。观测信息不足时，停下来可能是合理的保护行为，但只有任务结果判定进球才算射门成功。 |

{#report}
## 查看一次运行

把示例中的 `outputs/ppo-walk-test-01`（3.2 节的测试）换成你实际运行过的输出目录。命令会读取该目录中的结果文件，列出状态、随机种子、任务结果等关键字段；如果运行尚未结束、还没有结果文件，它会提示先查看运行日志。

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py report --run outputs/ppo-walk-test-01
```

提交问题时，请附上运行命令、随机种子、环境检查结果和第一条报错信息；只写“成功”或“失败”无法定位问题。请勿发送账户密码或访问令牌。
