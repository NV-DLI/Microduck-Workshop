---
myst:
  html_meta:
    description: 先用键盘指挥 Microduck 走路、转向和踢球，在同一个窗口里切换到轮滑，再看看 Isaac Sim 如何把场景、机器人和预训练策略连接起来。
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 2.1 认识 Microduck：先来踢一场球

:::{lesson-intro}
:meta: 第 2.1 节 · 预计学习时长 20 分钟

先用键盘指挥 Microduck 走路、转向和踢球，在同一个窗口里切换到轮滑，再看看 Isaac Sim 如何把场景、机器人和预训练策略连接起来。
:::

{#meet}
## 一只可以学习新技能的小机器人

Microduck 是 Pollen Robotics 与 Hugging Face 推出的开源双足机器人。它有 14 个主动关节，可以通过训练获得行走、踢腿等技能。我们用 NVIDIA Isaac Sim 搭建可反复试验的场景，用 Isaac Lab 组织训练与评测，逐步让机器人自己找到足球并完成射门。先不训练：加载官方已经训练好的行走与右脚踢球 ONNX 策略，体验一次手动射门。

{#play}
## 打开球场，用键盘试一试

在 1.3 节准备的课程容器终端里运行下面的命令（需要重新进入容器时，见 <a href="setup.html#start">1.3 节</a>）。它打开 Isaac Sim，场景中是第 4 部分自主射门同样的绿色球场、足球和球门。

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py manual \
  --seconds 600 \
  --output outputs/manual-football-01
```

窗口打开时是**步行模式**。把输入法切换到英文，用鼠标点击仿真窗口，就可以用下表的按键操作；按键没有反应时，通常是中文输入法拦截了按键。按 **M** 切换到轮滑模式，同一个窗口会重新搭好球场、换上轮滑脚，大约需要几十秒。按键说明也显示在窗口右侧的 “Microduck • Keyboard football” 标签页里。

:::{enrichment} 代码关键参数
- **模型。** `run.py manual` 不需要模型参数：它读取容器内的 `/workspace/Microduck-Workshop/models`（即宿主机仓库的 `models/`，只读）。步行模式用其中的 `walk.onnx` 和 `kick-teacher.onnx`，轮滑模式用 `official-browser/rollers.onnx`。
- **场景资产。** 球场、足球、球门和机器人来自仓库的 `lab/assets/`：`pitch_A.usda`、`official_ball.usda`、`steel_goal.usda`、`microduck_contact.usda`，轮滑脚为 `rollers/microduck_rollers_analytic.usdc`。
- **起步。** 小鸭子先锁住关节在球场上落稳，策略开始控制时再松开；M 键只在策略接管之后才生效。
- **切换模式。** 按 M 时当前场景关闭，同一个窗口里重新搭好球场，换入另一套机器人配置和对应策略。
- **输出。** `--output` 指定的 `outputs/manual-football-01` 在宿主机仓库的 `outputs/` 下。每次切换模式都会新建一个编号子文件夹，例如 `00-walking`、`01-rollers`，两种模式的记录分开保存。`--seconds` 是窗口最长运行的仿真时间。
:::

:::{course-image} media/isaac-ui-keyboard-walking.png
:alt: 键盘踢球的 Isaac Sim 完整界面（步行模式）：绿色球场上的 Microduck、足球和球门，右侧标签页列出步行模式的按键。

步行模式。截图来自课程在容器中运行上面这条命令的一次真实会话。
:::

| 按键 | 步行模式 | 轮滑模式 |
|---|---|---|
| W / S | 向前 / 向后走 | 推进 / 制动（持续按住 S 可能开始倒滑） |
| A / D | 向左 / 向右侧移 | 不使用 |
| Q / E | 向左 / 向右转 | 左转 / 右转 |
| 松开按键 | 零速度指令，回到站立 | 继续滑行，不保证立即停住 |
| 空格 | 一次右脚踢球，然后回到站立 | 一次有限时的滑行推射辅助，不会踢腿 |
| M | 切换到轮滑模式 | 切换到步行模式 |
| R / Esc | 重置当前场景 / 退出 | 重置当前场景 / 退出 |

**步行模式**：先走到球旁边并停下，让球位于右脚前方、身体朝向球门，再按空格出脚。踢空了就调整站位；失去平衡就按 R 重来。这里由你判断球在哪里、应该朝哪走，策略只负责把指令变成关节动作。没能进球时想一想：是站位偏了、朝向不对，还是出脚时机不合适？这些判断正是第 4 部分要交给自主射门系统的任务。松开按键后的站立也由行走策略完成：它一直在运行，收到零速度指令就保持站立。

<span id="rollers"></span>**轮滑模式**：四个轮子是被动轮，仍由原来的 14 个舵机调整腿部姿态完成推进、转向和制动，不会切换到走路或踢腿。先观察松开按键后仍会滑行，再练习提前减速；目标是把球碰进门，同时让小鸭子停在门前。W 给出的是推进量，并不对应以米每秒为单位的速度。比较两种方式：步行射门先停稳再出脚；轮滑射门通过运动中的接触传递动量，随后制动。

:::{course-image} media/isaac-ui-keyboard-rollers.png
:alt: 同一窗口按 M 后的轮滑模式：同一块球场上的 Microduck 换上了带轮子的脚，右侧标签页改为轮滑模式的按键。

按 M 之后的轮滑模式，与上一张截图来自同一次会话、同一个窗口：场景重新搭建，脚换成了轮滑脚，按键说明也随之改变。
:::

:::{admonition} 两种脚是两套不同的机器人配置
:class: note

普通脚掌和官方轮滑脚的结构、质量和碰撞形状都不同，所以切换模式时程序会重新搭建场景、换入另一套机器人配置和对应的官方策略。这相当于在实验台上换了一副脚，并不表示真机能在运动中自己变形。四个轮胎保留官方外观、质量和惯量；为了稳定计算滚动接触，碰撞体用直径 30 毫米、宽 7.6 毫米的圆柱近似。真实轮胎的摩擦与变形尚未标定。
:::

{#scene}
## Isaac Sim 如何构成这场实验

```{learning-flow}
label: 从场景到运动
nodes:
- [USD 场景, 机器人、球场、足球与球门]
- [物理配置, 质量、碰撞、摩擦与执行器]
- [策略与输入, 键盘速度 → ONNX → 关节目标]
details:
- USD 场景将机器人、球与球门放在同一空间。网格负责可见外形，几何与材质也会影响相机看到的图像。
- 质量、惯量、摩擦和碰撞体决定接触后的运动。相同外观可以对应不同物理配置，因此外观逼真不能代替物理检查。
- 键盘提供期望的运动方向与速度，ONNX 策略把观测和指令转换为关节目标。
caption: 画面、物理与策略各司其职：更换技能可以沿用同一个场景，移动足球也无需重新训练行走策略。
```

机器人 USD 除了可见网格，还包含关节、碰撞形状和质量属性：关节定义连接关系，碰撞形状决定接触位置，质量与惯量影响加速和转身，舵机模型决定目标角度如何变成实际运动。下面的代码按官网黑白足球设置半径 5 cm、质量 30 g。Isaac Sim 提供场景与渲染，Isaac Lab 的任务配置则接入 Newton 物理后端。

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/manual.py
language: python
code: |
  cfg.scene.ball = RigidObjectCfg(
      prim_path="{ENV_REGEX_NS}/Ball",
      spawn=sim_utils.SphereCfg(
          radius=0.05,
          mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
          collision_props=sim_utils.CollisionPropertiesCfg(),
          rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
      ),
      init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
  )
  # Apply the shared official-ball asset, goal and floor materials.
  scene_profile = configure_football(cfg, style="A")
steps:
- lines: [0, 2]
  title: 在场景中登记足球
  body: '`RigidObjectCfg` 声明一个刚体；`{ENV_REGEX_NS}` 让每个并行环境都有自己的一颗球。'
- lines: [2, 8]
  title: 设置形状与物理属性
  body: 半径 0.05 m、质量 0.03 kg，并启用碰撞与刚体动力学，Newton 才会计算球被踢后的运动。
- lines: [8, 10]
  title: 放到地面上
  body: 初始高度 0.05 m 正好等于半径，球刚好接触地面。
- lines: [10, 12]
  title: 套用共享的足球场景
  body: '`configure_football` 换入官网足球资产，加入球门与地面材质；`style="A"` 选择全课统一使用的绿色条纹球场。'
```

{#scene-contract}
## 先固定场景与硬件约定

键盘射门与第 4 部分的 GR00T 射门使用同一块 1 × 1.6 米的绿色球场、直径 10 厘米、质量 30 克的官网黑白足球和固定式钢网球门。进球判定使用球门的净开口：宽 38.8 厘米、高 24.4 厘米。第 3 部分使用各节说明的训练地面和测试地形；换场地、灯光或室内背景的影响留到 4.4 节讨论。

:::{enrichment} 场景参数
- **球门。** 外尺寸宽 40 × 深 20 × 高 25 厘米；门框直径按 6 毫米计，两根立柱的中心距为 39.4 厘米。定位用中心距，进球判定用净开口，两者不要混用。
- **地面。** 按刚性平面处理，没有模拟软垫的压缩或起翘；场地垫与实物地面的摩擦尚未标定。
:::

机器人使用官方 Microduck CAD、基础质量与关节参数，以及带齿隙的全身碰撞模型。课程锁定 XL330 的 BAM 电机拟合、200 Hz 物理和 50 Hz 策略周期；15 个实体舵机中，行走策略控制 14 个，嘴部不由这份策略驱动。全身碰撞让头、腿与电池等部位摔倒后也能被地面托住，不会穿过地面。

这套仿真配置与真机仍有差距：相机对应 GC2093，仿真按 960 × 540、15 Hz 采样，70° 水平视场角尚未经过镜头标定；ToF 和身体 IMU 也是近似。因此它不能证明最新固件、各种硬件误差或 RK3566 上的完整部署都已复现。移到真机前，应核对实际硬件版本、电压、关节零位、镜头参数与端到端延迟。

{#policies}
## 步行模式如何切换行走与踢球策略

| 文件 | 大小 | 作用 |
|---|---|---|
| walk.onnx | 793,705 字节，约 0.79 MB | 速度非零时行走；速度为零时站立 |
| kick-teacher.onnx | 793,685 字节，约 0.79 MB | 右脚踢球动作 |

步行与踢球使用固定版本的官方权重（版本链接见本页参考资料），可能与官网网页模拟器中最新的步行权重不同。轮滑实验使用官网的轮滑模型。

:::{network-diagram}
策略网络读入 61 维观测，依次经过 512、256、128 维隐藏层，输出 14 个关节动作。行走与踢球两个官方策略结构相同、权重不同，因此表现为不同的技能。训练时配合使用的价值网络不在图中。
:::

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/manual.py
language: python
code: |
  # Read the state once per policy step, then select the active skill.
  obs = env.observation_manager.compute(update_history=True)["policy"]
  action = walk(obs)

  if kick_start is not None:
      phase = (step - kick_start) * 0.02
      weight = 1.0 if phase < 0.14 else max(0.0, 1.0 - (phase - 0.14) / 0.12)
      if weight:
          action = torch.lerp(action, kick(obs), weight)

  # The action manager maps network outputs to joint targets.
  env.action_manager.process_action(action)
steps:
- lines: [0, 4]
  title: 每个策略步读取一次观察
  body: 观测管理器更新历史，给出 policy 观察；行走策略先算出默认动作。
- lines: [4, 7]
  title: 计算踢球权重
  body: 策略周期为 0.02 s（50 Hz），`phase` 是踢球开始后经过的秒数。前 0.14 s 权重为 1，之后 0.12 s 内线性降到 0。
- lines: [7, 10]
  title: 混合两个策略的动作
  body: '`torch.lerp` 按权重在行走动作与踢球动作之间插值；权重为 0 时直接沿用行走动作。'
- lines: [10, 12]
  title: 转换为关节目标
  body: 动作管理器把网络输出映射为关节目标，再交给执行器和物理仿真。
```

手动实验让踢球动作完整保持 0.14 秒，再用 0.12 秒平滑切回行走策略的站立动作。这两个时间只影响技能之间的切换，官方权重保持不变。

策略文件不到 1 MB，但运行完整仿真还要为场景、物理和渲染占用显存，不能按文件大小估算。1.3 节运行过的 `inspect-model` 会显示策略文件的 SHA-256、输入输出张量形状和用到的算子。

{#reflect}
## 刚才是谁在决定下一步

::::{quiz}
刚才您已经完成了哪些机器人还不会自主完成的判断？

:::{answer}
您识别了足球和球门、估计了距离、选择了接近路线和出脚位置，并决定何时踢球。第 4 部分训练 GR00T 从示范数据中学习这些判断，4.4 节再讨论环境变化与数据扩展。
:::
::::

2.2 节先看看这些动作背后的机械结构，以及策略和物理引擎按什么节奏协同工作。

{#references}
## 参考资料

{.references}
- [Pollen Robotics：课程采用的官方策略版本 ↗](https://huggingface.co/pollen-robotics/microduck-policies/tree/665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448)
- [NVIDIA：Isaac 平台与 Microduck ↗](https://nvidianews.nvidia.com/news/nvidia-advances-isaac-platform-for-general-purpose-robotics-with-new-reference-humanoid-frameworks-simulation-pipelines-and-data/)
