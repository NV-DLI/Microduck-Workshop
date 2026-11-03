---
myst:
  html_meta:
    description: 人工设计的板载策略怎样复用传感器估计完成射门：为什么采用板载策略，Nav2 路径规划和已训练的运动技能怎样由人工设计的阶段连接起来，每个阶段的核心代码，以及怎样在 Isaac Sim 中反复射门。
---

# 4.2 人工设计射门阶段

:::{lesson-intro}
:meta: 第 4.2 节 · 预计学习时长 35 分钟

[4.1 节](06-perception.md)把传感器读数变成了球和球门的位置。本节介绍人工设计的板载策略怎样用这些位置把导航、行走和踢球连接起来，阅读每个阶段的核心代码，并在 Isaac Sim 中运行它反复射门。
:::

{#why}
## 为什么采用板载策略

Microduck 使用基于 Rockchip RK3566 的主控板，其中的小型神经网络加速器（NPU）为单核 INT8，算力 0.8 TOPS；运动策略以 50 Hz 循环运行（[官方 README ↗](https://github.com/pollen-robotics/microduck)，[NPU 说明 ↗](https://github.com/pollen-robotics/microduck/blob/main/docs/project/npu-bringup.md)）。这样的算力难以承担 GR00T 这类约 30 亿参数的视觉-语言-动作模型的低延迟推理。因此板载策略选择小型检测器，配合几何计算、路径规划和已训练的行走与踢球技能。

{#pipeline}
## 从传感器到动作：Isaac Lab + Nav2

这里有两个独立的工具协同工作。**Isaac Lab**（基于 Isaac Sim，物理引擎为 Newton）仿真机器人、传感器和球场，并运行学到的策略。**Nav2** 是 ROS 2 的导航框架；课程只用它的路径规划器，作为同一容器里的一个 CPU 进程运行。课程代码把机器人的位置（在已知球场地图上由传感器估计）和目标点交给 Nav2，再拿回一串路径点。球场地图是预先测量的，当前 ToF 只补充附近的观测。

| 组成部分 | 做什么 |
|---|---|
| YOLO26n 检测器 | 在 RGB 图像中用识别框标出球和球门 |
| 几何计算 | 由识别框、ToF 地面和关节角算出球与球门的位置 |
| Nav2 规划器（NavFn，A*） | 在二维代价地图上找到一条到球后站位的低代价路径 |
| 路径跟随与阶段 | 把下一个路径点变成行走速度；按条件切换阶段 |
| 行走策略 | 把速度请求变成 14 个关节目标，50 Hz |
| 踢球与恢复 | 出脚击球，再回到站立 |

Nav2 返回的是地面上的点，从不输出关节角。一段简短的手写跟随程序把下一个路径点变成速度请求，平衡和迈步交给行走策略完成。

{#stages}
## 射门的六个阶段

设计阶段时，可以从成功条件倒推。踢球技能只在很小的范围内有效：球要在右脚前方，身体要朝向球门，机器人要先站稳。为了满足这些前提，需要先走到球后方的站位，再精确调整位置和朝向；而这一切的前提，是先知道球和球门在哪里。

再考虑传感器的不确定性。行走时相机晃动，近处测得的位置不够准，所以要走一小段、停稳、再看；踢球必须从静止站姿开始，并用最新的观测确认，所以单独设一个交接阶段。

```{learning-flow}
label: 板载策略的步行射门阶段
nodes:
- [寻找目标, 转头观察球门与足球]
- [规划接近, 选好球后站位，分段走近]
- [近距细调, 小步移动，停稳后再看球]
- [对准出脚位置, 球在右脚前，身体朝向球门]
- [交接给踢球, 停步，头部回正，身体静止]
- [踢球与恢复, 踢球技能出脚，回到站立]
details:
- 原地转头，确定机器人相对球门的位置，并找到足球。
- Nav2 规划到球后站位的路径，跟随程序沿路径行走。
- 近处按球的位置选择短步、短段连续移动或原地转向，每段结束后停稳再看球。
- 检查球是否在右脚前方、身体是否朝向球门。
- 停步，头部回正，确认静止和最新观测后交给踢球技能。
- 踢球技能出脚，随后回到站立；由评测程序判断是否进球。
layout: learning
loop: false
```

下面按顺序介绍每个阶段，并附上实际运行的核心代码。定位与路径使用球场地图坐标；球的近距位置使用以机器人为参考、与地面水平的坐标，x 向前、y 向左。代码会在需要时转换坐标。

{#stage-find}
### 寻找目标

机器人原地转头扫视，用 4.1 节的方法找到两根门柱的底点，由门柱位置算出自己相对球门的位置和朝向，再把看到的足球放到同一张地图上。单次估计可能有误差，所以要连续几次结果一致才接受。之后如果太久没看到球门，或者球的信息过旧，就回到这里重新观察。

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/football/controller.py
language: python
code: |
  measured = np.asarray(fix["pose_map_xyyaw"])
  self.goal_window.append((t, measured))
  if len(self.goal_window) == 3 and t - self.goal_window[0][0] < 0.5:
      centre = np.mean([p[:2] for _, p in self.goal_window], axis=0)
      a = np.array([p[2] for _, p in self.goal_window])
      yaw = math.atan2(np.sin(a).mean(), np.cos(a).mean())
      if (
          max(np.linalg.norm(p[:2] - centre) for _, p in self.goal_window)
          < 0.04
      ):
          if (
              self.pose is None
              or np.linalg.norm(self.pose[:2] - centre) < 0.20
          ):
              self.pose = np.array([*centre, yaw])
              self.goal_time = t
steps:
- lines: [0, 2]
  title: 记录一次定位
  body: 由门柱底点拟合出机器人在球场上的位置和朝向，放进最近三次的窗口。
- lines: [2, 6]
  title: 三次结果取平均
  body: 窗口满三次、而且都在 0.5 秒之内时，计算平均位置和平均朝向。
- lines: [6, 16]
  title: 一致才接受
  body: 三次位置与平均位置的偏差都不到 4 cm，才接受这组定位；已有位置估计时，还要求新旧位置相差不到 20 cm。随后更新位置，并记下这次看到球门的时刻。
```

{#stage-approach}
### 规划接近

目的是从远处走到球后方的准备站位。站位沿“球到球门”的方向退到球后约 0.22 m，再向一侧偏出，让最后球落在右脚前。远处只需大致走近，位置误差的影响较小，所以可以连续行走、反复重新规划；精确对准留给近距阶段。

Nav2 只负责规划路径：起点是估计的机器人位置，终点是这个站位。地图来自已测量的球场和当前 ToF：球场内部是空闲，边界是障碍，ToF 的有效回波补充沿途的空闲格和前方物体，既不在已知空闲球场内、也没有有效观测的格子保持未知。NavFn 用 A* 在这张图上找路，不穿过未知格，并与障碍保持机器人半径的余量，返回一串地面上的路径点。

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/football/controller.py
language: python
code: |
  path = self.last_plan["path_xy"]
  if not self.last_plan["envelope"]["accepted"] or not path:
      return self._output(t, reason="Nav2_path_not_admitted", gaze=gaze)
  near = min(
      range(len(path)), key=lambda i: math.dist(path[i], self.pose[:2])
  )
  way = next(
      (p for p in path[near:] if math.dist(p, self.pose[:2]) > 0.10), path[-1]
  )
  vec = np.asarray(rotate(-self.pose[2], np.asarray(way) - self.pose[:2]))
steps:
- lines: [0, 3]
  title: 只用被接受的路径
  body: Nav2 返回的路径要通过检查，空路径或未被接受时，这一拍不移动。
- lines: [3, 9]
  title: 选下一个路径点
  body: 先找路径上离机器人最近的点，再往前取第一个距离超过 10 cm 的点作为目标。
- lines: [9, 10]
  title: 求出机身坐标下的参考方向
  body: 把地图中的目标点转换到机身坐标，得到规划层的参考方向。后续调度根据观测与停稳状态，决定实际行走请求或停步，再交给训练好的行走策略执行。
```

走的同时会反复重新规划，并定期确认球门。离球足够近时进入近距细调；路径被拒绝或信息过旧时，机器人停步等待新的观测。课程没有验证动态障碍物避让。

{#stage-refine}
### 近距细调

离球很近时，行走中相机晃动造成的误差已经和需要的精度相当。这一阶段不再调用 Nav2，按球的位置选择短步、短段连续移动或原地转向；每段都有距离和时间上限，结束后停稳，再用新图像测球。球没有完整入画时，身体不动，只转头找球。

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/evaluation/football/neutral_fine_step.py
language: python
code: |
  if self.state == "SETTLE":
      if self.quiet(packet):
          if self.quiet_start is None:
              self.quiet_start = t
      else:
          self.quiet_start = None
      if self.quiet_start is not None and t - self.quiet_start >= 0.5 - 1e-9:
          self.state = "REACQUIRE"
      return self.status(t, memory)
steps:
- lines: [0, 6]
  title: 停下后判断是否静止
  body: '`quiet` 读取 IMU 和关节编码器：机身角速度不超过 0.1 rad/s、关节速度均方根不超过 0.25 rad/s、倾斜不超过 0.35 rad。一旦晃动，计时重新开始。'
- lines: [6, 9]
  title: 静止 0.5 秒后再观察
  body: 连续静止 0.5 秒，才进入重新观察；用这时的新图像决定下一小步，而不是用行走中晃动的旧结果。
```

{#stage-align}
### 对准出脚位置

近距细调的目标，是让球进入踢球技能能处理的小范围，同时让身体朝向球门。出脚之前，用停稳后的最新观测做最后一次检查：

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/onboard/relaxed_gate.py
language: python
code: |
  MARGIN_M=.005
  HEADING_MAX_RAD=math.radians(10.0)
  def exit_box(point):
      p=np.asarray(point,float)
      return bool(p.shape==(3,) and np.isfinite(p).all() and .070<=p[0]<=.110 and -.055<=p[1]<=-.015)
steps:
- lines: [0, 2]
  title: 余量与朝向上限
  body: 在原有出脚范围外留 5 mm 余量；球门方向与身体朝向的偏差不超过 10°。
- lines: [2, 5]
  title: 球在右脚前方的小范围内
  body: 球的位置要有效，且在机身前方 7–11 cm、右侧 1.5–5.5 cm 之间。不满足就拒绝出脚。
```

{#stage-handover}
### 交接给踢球

行走停止，头部回到踢球姿态。等身体静止 0.5 秒后，再确认球仍在上面的出脚范围内、球门观测不超过 0.5 秒、朝向满足要求，才把控制交给踢球技能。条件不满足就拒绝出脚；允许恢复时后退并重新观察，否则停止本次尝试。头部回正必须在 4.5 秒内完成。

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/evaluation/football/head_handoff.py
language: python
code: |
  self.head += np.clip(-self.head, -0.3 * dt, 0.3 * dt)
  if packet is None:
      self.fail("headreturn_public_sensor_unavailable")
      return self.status(now_s)
  gravity = np.asarray(packet["gravity_body"], float)
  gyro = np.asarray(packet["gyro_body_rad_s"], float)
  qd = np.asarray(packet["encoder_qd_rad_s"], float)
  tilt = math.atan2(float(np.linalg.norm(gravity[:2])), -float(gravity[2]))
  quiet = (
      np.max(np.abs(self.head)) <= 1e-9
      and np.linalg.norm(gyro) <= 0.1
      and np.sqrt(np.mean(qd**2)) <= 0.25
      and tilt <= 0.35
  )
  if quiet:
      if self.quiet_start is None:
          self.quiet_start = now_s
  else:
      self.quiet_start = None
steps:
- lines: [0, 4]
  title: 头部平稳回正
  body: 4 个头部关节的偏移每秒最多回收 0.3 rad；读不到传感器数据就放弃这次出脚。
- lines: [4, 14]
  title: 判断是否完全静止
  body: 头部已经回正，机身角速度、关节速度和倾斜都足够小，才算静止。
- lines: [14, 19]
  title: 记下静止开始的时刻
  body: 一旦晃动就重新计时；连续静止 0.5 秒后，才做上面那些出脚前的检查。
```

{#stage-kick}
### 踢球与恢复

踢球技能运行一次，持续 4 秒：官方右脚踢球策略出脚，随后平滑切换到站立。这里的站立动作来自官方行走策略在零速度指令下的输出，不加载额外的踢球修正网络。板载策略不判断是否进球，由单独的评测程序用仿真真值判断。

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/onboard/official_kick_runtime.py
language: python
code: |
  standing = self._infer(0, obs)
  if active:
      kicking = self._infer(1, obs)
      u = np.clip((phase - np.float32(HOLD_S)) / np.float32(BLEND_S), np.float32(0), np.float32(1))
      weight = np.float32(1) - u * u * (np.float32(3) - np.float32(2) * u)
      teacher = weight * kicking + (np.float32(1) - weight) * standing
steps:
- lines: [0, 3]
  title: 两个策略都计算
  body: 每 20 ms 用同一份 61 维观测分别运行站立和踢球两个官方策略。
- lines: [3, 6]
  title: 从踢球平滑过渡到站立
  body: 前 0.14 秒完全采用踢球动作，之后 0.12 秒内按平滑阶跃把权重从踢球移到站立。
```

阶段主要根据观测和身体状态切换：条件不满足，就停下、重新观察或退回前一阶段，所以细调常会重复几次。等待静止的时间和各阶段的时限则保证每次尝试不会无限进行下去。

{#training}
## 哪些经过训练，哪些没有

- **检测器。** 目标是在 960 × 540 的头部相机图像中找出两类物体：足球和球门。课程从一个 YOLO26n 检测器出发，用 Isaac Sim 渲染的球场图像继续训练，画面覆盖几种地面（包括课程的绿色球场）、不同的机位和球的位置。识别框标注由已知的场景几何投影到相机画面自动生成，并用渲染深度剔除被遮挡的部分。512 张用于训练，128 张用于挑选检查点，另外 128 张留作测试，测试集上 mAP50 为 0.978。这是图像检测指标，不是射门成功率。
- **行走。** 官方行走策略由 Pollen Robotics 用强化学习训练，也就是第 3 部分介绍的方法；课程直接使用，权重不变。
- **踢球与恢复。** 官方右脚踢球策略同样是强化学习训练的局部技能，权重不变；它与站立之间的切换时序由程序编写。
- **没有训练的部分。** Nav2 规划器、几何计算、路径跟随和阶段逻辑都是配置或编写出来的。

{#run}
## 在同一个窗口反复射门

先按 [1.2 节](setup.md) 准备好课程容器，再回到容器终端；容器已在运行时，可以在宿主机用 `docker exec -it microduck-workshop bash` 进入。然后运行：

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py football --seconds 60 --output outputs/onboard-football
```

机器人会自动观察、接近足球并尝试射门。把输入法切到英文，点击 Isaac Sim 窗口，按 **R** 就能重新开始。窗口保持打开，每轮从站立初始化开始，起点前后最多偏移 1.5 厘米、左右最多偏移 1 厘米；足球和球门的位置不变。场景重新初始化时稍等片刻，随后任务会自动开始。

建议先让每轮完整结束，再按 R，比较不同起点的表现。也可以在任务中途按 R；这次会单独记为“中断”，不计入完整任务的成功率。关掉 Isaac 窗口后，程序会检查已完成的回合并打印统计。

仿真要同时渲染第三人称视口、头部 RGB 相机和 ToF 等多路画面，运行速度取决于硬件性能，窗口里的动作看起来可能卡顿；任务按仿真时间推进。只想验证流程是否跑通时，可以不开窗口：

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py football --headless --seconds 60 --output outputs/onboard-football-headless
```

不开窗口时只运行一轮，结束后自动评测并打印结果。

| 参数 | 含义 |
|---|---|
| `--seconds 60` | 每轮任务时限，以仿真秒计，不含站立初始化；完成或失败会提前结束 |
| `--output` | 一个新目录；窗口模式下各轮记录分别保存在 `episode-001`、`episode-002` 等子目录 |
| `--headless` | 不打开窗口，运行一轮并评测 |
| `--single-run` | 打开窗口但只运行一轮，结束后关闭窗口并评测 |

:::{enrichment} 代码关键参数
- **重置。** `lab/onboard/session.py` 处理 R 按键，在同一个 Isaac Sim 进程中重建每轮环境；传感器、位置估计、动作历史和评分从头开始，旧结果保留。
- **初始化。** 关节暂时锁定在 HOME 姿态，身体自由落稳；传感器就绪且取得第一次真实策略输出后，在应用输出之前解除锁定。
- **频率。** `lab/onboard/executor.py` 使用 Newton 100 Hz 物理、50 Hz 关节策略、15 Hz RGB/ToF。每个策略动作持续两个 10 ms 物理步。
- **场景。** `lab/onboard_scene.py` 创建绿色球场、球门与上述初始化。
- **行走与踢球。** 行走和恢复使用官方 `models/walk.onnx` 的运行时副本 `velstand.onnx`；出脚使用 `models/kick-teacher.onnx` 的副本 `ball_kick_right.onnx`。
- **检测与定位。** `run.py` 核对 `models/scene-detector/deployment.json` 的哈希，再把检测器交给板载策略；`lab/scene_goal_points.py` 在识别框内寻找门柱底点。
- **Nav2。** `lab/scene_nav2/nav2.yaml` 配置 NavFn（A*，不穿过未知区域）与 2.5 厘米代价地图；`map_contract.py` 由已测量球场和当前 ToF 建图并检查路径。
:::

:::{course-image} media/isaac-ui-onboard-sensors.png
:alt: 板载射门的 Isaac Sim 完整界面，同时显示绿色球场、头部 RGB 图像和 8×8 ToF 距离。

左侧观察机器人，右侧对照它此刻看到的图像与距离：识别框是否稳定地框住球和球门，脚前的 ToF 读数是否随接近而变小。NR 表示没有回波，不能据此认为前方没有障碍。
:::

{#results}
## 实测结果

用上面的窗口配置连续运行 20 轮，轮与轮之间按 R，起点在前后 ±1.5 厘米、左右 ±1 厘米内小幅变化。完全成功要求脚触球、整球进门、没有摔倒，并恢复站立；未进球的完整回合也计入分母。

| 回合 | 完全成功 | 摔倒 |
|---|---|---|
| 1–5 | 5 / 5 | 0 / 5 |
| 6–10 | 4 / 5 | 0 / 5 |
| 11–15 | 4 / 5 | 0 / 5 |
| 16–20 | 4 / 5 | 0 / 5 |

这组实测为 **17/20（85%）**，摔倒 **0/20**，技术错误 0 次，没有剔除失败回合。<a href="_static/data/onboard-reset-evaluation.json">查看逐轮记录</a>。全部使用同一个球场和相同配置，这是小范围起点扰动下的样本，不是一般场景的成功率；请继续记录你自己的结果。

失败时先看停止在哪个阶段。`summary.json` 汇总各轮结果，单轮 `manifest.json` 的 `terminal_reason` 和失败细节说明停止原因。识别框仍在，并不表示几何位置已经通过检查；近距位置不够新鲜时，策略仍可能停步重新观察或结束这次尝试。也要检查路径是否可行、出脚条件是否满足，以及触球后是否恢复平衡。

{#design}
## 人工设计的边界

这套板载策略的每个阶段、每个切换条件和阈值，都是围绕“向这个球门射门”人工写出来的。它清楚、容易排查，但只适合这一个任务。换成“把球带到中圈停下”或“走到球旁但不要碰球”，感知大多可以复用，阶段、条件和阈值却要重新设计和调试。

如果模型推理可以离开机器人，放到 GPU 工作站上，或以更多网络延迟为代价放到云端服务器上，就能使用大得多的视觉-语言-动作模型：它从语言指令和示范中学习，而不是依靠手写的阶段，快速控制仍由已训练的运动技能负责。第 5 部分介绍 GR00T 怎样走这条路线。

::::{quiz}
板载射门中，哪些部分是从数据中学到的，哪些是人编写或配置的？

:::{answer}
检测器、官方行走与踢球策略经过训练。几何计算、Nav2 路径规划、路径跟随和阶段逻辑是配置或编写的。
:::
::::

::::{quiz}
想让 Microduck 把球带到中圈停下，现有的板载策略能直接复用吗？

:::{answer}
感知部分大多可以复用，但站位、阶段顺序、切换条件和交接都是为射门设计的，需要重新设计和调试。为每个新任务手工重写流程，正是第 5 部分转向从示范中学习的原因。
:::
::::

{#references}
## 参考资料

{.references}
- [Isaac Lab ↗](https://developer.nvidia.com/isaac/lab)
- [Nav2：配置 NavFn 规划器 ↗](https://docs.nav2.org/jazzy/configuration_and_development/configuration_guide/planners_plugins/configuring_navfn/)
- [Nav2 NavFn 规划器源码 ↗](https://github.com/ros-navigation/navigation2/blob/main/nav2_navfn_planner/README.md)
- [Microduck 开源项目 ↗](https://github.com/pollen-robotics/microduck)
- [Microduck 主控板系统镜像（microduck_yocto）↗](https://github.com/pollen-robotics/microduck_yocto)
