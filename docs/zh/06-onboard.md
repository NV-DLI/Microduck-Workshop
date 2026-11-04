---
myst:
  html_meta:
    description: 人工设计的板载策略怎样复用传感器估计完成射门：为什么采用板载策略，怎样沿直线接近足球并通过反馈调整位置，以及已训练的运动技能怎样由人工设计的阶段连接起来，每个阶段的核心代码，以及怎样在 Isaac Sim 中反复射门。
---

# 4.2 设计射门阶段

:::{lesson-intro}
:meta: 第 4.2 节 · 预计学习时长 35 分钟

[4.1 节](06-perception.md)把传感器读数变成了球和球门的位置。本节介绍人工设计的板载策略怎样用这些位置把直线接近、位置调整和踢球连接起来，阅读每个阶段的核心代码，并在 Isaac Sim 中运行它反复射门。
:::

{#why}
## 为什么采用板载策略

Microduck 使用基于 Rockchip RK3566 的主控板，其中的小型神经网络加速器（NPU）为单核 INT8，算力 0.8 TOPS；运动策略以 50 Hz 循环运行（[官方 README ↗](https://github.com/pollen-robotics/microduck)，[NPU 说明 ↗](https://github.com/pollen-robotics/microduck/blob/main/docs/project/npu-bringup.md)）。这样的算力难以承担 GR00T 这类约 30 亿参数的视觉-语言-动作模型（VLA）的低延迟推理。因此板载策略选择小型检测器，配合几何计算、直线接近和已训练的行走与踢球技能。

{#pipeline}
## 从传感器到动作：沿直线接近足球

**Isaac Lab**（基于 Isaac Sim，物理引擎为 Newton）仿真机器人、传感器和球场，并运行学到的策略。这里使用没有障碍物的球场：根据 RGB、ToF 和关节数据估计球的位置，在球后选一个朝向球门的准备站位，再把当前位置与站位连成直线。机器人沿这个方向接近，行走过程中持续用观测和位置记忆修正速度，接近足球时减速并调整。

| 组成部分 | 做什么 |
|---|---|
| YOLO26n 检测器 | 在 RGB 图像中用识别框标出球和球门 |
| 几何计算 | 由识别框、ToF 地面和关节角算出球与球门的位置 |
| 直线接近 | 连接当前位置与球后准备站位，检查直线路段是否可通行 |
| 反馈控制与阶段 | 根据球的位置和身体朝向更新行走速度；按条件切换阶段 |
| 行走策略 | 把速度请求变成 14 个关节目标，50 Hz |
| 踢球与恢复 | 出脚击球，再回到站立 |

直线提供接近的参考方向，反馈控制负责在行走中修正偏差。高层输出前进、侧移和转向的速度请求，平衡和迈步交给行走策略完成。

{#stages}
## 射门的六个阶段

设计阶段时，可以从成功条件倒推。踢球技能只在很小的范围内有效：球要在右脚前方，身体要朝向球门，机器人要先站稳。为了满足这些前提，需要先走到球后方的站位，再精确调整位置和朝向；而这一切的前提，是先知道球和球门在哪里。

再考虑传感器的不确定性。行走时相机晃动，所以用最近一次可靠的球位置，加上关节编码器和 IMU 估计的身体运动，持续更新球相对机器人的位置。估计可靠、位置和朝向合适时，可以连续走近；需要重新确认时才停步观察。踢球必须从静止站姿开始，所以单独设一个交接阶段。

```{learning-flow}
label: 板载策略的步行射门阶段
nodes:
- [寻找目标, 转头观察球门与足球]
- [直线接近, 连到球后站位，边走边修正]
- [近距细调, 减速，必要时停步或后退调整]
- [对准出脚位置, 球在右脚前，身体朝向球门]
- [交接给踢球, 停步，头部回正，身体静止]
- [踢球与恢复, 踢球技能出脚，回到站立]
details:
- 原地转头，确定机器人相对球门的位置，并找到足球。
- 连接当前位置与球后准备站位，持续更新速度请求，沿直线方向接近。
- 接近足球时减速；偏差较大时，停稳重看，或在后方可通行时后退转向，再接近。
- 检查球是否在右脚前方、身体是否朝向球门；近处看不到球时，还可使用连续更新的位置记忆判断到位。
- 停步，头部回正，确认静止和到位条件后交给踢球技能。
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
### 直线接近

目的是从远处走到球后方的准备站位。站位沿“球到球门”的方向退到球后，再向一侧偏出，让最后球落在右脚前。把估计的当前位置与这个站位连起来，就得到空球场中的直线接近方向。

代码先沿直线取点，再结合已知球场边界和当前 ToF 检查可通行范围。检查通过后开始行走。随着身体移动，球相对机器人的位置也在变化；控制器不断更新前进、侧移和转向请求，尽量保持合适的接近方向。

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/onboard/direct_path.py
language: python
code: |
  snap=snapshot(case,provenance)
  a,b=snap['pose'][:2],snap['goal']
  distance=math.dist(a,b)
  n=max(1,math.ceil(distance/.02))
  points=[[a[0]+(b[0]-a[0])*k/n,a[1]+(b[1]-a[1])*k/n] for k in range(n+1)]
  gate=check_path(points,snap)
steps:
- lines: [0, 3]
  title: 确定起点与终点
  body: 从传感器估计的状态中取当前位置和球后准备站位，计算二者的直线距离。
- lines: [3, 5]
  title: 沿直线取点
  body: 将这段直线分成不超过 2 cm 的小段，用于检查整条接近路段。
- lines: [5, 6]
  title: 检查可通行范围
  body: 结合球场边界和当前 ToF 检查这些点及机器人所需的空间；通过后才允许接近。
```

这条直线是行走的参考，不是一开始算好距离后就不再观察。传感器持续采集，位置记忆持续更新；接近足球时，如果位置和朝向满足条件，就连续转入近距细调。通行检查不通过或信息过旧时，先停步重新观察。

{#stage-refine}
### 近距细调

离球很近时，要同时考虑距离、左右偏差和身体朝向。控制器根据最近一小段球位变化估计接近速度，接近得越快就越早开始减速。偏差较大时，停稳后重新观察；需要较大转向且后方可通行时，可以边后退边转向，再向前接近。这样给双足行走留出调整空间。

停车观察只在需要时发生。下面的代码让身体连续静止 0.5 秒后，再使用新图像决定下一步。

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
  title: 静止 0.5 秒后进入重新观察阶段
  body: 传感器持续采集；需要停车重看时，连续静止 0.5 秒才进入重新观察阶段，用这时的新图像决定下一小步。
```

{#stage-align}
### 对准出脚位置

近距细调的目标，是让球进入右脚前方的出脚位置，同时让身体朝向球门。球还看得见时，用停稳后的观测检查位置。若球已经很近，低头仍看不到完整的球，就检查之前的可靠球位及其后连续更新的位置记忆：估计已到位、方向合适、身体与地面检查通过时，直接进入踢球交接，不再反复低头寻找。

下面是这一近距到位判断的位置范围；它还需要同时通过观测时效、连续运动估计和身体状态检查。仅仅“看不到球”不会触发出脚。

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/onboard/blind_arrival.py
language: python
code: |
  def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
  def vec(v,n):return isinstance(v,(list,tuple)) and len(v)==n and all(finite(x) for x in v)
  def inside(point):return vec(point,3) and .07<=point[0]<=.15 and -.10<=point[1]<=.01
steps:
- lines: [0, 2]
  title: 位置估计必须有效
  body: 需要三维有限数值；缺失的观测不能当成零距离。
- lines: [2, 3]
  title: 判断是否已到脚前
  body: 连续更新的球位估计在机身前方 7–15 cm、横向右侧 10 cm 到左侧 1 cm 之间，才满足这一到位条件；还需通过朝向等检查，再停稳交接。
```

{#stage-handover}
### 交接给踢球

行走停止，头部回到踢球姿态。等身体连续静止 0.5 秒后，再确认球的位置估计、球门方向和身体状态仍满足对应的到位条件，才把控制交给踢球技能。条件不满足时先停步，重新观察和对准；后退恢复也要先检查后方是否可通行。

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

踢球技能运行一次：官方右脚踢球策略出脚，随后平滑切换到站立，并继续观察球是否停稳。这里的站立动作来自官方行走策略在零速度指令下的输出，不加载额外的踢球修正网络。板载策略不判断是否进球，由单独的评测程序用仿真真值判断。

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
  title: 踢球期间运行两个策略
  body: 踢球已获授权的窗口内，每 20 ms 用同一份 61 维观测分别运行站立和踢球两个官方策略；窗口结束后只运行站立策略。
- lines: [3, 6]
  title: 从踢球平滑过渡到站立
  body: 前 0.14 秒完全采用踢球动作，之后 0.12 秒内按平滑阶跃把权重从踢球移到站立。
```

阶段主要根据观测和身体状态切换：条件不满足，就停下、重新观察或退回前一阶段，所以细调常会重复几次。局部动作有时间上限，整轮实验则由您按 R 重置。

{#training}
## 哪些经过训练，哪些没有

- **检测器。** 目标是在 960 × 540 的头部相机图像中找出两类物体：足球和球门。课程从一个 YOLO26n 检测器出发，用 Isaac Sim 渲染的球场图像继续训练，画面覆盖几种地面（包括课程的绿色球场）、不同的机位和球的位置。识别框标注由已知的场景几何投影到相机画面自动生成，并用渲染深度剔除被遮挡的部分。512 张用于训练，128 张用于挑选检查点，另外 128 张留作测试。
- **行走。** 官方行走策略由 Pollen Robotics 用强化学习训练，也就是第 3 部分介绍的方法；课程直接使用，权重不变。
- **踢球与恢复。** 官方右脚踢球策略同样是强化学习训练的局部技能，权重不变；这里实现了它与站立之间切换的时序逻辑。
- **没有训练的部分。** 几何计算、直线生成、速度反馈、减速和阶段逻辑都是编写出来的。

{#run}
## 射门！

先按 [1.2 节](setup.md) 准备好课程容器，再回到容器终端；容器已在运行时，可以在宿主机用 `docker exec -it microduck-workshop bash` 进入。然后运行：

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py football --output outputs/onboard-football
```

| 参数 | 含义 |
|---|---|
| `--output` | 一个新目录；各轮记录分别保存在 `episode-001`、`episode-002` 等子目录 |

机器人会自动观察、接近足球并尝试射门。第一轮从名义起点出发。点击 Isaac Sim 窗口，按 **R** 就能重新开始：窗口保持打开，每轮都从站立初始化开始，起点随机偏移，前后、左右各最多 5 厘米，朝向顺时针或逆时针最多偏转 15°；足球和球门的位置不变。场景重新初始化时稍等片刻，随后任务会自动开始。

学生实验不设总时限，进球、未进球或暂时无法出脚时都不会自动暂停仿真。传感器持续刷新；按 **R** 才会重置当前尝试。关闭 Isaac 应用，或在启动它的终端按 **Ctrl+C**，即可停止程序。感知、对准或交接失败只会暂停当前动作，并尝试重新观察。

仿真要同时渲染第三人称视口、头部 RGB 相机和 ToF 等多路画面，运行速度取决于硬件性能，窗口里的动作看起来可能卡顿；任务按仿真时间推进。

:::{course-image} media/isaac-ui-onboard-sensors.png
:alt: 板载射门的 Isaac Sim 完整界面，同时显示绿色球场、头部 RGB 图像和 8×8 ToF 距离。

左侧观察机器人，右侧对照它此刻看到的图像与距离：识别框是否稳定地框住球和球门，脚前的 ToF 读数是否随接近而变小。NR 表示没有回波，不能据此认为前方没有障碍。
:::

`summary.json` 汇总已保存的各轮记录；单轮 `manifest.json` 的 `navigation_retries` 记录中途重新观察的原因。识别框仍在，并不表示几何位置已经通过检查；近距位置不够新鲜时，策略会停步重新观察。也要检查接近时的左右偏差和朝向、出脚条件是否满足，以及触球后是否恢复平衡。

按 R 重置时，程序先保存当前记录，再开始下一轮。

{#design}
## 人工设计的边界

这套板载策略的每个阶段、每个切换条件和阈值，都是围绕“向这个球门射门”人工写出来的。它清楚、容易排查，但只适合这一个任务。换成“把球带到中圈停下”或“走到球旁但不要碰球”，感知大多可以复用，阶段、条件和阈值却要重新设计和调试。

如果模型推理可以离开机器人，放到 GPU 工作站上，或以更多网络延迟为代价放到云端服务器上，就能使用大得多的 VLA：它从语言指令和示范中学习，而不是依靠手写的阶段，快速控制仍由已训练的运动技能负责。第 5 部分介绍 GR00T 怎样走这条路线。

::::{quiz}
板载射门中，哪些部分是从数据中学到的，哪些是人编写或配置的？

:::{answer}
检测器、官方行走与踢球策略经过训练。几何计算、直线生成、速度反馈、减速和阶段逻辑是编写的。
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
- [Microduck 开源项目 ↗](https://github.com/pollen-robotics/microduck)
- [Microduck 主控板系统镜像（microduck_yocto）↗](https://github.com/pollen-robotics/microduck_yocto)
