---
myst:
  html_meta:
    description: 理解域随机化为什么能帮助策略适应真实差异，然后在 Isaac Lab 中启动行走训练、观察并行环境，再测试训练好的策略。
---

# 3.2 域随机化：训练并测试行走策略

:::{lesson-intro}
:meta: 第 3.2 节 · 操作实验 · 预计学习时长 45 分钟（观察训练几分钟即可；完整训练约 2 小时，可选）

理解域随机化为什么能帮助策略适应真实差异，然后在 Isaac Lab 中启动行走训练、观察上千个并行环境，再测试训练好的策略。
:::

```{symbols}
J_dr:
  tex: J(\theta)
  text: 域随机化下的学习目标：在许多组随机物理参数上平均后的期望折扣回报。
theta:
  tex: \theta
  text: 策略网络的参数（权重），训练时更新。
E_xi:
  tex: \mathbb E_{\xi\sim p(\xi)}
  text: 对随机抽取的物理参数组取平均。
xi:
  tex: \xi
  text: 一组仿真参数，例如脚底摩擦、躯干质量、关节摩擦和舵机延迟。
p_xi:
  tex: p(\xi)
  text: 参数的随机分布：每个参数在设定范围内抽样。
E_traj:
  tex: \mathbb E_{\tau\sim\pi_\theta,\,\xi}
  text: 在参数 $\xi$ 构成的仿真中，按当前策略采样得到的轨迹取平均。
pi_theta:
  tex: \pi_\theta
  text: 策略：根据观测给出动作分布的神经网络。
gamma:
  tex: \gamma
  text: 折扣因子，课程训练取 0.99。
t_idx:
  tex: t
  text: 控制步的序号。
T_horizon:
  tex: T
  text: 一个回合中计入回报的控制步数。
r_t:
  tex: r_t
  text: 当前控制步的即时奖励。
```

{#reality-gap}
## 为什么需要域随机化

3.1 节的强化学习完全在仿真里进行。仿真中每个参数都是一个确定的数：脚底摩擦系数、躯干质量、关节摩擦、舵机从收到目标到开始转动的延迟……而真实的 Microduck 各不相同：不同批次的舵机摩擦不一样，电池电量会改变舵机输出，编码器零位有微小偏差，地面也有软有硬。如果策略只在一组“完美”参数下练习，它可能学会依赖这组参数的细节，换一台机器人或换一块地面就站不稳。这种仿真与现实之间的差距，常被称为 sim-to-real 差距。

域随机化（Domain Randomization）的思路很直接：既然不知道真实参数究竟是多少，就在训练时让每个并行环境各自抽取一组不同的参数。策略同时面对上千台“略有不同的机器人”，只有学会对这些差异都有效的动作，才能在整体上获得高回报。

```{formula} 域随机化下的学习目标
\symx{J_dr}{J(\sym{theta})}=\sym{E_xi}\,\sym{E_traj}\!\left[\sum_{\sym{t_idx}=0}^{\sym{T_horizon}-1}\sym{gamma}^{\sym{t_idx}}\sym{r_t}\right]
```

与 3.1 节的目标相比，这里多了一层平均：先按分布 $\sym{p_xi}$ 抽取一组物理参数 $\sym{xi}$，再在这组参数下按策略 $\sym{pi_theta}$ 采样轨迹并计算折扣回报。PPO 的更新方式不变，变化的是数据：同一批样本里混合了许多种物理条件。

范围的选择是一种权衡。范围太窄，策略仍然只适应少数情形；范围太宽，任务可能变得过难，策略学会的是保守、迟缓的动作，甚至学不会。范围最好以测量或硬件规格为依据，例如舵机延迟的实测区间、电池电压的工作范围。域随机化能提高鲁棒性，但不能替代真机验证：它只覆盖你想到并写进配置的那些差异。

{#what-varies}
## 课程训练实际改变了什么

Isaac Lab 用事件管理器实现随机化。每个事件有一个触发时机：`startup` 在创建环境时抽样一次，之后保持不变；`reset` 在每个环境重置时重新抽样；`interval` 在回合进行中按时间间隔触发。下表来自完整训练保存的 `env.yaml`，是实际生效的数值。

| 随机化项目 | 范围 | 触发时机 | 模拟的真实差异 |
|---|---|---|---|
| 脚底静摩擦与动摩擦 | 0.7–1.3 | startup | 不同地面和鞋底材料 |
| 编码器零位偏差 | ±0.015 rad | startup | 装配与校准误差 |
| 躯干质量 | 名义值 × 0.95–1.05 | startup | 电池、线缆和外壳差异 |
| 躯干与头部重心 | 各方向 ±3 mm | reset | 质量分布不均 |
| 关节摩擦 | 名义值 × 0.9–1.1 | reset | 舵机个体差异与磨损 |
| 关节转动惯量（电枢） | 名义值 × 0.9–1.1 | reset | 减速齿轮与电机差异 |
| 外部推力 | 每 3–6 s 一次，水平速度突变 ±0.3 m/s | interval | 碰撞、被人推一下 |
| 舵机延迟 | 3–6 个物理步（15–30 ms） | 执行器模型 | 通信与控制延迟 |
| 舵机供电电压 | 6.5–8.2 V，另有随负载的电压跌落 | 执行器模型 | 电池电量变化 |
| 观测噪声 | 开启 | 每一步 | 传感器噪声 |
| 起始位置与朝向 | ±0.5 m，朝向任意 | reset | 从不同位置开始行走 |

```{code-story}
label: Python · 核心代码节选
source: https://github.com/isaac-sim/IsaacLab/blob/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/flat_env_cfg.py
language: python
code: |
  foot_friction = EventTerm(
      func=mdp.randomize_rigid_body_material,
      mode="startup",
      params={
          "asset_cfg": SceneEntityCfg("robot", body_names=MICRODUCK_FOOT_BODY_NAMES),
          "static_friction_range": (0.7, 1.3),
          "dynamic_friction_range": (0.7, 1.3),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 64,
      },
  )
  encoder_bias = EventTerm(func=mdp.randomize_encoder_bias, mode="startup", params={"bias_range": (-0.015, 0.015)})
  randomize_joint_friction = EventTerm(
      func=mdp.randomize_bam_friction, mode="reset", params={"scale_range": (0.9, 1.1)}
  )
  push_robot = EventTerm(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(3.0, 6.0),
      params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
  )
steps:
- lines: [0, 11]
  title: 脚底摩擦在创建环境时抽样
  body: '`mode="startup"` 表示每个环境开始时抽一次。静摩擦和动摩擦都在 0.7–1.3 之间，`num_buckets` 把取值分成 64 档，让上千个环境共用少量材料。'
- lines: [11, 12]
  title: 编码器零位偏差
  body: 每个关节读数加上 ±0.015 rad 以内的固定偏差，模拟装配和校准误差；策略看到的关节角因此与真实角度略有出入。
- lines: [12, 15]
  title: 每次重置时改变关节摩擦
  body: '`mode="reset"` 让环境每次重新开始时都换一组摩擦，同一个环境在训练中也会经历不同的舵机特性。'
- lines: [15, 21]
  title: 行走中随机推一下
  body: '`interval` 事件每隔 3–6 秒把机身水平速度突然改变最多 0.3 m/s，策略必须学会在受扰后恢复平衡。'
```

节选来自固定版本 Isaac Lab 的 Microduck 任务配置。注意：本节的测试和演示关闭了推力与参数随机化，在名义条件下比较策略；随机化只在训练中使用。

{#train}
## 启动完整训练，观察并行训练

在 1.3 节准备的课程容器终端里运行下面的命令（需要重新进入容器时，见 <a href="setup.html#start">1.3 节</a>）。它从随机初始化开始训练，不读取任何已有模型的权重；`--show` 打开 Isaac Sim 窗口，让你看到上千只小鸭子在各自随机化的环境里同时练习。`--output` 请使用尚不存在的新目录。

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py train-walk \
  --num-envs 1024 \
  --iterations 2500 \
  --show \
  --output outputs/walking-training-01
```

| 参数 | 含义 |
|---|---|
| `--num-envs 1024` | 同时在 GPU 上运行的环境数；每个环境各有一组随机参数 |
| `--iterations 2500` | PPO 采样与更新的循环次数，3.1 节已计算过对应的样本量 |
| `--show` | 打开 Isaac Sim 窗口显示并行环境；渲染会占用显存并让每轮训练变慢 |
| `--output` | 检查点、配置、TensorBoard 日志和导出模型的保存位置 |

:::{enrichment} 代码关键参数
- **起点。** 训练从随机初始化开始，不读取任何已有模型。
- **场景与任务。** 机器人模型是 `lab/assets/microduck_contact.usda`（全身碰撞模型）；训练任务、奖励和随机化来自容器启动时整理到 `/workspace/state` 的运行时代码。
- **输出。** 检查点（每 250 轮一个）、`env.yaml` 与 `agent.yaml`、TensorBoard 日志和导出的模型都写入 `outputs/walking-training-01`，也就是宿主机仓库的 `outputs/` 下。
:::

**完整训练需要多长时间？** 课程随附的预训练结果是在 DGX Spark 上不开窗口训练的：1024 个环境 × 2500 轮，训练阶段 7246 秒（约 2 小时），平均每 100 轮约 4.8 分钟，测量时 GPU 上还有其他任务。那次训练用的是官方原始碰撞模型，而上面的命令使用全身碰撞模型；打开 `--show` 时渲染也会拖慢训练，所以你的耗时会不同。

**你不需要等它训练完。** 窗口出现、终端开始逐轮打印回报和回合长度后，观察几分钟：前几十轮里大部分小鸭子很快摔倒、被单独重置，随着迭代推进，站得住的越来越多。

看够了就按 Ctrl+C 停止，容器仍在运行。已保存的检查点（每 250 轮一个）会保留，但只有训练跑完才会导出 `exported/policy.onnx`。下面默认测试课程预先训练好的策略；想让自己的训练跑完时，可以去掉 `--show` 加快速度，完成后再显式换成自己的模型。

:::{course-image} media/isaac-ui-ppo-training.png
:alt: 训练时打开 --show 的 Isaac Sim 完整界面：白色地面上许多 Microduck 副本处在不同的步态阶段，有的站立、有的迈步、有的正在倾倒，顶部显示 RTX 实时渲染与 Newton MJWarp 物理。

`--show` 打开的训练窗口，来自一次真实的训练运行，只展示训练刚开始的样子；你的命令使用 1024 个环境，窗口里的小鸭子会更多、更密。
:::

{#watch}
## 训练耗时与结果判断

课程仓库随附了这次完整训练的检查点、导出模型、配置和 TensorBoard 日志（`datasets/walking-ppo` 约 12 MB）。下图展示这份日志的 TensorBoard 曲线，绿框编号对应下方说明。

:::{course-image} media/tensorboard-walk-full.png
:alt: TensorBoard 标量页面，绿框标出四张曲线：1 平均回合长度，2 平均回合回报，3 线速度跟踪误差，4 转向角速度误差。横轴为训练迭代次数。
:no-zoom-label:

完整训练（1024 个环境 × 2500 轮）的真实 TensorBoard 记录。曲线经过 TensorBoard 默认的 0.6 平滑，浅色细线是原始值。
:::

按编号阅读：

1. **平均回合长度**从约 23 步升到约 870 步（一个回合最长 1000 步，即 20 秒）：机器人从“几乎立刻摔倒”变成大部分时间都能站住、走完回合。这是判断平衡是否学会的第一个信号。
2. **平均回合回报**在前 800 轮迅速上升，之后在 80 附近波动，不再明显增长。回报停止增长不等于技能完美，只说明继续用同样的设置训练收益变小。
3. **线速度跟踪误差**从约 0.38 m/s 降到约 0.22 m/s：前进与侧移的指令跟随有改善，但误差仍不小。
4. **转向角速度误差**从约 1.9 rad/s 降到约 0.7 rad/s，仍然偏大：这是一个提醒，转向可能没有学好，需要专门测试。

```{training-chart}
data: walk-full-training-scalars.json
title: 1024 个环境 · 2500 轮完整训练
badge: 训练阶段 7246 秒
metrics:
- key: Train/mean_episode_length
  label: 平均回合长度 / 控制步
- key: Train/mean_reward
  label: 平均回合回报
- key: Metrics/base_velocity/error_vel_xy
  label: 线速度跟踪误差 / (m/s)
- key: Metrics/base_velocity/error_vel_yaw
  label: 转向角速度误差 / (rad/s)
ticks: [1, 500, 1000, 1500, 2000, 2500]
x_label: 训练迭代次数
iteration_label: 迭代
caption: 每 25 轮取一个原始记录值（未平滑），数据来自同一次训练的 TensorBoard 事件文件，可以悬停查看具体数值。
```

曲线变好只说明训练目标在改善，机器人能否按指令行走要在独立测试中检查。课程用固定指令测试了这次训练导出的策略：每种指令 4 个环境、20 秒、关闭随机化。

| 指令 | 未跌倒 | 实际表现 |
|---|---|---|
| 前进 0.1 m/s | 4/4 | 能跟上指令，平均速度误差约 0.056 m/s |
| 侧移 0.06 m/s | 4/4 | 能站稳，但横向移动明显不足 |
| 左转 0.4 rad/s | 4/4 | 几乎不转，转向误差约 0.39 rad/s |

12 次测试都没有跌倒，前进能跟上指令，侧移和转向都没有做到，正好印证了上面第 4 张曲线的提醒：2500 轮、从随机初始化开始的训练学会了平衡和前进，但还没有学会按指令转向。官方模型来自 Pollen Robotics 的独立训练流程，后续各节继续把它作为固定参考。

| 观察到的现象 | 先检查什么 | 下一步验证 |
|---|---|---|
| 回报提高，仍然频繁跌倒 | 各奖励项是否相互掩盖 | 在固定指令下运行，记录跌倒时间 |
| 回合变长，但机器人几乎不移动 | 速度跟踪项与站姿奖励 | 比较实际速度与目标速度 |
| 前进可以，转向不行 | 转向奖励权重与训练长度 | 单独测试转向指令 |
| 损失或动作出现 NaN | 数值状态、配置和运行日志 | 保留失败输出，先修复问题再训练 |

训练太短时差距更明显：只训练几十轮时回报已经开始上升，导出的策略却往往不到一秒就跌倒。流程能跑通，不等于策略学会了技能。

:::{enrichment} 附录：自己打开 TensorBoard
TensorBoard 会读取训练输出目录里的事件文件。在第二个容器终端（宿主机运行 `docker exec -it microduck-workshop bash`）里运行 `python3 scripts/run.py tensorboard --logdir datasets/walking-ppo --port 6006`，再在宿主机浏览器打开终端显示的地址；容器使用宿主机网络。把 `--logdir` 换成 `outputs/walking-training-01`，就能看自己那次训练（包括中途停止的训练）已经记录下的曲线。搜索框输入 `Train/mean_(reward|episode_length)$|error_vel_(xy|yaw)` 可以筛出上面这四张图。
:::

{#test}
## 测试训练好的策略

下面的命令把课程预先训练好的策略（上面那次完整训练导出的 ONNX）放回 2.2 节同样的场景测试。`--model` 显式指定要测试的文件；不写这个参数的课程命令仍然使用官方模型。

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py walk \
  --model datasets/walking-ppo/exported/policy.onnx \
  --velocity 0.25 0 0 \
  --seconds 12 \
  --seed 202 \
  --output outputs/ppo-walk-test-01
```

:::{enrichment} 代码关键参数
- **`--model`。** 读取容器内 `/workspace/Microduck-Workshop/datasets/walking-ppo/exported/policy.onnx`，也就是宿主机仓库 `datasets/walking-ppo/` 里课程预先训练好的策略（只读）。这条命令只读取 `--model` 指定的这一个文件。
- **`--velocity 0.25 0 0`。** 前进、侧移和转向的速度指令；程序先发约 2 秒零速度，再持续发送这条指令。
- **`--seed`。** 与 2.2 节的录制示例取同一个值，两次结果可以直接对照。
- **起步。** 关节先临时锁在默认站姿，机器人在地面上自然落稳（机身不受外力支撑）；策略算出第一个动作后、执行之前再松开，避免从空中掉落造成的额外晃动。
- **输出。** `report.json` 和每 0.1 秒一条位置记录的 `trajectory.json` 写入 `--output` 指定的目录。
:::

:::{course-image} media/isaac-ui-ppo-walk-test.png
:alt: 用上面的命令测试预训练 PPO 策略时的 Isaac Sim 完整界面：视口中一只 Microduck 在平地上迈步，顶部显示 RTX 实时渲染与 Newton MJWarp 物理。

课程在容器中运行上面这条命令时的 Isaac Sim 窗口。
:::

设置与 2.2 节的录制示例相同，可以直接对照：先看有没有跌倒，再比较 `trajectory.json` 中的距离和方向。课程运行的那一次 12 秒内没有跌倒，前进指令期间走了约 1.11 m（平均约 0.11 m/s），但向右偏，整段约 26°、最后 2 秒约 37°；2.2 节的官方模型速度相近，却向左偏约 8°（最后 2 秒约 15°）。两者都能走，却都没有准确跟上指令。再把 `--velocity` 改成 `0 0 0.4` 测试转向，看看是否“会走不会转”。

**测试你自己的策略。** 只有当你的训练完整跑完、`outputs/walking-training-01/exported/policy.onnx` 已经生成时，才把命令里的 `--model` 手动换成这个路径，并换一个新的 `--output` 目录。一次成功的运行只是一个样本；要下结论，应换几组指令、从不同起点多次重复测试，并保留失败的结果。

::::{quiz}
为什么测试时要关闭训练中使用的随机推力和参数随机化？

:::{answer}
本节先关闭随机推力与参数随机化，在名义条件下比较策略的基本响应。评估鲁棒性时，应另外预先固定随机分布和测试种子，让不同策略经历同一组参数与扰动，并统计全部结果。随机化并不等于不可复现。
:::
::::

{#next}
## 下一步：从平地走向坡地

域随机化让策略面对参数的变化，但地面仍然是平的。即使一个策略在平地上走得很好，遇到坡道时，原来的抬脚高度、落脚位置和重心调整还够用吗？3.3 节先讨论奖励应该如何设计，再把随机化扩展到地形，让步行走稳上下坡。

{#references}
## 参考资料

{.references}
- [Isaac Lab：Microduck 任务配置 ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
- [Tobin 等：用于 sim-to-real 迁移的域随机化 ↗](https://arxiv.org/abs/1703.06907)
- [PPO 原论文 ↗](https://arxiv.org/abs/1707.06347)
