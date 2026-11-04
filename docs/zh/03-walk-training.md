---
myst:
  html_meta:
    description: 理解域随机化的作用、类别和本次训练的实际配置，阅读奖励与 PPO 的核心代码，在 Isaac Lab 中启动行走训练、观察并行环境，再测试训练好的策略。
---

# 3.3 训练并测试行走策略

:::{lesson-intro}
:meta: 第 3.3 节 · 操作实验 · 预计学习时长 45 分钟（观察训练几分钟即可；完整训练约 2 小时，可选）

先理解域随机化：它是什么、有哪些类别、为什么行走需要它、本次训练实际随机化了哪些量。再阅读奖励与 PPO 的核心代码，在 Isaac Lab 中启动行走训练、观察上千个并行环境，最后测试训练好的策略。
:::

```{symbols}
J_dr:
  tex: J(\theta)
  text: 域随机化下的学习目标：在许多组随机仿真参数上平均后的期望折扣回报。
theta:
  tex: \theta
  text: 策略网络的参数（权重），训练时更新。
E_xi:
  tex: \mathbb E_{\xi\sim p(\xi)}
  text: 对随机抽取的参数组取平均。
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
  text: 当前控制步的即时奖励：各奖励项加权求和后的值。
r_v:
  tex: r_t^{v}
  text: 当前控制步的线速度跟踪奖励，取值在 0 到 1 之间。
exp:
  tex: \exp
  text: 指数函数：误差为 0 时结果为 1，误差越大越接近 0。
v_cmd:
  tex: \mathbf v_{xy}^{\mathrm{cmd}}
  text: 速度指令的水平分量（m/s）。
v_xy:
  tex: \mathbf v_{xy}
  text: 机身实际的水平速度（m/s）。
v_z:
  tex: v_z
  text: 机身的竖直速度（m/s）；它的平方也计入误差，用来抑制上下跳动。
sigma_v:
  tex: \sigma_v
  text: 速度误差的容忍尺度，对应代码中的 std。
dt:
  tex: \Delta t
  text: 一个控制步的时长，0.02 s：各项奖励按时间步长累加。
i:
  tex: i
  text: 奖励项的编号。
w_i:
  tex: w_i
  text: 某一奖励项的权重。
r_it:
  tex: r_{i,t}
  text: 某一奖励项在当前控制步的值。
N_batch:
  tex: N_{\mathrm{batch}}
  text: 每轮训练迭代采集的样本数。
N_env:
  tex: N_{\mathrm{env}}
  text: 并行环境数；本节的完整训练为 1024。
T_rollout:
  tex: T_{\mathrm{rollout}}
  text: 每个环境每轮采集的控制步数，为 24。它和回合长度是两个不同的量。
N_total:
  tex: N_{\mathrm{total}}
  text: 2500 轮迭代一共采集的样本数。
```

{#concept}
## 什么是域随机化

仿真中每个参数都是一个确定的数：脚底摩擦系数、躯干质量、舵机从收到目标到开始转动的延迟……真实的机器人却各不相同。域随机化（Domain Randomization）的思路是：既然不知道真实参数究竟是多少，就在训练时让每个并行环境各自抽取一组不同的参数。策略同时面对上千台“略有不同的机器人”，只有学会对这些差异都有效的动作，才能在整体上获得高回报。

```{formula} 域随机化下的学习目标
\symx{J_dr}{J(\sym{theta})}=\sym{E_xi}\,\sym{E_traj}\!\left[\sum_{\sym{t_idx}=0}^{\sym{T_horizon}-1}\sym{gamma}^{\sym{t_idx}}\sym{r_t}\right]
```

与 3.1 节的目标相比，这里多了一层平均：先按分布 $\sym{p_xi}$ 抽取一组参数 $\sym{xi}$，再在这组参数下按策略 $\sym{pi_theta}$ 采样轨迹。PPO 的更新方式不变，变化的是数据：同一批样本里混合了许多种条件。

{#categories}
## 可以随机化什么

| 类别 | 例子 | 本次行走训练 |
|---|---|---|
| 动力学 | 摩擦、质量与重心、关节摩擦与惯量、舵机延迟与电压、外部推力 | 启用 |
| 观测 | 传感器噪声、编码器零位偏差 | 启用 |
| 初始状态与指令 | 起始位置与朝向、速度指令的大小和切换时机 | 启用 |
| 地形 | 坡度、台阶、起伏 | 未启用（平地）；3.4 节的坡地训练使用 |
| 视觉 | 光照、材质、背景、相机参数 | 未启用：行走策略不读取图像；5.3 节讨论 |

{#why}
## 行走为什么需要它

行走全靠脚与地面的接触和舵机的响应，而这些正是仿真最难做准的部分：不同批次的舵机摩擦不一样，电池电量会改变舵机输出，编码器零位有微小偏差，地面也有软有硬。如果策略只在一组“完美”参数下练习，它可能依赖这组参数的细节，换一台机器人或一块地面就站不稳。这种仿真与现实之间的差距，常被称为 sim-to-real 差距。

范围的选择是一种权衡。范围太窄，策略仍然只适应少数情形；范围太宽，任务可能变得过难，策略学会的是保守、迟缓的动作，甚至学不会。范围最好以测量或硬件规格为依据，例如舵机延迟的实测区间、电池电压的工作范围。域随机化只覆盖您想到并写进配置的那些差异，不能替代真机验证。

{#what-varies}
## 本次训练使用哪些随机化，怎样配置

Isaac Lab 用事件管理器实现随机化。每个事件有一个触发时机：`startup` 在创建环境时抽样一次，之后保持不变；`reset` 在每个环境重置时重新抽样；`interval` 在回合进行中按时间间隔触发。下表来自完整训练保存的 `env.yaml`。注意它同时保存初始值和课程（curriculum）设置：有些范围会随训练逐级扩大，所以不能只读初始字段。

| 类别 | 随机化项目 | 范围 | 触发时机 | 模拟的真实差异 |
|---|---|---|---|---|
| 动力学 | 脚底静摩擦与动摩擦 | 0.7–1.3 | startup | 不同地面和鞋底材料 |
| 动力学 | 躯干质量 | 名义值 × 0.95–1.05 | startup | 电池、线缆和外壳差异 |
| 动力学 | 躯干与头部重心 | 初始各方向 ±3 mm，随训练逐级扩大：躯干最多 ±15 mm，头部最多 ±10 mm | reset | 质量分布不均 |
| 动力学 | 关节摩擦 | 名义值 × 0.9–1.1 | reset | 舵机个体差异与磨损 |
| 动力学 | 关节转动惯量（电枢） | 名义值 × 0.9–1.1 | reset | 减速齿轮与电机差异 |
| 动力学 | 外部推力 | 每 3–6 s 一次，水平速度突变 ±0.3 m/s | interval | 碰撞、被人推一下 |
| 动力学 | 舵机延迟 | 3–6 个物理步（15–30 ms） | 执行器模型 | 通信与控制延迟 |
| 动力学 | 舵机供电电压 | 6.5–8.2 V，另有随负载的电压跌落 | 执行器模型 | 电池电量变化 |
| 观测 | 编码器零位偏差 | ±0.015 rad | startup | 装配与校准误差 |
| 观测 | 观测噪声 | 开启 | 每一步 | 传感器噪声 |
| 初始状态与指令 | 起始位置与朝向 | ±0.5 m，朝向任意 | reset | 从不同位置开始行走 |
| 初始状态与指令 | 速度指令 | 每 3–8 s 重新抽取；练习站立的环境从 2% 逐级增至 25% | 指令管理器 | 指令随时变化 |

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

节选来自固定版本 Isaac Lab 的 Microduck 任务配置。舵机延迟和供电电压不是事件，而是写在执行器模型里；观测噪声写在观测配置里。本节后面的测试关闭了推力与参数随机化，在名义条件下比较策略。

Pollen Robotics 训练官方策略时同样使用域随机化。官方训练仓库 microduck_rl 基于 mjlab，而不是本课的 Isaac Lab 与 Newton，所以两者不是同一份配置；但它的行走任务也随机化脚底摩擦（同为 0.7–1.3）、质量与惯量、躯干和头部重心、关节摩擦与转动惯量、编码器零位偏差和 IMU 安装角度，并定时施加推力（[官方行走任务配置 ↗](https://github.com/pollen-robotics/microduck_rl/blob/bae39f5900b318f7e87fc485c30a31213cfa14eb/src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py)）。官方 README 还说明，它的舵机模型会随机化供电电压、电压跌落、指令延迟和摩擦（[microduck_rl ↗](https://github.com/pollen-robotics/microduck_rl/tree/bae39f5900b318f7e87fc485c30a31213cfa14eb)）。

{#implementation}
## 奖励与 PPO 的核心实现

3.2 节说明了奖励由哪些项组成。以线速度跟踪项为例，看它在代码里怎样写：

```{formula} 线速度跟踪奖励与总奖励
\sym{r_v}=\sym{exp}\!\left(-\frac{\|\sym{v_cmd}-\sym{v_xy}\|_2^2+\sym{v_z}^2}{\sym{sigma_v}^2}\right),\qquad \sym{r_t}=\sym{dt}\sum_{\sym{i}}\sym{w_i}\sym{r_it}
```

```{code-story}
label: Python · 核心代码节选
source: https://github.com/isaac-sim/IsaacLab/blob/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck/mdp/rewards.py
language: python
code: |
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  return zero_non_finite(torch.exp(-(xy_error + z_error) / std**2))
steps:
- lines: [0, 2]
  title: 计算速度误差
  body: 对每个环境，`xy_error` 是水平目标速度与实际速度之差的平方和；`z_error` 是竖直速度的平方，竖直方向的目标是 0。
- lines: [2, 3]
  title: 转换为 0 到 1 的奖励
  body: 误差之和除以 `std` 的平方后取负指数，误差为 0 时奖励为 1；`zero_non_finite` 把数值异常的结果记为 0。奖励管理器再把各项乘以权重，按控制步长 $\sym{dt}$ 累加。
```

PPO 使用 RSL-RL 5.5.1 的实现。下面这段就是 3.1 节的裁剪目标：

```{code-story}
label: Python · 核心代码节选
source: https://github.com/leggedrobotics/rsl_rl/blob/v5.5.1/rsl_rl/algorithms/ppo.py
language: python
code: |
  ratio = torch.exp(
      actions_log_prob - torch.squeeze(batch.old_actions_log_prob)
  )
  surrogate = -torch.squeeze(batch.advantages) * ratio
  surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
      ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
  )
  surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
steps:
- lines: [0, 3]
  title: 计算新旧策略概率比
  body: 对数概率相减再取指数，得到同一动作在新旧策略下的概率比。旧策略的对数概率在采样时保存。
- lines: [3, 7]
  title: 裁剪
  body: 优势与概率比相乘得到代理目标；`clip_param` 是裁剪阈值 0.2。
- lines: [7, 8]
  title: 聚合损失
  body: 优化器做最小化，所以代码给目标加负号、取较大项，与“最大化较小项”等价。
```

:::{enrichment} 参数详解

- **折扣因子、GAE 参数**：0.99、0.95
- **裁剪阈值**：0.2
- **每个环境每轮采样步数**：24
- **每批数据的更新遍数、小批次数**：5、4
- **学习率**：0.001，按 KL 散度自适应
- **价值损失系数、熵系数**：1.0、0.01
:::

下面的命令用 1024 个环境训练 2500 轮，采样量是：

```{formula} 每轮采样量与总样本数
\sym{N_batch}=\sym{N_env}\times \sym{T_rollout}=1024\times24=24{,}576,\qquad \sym{N_total}=2500\times24{,}576=61{,}440{,}000
```

按 50 Hz 换算，约 6144 万条交互相当于在仿真中累计练习约 341 小时；并行环境让这些经验在约 2 小时内收集完。训练入口把这些部件连在一起：

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/gait/student_train.py
language: python
code: |
  base = gym.make(task, cfg=cfg).unwrapped
  env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
  dump_yaml(str(args.output / "env.yaml"), cfg)
  dump_yaml(str(args.output / "agent.yaml"), agent)
  runner = OnPolicyRunner(
      env, to_dict(agent), log_dir=str(args.output), device="cuda:0"
  )
steps:
- lines: [0, 2]
  title: 创建训练环境
  body: 按 Isaac Lab 的任务配置创建并行环境，再包装成 RSL-RL 能读取的向量环境。
- lines: [2, 4]
  title: 保存实际配置
  body: 把本次使用的环境配置（观测、奖励、随机化与课程设置）和算法配置写进输出目录，事后可以逐项核对。
- lines: [4, 7]
  title: 连接 PPO 训练器
  body: '`OnPolicyRunner` 负责反复采样和更新；之后调用 `runner.learn` 按指定的迭代次数训练。'
```

{#train}
## 启动完整训练，观察并行训练

在 1.2 节准备的课程容器终端里运行下面的命令（需要重新进入容器时，见 <a href="setup.html#start">1.2 节</a>）。它从随机初始化开始训练，不读取任何已有模型的权重；`--show` 打开 Isaac Sim 窗口，让您看到上千只小鸭子在各自随机化的环境里同时练习。`--output` 请使用尚不存在的新目录。

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py train-walk \
  --num-envs 1024 \
  --iterations 2500 \
  --show \
  --output outputs/walking-training-01
```

:::{enrichment} 参数详解

- **`--num-envs 1024`**：同时在 GPU 上运行的环境数；每个环境各有一组随机参数
- **`--iterations 2500`**：PPO 采样与更新的循环次数，上面已计算过对应的样本量
- **`--show`**：打开 Isaac Sim 窗口显示并行环境；渲染会占用显存并让每轮训练变慢
- **`--output`**：检查点、配置、TensorBoard 日志和导出模型的保存位置
:::

**完整训练需要多长时间？** 课程随附的预训练结果是在 DGX Spark 上不开窗口训练的：1024 个环境 × 2500 轮，训练阶段 7246 秒（约 2 小时），平均每 100 轮约 4.8 分钟，测量时 GPU 上还有其他任务。那次训练用的是官方原始碰撞模型，而上面的命令使用全身碰撞模型；打开 `--show` 时渲染也会拖慢训练，所以您的耗时会不同。

**您不需要等它训练完。** 窗口出现、终端开始逐轮打印回报和回合长度后，观察几分钟：前几十轮里大部分小鸭子很快摔倒、被单独重置，随着迭代推进，站得住的越来越多。

看够了就按 Ctrl+C 停止，容器仍在运行。已保存的检查点（每 250 轮一个）会保留，但只有训练跑完才会导出 `exported/policy.onnx`。下面默认测试课程预先训练好的策略；想让自己的训练跑完时，可以去掉 `--show` 加快速度，完成后再显式换成自己的模型。

:::{course-image} media/isaac-ui-ppo-training.png
:alt: 训练时打开 --show 的 Isaac Sim 完整界面：白色地面上许多 Microduck 副本处在不同的步态阶段，有的站立、有的迈步、有的正在倾倒，顶部显示 RTX 实时渲染与 Newton MJWarp 物理。

`--show` 打开的训练窗口，训练刚开始：各副本处在不同的步态阶段，摔倒的副本会被单独重置。您的命令使用 1024 个环境，窗口里的小鸭子会更多、更密。
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

{#outputs}
## 训练输出的文件

训练会把几类文件写进输出目录，用途各不相同：`model_*.pt` 是每 250 轮保存的训练检查点，`env.yaml` 和 `agent.yaml` 记录本次实际使用的环境与算法配置，`exported/policy.onnx` 是只含 Actor 的推理文件。检查点包含优化器等训练状态，ONNX 只保留推理所需的计算图，两者不能互相替代。

::::{quiz}
只保存 ONNX 文件，能否保证从中精确续训？

:::{answer}
不能。ONNX 里只有 Actor 的推理计算图；续训还需要 Critic、优化器状态、归一化状态和相同的配置，所以要保留完整检查点和训练配置。
:::
::::

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

:::{enrichment} 参数详解

- **`--model`**：要测试的策略文件；这里是课程预先训练好的 `datasets/walking-ppo/exported/policy.onnx`（只读）
- **`--velocity 0.25 0 0`**：前进、侧移和转向的速度指令；程序先发约 2 秒零速度，再持续发送这条指令
- **`--seconds 12`**：运行的仿真时长
- **`--seed 202`**：与 2.2 节的录制示例相同，两次结果可以直接对照
- **`--output`**：`report.json` 和每 0.1 秒一条位置记录的 `trajectory.json` 的保存目录
:::

:::{course-image} media/isaac-ui-ppo-walk-test.png
:alt: 用上面的命令测试预训练 PPO 策略时的 Isaac Sim 完整界面：视口中一只 Microduck 在平地上迈步，顶部显示 RTX 实时渲染与 Newton MJWarp 物理。

测试窗口：观察机器人起步后是否一直站稳、是否沿指令方向直行，还是逐渐偏向一侧。
:::

设置与 2.2 节的录制示例相同，可以直接对照：先看有没有跌倒，再比较 `trajectory.json` 中的距离和方向。课程运行的那一次 12 秒内没有跌倒，前进指令期间走了约 1.11 m（平均约 0.11 m/s），但向右偏，整段约 26°、最后 2 秒约 37°；2.2 节的官方模型速度相近，却向左偏约 8°（最后 2 秒约 15°）。两者都能走，却都没有准确跟上指令。再把 `--velocity` 改成 `0 0 0.4` 测试转向，看看是否“会走不会转”。

**测试您自己的策略。** 只有当您的训练完整跑完、`outputs/walking-training-01/exported/policy.onnx` 已经生成时，才把命令里的 `--model` 手动换成这个路径，并换一个新的 `--output` 目录。一次成功的运行只是一个样本；要下结论，应换几组指令、从不同起点多次重复测试，并保留失败的结果。

::::{quiz}
为什么测试时要关闭训练中使用的随机推力和参数随机化？

:::{answer}
本节先关闭随机推力与参数随机化，在名义条件下比较策略的基本响应。评估鲁棒性时，应另外预先固定随机分布和测试种子，让不同策略经历同一组参数与扰动，并统计全部结果。随机化并不等于不可复现。
:::
::::

{#next}
## 下一步：从平地走向坡地

域随机化让策略面对参数的变化，但地面仍然是平的。即使一个策略在平地上走得很好，遇到坡道时，原来的抬脚高度、落脚位置和重心调整还够用吗？3.4 节先讨论奖励应该如何设计，再把随机化扩展到地形，让步行走稳上下坡。

{#references}
## 参考资料

{.references}
- [Isaac Lab：Microduck 任务配置 ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
- [Tobin 等：用于 sim-to-real 迁移的域随机化 ↗](https://arxiv.org/abs/1703.06907)
- [PPO 原论文 ↗](https://arxiv.org/abs/1707.06347)
