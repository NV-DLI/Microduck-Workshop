---
myst:
  html_meta:
    description: 强化学习怎样从试错中学会行走，PPO 为什么要约束每次策略更新，以及这些公式在 Isaac Lab 训练代码中的对应位置。
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 3.1 强化学习与 PPO

:::{lesson-intro}
:meta: 第 3.1 节 · 理论 · 预计学习时长 30 分钟

强化学习怎样从试错中学会行走，PPO 为什么要约束每次策略更新，以及这些公式在 Isaac Lab 训练代码中的对应位置。本节只讲原理，3.2 节再动手启动训练并测试训练好的策略。
:::

```{symbols}
J:
  tex: J(\theta)
  text: 学习目标：按当前策略采样时的期望折扣回报；训练就是调整策略参数让它增大。
E_traj:
  tex: \mathbb E_{\tau\sim\pi_\theta}
  text: 期望：对按当前策略采样得到的许多条轨迹取平均。
tau:
  tex: \tau
  text: 一段交互轨迹：一个回合中依次出现的观测、动作和奖励。
pi_theta:
  tex: \pi_\theta
  text: 策略：由网络参数决定的神经网络，根据观测给出动作的概率分布。
theta:
  tex: \theta
  text: 策略网络的参数（权重），训练时更新。
t_idx:
  tex: t
  text: 控制步的序号。
T_horizon:
  tex: T
  text: 这段轨迹中计入回报的控制步数；与后面每轮采样的步数是两个不同的量。
gamma:
  tex: \gamma
  text: 折扣因子：越接近 1，越重视远期奖励；课程训练取 0.99。
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
  text: 一个控制步的时长：各项奖励按时间步长累加。
i:
  tex: i
  text: 奖励项的编号。
w_i:
  tex: w_i
  text: 某一奖励项的权重。
r_it:
  tex: r_{i,t}
  text: 某一奖励项在当前控制步的值。
delta_t:
  tex: \delta_t
  text: 时序差分误差：这一步的实际结果比价值网络原先的估计好多少。
m_t:
  tex: m_t
  text: 继续标记：回合在这一步终止时为 0，否则为 1。
V:
  tex: V
  text: 价值网络：估计从某个状态出发能获得的折扣回报。
s_t:
  tex: s_t
  text: 当前控制步的状态。
s_next:
  tex: s_{t+1}
  text: 下一步的状态。
A_hat:
  tex: \hat A_t
  text: 当前控制步的优势估计：这个动作比平均预期好多少。
A_next:
  tex: \hat A_{t+1}
  text: 下一步的优势估计；递推时从后往前累积。
lambda:
  tex: \lambda
  text: GAE 参数：控制多步误差的权重；课程训练取 0.95。
rho:
  tex: \rho_t
  text: 概率比：同一个已采样动作在新策略与旧策略下的概率密度之比。
pi_old:
  tex: \pi_{\theta_{\mathrm{old}}}
  text: 旧策略：采样这批数据时的策略，更新期间保持不变。
a_t:
  tex: a_t
  text: 当前控制步实际采样的动作。
o_t:
  tex: o_t
  text: 当前控制步的观测。
L_clip:
  tex: L^{\mathrm{CLIP}}
  text: PPO 的裁剪代理目标：训练要让它增大。
E_t:
  tex: \mathbb E_t
  text: 对这一批样本的各个时间步取平均。
min:
  tex: \min
  text: 取两项中较小的一项，只保留更保守的估计。
clip:
  tex: \operatorname{clip}
  text: 截断：把概率比限制在 $1-\epsilon$ 到 $1+\epsilon$ 之间。
epsilon:
  tex: \epsilon
  text: 裁剪范围；课程训练取 0.2。
L_total:
  tex: \mathcal L
  text: 优化器实际最小化的总损失。
c_V:
  tex: c_V
  text: 价值损失系数；当前配置为 1.0。
L_V:
  tex: L_V
  text: 价值损失：价值网络预测回报的误差。
c_H:
  tex: c_H
  text: 熵系数；当前配置为 0.01。
H:
  tex: H
  text: 熵：动作分布的随机程度；加入这一项可以鼓励继续探索。
N_batch:
  tex: N_{\mathrm{batch}}
  text: 每轮训练迭代采集的样本数。
N_env:
  tex: N_{\mathrm{env}}
  text: 并行环境数；3.2 节的完整训练为 1024。
T_rollout:
  tex: T
  text: 每个环境每轮采集的控制步数；3.2 节的训练为 24。它和回合的时间上限是两个不同的量。
N_total:
  tex: N_{\mathrm{total}}
  text: 2500 轮迭代一共采集的样本数。
```

{#loop}
## 任务：让机器人按速度指令稳定行走

2.1 和 2.2 节只运行了官方训练好的固定策略。现在换个问题：从随机初始化出发，策略怎样通过仿真中的反复尝试学会行走？本节把学习目标、奖励、优势估计和 PPO 更新对应到 Isaac Lab 与 RSL-RL 的实际代码；3.2 节再启动训练，并测试课程预先训练好的策略。

我们希望机器人接收前进、横移和转向指令后，自己协调 14 个关节：迈步时保持平衡，支撑脚尽量不打滑，动作平顺连贯。给每个关节手写固定的周期轨迹，很难同时适应不同的速度、受力和初始状态。

强化学习通过“尝试动作 → 得到反馈 → 调整策略”学习控制，不需要“每个状态对应哪个关节角度”的示例数据：Newton 生成状态转移，Isaac Lab 的奖励管理器描述期望的行为。

```{learning-flow}
label: 强化学习的在线交互与更新
nodes:
- [并行仿真环境, 根据当前状态构造观测]
- [当前策略 πθ, 采样关节动作 aₜ]
- [执行与评价, 得到奖励 rₜ 和下一状态]
- [保存一批交互, 观测、动作、奖励、终止标记]
- [PPO 更新, 估计优势，优化策略与价值网络]
- [继续采样, 用更新后的策略收集新数据]
details:
- 多个环境同时产生观测。某个副本跌倒或超时后单独重置，其他副本继续执行。
- 根据当前观测采样动作。训练时的随机探索让策略见到不同动作的后果；部署时按选定的推理方式输出动作。
- 把动作送入物理环境，再计算速度跟踪、姿态与能耗等奖励。奖励只评价行为好坏，具体的关节动作由策略自己摸索。
- 保存一段交互中的观测、动作、奖励和终止信息。按时间顺序排列的记录用于计算回报和优势。
- 用收集的样本更新策略与价值网络。裁剪目标限制单次更新的幅度，避免策略突变。
- 更新后的策略会产生新的动作和状态分布，因此要重新采样。交互与更新反复交替，就是强化学习训练。
loop: true
caption: Isaac Lab 在 GPU 上把多个环境组织成批量状态，观测、动作、奖励和终止标记都按环境索引存放。策略一变，数据分布也随之改变，所以 PPO 每次只用一批新采集的数据做有限次更新，然后重新与环境交互。
```

```{formula} 学习目标：提高长期折扣回报
\symx{J}{J(\sym{theta})}=\symx{E_traj}{\mathbb E_{\sym{tau}\sim\sym{pi_theta}}}\!\left[\sum_{\sym{t_idx}=0}^{\sym{T_horizon}-1}\sym{gamma}^{\sym{t_idx}} \sym{r_t}\right]
```

$\sym{theta}$ 是策略参数，$\sym{tau}$ 是一段交互轨迹，$\sym{r_t}$ 是即时奖励，$\sym{gamma}$ 是折扣因子。优化长期回报意味着：某一步动作是否有价值，要结合后续能否站稳来判断，不能只看当前帧的奖励。

训练时，策略网络根据 61 维可部署观测选择动作；价值网络还可以读取额外的仿真信息来估计回报。部署时只导出读取这 61 维观测的策略网络；价值网络额外用到的仿真信息（例如机身线速度、足部接触力）只在训练时使用。

{#reward-design}
## 把“走得好”写成奖励

只奖励速度接近指令，机器人可能学到不稳定的移动方式；只奖励站直，又可能学会原地不动。因此，任务会组合速度跟踪、姿态、脚部运动和动作变化等项目。下面给出一个实际奖励项及其组合形式；完整项目与权重以本次训练保存的 `env.yaml` 为准。

```{formula} 线速度跟踪奖励
\sym{r_v}=\sym{exp}\!\left(-\frac{\|\sym{v_cmd}-\sym{v_xy}\|_2^2+\sym{v_z}^2}{\sym{sigma_v}^2}\right),\qquad \sym{r_t}=\sym{dt}\sum_{\sym{i}}\sym{w_i}\sym{r_it}
```

速度误差越小，这个指数奖励越接近 1；竖直速度也受到惩罚，避免把上下跳动当成理想平移。$\sym{sigma_v}$ 控制误差的容忍范围。奖励管理器把各项乘以权重，并按控制时间步 $\sym{dt}$ 汇总；比较奖励曲线时，要同时核对权重和时间步是否一致。

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
  body: 误差之和除以 `std` 的平方后取负指数，误差为 0 时奖励为 1；`zero_non_finite` 把数值异常的结果记为 0。
```

节选来自固定版本 Isaac Lab 的 `microduck/mdp/rewards.py`。`command` 对应目标速度，`actual` 对应实际速度，`std` 对应公式中的 $\sym{sigma_v}$。训练开始时，线速度跟踪权重为 2.0，动作变化惩罚权重为 −0.1。继续长训时，还要查看课程学习设置：动作变化惩罚会随训练进展加大。这些值针对本任务设定，换成其他机器人需要重新调整。

检查奖励设计时，先看行为，再看总分。若奖励上升但跌倒更频繁，应逐项检查速度、姿态、动作变化和终止情况，找出是哪个奖励项掩盖了问题。

{#ppo}
## 为什么选择 PPO：约束每次策略更新的幅度

策略更新幅度过大时，刚学会的平衡行为可能在下一次采样中丢失。PPO（近端策略优化）通过裁剪代理目标，让过大的概率变化不再带来额外收益。它适合连续关节控制，也便于在大量并行仿真环境中收集数据；这里采用现成的 RSL-RL 实现，方便把公式、配置与训练日志对应起来。PPO 让更新更稳定，但单次更新仍可能让行为变差。

先估计某个动作比当前状态下的平均预期好多少，这个差值称为优势。正优势鼓励提高该动作的概率，负优势鼓励降低其概率。价值网络 $\sym{V}$ 提供状态回报基线；广义优势估计（GAE）组合多步时序差分误差，以权衡估计的偏差与方差。

```{formula} 时序差分误差与广义优势估计
\sym{delta_t}=\sym{r_t}+\sym{gamma}\sym{m_t}\sym{V}(\sym{s_next})-\sym{V}(\sym{s_t}),\qquad \sym{A_hat}=\sym{delta_t}+\sym{gamma}\sym{lambda}\sym{m_t}\sym{A_next}
```

$\sym{m_t}$ 在任务终止时为 0，否则为 1；$\sym{gamma}$ 控制未来奖励的折扣，$\sym{lambda}$ 控制多步误差的权重。达到时间上限的截断与跌倒终止含义不同：实现会对截断单独做价值自举。课程训练配置使用 $\sym{gamma} = 0.99$、$\sym{lambda} = 0.95$。

```{formula} PPO 的裁剪代理目标
\sym{rho}(\sym{theta})=\frac{\sym{pi_theta}(\sym{a_t}\mid \sym{o_t})}{\sym{pi_old}(\sym{a_t}\mid \sym{o_t})},\qquad \sym{L_clip}=\sym{E_t}\!\left[\sym{min}\!\left(\sym{rho}\sym{A_hat},\sym{clip}(\sym{rho},1-\sym{epsilon},1+\sym{epsilon})\sym{A_hat}\right)\right]
```

$\sym{rho}$ 是同一已采样动作在新旧策略下的概率密度之比，$\sym{epsilon} = 0.2$ 定义裁剪范围。裁剪只作用于优化目标：概率比本身仍可能越出这个范围，机器人的关节角度也不受它约束。

```{concept-diagram}
kind: ppo
label: PPO 裁剪目标：优势正负与概率比的关系
caption: 灰色虚线是未裁剪的目标，绿色实线是 PPO 目标，浅绿色区域是 $[1-\sym{epsilon},\,1+\sym{epsilon}]$。优势为正时，增加动作概率超过上界不再提高该样本的目标；优势为负时，对应的平台出现在下界外。
```

尝试把优势从 +1 切换为 −1，再移动概率比滑块。目标取两项中的较小值，所以平台只出现在一侧，位置取决于优势的正负。图中只分析单个样本的策略目标；实际更新还会同时受到其他样本、价值损失和熵项的影响。

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
  body: 对数概率相减再取指数，得到同一动作在新旧策略下的概率比。
- lines: [3, 7]
  title: 限制更新幅度
  body: 优势估计与概率比共同决定代理目标；裁剪概率比，限制过大的策略变化。
- lines: [7, 8]
  title: 聚合损失
  body: 代码在负号形式的损失中取较大项，再对样本求平均；优化器最小化这个损失。
```

代码中的 `ratio` 对应 $\sym{rho}$，`batch.advantages` 对应 $\sym{A_hat}$，`clip_param` 对应 $\sym{epsilon}$。由于优化器执行最小化，实现会给目标加负号并使用 `max`，这与公式中的“最大化 min”一致。旧策略的对数概率在采样时保存，更新期间保持不变。

```{formula} 实际优化的总损失
\sym{L_total}=-\sym{L_clip}+\sym{c_V} \sym{L_V}-\sym{c_H}\sym{E_t}[\sym{H}(\sym{pi_theta}(\cdot\mid \sym{o_t}))]
```

价值损失 $\sym{L_V}$ 用于训练回报预测，熵 $\sym{H}$ 用于鼓励探索。当前配置的价值损失系数为 1.0、熵系数为 0.01，并启用了价值裁剪。学习率初值为 0.001，随后根据 KL 散度自适应调整；这些设置都可在输出目录的 `agent.yaml` 中核对。

原理参考：[PPO 论文](https://arxiv.org/abs/1707.06347)；实现参考：[RSL-RL 5.5.1 算法源码](https://github.com/leggedrobotics/rsl_rl/tree/v5.5.1/rsl_rl/algorithms)。

{#rollout-data}
## 训练数据从哪里来，参数意味着什么

PPO 不需要预先下载数据集：训练数据是在训练过程中由仿真实时产生的。3.2 节的完整训练从随机初始化开始，不读取固定行走模型的权重。脚本按 Isaac Lab 的任务配置创建 1024 个 Newton 环境，每个环境各采集 24 步交互，组成一轮采样数据。每条记录包含观测、动作、奖励、旧策略对数概率、价值估计和结束标记，保存在 GPU 内存的缓冲区里，用完即被下一轮数据替换。

```{formula} 每轮采样量与总样本数
\sym{N_batch}=\sym{N_env}\times \sym{T_rollout}=1024\times24=24576,\qquad \sym{N_total}=2500\times24576=61{,}440{,}000
```

| 配置项 | 3.2 节的完整训练 | 主要影响 |
|---|---|---|
| 并行环境 | 1024 | 每轮可采集的并行交互量 |
| 每个环境的采样步数 | 24 | 每轮轨迹片段长度，不等于完整回合长度（最长 20 秒，即 1000 个控制步） |
| 每轮更新的 epoch 数 | 5 | 一批数据用于优化的遍数 |
| 小批次数 | 4 | 每次划分为 4 个小批次，每批 6144 条 |
| 训练迭代次数 | 2500 | 采样与更新循环的次数；不是 2500 个动作 |
| 学习率 | 0.001，按 KL 散度自适应 | 每次参数更新的步长 |

这份预算会产生约 6144 万条环境交互，执行 2500 × 5 × 4 = 50,000 次优化器更新。按 50 Hz 控制频率换算，相当于机器人在仿真中累计“练习”了约 341 小时；并行环境与 GPU 让这些经验在约 2 小时内收集完。训练环境会在回合终止后单独重置该环境，其余环境继续采样。

```{code-story}
label: Python · 核心代码节选
source: https://github.com/NV-DLI/Microduck-Workshop/blob/main/lab/runtime/simulation/core/gait/student_train.py
language: python
code: |
  base = gym.make(task, cfg=cfg).unwrapped
  env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
  runner = OnPolicyRunner(
      env, to_dict(agent), log_dir=str(args.output), device="cuda:0"
  )
  runner.learn(
      num_learning_iterations=args.iterations, init_at_random_ep_len=True
  )
steps:
- lines: [0, 2]
  title: 创建训练环境
  body: 构建任务并包装向量环境，让算法同时读取多个环境的观察和奖励。
- lines: [2, 5]
  title: 连接 PPO 训练器
  body: 交给训练器环境、算法参数、输出目录和运行设备。
- lines: [5, 8]
  title: 开始有限轮次的学习
  body: 迭代次数决定本次预算；随机初始回合长度可分散各环境的终止时刻。
```

Isaac Lab 任务提供状态转移、管理器和批量重置，包装器将其接入 RSL-RL，`OnPolicyRunner` 负责反复采样和更新。Newton 的 Warp/CUDA 路径让这些环境在 GPU 上并行推进，速度优势来自批量执行和统一的任务接口。

{#read-curves}
## 检查点、配置与导出文件

训练会把几类文件写进输出目录，用途各不相同。`initial.pt` 保存训练起点，`model_*.pt` 是每隔若干轮保存的训练检查点，`env.yaml` 和 `agent.yaml` 记录本次实际使用的环境与优化配置，`exported/policy.onnx` 是供推理使用的导出文件。检查点可以包含优化器等训练状态，ONNX 通常只保留推理所需的计算图，两者用途不同，不能互相替代。

奖励由速度跟踪、机身姿态、动作平滑性和接触相关项共同组成。两个总奖励相同的策略，可能一个靠“速度更准确”得分，另一个靠“少动更稳定”得分，因此看训练结果时要同时检查实际行为与各项奖励。

::::{quiz}
只保存 ONNX 文件，能否保证从中精确续训？

:::{answer}
不能。精确续训还需要优化器状态、归一化状态、随机状态以及相同的配置等信息，因此必须保留完整检查点和训练配置。
:::
::::

::::{quiz}
为什么不能把训练过程中回报最高的一次当作独立测试结果？

:::{answer}
因为这次结果已经参与学习或模型选择，而且奖励高并不必然等于任务成功。独立测试应预先确定场景和成功标准，并在未参与选型的条件下统计全部结果。
:::
::::

{#beyond-flat}
## 下一步：让训练面对变化

现在已经知道目标、奖励、优势和裁剪更新是怎样一起工作的。还有一个问题：仿真里的摩擦、质量和舵机延迟都是确定的数值，真实机器人却各不相同，在一个“完美”仿真里学到的策略能适应这些差异吗？3.2 节先介绍域随机化，再启动训练、观察并行环境，并测试课程预先训练好的策略。

{#references}
## 参考资料

{.references}
- [PPO 原论文 ↗](https://arxiv.org/abs/1707.06347)
- [Isaac Lab：Microduck 任务源码 ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
