---
myst:
  html_meta:
    description: 强化学习的通用方法：目标、奖励与回报，Actor-Critic 中 Actor 与 Critic 的分工，优势估计，以及 PPO 为什么要限制每次更新的幅度。
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 3.1 强化学习与 PPO

:::{lesson-intro}
:meta: 第 3.1 节 · 理论 · 预计学习时长 15 分钟

强化学习的通用方法：目标、奖励与回报，Actor-Critic 中 Actor 与 Critic 的分工，优势估计，以及 PPO 为什么要限制每次更新的幅度。3.2 节再把这些概念对应到 Microduck 的行走任务。
:::

```{symbols}
G_t:
  tex: G_t
  text: 回报：从第 $t$ 步开始，之后所有奖励按折扣累加的总和。
gamma:
  tex: \gamma
  text: 折扣因子，取 0 到 1 之间：越接近 1，越重视远期奖励。
k_idx:
  tex: k
  text: 从当前步往后数的步数。
t_idx:
  tex: t
  text: 时间步的序号。
r_t:
  tex: r_t
  text: 第 $t$ 步的即时奖励。
delta_t:
  tex: \delta_t
  text: 时序差分误差：这一步的实际结果比 Critic 原先的估计好多少。
m_t:
  tex: m_t
  text: 继续标记：回合在这一步终止时为 0，否则为 1。
V:
  tex: V
  text: 价值函数（Critic）：估计从某个状态出发、按当前策略继续行动能得到的回报。
s_t:
  tex: s_t
  text: 当前步的状态。
s_next:
  tex: s_{t+1}
  text: 下一步的状态。
A_hat:
  tex: \hat A_t
  text: 优势估计：这一步实际采取的动作比 Critic 的预期好多少。
A_next:
  tex: \hat A_{t+1}
  text: 下一步的优势估计；递推时从后往前累积。
lambda:
  tex: \lambda
  text: GAE 参数：在只看一步与看很多步之间折中。
rho:
  tex: \rho_t
  text: 概率比：同一个已采样动作在新策略与旧策略下的概率密度之比。
theta:
  tex: \theta
  text: 策略网络的参数（权重），训练时更新。
pi_theta:
  tex: \pi_\theta
  text: 新策略（Actor）：根据观测给出动作的概率分布。
pi_old:
  tex: \pi_{\theta_{\mathrm{old}}}
  text: 旧策略：采样这批数据时的策略，更新期间保持不变。
a_t:
  tex: a_t
  text: 当前步实际采样的动作。
o_t:
  tex: o_t
  text: 当前步的观测。
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
  text: 截断：把概率比截在 $1-\epsilon$ 到 $1+\epsilon$ 之间再计算目标。
epsilon:
  tex: \epsilon
  text: 裁剪阈值，常用 0.2。
```

{#loop}
## 从试错中学习

强化学习研究的是：一个智能体在环境中反复行动，怎样根据得到的反馈学会更好的行为。它不需要“每个状态该做什么动作”的标准答案，只需要一个能评价行为好坏的信号。

```{learning-flow}
label: 强化学习的交互与更新循环
nodes:
- [环境, 给出当前观测]
- [策略, 根据观测选择动作]
- [执行与评价, 得到奖励和下一观测]
- [收集一批经验, 观测、动作、奖励、终止]
- [更新策略, 提高好动作的概率]
- [继续交互, 用新策略收集新数据]
details:
- 环境可以是真实世界，也可以是仿真器。智能体只能通过观测了解环境的状态。
- 策略是从观测到动作的映射。训练时它带有随机性，才能尝试不同的动作。
- 环境执行动作后进入新状态，并给出一个奖励，说明这一步的结果有多好。
- 按时间顺序保存一段交互，用来计算回报和优势。
- 用这批经验调整策略参数：让带来好结果的动作更常出现，坏结果的动作更少出现。
- 策略变了，它遇到的状态也会变，所以要用新策略重新收集数据。交互与更新交替进行，就是训练。
loop: true
caption: 强化学习的基本循环。数据来自策略自己的尝试，策略一变，数据分布也跟着改变。
```

{#return}
## 目标、奖励与回报

**目标**是我们希望智能体最终做到的事。**奖励**是每一步的即时反馈，是把目标翻译成数字的方式。好的动作往往要过一段时间才显出效果，所以学习看的是**回报**：从某一步开始，之后所有奖励按折扣累加。

```{formula} 折扣回报
\sym{G_t}=\sum_{\sym{k_idx}=0}^{\infty}\sym{gamma}^{\sym{k_idx}}\,\symx{r_t}{r_{\sym{t_idx}+\sym{k_idx}}}
```

折扣因子 $\sym{gamma}$ 越接近 1，越重视远期结果。训练的目标就是调整策略，让期望回报尽量大。

奖励只是训练信号，不等于任务成功。如果奖励没有写好，智能体可能找到拿高分却没完成任务的办法，所以训练后要用单独的成功判据检验结果。

{#actor-critic}
## Actor-Critic：训练时一个负责行动，一个负责评价

直接用回报调整策略，结果会很不稳定：同一个动作，有时因为后面的运气好拿到高回报，有时则相反。Actor-Critic 是一种训练范式，训练期间让两个网络分工：

| 网络 | 输出 | 作用 |
|---|---|---|
| Actor（策略） | 动作 | 决定做什么；训练结束后部署的就是它 |
| Critic（价值函数） | 一个数：预期回报 | 评价“现在的处境有多好”，只在训练时使用 |

有了 Critic 的预期，就能判断某个动作是否比预期更好，这个差值称为**优势**。优势为正，Actor 提高这个动作的概率；优势为负，就降低它。Critic 本身也在训练：它的预测要尽量接近实际得到的回报。训练结束后只保留 Actor，Critic 的任务也随之完成。

{#ppo}
## PPO：每次只改一点

用一批数据更新策略时，如果一步改得太多，策略可能突然变差，而下一批数据又是这个变差的策略采集的，训练就容易崩溃。PPO（近端策略优化）比较新旧策略选择同一个动作的概率：概率变化超过裁剪阈值 $\sym{epsilon}$ 后，继续朝这个方向改不会再带来额外收益，于是每次更新自然保持在较小幅度。

```{concept-diagram}
kind: ppo
label: PPO 裁剪目标：优势正负与概率比的关系
caption: 灰色虚线是未裁剪的目标，绿色实线是 PPO 目标，浅绿色区域是 $[1-\sym{epsilon},\,1+\sym{epsilon}]$。优势为正时，平台出现在上界右侧；优势为负时，出现在下界左侧。
```

把优势从 +1 切换为 −1，再移动概率比滑块，观察平台出现在哪一侧。裁剪只作用于优化目标，并不硬性限制实际的概率比；PPO 让更新更稳，但不保证每次更新都让行为变好。它实现简单，适合大量并行仿真采集数据，所以被广泛用于机器人运动控制。

::::{enrichment} 拓展：优势估计与 PPO 目标的公式
Critic 用时序差分误差 $\sym{delta_t}$ 比较“这一步的奖励加上下一状态的预期”与“这一状态原本的预期”。广义优势估计（GAE）把多步误差按 $\sym{gamma}\sym{lambda}$ 衰减后累加，得到更平稳的优势；回合终止时 $\sym{m_t}=0$，终止之后的预期不再计入。

```{formula} 时序差分误差与广义优势估计
\sym{delta_t}=\sym{r_t}+\sym{gamma}\sym{m_t}\sym{V}(\sym{s_next})-\sym{V}(\sym{s_t}),\qquad \sym{A_hat}=\sym{delta_t}+\sym{gamma}\sym{lambda}\sym{m_t}\sym{A_next}
```

PPO 用概率比 $\sym{rho}$ 乘以优势作为目标，并在未裁剪与裁剪两项中取较小值：

```{formula} PPO 的裁剪代理目标
\sym{rho}(\sym{theta})=\frac{\sym{pi_theta}(\sym{a_t}\mid \sym{o_t})}{\sym{pi_old}(\sym{a_t}\mid \sym{o_t})},\qquad \sym{L_clip}=\sym{E_t}\!\left[\sym{min}\!\left(\sym{rho}\sym{A_hat},\sym{clip}(\sym{rho},1-\sym{epsilon},1+\sym{epsilon})\sym{A_hat}\right)\right]
```
::::

::::{quiz}
部署一个训练好的策略时，还需要 Critic 吗？

:::{answer}
不需要。Critic 只在训练时用来估计预期回报、计算优势；部署时由 Actor 根据观测输出动作。
:::
::::

::::{quiz}
为什么不能把训练中回报最高的那一次当作测试结果？

:::{answer}
回报只说明策略拿到了多少奖励，奖励不等于任务成功；而且这次结果已经参与了训练或挑选。测试要预先定好场景和成功判据，统计全部结果。
:::
::::

{#next}
## 下一步

3.2 节把这些概念对应到 Microduck 的行走任务：观测是什么、动作是什么、奖励怎样写，以及训练时 Actor 和 Critic 各读取哪些信息、怎样配合。

{#references}
## 参考资料

{.references}
- [PPO 原论文 ↗](https://arxiv.org/abs/1707.06347)
- [GAE：广义优势估计 ↗](https://arxiv.org/abs/1506.02438)
- [Sutton 与 Barto：强化学习导论 ↗](http://incompleteideas.net/book/the-book-2nd.html)
