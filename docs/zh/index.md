---
myst:
  html_meta:
    description: 本课程专注于在仿真环境中训练和验证 Microduck：从机械结构与传感器，到强化学习、运动技能和任务控制，逐步建立可检查的机器人实验。自主射门是串联感知与动作的一个综合案例，课程也涵盖行走、上下坡和更多任务的设计思路。
---

# 1.1 课程概览

:::{lesson-intro}
:meta: 6 个部分 · 18 节 · 仿真实践与前沿方法
:hero: media/isaac-football-hero-still.webp
:hero-animation: media/isaac-football-hero.webp
:hero-size: 1600x1000
:hero-alt: Isaac 完整窗口的运行录像，按仿真时间播放：板载策略控制 Microduck 在绿色球场踢球进门，旁边同步显示头部 RGB 和 ToF 传感器画面。

本课程专注于在仿真环境中训练和验证 Microduck：从机械结构与传感器，到强化学习、运动技能和任务控制，逐步建立可检查的机器人实验。自主射门是串联感知与动作的一个综合案例，课程也涵盖行走、上下坡和更多任务的设计思路。
:::

```{toctree}
:caption: 1 概览
:hidden:

self
setup
```

```{toctree}
:caption: 2 认识 Microduck
:hidden:

01-play
02-anatomy
```

```{toctree}
:caption: 3 使用强化学习让 Microduck 学会技能
:hidden:

03-learning
03-microduck-rl
03-walk-training
04-terrain
05-skills
```

```{toctree}
:caption: 4 板载策略完成射门任务
:hidden:

06-perception
06-onboard
06-shoot
```

```{toctree}
:caption: 5 （拓展）使用 GR00T 自主完成射门
:hidden:

07-groot
08-groot-interface
09-worlds
```

```{toctree}
:caption: 6 参考与帮助
:hidden:

models
troubleshooting
references
```

{#simulation-first}
## 为什么先在仿真中训练与验证

本课编写时（2026 年 11 月），Microduck 仍处于预售阶段，首批尚未发货，我们还无法拿到真机开展课程实验。您可以先使用课程提供的机器人模型，在 Isaac Sim、Isaac Lab 与 Newton 构成的仿真环境中学习、训练和验证，无需等待真机。[官方发货说明](https://store.pollen-robotics.com/policies/shipping-policy)

先在仿真中训练，也是 sim2real（从仿真迁移到真实环境）范式的关键步骤。仿真允许反复试验、重置场景和并行采集数据，并通过域随机化改变摩擦、执行器特性或传感器条件，帮助策略适应变化。等真机具备条件后，再核对传感器与动作接口、校准物理参数，并逐步开展真实环境验证。

本课完成的是这条路径中的仿真训练与验证阶段。仿真中的成功是后续迁移的起点，仍需在真机上检查建模误差、传感器噪声、控制延迟与接触差异。

{#outcome}
## 学习目标

课程先认识 Microduck 的机构与控制闭环，再学习强化学习和域随机化，训练与验证行走、上下坡等运动技能。随后以射门为例，把传感器、人工设计的阶段和已训练的技能组合成完整任务，并了解 GR00T 怎样从示范中学习高层请求。最后讨论环境变化，以及如何借助智能体设计自己的机器人任务。所有动手实验都在课程容器的仿真环境中进行。

{.pipeline}
1. 课程概览与容器环境准备
2. 认识 Microduck
3. 用强化学习训练技能
4. 板载策略完成射门任务
5. （拓展）使用 GR00T 自主完成射门
6. 参考与帮助

{#learn}
## 学完后能够

- 解释机构、执行器、观测、动作与奖励的关系。
- 在课程容器中使用 Isaac Lab、Newton 与 Isaac Sim 运行固定策略，完成 PPO 训练与域随机化实验。
- 说明板载策略怎样用 RGB 检测、ToF 测距和人工设计的阶段完成射门，以及它与视觉-语言-动作模型（VLA）的动作预测各自负责的工作。
- 说明 GR00T 的 8 维高层请求与底层技能的分工，以及示范数据、动作头训练和闭环验证的思路。
- 设计环境变化与独立评测，分清“流程能跑通”“模型指标改善”与“完整任务成功”。

{#route}
## 课程路线

| 部分 | 小节 |
|---|---|
| 1 概览 | 1.1 课程概览 · 1.2 准备环境与获取代码 |
| 2 认识 Microduck | 2.1 认识 Microduck：先来踢一场球 · 2.2 拆解机械结构与控制周期 |
| 3 使用强化学习让 Microduck 学会技能 | 3.1 强化学习与 PPO · 3.2 Microduck 的强化学习 · 3.3 训练并测试行走策略 · 3.4 延伸：走稳上下坡 · 3.5 （拓展）社区技能：奖励设计 |
| 4 板载策略完成射门任务 | 4.1 识别球与球门：RGB 检测与 ToF 测距 · 4.2 设计射门阶段 · 4.3 射门！ |
| 5 （拓展）使用 GR00T 自主完成射门 | 5.1 视觉-语言-动作模型（VLA）与 GR00T · 5.2 使用 GR00T 自主完成射门 · 5.3 从球场走向更多环境 |
| 6 参考与帮助 | 6.1 让智能体来帮忙 · 6.2 故障排查 · 6.3 继续学习与参考资料 |
