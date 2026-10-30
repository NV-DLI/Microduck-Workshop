---
myst:
  html_meta:
    description: 先体验 Microduck 的运动，再用强化学习训练技能，然后通过示范后训练 GR00T、在仿真中验证步行射门表现，并讨论怎样把它带到更多环境。
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 1.1 让 Microduck 学会自主射门

:::{lesson-intro}
:meta: 5 个部分 · 16 节 · 仿真实践与前沿方法
:hero: media/football-success-hero-still.webp
:hero-animation: media/football-success-hero.webp
:hero-size: 1200x584
:hero-alt: Isaac Sim 仿真中的 Microduck 在绿色球场上走到足球前，把球踢进球门。

先体验 Microduck 的运动，再用强化学习训练技能，然后通过示范后训练 GR00T、在仿真中验证步行射门表现，并讨论怎样把它带到更多环境。
:::

```{toctree}
:caption: 1 概览
:hidden:

self
how-to-learn
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
03-walk-training
04-terrain
05-skills
```

```{toctree}
:caption: 4 训练 GR00T 自主完成任务
:hidden:

07-groot
08-groot-interface
08-groot-lab
09-worlds
```

```{toctree}
:caption: 5 参考与帮助
:hidden:

models
troubleshooting
references
```

{#outcome}
## 学习目标

整门课围绕一台 Microduck、一块绿色足球场、一个足球和一个球门展开：先认识机器人并体验运动技能，再用强化学习与域随机化训练行走、坡地适应等技能，最后用示范后训练 GR00T，在仿真中验证它的步行射门表现，并讨论如何适应更多环境。所有实验都在同一个课程容器里运行。

{.pipeline}
1. 课程概览与容器环境准备
2. 认识 Microduck
3. 用强化学习训练技能
4. 训练 GR00T 自主完成任务
5. 参考与帮助

{#learn}
## 学完后能够

- 解释机构、执行器、观测、动作与奖励的关系。
- 在课程容器中使用 Isaac Lab、Newton 与 Isaac Sim 运行固定策略，完成 PPO 训练与域随机化实验。
- 区分机载策略（Onboard）与 VLA 动作预测各自负责的工作。
- 说明 GR00T 的 8 维高层请求接口、示范数据设计与动作头训练，并验证训练好的检查点。
- 设计环境变化与独立评测，分清“流程能跑通”“模型指标改善”与“完整任务成功”。

{#route}
## 课程路线

| 部分 | 小节 |
|---|---|
| 1 概览 | 1.1 让 Microduck 学会自主射门 · 1.2 如何学习本课程 · 1.3 准备环境与获取代码 |
| 2 认识 Microduck | 2.1 认识 Microduck：先来踢一场球 · 2.2 拆解机械结构与控制周期 |
| 3 使用强化学习让 Microduck 学会技能 | 3.1 强化学习与 PPO · 3.2 域随机化：训练并测试行走策略 · 3.3 奖励设计：走稳上下坡 · 3.4 社区技能：从目标到奖励 |
| 4 训练 GR00T 自主完成任务 | 4.1 视觉-语言-动作模型与 GR00T · 4.2 步行射门：GR00T 的输入、输出与数据 · 4.3 训练与验证 GR00T 步行射门 · 4.4 从球场走向更多环境 |
| 5 参考与帮助 | 5.1 模型与数据 · 5.2 故障排查 · 5.3 继续学习与参考资料 |
