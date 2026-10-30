---
myst:
  html_meta:
    description: 按课程主题查阅官方源码、文档和论文。
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 5.3 继续学习与参考资料

:::{lesson-intro}
:meta: 阅读资源

按课程主题查阅官方源码、文档和论文。
:::

{#robotics}
## 机器人与仿真

建议先从任务配置入手，理解观测、动作和奖励，再研究物理引擎与传感器接口。课程命令对应固定版本，官方文档可能描述的是更新的版本，比较参数前请先核对版本。

{#next}
## 建议的延伸练习

- 调整 3.2 节的训练预算，并在相同的独立测试种子上比较表现。
- 分析 4.3 节验证中的失败回合：哪些来自观察，哪些来自动作请求与技能衔接？
- 阅读 [SO-101 Sim-to-Real 课程的 GR00T 章节 ↗](https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/10-groot.html)，对照 4.2 节的示范数据设计，列出 Microduck 与 SO-101 的共同步骤和不同动作接口。
- 用 4.4 节的组合计数器规划陌生房间评测，说明哪些条件用于训练、哪些留作测试。

{#references}
## 参考资料

{.references}
- [Pollen Robotics：Microduck 硬件 ↗](https://github.com/pollen-robotics/microduck)
- [Isaac Lab：Microduck 任务源码 ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
- [Newton 物理引擎 ↗](https://github.com/newton-physics/newton)
- [PPO 原论文 ↗](https://arxiv.org/abs/1707.06347)
- [RSL-RL：强化学习实现 ↗](https://github.com/leggedrobotics/rsl_rl)
- [Nav2 文档 ↗](https://docs.nav2.org/)
- [Ultralytics 文档 ↗](https://docs.ultralytics.com/)
- [VL53L8CX 传感器资料 ↗](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8cx.html)
