---
myst:
  html_meta:
    description: 运行板载策略，观察感知、阶段决策与底层技能怎样在控制周期中协同完成射门。
---

# 4.3 射门！

:::{lesson-intro}
:meta: 第 4.3 节 · 预计学习时长 20 分钟

运行板载策略，观察感知、阶段决策与底层技能怎样在控制周期中协同完成射门。
:::

先回顾 [4.2 节](06-onboard.md) 的射门阶段，再运行实验，观察这些阶段怎样连续切换。

{#run}
## 射门！

先按 [1.2 节](setup.md) 准备好课程容器，再回到容器终端；容器已在运行时，可以在宿主机用 `docker exec -it microduck-workshop bash` 进入。然后运行：

```{code-block} bash
:caption: Terminal · 容器内
:class: terminal

python3 scripts/run.py football --output outputs/onboard-football
```

:::{enrichment} 参数详解

- **`--output`**：一个新目录；各轮记录分别保存在 `episode-001`、`episode-002` 等子目录
:::

机器人会自动观察、接近足球并尝试射门。第一轮从名义起点出发。点击 Isaac Sim 窗口，按 **R** 就能重新开始：窗口保持打开，每轮都从站立初始化开始，起点随机偏移，前后、左右各最多 5 厘米，朝向顺时针或逆时针最多偏转 15°；足球和球门的位置不变。场景重新初始化时稍等片刻，随后任务会自动开始。

学生实验不设总时限，进球、未进球或暂时无法出脚时都不会自动暂停仿真。传感器持续刷新；按 **R** 才会重置当前尝试。关闭 Isaac 应用，或在启动它的终端按 **Ctrl+C**，即可停止程序。感知、对准或交接失败只会暂停当前动作，并尝试重新观察。

仿真要同时渲染第三人称视口、头部 RGB 相机和 ToF 等多路画面，运行速度取决于硬件性能，窗口里的动作看起来可能卡顿；任务按仿真时间推进。

:::{course-image} media/isaac-ui-onboard-sensors.png
:alt: 板载射门的 Isaac Sim 完整界面，同时显示绿色球场、头部 RGB 图像和 8×8 ToF 距离。

左侧观察机器人，右侧对照它此刻看到的图像与距离：识别框是否稳定地框住球和球门，脚前的 ToF 读数是否随接近而变小。NR 表示没有回波，不能据此认为前方没有障碍。
:::

`summary.json` 汇总已保存的各轮记录；单轮 `manifest.json` 的 `navigation_retries` 记录中途重新观察的原因。识别框仍在，并不表示几何位置已经通过检查；近距位置不够新鲜时，策略会停步重新观察。也要检查接近时的左右偏差和朝向、出脚条件是否满足，以及触球后是否恢复平衡。

按 R 重置时，程序先保存当前记录，再开始下一轮。

{#timing}
## 板载策略的控制周期

```{control-timing}
kind: onboard-theory
```

在 <a href="08-groot-interface.html#timing">5.2 节</a>，对比用 GR00T 生成高层请求时，感知与动作更新的周期有哪些变化。
