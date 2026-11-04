---
myst:
  html_meta:
    description: 阅读两个社区技能（单脚站立循环、篮球平衡）的真实奖励代码：目标是什么、读取哪些量、公式怎样写、可能怎样失败、作者怎样检验，再用同样的步骤构思新任务。
---

# 3.5 （拓展）社区技能：奖励设计

:::{lesson-intro}
:meta: 第 3.5 节 · 阅读与分析 · 预计学习时长 45 分钟

阅读两个社区技能（单脚站立循环、篮球平衡）的真实奖励代码：目标是什么、读取哪些量、公式怎样写、可能怎样失败、作者怎样检验，再用同样的步骤构思新任务。
:::

```{symbols}
dt:
  tex: \Delta t
  text: 控制步长：两次策略输出之间的时间。本节两个技能都是 0.02 s（50 Hz）。
alpha_f:
  tex: \alpha
  text: 姿势混合进度：0 表示双脚站立，1 表示单脚站立。指令切换后，它以 1.5 s 从一端平滑走到另一端。它只在奖励中使用，策略观测里只有指令本身。
p_target:
  tex: p^{*}_{xy}
  text: 质心的目标水平位置（m）：从两脚中点逐渐移到支撑脚上方。
p_mid:
  tex: p_{\mathrm{mid}}
  text: 两只脚的中点（水平位置，m）。
p_stance:
  tex: p_{\mathrm{stance}}
  text: 支撑脚的水平位置（m）。
c_com:
  tex: c_{xy}
  text: 整机质心的水平位置（m），由仿真对全身所有部件计算。
sigma_c:
  tex: \sigma_c
  text: 质心误差尺度：0.03 m，训练第 700 轮后收紧到 0.02 m。
r_com:
  tex: r_{\mathrm{com}}
  text: 质心目标项（权重 3）。
S:
  tex: S
  text: 平滑阶跃（smoothstep）：混合进度在区间 $[a,b]$ 内时从 0 平滑升到 1，之前为 0，之后为 1。
tau:
  tex: \tau
  text: 平滑阶跃内部的归一化进度，截断在 0 到 1 之间。
a_lo:
  tex: a
  text: 平滑阶跃开始上升的位置。
b_hi:
  tex: b
  text: 平滑阶跃升到 1 的位置。
clip:
  tex: \operatorname{clip}
  text: 截断：把数值限制在给定区间内。
k_swing:
  tex: k_{\mathrm{swing}}
  text: 要抬起的那只脚此刻是否接触地面（1 为接触，0 为离地）。
k_stance:
  tex: k_{\mathrm{stance}}
  text: 支撑脚此刻是否接触地面（1 为接触）。
z_swing:
  tex: z_{\mathrm{swing}}
  text: 要抬起的那只脚离地面的高度（m）。
r_contact:
  tex: r_{\mathrm{contact}}
  text: 抬起脚的接触项（权重 0.5）：该站立时奖励接触，该抬脚时惩罚接触。
r_clear:
  tex: r_{\mathrm{clear}}
  text: 抬脚高度项（权重 1.5）：目标 0.10 m，尺度 0.06 m。
p_ball:
  tex: p_{\mathrm{ball}}
  text: 篮球球心位置（m），直接从仿真状态读取；部署的策略看不到它。
p_root:
  tex: p_{\mathrm{root}}
  text: 机器人根部（机身）位置（m）。
p_foot:
  tex: p_{\mathrm{foot}}
  text: 某只脚的足底参考点位置（m）。
z_root:
  tex: z_{\mathrm{root}}
  text: 机身根部高度（m）。
z_ball:
  tex: z_{\mathrm{ball}}
  text: 球心高度（m）。
R_ball:
  tex: R
  text: 篮球半径，0.12 m（7 号球）。
g_x:
  tex: g_x
  text: 重力方向在机身坐标系中的前后分量（无单位）；站直时为 0。
g_y:
  tex: g_y
  text: 重力方向在机身坐标系中的左右分量（无单位）；站直时为 0。
v_ball:
  tex: v_{\mathrm{ball}}
  text: 篮球的水平速度（m/s），仿真状态。
r_center:
  tex: r_{\mathrm{center}}
  text: 机身对准球心项（权重 2）。
r_feet:
  tex: r_{\mathrm{feet}}
  text: 双脚贴在球面上项（权重 1），两只脚取平均。
r_height:
  tex: r_{\mathrm{height}}
  text: 站在球顶的高度项（权重 1）。
r_up:
  tex: r_{\mathrm{up}}
  text: 机身直立项（权重 1）。
r_ball:
  tex: r_{\mathrm{ball}}
  text: 球速平方（权重 −0.05）：零指令时让球保持安静，权重小，指令要求滚动时仍然划算。
exp:
  tex: \exp
  text: 指数函数：把平方误差变成 0 到 1 之间的平滑奖励。
```

{#community}
## 同一机器人，可以学什么

3.4 节用六步设计了上下坡的奖励。社区作者用同一只 Microduck 训练了很多别的技能，其中不少公开了策略、训练代码和评测记录。可以从这两个入口浏览：

- [Microduck 官方网页模拟器 ↗](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)：Pollen Robotics 发布的浏览器模拟器，可以直接在网页里试用官方策略，不需要安装。
- [Hugging Face 上的 Microduck 模型 ↗](https://huggingface.co/models?search=microduck)：搜索 “microduck” 可以看到社区上传的各种策略，每个模型卡说明它的指令格式、训练方法和已知局限。

本节精读其中两个：单脚站立循环用门控安排动作的先后，篮球平衡区分奖励里的特权信息与部署时的观测。硬件相同，变化的是目标、奖励、初始状态与结束条件。它们都在作者自己的 MuJoCo / mjlab 环境中训练和评测，下面的成绩是作者报告的仿真结果，本课程没有在 Newton 场景或真机上复测。

这两个模型和它们的训练代码都不在课程仓库和容器中，课程命令也不会加载它们；代码节选来自作者公开的固定版本，只用于阅读，链接指向那个版本的完整文件。演示视频是课程网页保存的录像副本。

{#how-to-read}
## 遵循这样的思路分析技能

每个技能都按同样的顺序读：

| 读什么 | 要回答的问题 |
|---|---|
| 目标 | 作者想让机器人做成什么？成功怎样判定？ |
| 可观测量 | 奖励读取了哪些量？单位和坐标系是什么？哪些是策略看不到的仿真状态？ |
| 公式 | 每一项是什么形状？尺度、符号和门控是什么？ |
| 代码 | 公式在作者代码里怎样实现？实际传入的参数是多少？ |
| 失败方式 | 作者记录过哪些“钻空子”或失败？用什么办法堵上？ |
| 成功检查 | 作者用什么独立评测判断结果？还有哪些没有验证？ |

读社区奖励代码前，先知道三件事：

- **奖励按步长缩放。** mjlab 1.3.0 的奖励管理器和 Isaac Lab 一样，每一步把各项原始值乘以权重和控制步长 $\sym{dt}$ 再相加（[源码 ↗](https://github.com/mujocolab/mjlab/blob/19d6c06c17572dafcb31ad5110a902006b58d944/src/mjlab/managers/reward_manager.py)）。所以一项如果只在某一步返回“数值 ÷ $\sym{dt}$”，乘回 $\sym{dt}$ 之后正好一次性支付“权重 × 数值”。
- **符号要逐项看。** 多数惩罚项返回非负值，再配负权重；但也有例外，例如单脚站立里有一项返回负值、配正权重。只看权重的正负会读错。
- **函数默认值不等于实际参数。** 奖励函数的默认参数常常和配置里传入的不同，要以注册奖励项时的 `params` 为准。

{#flamingo}
## 单脚站立循环：用门控安排动作的先后

:::{course-video} media/community-flamingo-hd
:alt: 社区单脚站立循环的作者预览：Microduck 先双脚站立，再把重心移到一只脚上、抬起另一只脚保持，放下双脚站好后，换另一侧单脚站立。

社区单脚站立循环的作者预览：作者在 MuJoCo 中运行导出的 ONNX 策略，按预设的指令顺序先用一侧单脚站立，放下后再换另一侧，录像在第二次单脚站立中结束。这是原作者的仿真环境，不是本课程在 Newton 中的运行。观察重心先移到支撑脚、再抬脚，放下时顺序相反。
:::

**目标。** [单脚站立循环 ↗](https://huggingface.co/RemiFabre/microduck-flamingo-cycle) 按指令在双脚站立和单脚站立之间切换：指令 `[flag, side, 0]` 中 `flag = 1` 时把重心移到 `side` 指定的支撑脚，抬起另一只脚约 9 cm 并保持；`flag = 0` 时放下脚、双脚站好。每次过渡约 1.5 s，任何时候都可以打断。上面是作者模型卡附带的仿真预览，可以先看一遍动作。

**可观测量。** 整机质心（作者特意用全身质心，而不是机身单个部件的质心：头部约占总质量的 38%，抬腿后两者相差几厘米）、两只脚的位置和接触、重力在机身坐标系中的方向、关节角与角速度。奖励中还用到一个内部量：混合进度 $\sym{alpha_f}$。它在指令切换后用 1.5 s 从 0 走到 1（或反过来），策略只观测指令本身。

**公式。** 作者没有写死“先移重心、再抬脚”的时间表，而是让每一项随 $\sym{alpha_f}$ 打开或关闭：

```{formula} 质心目标随混合进度移动（权重 3）
\sym{p_target}=(1-\sym{alpha_f})\,\sym{p_mid}+\sym{alpha_f}\,\sym{p_stance},\qquad \sym{r_com}=\sym{exp}\!\left(-\frac{\|\sym{c_com}-\sym{p_target}\|^2}{\sym{sigma_c}^2}\right)
```

```{formula} 平滑阶跃门控
\symx{S}{S(\sym{alpha_f};\sym{a_lo},\sym{b_hi})}=\sym{tau}^2(3-2\sym{tau}),\qquad \sym{tau}=\sym{clip}\!\left(\frac{\sym{alpha_f}-\sym{a_lo}}{\sym{b_hi}-\sym{a_lo}},0,1\right)
```

```{formula} 抬起脚的接触项（权重 0.5）与抬脚高度项（权重 1.5）
\sym{r_contact}=\sym{k_swing}\left[1-2\,\symx{S}{S(\sym{alpha_f};0.4,0.9)}\right],\qquad \sym{r_clear}=\sym{exp}\!\left[-\left(\frac{\sym{z_swing}-0.10}{0.06}\right)^2\right]\sym{k_stance}\,\symx{S}{S(\sym{alpha_f};0.6,1.0)}
```

于是动作的先后自然出现：质心目标先移向支撑脚；$\sym{alpha_f}$ 还小于 0.4 时，抬起脚触地得 +1，超过 0.9 后触地变成 −1，两者之间连续过渡，$\sym{alpha_f} = 0.65$ 时正好为 0；抬脚高度要等 $\sym{alpha_f}$ 超过 0.6、并且支撑脚着地时才计分，避免靠单脚蹦跳拿分。放下脚是同一过程的反向。

```{code-story}
label: Python · 社区作者源码节选
source: https://github.com/pollen-robotics/microduck_rl/blob/0bf989701f14d4008c866ffd6f09067870f99379/src/mjlab_microduck/tasks/mdp.py#L7523-L7696
language: python
code: |
  def _smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
      t = torch.clamp((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
      return t * t * (3.0 - 2.0 * t)
  def fl_com_target(
      env: ManagerBasedRlEnv,
      command_name: str,
      left_cfg: SceneEntityCfg,
      right_cfg: SceneEntityCfg,
      std: float = 0.02,
  ) -> torch.Tensor:
      """Gaussian on CoM_xy vs the BLENDED support target: (1−α)·midpoint(feet) +
      α·stance foot. The leader of the stack: the weight shift IS the transition."""
      flag, side, alpha = _fl_ctx(env, command_name)
      asset, left, right = _fl_site_xyz(env, left_cfg, right_cfg)
      stance = torch.where((side > 0).unsqueeze(1), right, left)
      mid = 0.5 * (left + right)
      a = alpha.unsqueeze(1)
      target_xy = (1.0 - a) * mid[:, :2] + a * stance[:, :2]
      com_xy = _robot_com_xy(env, asset)
      dist2 = ((com_xy - target_xy) ** 2).sum(dim=-1)
      return torch.exp(-dist2 / (std ** 2))
  def fl_swing_foot_contact(
      env: ManagerBasedRlEnv, command_name: str, sensor_name: str,
      lo: float = 0.4, hi: float = 0.9,
  ) -> torch.Tensor:
      """Swing-foot contact: +1 when standing is commanded (α ≤ lo), −1 once the
      lift is due (α ≥ hi), neutral in between. Mixed sign by design."""
      _, side, alpha = _fl_ctx(env, command_name)
      _, swing = _fl_feet_found(env, sensor_name, side)
      return swing * (1.0 - 2.0 * _smoothstep(alpha, lo, hi))


  def fl_swing_foot_clear(
      env: ManagerBasedRlEnv,
      command_name: str,
      left_cfg: SceneEntityCfg,
      right_cfg: SceneEntityCfg,
      sensor_name: str,
      target: float = 0.05,
      std: float = 0.03,
      lo: float = 0.6,
      hi: float = 1.0,
  ) -> torch.Tensor:
      """Gaussian on the swing-foot height around ``target``, gated on stance
      contact (no hopping) and on the lift being due (smoothstep of α)."""
      _, side, alpha = _fl_ctx(env, command_name)
      _, left, right = _fl_site_xyz(env, left_cfg, right_cfg)
      swing = torch.where((side > 0).unsqueeze(1), left, right)
      z = swing[:, 2] - env.scene.terrain.env_origins[:, 2]
      reward = torch.exp(-((z - target) / std) ** 2)
      stance_found, _ = _fl_feet_found(env, sensor_name, side)
      return reward * stance_found * _smoothstep(alpha, lo, hi)
steps:
- lines: [0, 3]
  title: 平滑阶跃
  body: 把混合进度先截断到 0–1，再用 $\sym{tau}^2(3-2\sym{tau})$ 平滑过渡，两端斜率为零，门控打开时奖励不会突然跳变。
- lines: [3, 21]
  title: 质心目标在两脚中点与支撑脚之间移动
  body: 作者在注释里称它为整组奖励的“领头项”：重心转移本身就是过渡。注意函数默认 `std=0.02`，实际配置传入 0.03 m，训练第 700 轮后再收紧到 0.02 m。
- lines: [21, 32]
  title: 正负号随进度翻转的接触项
  body: 抬起脚触地时为 1，乘以 $1-2S$：该站立时为 +1，该抬脚时为 −1，中间连续过渡，混合进度为 0.65 时正好为 0。作者注明这是有意设计的混合符号。
- lines: [32, 52]
  title: 双重门控的抬脚高度
  body: 高度高斯核再乘以“支撑脚着地”和“该抬脚了”两道门。函数默认目标 0.05 m，实际配置传入 0.10 m、尺度 0.06 m。
```

其余奖励约束支撑、姿态和动作平稳。注意防止向支撑侧倾倒过多的那一项返回非正值，所以配的是正权重 4.0：只看权重的正负，判断不了一项是奖励还是惩罚。完整权重见作者的[配置 ↗](https://github.com/pollen-robotics/microduck_rl/blob/0bf989701f14d4008c866ffd6f09067870f99379/src/mjlab_microduck/tasks/microduck_flamingo_cycle_env_cfg.py)。

**作者记录的失败与对策。**

- 第一阶段要求抬脚高度 0.05 ± 0.03 m，但目标姿态里脚实际抬起约 0.17 m，两项奖励互相矛盾；改为 0.10 ± 0.06 m 后，在 5 cm 处得 0.5，10 cm 处得 1，17 cm 处仍有 0.26。
- 第一阶段在 −1.0 的动作变化惩罚上再叠加 0.25 m/s 的推力，回报减半且再也没恢复；循环任务因此把推力先限制在 0.15 m/s、动作变化惩罚限制在 −0.5。第一次循环训练遇到向后 0.18 m/s 以上的推力会直接跌倒，所以发布的 Hard 版本在第 1400 轮之后再加入 0.25 m/s 的推力阶段。
- 被推得太狠时，抬起脚触地只是一笔小代价、不会终止回合，而保持站立的价值大得多，于是学到的应对是先放下脚、再重新抬起。作者把这写成设计意图，而不是漏洞。

**成功检查与局限。** 作者在带 BAM XL330 舵机模型的 CPU MuJoCo 中运行了 10 个固定情景，全部通过（左右两侧循环、向两侧 0.1 m/s 和向前 0.15 m/s 的推力、向抬脚一侧 0.3 m/s 的推力后短暂放脚再抬起、从静止单脚站立放下、保持 10 秒）；24 次 0.05–0.25 m/s 的随机推力测试中 20 次保持、2 次放脚后恢复、2 次跌倒。已知局限：向后推力达到 0.18 m/s 会跌倒；向支撑侧推力超过约 0.15 m/s 会放下脚；只在仿真中运行，从未上真机。

{#basketball}
## 篮球平衡：奖励能看见球，策略看不见

:::{course-video} media/community-balance-hd
:alt: 社区篮球平衡演示。观察脚下支撑面随球转动而变化，机器人怎样持续调整姿态。

社区篮球平衡演示（早期版本的录像）。观察脚下支撑面随球转动而变化，机器人怎样持续调整姿态。下文分析的是作者最新发布的 b11 版本。
:::

**目标。** [篮球平衡 ↗](https://huggingface.co/HannesVonEssen/microduck-basketball) 让机器人站在一个自由滚动的 7 号篮球（直径 0.24 m、质量 0.62 kg）顶上保持平衡，并按速度指令移动；指令为 `[0, 0, 0]` 时原地平衡。回合从机器人被放在球顶开始，它不会自己爬上球，也不会从地面爬起。

**可观测量。** 奖励读取球心位置和球速、机身位置、两只脚的足底参考点、重力在机身坐标系中的方向。关键在于：**部署的策略看不见球**。作者最新发布的 b11 版本中，策略只用本体感知和速度指令，经过一层 LSTM 记忆，从自身状态的变化里推测脚下的球；训练时的价值网络（critic）可以读取球的状态，但它不进入导出的 ONNX。奖励在仿真里计算，所以可以直接使用球的位置。

```{formula} 球上站姿：对准球心与双脚贴球（权重 2、1）
\sym{r_center}=\sym{exp}\!\left(-\frac{\|\sym{p_ball}-\sym{p_root}\|_{xy}^2}{0.04^2}\right),\qquad \sym{r_feet}=\frac{1}{2}\sum_{\mathrm{L,R}}\sym{exp}\!\left[-\left(\frac{\|\sym{p_foot}-\sym{p_ball}\|-(\sym{R_ball}+0.012)}{0.02}\right)^2\right]
```

```{formula} 球上站姿：高度与直立（权重 1、1）
\sym{r_height}=\sym{exp}\!\left[-\left(\frac{\sym{z_root}-\sym{z_ball}-(\sym{R_ball}+0.125)}{0.03}\right)^2\right],\qquad \sym{r_up}=\sym{exp}\!\left(-\frac{\sym{g_x}^2+\sym{g_y}^2}{0.3^2}\right)
```

```{formula} 球速惩罚（权重 −0.05）
\sym{r_ball}=\|\sym{v_ball}\|^2
```

尺度都带长度单位：机身偏离球心 4 cm、脚底离开球面 2 cm、高度偏差 3 cm 时，对应项降到 $e^{-1}\approx 0.37$。$\sym{r_ball}$ 的权重很小，零指令时让球安静，但指令要求滚动时，速度跟踪奖励仍然更划算。终止条件也读球：机身相对球心太低（低于 $\sym{R_ball}$ + 0.05 m）、水平偏离超过 0.16 m 或倾斜超过 55° 即判为从球上掉落。

```{code-story}
label: Python · 社区作者源码节选
source: https://github.com/Vottivott/microduck-playground/blob/aa5bd7909873a807fb9f4045f491519c443142e7/src/mjlab_microduck/tasks/mdp.py#L8137-L8186
language: python
code: |
  def basketball_centered(env: ManagerBasedRlEnv, std: float = 0.04) -> torch.Tensor:
      """Gaussian on the horizontal offset root <-> ball centre (mass over the ball)."""
      d = _bb_ball_pos(env)[:, :2] - env.scene["robot"].data.root_link_pos_w[:, :2]
      return torch.exp(-(d * d).sum(dim=-1) / (std * std))


  def basketball_feet_on_ball(
      env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg, ball_radius: float = 0.12, sole_gap: float = 0.012, std: float = 0.02
  ) -> torch.Tensor:
      """Per-foot Gaussian on (|foot site - ball centre| - (R + gap)); both soles
      resting on the ball pay 1, a lifted or dangling foot pays ~0 for that foot."""
      robot = env.scene[asset_cfg.name]
      feet = robot.data.site_pos_w[:, asset_cfg.site_ids, :]
      d = torch.linalg.norm(feet - _bb_ball_pos(env)[:, None, :], dim=-1) - (ball_radius + sole_gap)
      return torch.exp(-(d * d) / (std * std)).mean(dim=-1)


  def basketball_height(
      env: ManagerBasedRlEnv, ball_radius: float = 0.12, root_height: float = 0.125, std: float = 0.03
  ) -> torch.Tensor:
      """Gaussian on the root height above the ball centre (standing tall on the apex)."""
      dz = env.scene["robot"].data.root_link_pos_w[:, 2] - _bb_ball_pos(env)[:, 2] - (ball_radius + root_height)
      return torch.exp(-(dz * dz) / (std * std))


  def basketball_ball_speed_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
      """Squared horizontal ball speed (>= 0, use a negative weight): keeps the
      ball quiet at zero command; small weight so commanded rolling still pays."""
      v = torch.nan_to_num(env.scene["ball"].data.root_link_lin_vel_w[:, :2], nan=0.0)
      return (v * v).sum(dim=-1)


  # --- terminations -----------------------------------------------------------------


  def basketball_fell(
      env: ManagerBasedRlEnv,
      ball_radius: float = 0.12,
      fall_tilt_deg: float = 55.0,
      min_height: float = 0.05,
      max_offset: float = 0.16,
  ) -> torch.Tensor:
      """Off the ball: root too low above the ball centre, root horizontally off
      the ball, or tilted past ``fall_tilt_deg``."""
      robot = env.scene["robot"]
      ball = _bb_ball_pos(env)
      root = torch.nan_to_num(robot.data.root_link_pos_w, nan=0.0)
      dz = root[:, 2] - ball[:, 2]
      off = torch.linalg.norm(root[:, :2] - ball[:, :2], dim=-1)
      tilt = wrestle_tilt(env, "robot")
      return (dz < ball_radius + min_height) | (off > max_offset) | (tilt > math.radians(fall_tilt_deg))
steps:
- lines: [0, 6]
  title: 机身对准球心
  body: '`_bb_ball_pos` 直接读取仿真中的球心位置，这是策略观测里没有的特权信息。水平偏差越小，奖励越接近 1；权重 2。'
- lines: [6, 16]
  title: 双脚贴在球面上
  body: 每只脚到球心的距离应等于半径加 12 mm 的鞋底间隙；两只脚各算一个高斯核再取平均，悬空的那只脚得分接近 0。权重 1。
- lines: [16, 23]
  title: 站在球顶的高度
  body: 机身应在球心上方“半径 + 0.125 m”处，即普通站立高度放到球顶。权重 1。
- lines: [23, 30]
  title: 球速惩罚
  body: 返回非负值，注册时配 −0.05。注释写明权重故意很小。
- lines: [30, 51]
  title: 终止条件
  body: 掉落判定同样读球：太低、偏离球顶或倾斜过大都结束回合。终止让“掉下球”之后本可获得的奖励全部消失，所以提前跌落也会影响策略学到的平衡行为。
```

其余沿用步行配方的速度跟踪、姿态、直立、动作平滑和关节限位等项，完整权重见作者的[配置 ↗](https://github.com/Vottivott/microduck-playground/blob/aa5bd7909873a807fb9f4045f491519c443142e7/src/mjlab_microduck/tasks/microduck_basketball_env_cfg.py)。训练初期还给球加了一个可调的“弹簧固定”，从完全固定逐步放松到自由滚动，存活够久的环境升级、很快掉下的降级。

**作者记录的失败与对策。** 根据作者的[训练记录 ↗](https://github.com/Vottivott/microduck-playground/blob/aa5bd7909873a807fb9f4045f491519c443142e7/experiments/basketball/TRAINING.md)，去掉球的直接观测后，策略更难保持平衡；加入 LSTM 记忆改善了表现，教师-学生蒸馏则没能保住平衡。之后在**奖励保持不变**时只调整学习率和推力练习的频率，表现仍有提高。奖励只是手段之一：策略能看到什么、记得什么、练过什么，同样决定结果。

**成功检查与局限。** 作者用 3 个种子 × 1024 个环境、共 3072 次 60 秒测试，首次掉落即算失败：b11 存活 20 秒 98.86%，存活 60 秒 97.01%；看得见球的参考版本为 99.67% 和 99.12%。转向跟踪误差 1.26 rad/s，明显差于参考版本的 0.76 rad/s，动作也更不平滑；这些跟踪指标只统计掉落之前的样本。导出的 ONNX 需要额外输入和输出 LSTM 状态，普通的前馈加载器不能直接运行它。板载计算的实时性和真机表现都没有测试。

{#compare}
## 两个技能放在一起比较

| | 单脚站立循环 | 篮球平衡 |
|---|---|---|
| 目标类型 | 按指令在两种姿势间切换 | 持续平衡并跟随速度 |
| 关键写法 | 用混合进度门控每一项 | 用球的真实状态写奖励和终止 |
| 特权信息 | 整机质心、接触 | 球心位置与球速（策略看不到） |
| 终止 | 沿用摔倒终止 | 从球上掉落 |
| 作者的检查 | 10 个情景 + 24 次随机推力 | 3 种子 × 1024 环境 × 60 秒 |
| 都没有验证 | 真机 | 真机与板载实时性 |

检查奖励时问四件事：目标是否有对应的奖励或验收；能否不完成任务也拿高分；奖励用到的特权信息与部署时的观测是否分清；最后是否用独立判据评测。

{#new-task}
## 为新任务设计奖励

挑一个这两个之外的新任务，例如“原地转一圈后面向原方向站定”或“坐下再站起来”，按 3.4 节的六步写在纸上：

1. 用可测量的量写出成功判据和失败条件，在动笔写奖励之前固定下来。
2. 列出需要的量，标明单位、坐标系，以及部署时策略能否得到。
3. 为每条要求选形状：跟踪、惩罚、一次性事件；需要先后顺序时，参考单脚站立的门控。
4. 估算每一项的典型大小，检查有没有一项会压倒其他项。
5. 写出至少两种可能被钻的空子，以及终止条件会怎样改变奖励的含义。
6. 计划独立评测与消融：用哪些种子、多少次、比较哪些指标，同样的训练预算。

::::{quiz}
一个新任务的奖励曲线持续上升，导出的 ONNX 也通过了 61 维输入、14 维输出的接口检查。这能说明任务完成了吗？

:::{answer}
不能。奖励上升只说明策略更会拿到这组奖励的分数；接口检查只说明文件能被控制器调用。要用训练前固定的验收判据、独立种子和实际观察来判断，描述里想要的效果如果没有写成奖励或验收判据，就不能指望训练自动得到。
:::
::::

{#compatible}
## 同样的接口，不同的约定

ONNX 是一种模型文件格式，记录网络结构和权重，但不规定机器人技能怎样被调用。单脚站立和官方行走策略都使用 61 维输入、14 维输出的共同接口，却不能互相替换：单脚站立把速度指令的前两个分量当作“是否抬脚”和“哪一侧”，行走策略才把它们当作速度。篮球策略在同样的 61 维观测之外，还要输入并传回 LSTM 的记忆状态。课程中的官方行走与踢球模型也一样：都读取 61 维输入、输出 14 个数，但训练任务不同，仍不能随意互换。

文件能成功加载，只说明格式可读。要让机器人完成动作，指令每个分量的含义、单位、关节顺序、动作缩放和结束条件都必须与控制器一致；[模型与数据页](models.md)列出了课程各固定权重的用途。技能之间的衔接也要单独设计：2.1 节的键盘射门在踢球动作结束后，用 0.12 s 平滑切回行走策略的站立动作，这个过渡并不在踢球模型里。

{#eyes}
## 会执行动作，还不等于会完成任务

到目前为止，都是我们告诉机器人“向前走”“抬起左脚”或“在球上保持平衡”。策略利用关节和 IMU 反馈保持平衡，但足球在哪里、该走哪条路，还要由别的模块来判断。第 4 部分先由人把这些判断写成板载策略；第 5 部分再训练 GR00T，让它根据相机图像、指令和身体状态发出技能请求，把这些判断交给可学习的模型。

{#references}
## 参考资料

{.references}
- [Microduck 官方网页模拟器（Hugging Face Space）↗](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)
- [Hugging Face：搜索 Microduck 模型 ↗](https://huggingface.co/models?search=microduck)
- [Isaac Lab：Microduck 任务源码 ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
- [mjlab 1.3.0：奖励管理器源码 ↗](https://github.com/mujocolab/mjlab/blob/19d6c06c17572dafcb31ad5110a902006b58d944/src/mjlab/managers/reward_manager.py)
- [社区单脚站立循环 ↗](https://huggingface.co/RemiFabre/microduck-flamingo-cycle)
- [社区单脚站立循环：训练源码（固定版本）↗](https://github.com/pollen-robotics/microduck_rl/tree/0bf989701f14d4008c866ffd6f09067870f99379/src/mjlab_microduck/tasks)
- [社区篮球平衡：模型卡与评测 ↗](https://huggingface.co/HannesVonEssen/microduck-basketball)
- [社区篮球平衡：训练源码 ↗](https://github.com/Vottivott/microduck-playground/tree/main/experiments/basketball)
