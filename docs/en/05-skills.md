---
myst:
  html_meta:
    description: "Read the real reward code of three community skills (flamingo cycle, basketball balance and moonwalk): what the goal is, which quantities are read, how the formulas are written, how they can fail and how the authors checked them, then use the same steps to plan a new task."
---

<!-- Course source of truth: edit this Markdown directly. Migrated from content/course.json sha256 05a2fa050056450b… (zh-depth b42bebdaf018…) on 2026-10-29; the JSON is a read-only archive. -->

# 3.4 Community skills: from goal to reward

:::{lesson-intro}
:meta: Section 3.4 · Reading and analysis · Estimated learning time: 45 minutes

Read the real reward code of three community skills (flamingo cycle, basketball balance and moonwalk): what the goal is, which quantities are read, how the formulas are written, how they can fail and how the authors checked them, then use the same steps to plan a new task.
:::

```{symbols}
dt:
  tex: \Delta t
  text: 'Control step: the time between two policy outputs. All three skills here use 0.02 s (50 Hz).'
alpha_f:
  tex: \alpha
  text: 'Posture blend: 0 means standing on two feet, 1 means standing on one foot. After a command change it moves smoothly from one end to the other in 1.5 s. It is used only in the reward; the policy observes only the command itself.'
p_target:
  tex: p^{*}_{xy}
  text: 'Target horizontal position of the centre of mass (m): moves from the midpoint of the feet to above the stance foot.'
p_mid:
  tex: p_{\mathrm{mid}}
  text: Midpoint between the two feet (horizontal position, m).
p_stance:
  tex: p_{\mathrm{stance}}
  text: Horizontal position of the stance foot (m).
c_com:
  tex: c_{xy}
  text: Horizontal position of the whole-robot centre of mass (m), computed by the simulator over all parts.
sigma_c:
  tex: \sigma_c
  text: 'Centre-of-mass error scale: 0.03 m, tightened to 0.02 m after training iteration 700.'
r_com:
  tex: r_{\mathrm{com}}
  text: Centre-of-mass target term (weight 3).
S:
  tex: S
  text: 'Smoothstep: rises smoothly from 0 to 1 while the blend is inside $[a,b]$; 0 before, 1 after.'
tau:
  tex: \tau
  text: Normalized progress inside the smoothstep, clamped to between 0 and 1.
a_lo:
  tex: a
  text: Where the smoothstep starts to rise.
b_hi:
  tex: b
  text: Where the smoothstep reaches 1.
clip:
  tex: \operatorname{clip}
  text: 'Clamp: limits a value to a given interval.'
k_swing:
  tex: k_{\mathrm{swing}}
  text: Whether the foot to be lifted touches the ground right now (1 contact, 0 airborne).
k_stance:
  tex: k_{\mathrm{stance}}
  text: Whether the stance foot touches the ground right now (1 contact).
z_swing:
  tex: z_{\mathrm{swing}}
  text: Height of the foot to be lifted above the ground (m).
r_contact:
  tex: r_{\mathrm{contact}}
  text: 'Swing-foot contact term (weight 0.5): rewards contact while standing is due, penalizes it once the lift is due.'
r_clear:
  tex: r_{\mathrm{clear}}
  text: 'Swing-foot height term (weight 1.5): target 0.10 m, scale 0.06 m.'
p_ball:
  tex: p_{\mathrm{ball}}
  text: Position of the basketball centre (m), read directly from the simulation state; the deployed policy cannot see it.
p_root:
  tex: p_{\mathrm{root}}
  text: Position of the robot root (body) (m).
p_foot:
  tex: p_{\mathrm{foot}}
  text: Position of one foot's sole reference point (m).
z_root:
  tex: z_{\mathrm{root}}
  text: Height of the robot root (m).
z_ball:
  tex: z_{\mathrm{ball}}
  text: Height of the ball centre (m).
R_ball:
  tex: R
  text: Basketball radius, 0.12 m (size 7).
g_x:
  tex: g_x
  text: Forward/backward component of the gravity direction in the body frame (dimensionless); 0 when upright.
g_y:
  tex: g_y
  text: Left/right component of the gravity direction in the body frame (dimensionless); 0 when upright.
v_ball:
  tex: v_{\mathrm{ball}}
  text: Horizontal velocity of the basketball (m/s), simulation state.
r_center:
  tex: r_{\mathrm{center}}
  text: Body-over-ball-centre term (weight 2).
r_feet:
  tex: r_{\mathrm{feet}}
  text: Feet-resting-on-the-ball term (weight 1), averaged over both feet.
r_height:
  tex: r_{\mathrm{height}}
  text: Standing-height-on-the-apex term (weight 1).
r_up:
  tex: r_{\mathrm{up}}
  text: Upright-body term (weight 1).
r_ball:
  tex: r_{\mathrm{ball}}
  text: 'Squared ball speed (weight −0.05): keeps the ball quiet at zero command; the weight is small so commanded rolling still pays.'
v_cmd:
  tex: v^{*}_{xy}
  text: Horizontal velocity command (m/s). In the moonwalk training the forward component is −0.25 to −0.08 m/s, i.e. backward.
v_xy:
  tex: v_{xy}
  text: Horizontal velocity in the body frame (m/s).
v_z:
  tex: v_z
  text: Vertical velocity in the body frame (m/s); 0 for ideal translation.
omega_cmd:
  tex: \omega^{*}_z
  text: Turn-rate command (rad/s).
sigma_v:
  tex: \sigma_v
  text: Velocity-tracking scale, $\sigma_v^2 = 0.1\ (\mathrm{m/s})^2$.
r_lin:
  tex: r_{\mathrm{lin}}
  text: Linear-velocity tracking term (weight 2 in the base walking recipe).
v_foot:
  tex: v_{\mathrm{foot}}
  text: Horizontal velocity of one foot in the world frame (m/s).
k_foot:
  tex: k_{\mathrm{foot}}
  text: Whether this foot touches the ground (1 contact).
r_slip:
  tex: r_{\mathrm{slip}}
  text: 'Foot-slip cost: counts feet that still move horizontally while in contact; negative weight.'
one:
  tex: \mathbf{1}
  text: 'Indicator: 1 when the condition in brackets holds, otherwise 0.'
exp:
  tex: \exp
  text: 'Exponential: turns a squared error into a smooth reward between 0 and 1.'
```

{#community}
## What else can the same robot learn?

Section 3.3 used six steps to design rewards for slopes. Community authors have trained many other skills on the same Microduck, and several have published their policies, training code and evaluation records. Start from these two entry points:

- [Official Microduck web simulator ↗](https://huggingface.co/spaces/pollen-robotics/microduck-simulator): a browser simulator published by Pollen Robotics where you can switch between and try the official walking, roller and other policies without installing anything.
- [Microduck models on Hugging Face ↗](https://huggingface.co/models?search=microduck): searching for “microduck” lists policies uploaded by the community; each model card describes its command format, training method and known limits.

This section reads three of them closely: the flamingo (one-leg) cycle uses gates to order its movements, basketball balance separates privileged information in the reward from what the deployed policy observes, and the backward glide shows that a written goal is not a verified capability. The hardware is the same; the goals, rewards, initial states and termination conditions change. All three were trained and evaluated in their authors' own MuJoCo / mjlab environments; the results below are the authors' simulation reports and have not been retested in the course's Newton scene or on the robot.

These three models and their training code are not in the course repository or container, and no course command loads them; every code excerpt comes from a pinned public revision of the author's work, is for reading only and links to the complete file at that revision. The demonstration videos are copies of recordings stored with the course pages.

{#how-to-read}
## Read three skills with one table

Read every skill in the same order:

| What to read | Question to answer |
|---|---|
| Goal | What should the robot achieve? How is success decided? |
| Observable quantities | Which quantities does the reward read? In which units and frames? Which are simulation states the policy cannot see? |
| Formula | What shape is each term? What are its scale, sign and gate? |
| Code | How does the author's code implement the formula? Which parameters are actually passed in? |
| Failure modes | Which loopholes or failures did the author record, and how were they closed? |
| Success checks | Which independent evaluation did the author use? What remains unverified? |

Before reading community reward code, know three things:

- **Rewards are scaled by the step.** Like Isaac Lab, the mjlab 1.3.0 reward manager multiplies each raw term by its weight and the control step $\sym{dt}$ before adding them up ([source ↗](https://github.com/mujocolab/mjlab/blob/19d6c06c17572dafcb31ad5110a902006b58d944/src/mjlab/managers/reward_manager.py)). A term that returns “value ÷ $\sym{dt}$” at one step therefore pays exactly “weight × value” once.
- **Check signs term by term.** Most penalties return non-negative values with a negative weight, but there are exceptions: one term in the flamingo cycle returns negative values with a positive weight. The sign of the weight alone can mislead you.
- **Function defaults are not the actual parameters.** Reward functions often have defaults that differ from what the configuration passes; the `params` used when the term is registered are what count.

{#flamingo}
## Flamingo cycle: gates set the order of movements

:::{course-video} media/community-flamingo-hd
:alt: Author preview of the community flamingo cycle: Microduck stands on both feet, shifts its weight onto one foot, lifts and holds the other, puts it down, then stands on the other foot.

Author preview of the community flamingo cycle: the author ran the exported ONNX policy in MuJoCo with a preset sequence of commands, standing on one foot, putting it down, then standing on the other; the recording ends during the second one-foot stance. This is the author's original simulation environment, not a Newton run of this course. Watch the weight move onto the stance foot before the other foot lifts, and the reverse order when it comes down.
:::

**Goal.** The [flamingo cycle ↗](https://huggingface.co/RemiFabre/microduck-flamingo-cycle) switches on command between standing on two feet and standing on one: in the command `[flag, side, 0]`, `flag = 1` shifts the weight onto the stance foot chosen by `side`, lifts the other foot about 9 cm and holds; `flag = 0` lowers it and stands on both feet. Each transition takes about 1.5 s and can be interrupted at any time. The author's simulation preview from the model card is above, so you can watch the motion first.

**Observable quantities.** The whole-robot centre of mass (the author deliberately uses the whole-body centre of mass rather than that of the single body part: the head is about 38% of the mass, and with a lifted leg the two differ by centimetres), the positions and contacts of both feet, the gravity direction in the body frame, joint angles and velocities. The reward also uses an internal quantity, the blend $\sym{alpha_f}$, which moves from 0 to 1 (or back) over 1.5 s after a command change; the policy observes only the command.

**Formula.** Instead of scripting “shift the weight, then lift the foot” as a timetable, the author lets each term switch on or off with $\sym{alpha_f}$:

```{formula} The centre-of-mass target moves with the blend (weight 3)
\sym{p_target}=(1-\sym{alpha_f})\,\sym{p_mid}+\sym{alpha_f}\,\sym{p_stance},\qquad \sym{r_com}=\sym{exp}\!\left(-\frac{\|\sym{c_com}-\sym{p_target}\|^2}{\sym{sigma_c}^2}\right)
```

```{formula} Smoothstep gate
\symx{S}{S(\sym{alpha_f};\sym{a_lo},\sym{b_hi})}=\sym{tau}^2(3-2\sym{tau}),\qquad \sym{tau}=\sym{clip}\!\left(\frac{\sym{alpha_f}-\sym{a_lo}}{\sym{b_hi}-\sym{a_lo}},0,1\right)
```

```{formula} Swing-foot contact (weight 0.5) and swing-foot height (weight 1.5)
\sym{r_contact}=\sym{k_swing}\left[1-2\,\symx{S}{S(\sym{alpha_f};0.4,0.9)}\right],\qquad \sym{r_clear}=\sym{exp}\!\left[-\left(\frac{\sym{z_swing}-0.10}{0.06}\right)^2\right]\sym{k_stance}\,\symx{S}{S(\sym{alpha_f};0.6,1.0)}
```

The order of movements then emerges by itself: the centre-of-mass target moves toward the stance foot first; while $\sym{alpha_f}$ is below 0.4, contact of the foot to be lifted earns +1, above 0.9 it costs −1, and in between it changes continuously, reaching exactly 0 at $\sym{alpha_f} = 0.65$; foot height counts only once $\sym{alpha_f}$ exceeds 0.6 and the stance foot is down, so hopping on one foot cannot earn it. Lowering the foot is the same process in reverse.

```{code-story}
label: Python · Community author source excerpt
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
  title: Smoothstep
  body: Clamp the blend to 0–1, then use $\sym{tau}^2(3-2\sym{tau})$ for a smooth transition with zero slope at both ends, so the reward does not jump when the gate opens.
- lines: [3, 21]
  title: The centre-of-mass target moves between the feet midpoint and the stance foot
  body: 'The author calls it the “leader” of the whole stack: the weight shift is the transition. Note the function default `std=0.02`; the configuration passes 0.03 m and tightens it to 0.02 m after iteration 700.'
- lines: [21, 32]
  title: A contact term whose sign flips with the blend
  body: 'Contact of the foot to be lifted is 1, multiplied by $1-2S$: +1 while standing is due, −1 once the lift is due, changing continuously in between and exactly 0 at a blend of 0.65. The author notes that the mixed sign is by design.'
- lines: [32, 52]
  title: Foot height behind two gates
  body: The height Gaussian is multiplied by two gates, “stance foot down” and “lift is due”. The function default target is 0.05 m; the configuration passes 0.10 m with scale 0.06 m.
```

The remaining terms constrain support, posture and smoothness. Note that the term against rolling too far toward the stance side returns non-positive values and therefore carries a positive weight of 4.0: the sign of a weight alone does not tell you whether a term rewards or penalizes. See the author's [configuration ↗](https://github.com/pollen-robotics/microduck_rl/blob/0bf989701f14d4008c866ffd6f09067870f99379/src/mjlab_microduck/tasks/microduck_flamingo_cycle_env_cfg.py) for all weights.

**Failures and fixes recorded by the author.**

- The first stage asked for a lifted-foot height of 0.05 ± 0.03 m, while the target pose lifts the foot about 0.17 m, so two reward terms contradicted each other; with 0.10 ± 0.06 m the term pays 0.5 at 5 cm, 1 at 10 cm and still 0.26 at 17 cm.
- In the first stage, a 0.25 m/s push stage on top of a −1.0 action-change penalty halved the reward, and the policy never recovered; the cycle task therefore limits pushes to 0.15 m/s and the action-change penalty to −0.5. The first cycle run fell outright under backward pushes of 0.18 m/s or more, so the released Hard variant adds a 0.25 m/s push stage after iteration 1400.
- When pushed too hard, a touchdown of the lifted foot is only a small cost and never ends the episode, while staying upright is worth far more, so the learned answer is to step down and lift again. The author describes this as intended, not as a loophole.

**Success checks and limits.** In CPU MuJoCo with the BAM XL330 servo model, the author ran 10 fixed scenarios and all passed (cycles on both sides, 0.1 m/s pushes toward either side and 0.15 m/s forward, a 0.3 m/s push toward the lifted side with a brief touchdown and re-lift, lowering from a static hold, a 10 s hold); in 24 random pushes of 0.05–0.25 m/s it held 20 times, stepped down and recovered twice and fell twice. Known limits: backward pushes of 0.18 m/s make it fall; pushes toward the stance side above about 0.15 m/s end in a step-down; it has run only in simulation and never on hardware.

{#basketball}
## Basketball balance: the reward sees the ball, the policy does not

:::{course-video} media/community-balance-hd
:alt: Community basketball-balance demo. Watch how the robot keeps adjusting its posture as the support surface rolls under it.

Community basketball-balance demo (recording of an earlier version). Watch how the robot keeps adjusting its posture as the support surface rolls under it. The analysis below covers the author's latest release, b11.
:::

**Goal.** In [basketball balance ↗](https://huggingface.co/HannesVonEssen/microduck-basketball) the robot balances on a free-rolling size-7 basketball (0.24 m diameter, 0.62 kg) and moves according to velocity commands; `[0, 0, 0]` means balance in place. Each episode starts with the robot placed on the apex of the ball; it does not climb onto the ball or get up from the floor.

**Observable quantities.** The reward reads the ball centre and ball speed, the body position, the sole reference points of both feet and the gravity direction in the body frame. The key point: **the deployed policy cannot see the ball**. In the author's latest release, b11, the policy uses only proprioception and velocity commands, passed through one LSTM layer of memory, and infers the ball under its feet from how its own state changes; the value network (critic) can read the ball state during training, but it is not part of the exported ONNX. The reward is computed in simulation, so it can use the ball position directly.

```{formula} Standing on the ball: over the centre, feet on the ball (weights 2 and 1)
\sym{r_center}=\sym{exp}\!\left(-\frac{\|\sym{p_ball}-\sym{p_root}\|_{xy}^2}{0.04^2}\right),\qquad \sym{r_feet}=\frac{1}{2}\sum_{\mathrm{L,R}}\sym{exp}\!\left[-\left(\frac{\|\sym{p_foot}-\sym{p_ball}\|-(\sym{R_ball}+0.012)}{0.02}\right)^2\right]
```

```{formula} Standing on the ball: height and upright (weights 1 and 1)
\sym{r_height}=\sym{exp}\!\left[-\left(\frac{\sym{z_root}-\sym{z_ball}-(\sym{R_ball}+0.125)}{0.03}\right)^2\right],\qquad \sym{r_up}=\sym{exp}\!\left(-\frac{\sym{g_x}^2+\sym{g_y}^2}{0.3^2}\right)
```

```{formula} Ball-speed penalty (weight −0.05)
\sym{r_ball}=\|\sym{v_ball}\|^2
```

The scales all carry length units: with the body 4 cm off the ball centre, a sole 2 cm off the ball surface or a 3 cm height error, the corresponding term drops to $e^{-1}\approx 0.37$. $\sym{r_ball}$ has a small weight: it keeps the ball quiet at zero command, but when a command asks for rolling, velocity tracking still pays more. The termination reads the ball too: the body too low relative to the ball centre (below $\sym{R_ball}$ + 0.05 m), more than 0.16 m off horizontally, or tilted beyond 55° counts as falling off the ball.

```{code-story}
label: Python · Community author source excerpt
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
  title: Body over the ball centre
  body: '`_bb_ball_pos` reads the ball centre straight from the simulation, privileged information that is not in the policy observations. The smaller the horizontal offset, the closer the reward is to 1; weight 2.'
- lines: [6, 16]
  title: Both feet resting on the ball
  body: Each foot's distance to the ball centre should equal the radius plus a 12 mm sole gap; each foot gets its own Gaussian and the two are averaged, so a dangling foot scores close to 0. Weight 1.
- lines: [16, 23]
  title: Standing height on the apex
  body: The body should sit “radius + 0.125 m” above the ball centre, i.e. the normal standing height placed on top of the ball. Weight 1.
- lines: [23, 30]
  title: Ball-speed penalty
  body: Returns a non-negative value and is registered with −0.05. The comment says the weight is small on purpose.
- lines: [30, 51]
  title: Termination
  body: 'Falling off is judged from the ball as well: too low, off the apex or tilted too far ends the episode. Termination removes all reward the policy could have earned after “falling off the ball”, so ending early also shapes the learned balance behavior.'
```

The walking recipe's velocity tracking, pose, upright, smoothness and joint-limit terms are also retained; see the author's [configuration ↗](https://github.com/Vottivott/microduck-playground/blob/aa5bd7909873a807fb9f4045f491519c443142e7/src/mjlab_microduck/tasks/microduck_basketball_env_cfg.py) for all weights. Early training also puts an adjustable “spring hold” on the ball, relaxed step by step from fully held to free rolling; environments that survive long enough are promoted and quick falls are demoted.

**Failures and fixes recorded by the author.** According to the author's [training record ↗](https://github.com/Vottivott/microduck-playground/blob/aa5bd7909873a807fb9f4045f491519c443142e7/experiments/basketball/TRAINING.md), removing direct observation of the ball made balancing harder; LSTM memory improved it, while a teacher–student distillation attempt failed to preserve balance. With **the reward unchanged**, adjusting only the learning rate and the frequency of push practice improved it further. The reward is only one lever: what the policy can see, remember and has practised also decides the result.

**Success checks and limits.** The author ran 3 seeds × 1024 environments, 3072 trials of 60 seconds, with the first fall counted as failure: b11 survived 20 s in 98.86% and 60 s in 97.01% of trials; the reference version that sees the ball reached 99.67% and 99.12%. Its turn-tracking error of 1.26 rad/s is clearly worse than the reference's 0.76 rad/s, and its actions are less smooth; these tracking metrics count only samples before a fall. The exported ONNX needs extra LSTM-state inputs and outputs, so a plain feed-forward loader cannot run it. Onboard real-time timing and real-robot behaviour are untested.

{#moonwalk}
## Moonwalk: in the prompt, not in the reward

**Goal.** [Moonwalk backward ↗](https://huggingface.co/fffiloni/microduck-moonwalk-backward-55e6af) was generated by an automated tool from a text prompt: “a moonwalk-style backward gliding gait, alternating feet, creating the visual illusion of sliding rather than ordinary backward walking, upper body stable and upright, avoiding jumps or large vertical motion, balanced throughout, finishing in a stable standing pose after a few seconds.”

**What was actually trained.** The tool chose the walking velocity-tracking recipe from its existing recipes and made only three changes: the forward velocity command became backward, −0.25 to −0.08 m/s (with only small sideways and turning ranges); the foot-slip penalty was weakened from −0.1 to −0.015; and left–right symmetry augmentation was enabled.

```{formula} Two terms inherited from the walking recipe (mjlab 1.3.0)
\sym{r_lin}=\sym{exp}\!\left(-\frac{\|\sym{v_cmd}-\sym{v_xy}\|^2+\sym{v_z}^2}{\sym{sigma_v}^2}\right),\qquad \sym{r_slip}=\sym{one}\!\left[\|\sym{v_cmd}\|+|\sym{omega_cmd}|>0.01\right]\sum_{\mathrm{L,R}}\sym{k_foot}\,\|\sym{v_foot}\|^2
```

```{code-story}
label: Python · Automatically generated task configuration (complete)
source: https://huggingface.co/fffiloni/microduck-moonwalk-backward-55e6af/blob/951073cb7791a898e3e5ecb242c5bcccf40aa72f/compiled_motion.json
language: python
code: |
  from copy import deepcopy
  from mjlab_microduck.tasks.microduck_velocity_env_cfg import make_microduck_velocity_env_cfg, MicroduckRlCfg
  from mjlab_microduck.tasks.symmetry import SYMMETRY_CFG

  def make_generated_env_cfg(play: bool = False, rough: bool = False):
      cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)
      command = cfg.commands["twist"]
      command.ranges.lin_vel_x = (-0.25, -0.08)
      command.ranges.lin_vel_y = (-0.03, 0.03)
      command.ranges.ang_vel_z = (-0.15, 0.15)
      command.rel_standing_envs = 0.08
      command.rel_turn_in_place_envs = 0.0
      cfg.rewards['foot_slip'].weight = -0.015
      return cfg

  GeneratedRlCfg = deepcopy(MicroduckRlCfg)
  GeneratedRlCfg.experiment_name = 'motionlab_moonwalk_backward'
  GeneratedRlCfg.run_name = 'moonwalk_backward'
  GeneratedRlCfg.algorithm.symmetry_cfg = SYMMETRY_CFG
steps:
- lines: [0, 6]
  title: Start from the walking recipe
  body: Observations, actuators, domain randomization, terminations and every reward term come from the official walking recipe (pinned revision `d424a0c`). This code is stored in `compiled_motion.json` on the model card.
- lines: [6, 12]
  title: 'The only “moonwalk” information: a backward velocity command'
  body: Forward velocity is sampled only between −0.25 and −0.08 m/s, and 8% of environments practise standing. The velocity-tracking reward therefore asks the robot to move backward.
- lines: [12, 14]
  title: Relax the foot-slip penalty
  body: The weight changes from −0.1 to −0.015, allowing more sliding while a foot touches the ground. It merely objects less to sliding; it does not reward “looking like a glide”.
- lines: [14, 19]
  title: Enable symmetry augmentation
  body: Training uses left–right mirrored data to keep the legs' motions symmetric.
```

Compare with the goal item by item: moving backward and an upright upper body have matching reward terms (velocity tracking and the inherited upright term); “alternating feet” can come only from the walking recipe's existing gait terms; “finish standing” relies on switching the command back to 0 at deployment and on the recipe's existing zero-command standing practice; and “the visual illusion of sliding” is described by no reward term or check at all. The tool's generated notes also say that vertical motion is suppressed by a body-pose reward, but in this base recipe the body-pose tracking term has weight 0.

**Success checks and limits.** The model card records that the export check and an ONNX rehearsal in CPU MuJoCo passed (61 inputs, 14 outputs); in a 5-second preview (`preview.mp4` on the model card) the maximum horizontal displacement was 0.33 m and the largest joint change 14.7°; when you watch it, check the goal item by item. But the automatic motion-quality gate says “needs review”, and the card explicitly marks the semantic match as unverified. The file loads, the interface is correct and the robot moves, but there is no evidence that it learned a moonwalk. The lesson: an effect that is wanted in the prompt, but written into neither a reward nor an acceptance test, cannot be expected to emerge from training.

:::{enrichment} Going further: if you wanted to really test a “moonwalk”
What follows is a direction proposed by the course, not the author's implementation, and it has not been trained or validated. Write the acceptance test first: for example, over a fixed duration the body really moves backward; within each gait cycle the foot in contact slides backward relative to the ground while the lifted foot moves forward; the vertical motion of the centre of mass stays below a threshold; the robot ends in its standing pose. Then ask: can these quantities be read from the simulation? Should each become a tracking term, a penalty or only a final check? Which are suitable as rewards, and which are better kept as independent evaluation metrics?
:::

{#compare}
## The three skills side by side

| | Flamingo cycle | Basketball balance | Moonwalk |
|---|---|---|---|
| Type of goal | Switching between two postures on command | Continuous balance while following velocity | Stylized continuous gait |
| Key technique | Gate every term with the blend | Use the true ball state in rewards and termination | Change only the command range and one weight |
| Privileged information | Whole-body centre of mass, contacts | Ball position and speed (invisible to the policy) | Same as the walking recipe |
| Termination | Inherited fall termination | Falling off the ball | Inherited from the walking recipe |
| Author's check | 10 scenarios + 24 random pushes | 3 seeds × 1024 environments × 60 s | Export and interface pass; semantics unverified |
| Not verified by any | Real robot | Real robot and onboard timing | Whether the motion matches the prompt |

Check four things: whether the goal has matching rewards or acceptance criteria; whether high scores are possible without success; whether privileged reward inputs are separated from deployed observations; and whether an independent test judges the result.

{#new-task}
## Design a reward for a new task

Pick a task outside these three, for example “turn once on the spot and stop facing the original direction” or “sit down and stand up again”, and work through the six steps of Section 3.3 on paper:

1. Write the success and failure criteria in measurable quantities and fix them before writing any reward.
2. List the quantities you need, with units, frames, and whether the policy will have them at deployment.
3. Choose a shape for each requirement: tracking, penalty or one-time event; if the order of movements matters, look at the flamingo gates.
4. Estimate the typical size of each term and check that none outweighs the rest.
5. Write down at least two loopholes, and how the termination conditions change what the reward means.
6. Plan the independent evaluation and the ablation: which seeds, how many trials, which metrics, and the same training budget.

::::{quiz}
The reward curve of a new task keeps rising, and the exported ONNX passes the 61-input, 14-output interface check. Does that show the task is done?

:::{answer}
No. A rising reward only shows that the policy got better at collecting this reward set; an interface check only shows that the controller can call the file. The moonwalk meets both, yet its semantic match is unverified; the roller training in Section 3.3 raised its reward sharply while the ramp verification gave 0/5. Judge with acceptance criteria fixed before training, independent seeds and actual observation.
:::
::::

{#compatible}
## Same interface, different conventions

ONNX is a model file format: it stores the network structure and weights but does not define how a robot skill is called. The flamingo cycle and the moonwalk both use the shared 61-input, 14-output interface, yet neither can replace the other: the flamingo cycle uses the first two velocity-command slots for “lift a foot” and “which side”, and only the moonwalk uses them as a real velocity command. The basketball policy additionally needs its LSTM memory state passed in and back out alongside the same 61 observations. The course walking model takes 61 inputs and the kicking residual takes 62; both output 14 numbers, and they cannot be swapped either.

A file that loads is only readable. For the robot to perform the motion, the meaning of every command slot, the units, joint order, action scaling and termination conditions must all match the controller. The [models page](models.md) lists the fixed course weights and their roles. Handovers between skills also need their own design: after the kick in the Section 2.1 keyboard game, a 0.12 s smooth blend returns to the walking policy's standing pose, and that blend is not part of the kick model.

{#kick-example}
## A skill’s start and end matter equally

Walking continuously accepts velocity commands, while a kick is a short skill with a clear start and end. To add a kick to an autonomous task, besides the motion itself you must define when it may start, which state it ends in, and how it stops on failure.

:::{course-video} media/kick-contact-hd
:alt: Close-up of the swing, ball contact and return to standing. The clip uses the combined kicking policy from the complete football task.

Close-up of the swing, ball contact and return to standing. The clip uses the combined kicking policy from the complete football task.
:::

| Phase | What to check |
|---|---|
| Prepare | Ball in the kickable zone, suitable body heading, observations still fresh |
| Contact | The ball moves only through physical foot contact; the simulation never overwrites its state |
| Retract | End the swing promptly and return to support |
| Recover | The robot stays stable, without falling or exceeding joint limits |

A shot requires the ball in a useful right-foot position, the body facing the goal, stable support and fresh target information. Execution combines the kicking teacher with a residual correction; after touching the ball, the robot must return to stable standing or walking before the skill ends. A residual is a constrained correction to an existing action: it narrows what must be learned from scratch and also inherits the teacher’s operating range, so a local kick started far from the ball will not search for it. When watching the clip, use the physical contact record to confirm that the foot, not another body part, touches the ball, then check that the centre of mass returns over the support area and body tilt settles.

::::{quiz}
The robot is beside the ball, but its remembered position has expired. Should the kick start immediately?

:::{answer}
No. Observe again and confirm the entry conditions. Proximity alone does not guarantee valid inputs.
:::
::::

{#eyes}
## Executing a skill is not completing a task

So far we have told the robot when to walk, lift a foot or glide backward. The policy uses joint and IMU feedback to balance, but where the ball is and which route to take must be decided by other modules. Part 4 trains GR00T to issue skill requests from camera images, instructions and body state, handing these decisions to a learned model.

{#references}
## References

{.references}
- [Official Microduck web simulator (Hugging Face Space) ↗](https://huggingface.co/spaces/pollen-robotics/microduck-simulator)
- [Hugging Face: search Microduck models ↗](https://huggingface.co/models?search=microduck)
- [Isaac Lab: Microduck task source ↗](https://github.com/isaac-sim/IsaacLab/tree/5088b326c7fc81caa7899395b0176ad08b89e9e1/source/isaaclab_tasks/isaaclab_tasks/contrib/microduck)
- [mjlab 1.3.0: reward manager source ↗](https://github.com/mujocolab/mjlab/blob/19d6c06c17572dafcb31ad5110a902006b58d944/src/mjlab/managers/reward_manager.py)
- [Community flamingo cycle ↗](https://huggingface.co/RemiFabre/microduck-flamingo-cycle)
- [Community flamingo cycle: training source (pinned revision) ↗](https://github.com/pollen-robotics/microduck_rl/tree/0bf989701f14d4008c866ffd6f09067870f99379/src/mjlab_microduck/tasks)
- [Community basketball balance: model card and evaluation ↗](https://huggingface.co/HannesVonEssen/microduck-basketball)
- [Community basketball balance: training source ↗](https://github.com/Vottivott/microduck-playground/tree/main/experiments/basketball)
- [Community moonwalk backward ↗](https://huggingface.co/fffiloni/microduck-moonwalk-backward-55e6af)
