# Microduck 机械结构 / Mechanical anatomy

`microduck.blend` 是可编辑的逐组件模型。用 Blender 打开后，时间轴第 1 帧为装配状态，第 61 帧为爆炸状态。每个零件可独立选中，按躯干、头部、左腿和右腿分组。

Open `microduck.blend` in Blender. Frame 1 is assembled; frame 61 is exploded. Each component is independently selectable and grouped into torso, head, left leg and right leg.

- 70 个零件实例，38 种原始 CAD 网格 / 70 component instances, 38 source CAD meshes.
- 15 个实体舵机；源模型具有 14 个仿真关节，喙部在此导出中固定 / 15 physical servo units; the source has 14 simulated joints and a fixed beak.
- 尺寸约 123 × 142 × 272 mm；单位为米 / Approximately 123 × 142 × 272 mm; units are meters.
- 爆炸图和 GIF 是机械结构示意，不是物理仿真或策略运行 / The exploded view and GIF illustrate anatomy, not physics or a learned policy.

## 重新生成 / Rebuild

在仓库根目录执行，使用 Blender 4.0.2 或兼容版本。渲染使用 CPU，不依赖 Isaac Lab 或训练模型。

Run from the repository root with Blender 4.0.2 or a compatible release. Rendering uses the CPU and requires neither Isaac Lab nor trained models.

```bash
blender --background --python artwork/build_microduck.py -- \
  --source artwork/source \
  --output artwork/rebuilt \
  --samples 192
```

输出包括 `microduck.glb`、`microduck.blend`、双语 `components.json`，以及装配和爆炸状态的 PNG。`--no-render` 可只生成模型和元数据。

Outputs include GLB, editable Blender scene, bilingual component metadata, and assembled/exploded PNGs. Add `--no-render` to build geometry and metadata only.

网页中的 GLB 使用 Y 轴向上。每个根 mesh 的 `extras` 包含双语名称、用途、零件组和 `explode` 三维平移量。展开比例 `t` 从 0 到 1 时，位置为「初始位置 + t × explode」。

The GLB is Y-up. Each root mesh carries bilingual names, descriptions, a group, and an `explode` translation. For expansion fraction `t` from 0 to 1, set position to initial position plus `t × explode`.

## 来源 / Source

几何来自 [Pollen Robotics microduck_rl](https://github.com/pollen-robotics/microduck_rl/tree/2b581c641406a48346e696212930ea881c222c52/src/mjlab_microduck/robot/microduck)，保留原始装配变换和 STAND 姿态。只有 42 个零面积三角面被移除；没有简化或编造零件。电子器件外形对应这份 CAD 的 Raspberry Pi Zero 2 W、Robot HAT 和 NP-F970，不是其他硬件版本的工程图。

Geometry comes from the pinned Pollen Robotics source, preserving its assembly transforms and STAND pose. Only 42 zero-area triangles are removed; no parts are simplified or invented. Electronics envelopes represent the Raspberry Pi Zero 2 W, Robot HAT and NP-F970 in that CAD, not another hardware revision.

材质与灯光用于教学展示，不是制造材料规范。原提交的 README 没有列出许可版本；上游当前 [LICENSE-MODELS.md](source/LICENSE-MODELS.md) 已明确 STL 及其派生模型使用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)，XML 和软件使用 Apache-2.0。本目录的 Blender、GLB、图片和示意动画遵循 CC BY-NC-SA 4.0，保留 Pollen Robotics 署名。

Materials and lighting are illustrative, not manufacturing specifications. The historical README omitted a version; the current upstream LICENSE-MODELS.md explicitly licenses STL assets and adaptations under CC BY-NC-SA 4.0, with XML and software under Apache-2.0. The Blender, GLB, images and illustrative animation here retain Pollen Robotics attribution and are provided under CC BY-NC-SA 4.0.
