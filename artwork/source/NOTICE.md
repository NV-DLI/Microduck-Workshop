# Original Microduck model attribution / 模型来源

Source / 来源: Pollen Robotics, https://github.com/pollen-robotics/microduck_rl

Revision / 版本: 2b581c641406a48346e696212930ea881c222c52

Path / 路径: src/mjlab_microduck/robot/microduck

The upstream README states: “3D model files are licensed under Creative Commons BY-SA-NC.” It does not specify a version in that statement. The separately included Apache-2.0 LICENSE covers the software and must not be taken as replacing this model-specific notice. The upstream README is included unchanged for reference. No broader permission is asserted here.

上游 README 单独将 3D 模型标注为 Creative Commons BY-SA-NC，该声明没有列出具体版本。其软件 Apache-2.0 许可证不能代替模型的许可说明。本文件不声明额外授权。公开或商业课程使用时应核对模型许可及所需授权。

This deliverable preserves the original STL geometry and assembles the visual parts using the source transforms and STAND joint values. New fields, goals, cords, putting platform and camera/lighting are course visualization additions. The authored animation does not demonstrate a learned policy or physical transfer.

交付保留官方 STL 几何，并根据原始装配变换与 STAND 关节值还原姿态。场地、球门、低位绳线、高尔夫球道、灯光与相机为新增课程可视化内容。动画不代表策略训练或物理迁移结果。

## Upstream clarification checked 2026-10-22 / 上游许可补充

The historical statement above is preserved. The current upstream LICENSE-MODELS.md explicitly licenses the Microduck STL assets under CC BY-NC-SA 4.0 and the MJCF XML, export configs and .part metadata under Apache-2.0. See LICENSE-MODELS.md in this directory and https://raw.githubusercontent.com/pollen-robotics/microduck_rl/develop/LICENSE-MODELS.md . Derivative Blender/GLB/images/animations in this workshop retain Pollen Robotics attribution and are provided under CC BY-NC-SA 4.0.
