# Sources and licenses

This repository combines original course material with third-party robot geometry, policies, and runtime dependencies. Their terms remain separate. New software is AGPL-3.0; see `LICENSE` and `licenses/AGPL-3.0.txt`.

| Material | Source and terms |
|---|---|
| Microduck STL geometry and derivatives, including GLB, Blender scene, assembly renders, and robot appearances in recordings | Copyright Pollen Robotics. [microduck_rl](https://github.com/pollen-robotics/microduck_rl), source revision and asset list in `artwork/`. Upstream [LICENSE-MODELS.md](https://github.com/pollen-robotics/microduck_rl/blob/develop/LICENSE-MODELS.md) specifies CC BY-NC-SA 4.0. Mesh cleanup, assembly conversion, materials and exploded presentation are modifications. See `artwork/NOTICE.md`. |
| Robot XML and community source adaptations | Pollen Robotics, Apache-2.0. Preserve upstream notices. `licenses/Apache-2.0.txt` contains the license. |
| `models/walk.onnx`, `models/kick-teacher.onnx` | Pollen Robotics [microduck-policies v8](https://huggingface.co/pollen-robotics/microduck-policies/tree/v8), revision `665ae4eb20bfe13bca1ef4ea6d4bb6b4a096e448`, Apache-2.0 model card saved in `licenses/community-model-card.md`. Original filenames: `velstand.onnx` and `ball_kick_right.onnx`. SHA-256 identities are in `models/manifest.json`. |
| Kick residual and checkpoint | Retained course-trained parameters applied with the community teachers. Runtime skill contracts are under `lab/runtime/simulation/core/skills/`; deployment configuration and model provenance are retained with the course models. Provided under AGPL-3.0. |
| Detector weights and detector integration | Course-trained YOLO26n from random initialization using Ultralytics 8.4.182. Ultralytics AGPL-3.0 terms apply; [upstream source](https://github.com/ultralytics/ultralytics/tree/v8.4.182), [license](https://www.ultralytics.com/legal/agpl-3-0-software-license). Training scripts and configurations are included in the perception source bundle. The 40-image sample is a teaching subset, not the full original training dataset. |
| Sample simulation images and robot recordings | Course-generated simulation captures, with Pollen Robotics CAD-derived robot appearances. Preserve Pollen Robotics attribution and CC BY-NC-SA 4.0 terms for those appearances. No real-world detector or hardware-performance claim is attached. |
| Open Sans and Noto Sans SC | Self-hosted variable fonts, SIL Open Font License 1.1. Licenses in `web/assets/fonts/`. Sources: [Open Sans](https://github.com/google/fonts/tree/main/ofl/opensans), [Noto Sans SC](https://github.com/google/fonts/tree/main/ofl/notosanssc). |
| Highlight.js 11.10.0 | BSD-3-Clause. Locally bundled code highlighter and GitHub Dark theme; license at `web/assets/vendor/highlight/LICENSE`. |
| KaTeX 0.16.22 | MIT, local equation rendering with bundled fonts. License: `web/assets/vendor/katex/LICENSE`. [Source](https://github.com/KaTeX/KaTeX/tree/v0.16.22). |
| Teaching code excerpts | RSL-RL 5.5.1 PPO and Isaac Lab velocity reward snippets retain their BSD-3-Clause terms; see `licenses/RSL-RL-BSD-3-Clause.txt` and `licenses/Isaac-Lab-BSD-3-Clause.txt`. New explanatory prose and diagrams are course-authored. |
| Three.js | MIT, vendored for an offline-capable viewer; license at `web/assets/vendor/THREE-LICENSE.txt`. |
| Isaac Lab / Newton / Warp | Downloaded during setup, not redistributed as installed binaries. See their upstream licenses and pinned source / dependency locks in `lab/environment/`. Isaac Lab: [source](https://github.com/isaac-sim/IsaacLab); Newton: [source](https://github.com/newton-physics/newton); Warp: [source](https://github.com/NVIDIA/warp). |
| Isaac Sim | Installed separately under NVIDIA's terms. See [Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/latest/common/licenses.html). |
| ROS 2 / Nav2 packages | Original official package URLs, versions and hashes are listed in `lab/nav2/`; package binaries are downloaded at setup and retain their respective upstream licenses. |

The course structure was informed by [NemoClawDLI](https://github.com/NVDLI/NemoClawDLI), NVIDIA DLI c-fx-30-v1, and the supplied sim2real tutorial. Their private archives, credentials, and course source were not copied into this repository. Lessons and frontend implementation here are newly authored.

The 3D model reproduces the provided CAD revision, including its electronics envelopes. It does not assert that older Raspberry Pi / Robot HAT geometry depicts the newer electronics used by the simulation sensor configuration.
