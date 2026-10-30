"""Isaac Sim lighting and a fixed camera for the terrain lab."""

import math


def configure_scene(cfg):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from scene_style import configure_lighting

    configure_lighting(cfg)
    # Paint-like visual strips follow the physical ramp without adding contacts.
    angle = cfg.scene.terrain.terrain_generator.sub_terrains["ramp"].slope_deg
    tangent = math.tan(math.radians(angle))
    # The 6×4 m tile is centered; ramp_mesh places the spawn at local (2,2).
    size = cfg.scene.terrain.terrain_generator.size
    offset = (2.0 - size[0] / 2, 2.0 - size[1] / 2)
    for index in range(18):
        x = 0.05 + index * 0.1
        z = max(0.0, min(1.0, x - 0.35)) * tangent + 0.0008
        tilt = math.radians(angle) if 0.35 < x < 1.35 else 0.0
        for side, y in enumerate([-0.32, 0.32]):
            setattr(
                cfg.scene,
                f"lane_{index}_{side}",
                AssetBaseCfg(
                    prim_path="{ENV_REGEX_NS}/Lane" + str(index) + "_" + str(side),
                    spawn=sim_utils.CuboidCfg(
                        size=(0.065, 0.008, 0.001),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.88, 0.75, 0.37), roughness=0.7
                        ),
                    ),
                    init_state=AssetBaseCfg.InitialStateCfg(
                        pos=(x + offset[0], y + offset[1], z),
                        rot=(0.0, -math.sin(tilt / 2), 0.0, math.cos(tilt / 2)),
                    ),
                ),
            )
    angle_rad = math.radians(angle)
    cfg.scene.ramp_tint = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RampTint",
        spawn=sim_utils.CuboidCfg(
            size=(1.0 / math.cos(angle_rad), 0.62, 0.0004),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.29, 0.35), roughness=0.8
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.85 + offset[0], offset[1], 0.5 * tangent + 0.0006),
            rot=(0.0, -math.sin(angle_rad / 2), 0.0, math.cos(angle_rad / 2)),
        ),
    )
    cfg.scene.finish = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Finish",
        spawn=sim_utils.CuboidCfg(
            size=(0.015, 0.64, 0.001),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.38, 0.65, 0.2), roughness=0.7
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.45 + offset[0], offset[1], tangent + 0.0008)
        ),
    )


def camera_prestartup(env, env_ids, width=1920, height=1080):
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    import isaaclab.sim as sim_utils

    focal = width / (2 * math.tan(math.radians(52) / 2))
    env.terrain_camera = Camera(
        CameraCfg(
            prim_path="/World/envs/env_0/TerrainCamera",
            width=width,
            height=height,
            update_period=0.0,
            data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                [focal, 0, width / 2, 0, focal, height / 2, 0, 0, 1],
                width,
                height,
                clipping_range=(0.02, 100.0),
                focal_length=20.0,
            ),
        )
    )


def start_recording(
    env,
    output,
    *,
    eye_offset=(2.0, -3.7, 1.5),
    target_offset=(1.45, 0.35, 0.10),
    filename="terrain.mp4",
    fps=25,
):
    import torch
    import subprocess

    camera = env.terrain_camera
    camera.reset()
    origin = env.scene.env_origins[0]
    origin = getattr(origin, "torch", origin)
    from pxr import Gf, UsdGeom
    import numpy as np
    from scipy.spatial.transform import Rotation
    import usdrt
    import isaaclab.sim as sim_utils

    eye = (origin + torch.tensor(eye_offset, device=env.device)).cpu().numpy()
    target = (origin + torch.tensor(target_offset, device=env.device)).cpu().numpy()
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    ros = Rotation.from_matrix(np.column_stack([right, down, forward]))
    camera.set_world_poses(
        positions=eye[None], orientations=ros.as_quat()[None], convention="ros"
    )
    gl = (ros * Rotation.from_rotvec([np.pi, 0.0, 0.0])).as_quat()
    matrix = Gf.Matrix4d().SetRotate(
        Gf.Rotation(Gf.Quatd(float(gl[3]), Gf.Vec3d(*gl[:3].tolist())))
    )
    matrix.SetTranslateOnly(Gf.Vec3d(*eye.tolist()))
    UsdGeom.Xformable(
        env.sim.stage.GetPrimAtPath("/World/envs/env_0/TerrainCamera")
    ).MakeMatrixXform().Set(matrix)
    stage = usdrt.Usd.Stage.Attach(sim_utils.get_current_stage_id())
    transform = usdrt.Rt.Xformable(
        stage.GetPrimAtPath("/World/envs/env_0/TerrainCamera")
    )
    transform.SetLocalXformFromUsd()
    transform.SetWorldXformFromUsd()
    rt_matrix = usdrt.Gf.Matrix4d(
        *[float(matrix[i][j]) for i in range(4) for j in range(4)]
    )
    transform.CreateFabricHierarchyLocalMatrixAttr().Set(rt_matrix)
    transform.CreateFabricHierarchyWorldMatrixAttr().Set(rt_matrix)
    env.sim.set_setting("/rtx/sdg/force/disableColorRender", False)
    for _ in range(8):
        env.sim.render()
    import json

    width, height = int(camera.cfg.width), int(camera.cfg.height)

    root = env.scene["robot"].data.root_pos_w
    root = getattr(root, "torch", root)
    bounds = (
        UsdGeom.BBoxCache(0.0, ["default", "render"])
        .ComputeWorldBound(env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot"))
        .ComputeAlignedBox()
    )
    (output / "camera.json").write_text(
        json.dumps(
            dict(
                origin=origin.cpu().tolist(),
                eye=eye.tolist(),
                target=target.tolist(),
                root=root.cpu().tolist(),
                usd_robot_bounds=str(bounds),
                camera_matrix=str(matrix),
                width=width,
                height=height,
                fps=fps,
            ),
            indent=2,
        )
    )
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-crf",
            "18",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / filename),
        ],
        stdin=subprocess.PIPE,
    )
    return camera, process


def capture(env, camera, process, renders=4):
    """Write one frame. renders=4 (default) renders the scene four more times first;
    renders=0 reads the frame of the render that already happened this step (the caller
    renders once per frame: sim.step(render=True) with a window, or the camera update's own
    single Kit update when headless). Returns the frame bytes."""
    for _ in range(renders):
        env.sim.render()
    camera.update(0.0, force_recompute=True)
    rgb = camera.data.output["rgb"]
    rgb = getattr(rgb, "torch", rgb)
    frame = rgb[0, :, :, :3].cpu().numpy().tobytes()
    if len(frame) != int(camera.cfg.width) * int(camera.cfg.height) * 3:
        raise RuntimeError(f"camera frame has {len(frame)} bytes, expected {camera.cfg.width}x{camera.cfg.height} RGB")
    process.stdin.write(frame)
    return frame
