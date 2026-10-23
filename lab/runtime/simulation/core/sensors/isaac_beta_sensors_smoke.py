#!/usr/bin/env python3
"""Capture actual head-mounted RTX Beta sensor snapshots in Newton.

Static, explicitly posed head configurations demonstrate extrinsics and optics.
This does not execute or validate a walking policy, hardware timing, or football.
"""
from __future__ import annotations
import argparse, hashlib, html, json, math, time, traceback, sys
from pathlib import Path
from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--frames", type=int, default=6)
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--noise-std-m", type=float, default=0.0)
parser.add_argument("--dropout-probability", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=38111)
parser.add_argument(
    "--mount-profile",
    required=True,
    choices=["rl-jaw-soft-20261012", "runtime-bottom-head-shell-20261017"],
)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=540)
parser.add_argument("--assumed-hfov", type=float, default=70.0)
parser.add_argument(
    "--assumed-distortion", type=float, nargs=5, default=[0.0, 0.0, 0.0, 0.0, 0.0]
)
parser.add_argument(
    "--robot-usd",
    default=None,
    help="Optional local copy of the same official asset; record provenance separately",
)
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if args.smoke:
    args.frames = 3
if args.frames < 1 or args.frames > 60:
    parser.error("Choose 1–60 selected pose snapshots")
if not math.isfinite(args.noise_std_m) or args.noise_std_m < 0:
    parser.error("Noise must be finite and nonnegative")
if not 0 <= args.dropout_probability <= 1:
    parser.error("Dropout probability must be between zero and one")
if args.output.exists():
    parser.error("Choose a new output directory to preserve measurements")
runtime_context = launch_simulation(None, {**vars(args), "require_kit": True})
runtime_context.__enter__()
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.sim import SimulationContext
from copy import deepcopy
from isaaclab_assets.robots.microduck import MICRODUCK_BACKLASH_CFG
from isaaclab_tasks.contrib.microduck.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from isaac_beta_sensor_rig import IsaacBetaSensors, array
from beta_sensor_profile import BetaCameraProfile, PROFILE_ID


def backend_evidence(sim):
    manager = sim.backend_manager
    return {
        "backend_manager": type(manager).__module__ + "." + type(manager).__name__,
        "backend_name": getattr(manager, "backend_name", None),
        "solver_cfg_type": type(sim.cfg.physics.solver_cfg).__name__,
        "native_newton_actuators_enabled": sim.cfg.use_newton_actuators,
    }


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def image_evidence(rgb):
    """Photometric diagnostic of rendered fixtures, not semantic teacher data."""
    pixels = rgb.astype(np.float64)
    r, g, b = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
    masks = {
        "red": (r > 100) & (r > 1.8 * g) & (r > 1.8 * b),
        "green": (g > 120) & (g > 1.6 * r) & (g > 1.4 * b),
        "blue": (b > 100) & (b > 1.7 * r) & (b > 1.25 * g),
    }
    markers = {}
    for color, mask in masks.items():
        y, x = np.nonzero(mask)
        markers[color] = {
            "pixels": int(len(x)),
            "centroid_xy_px": [float(x.mean()), float(y.mean())] if len(x) else None,
        }
    return {
        "mean": float(pixels.mean()),
        "std": float(pixels.std()),
        "nonblack_fraction": float((pixels.max(axis=2) > 8).mean()),
        "markers": markers,
    }


def validate_samples(samples):
    failures = []
    for index, sample in enumerate(samples):
        quality = sample["image_evidence"]
        if (
            quality["mean"] < 2
            or quality["std"] < 3
            or quality["nonblack_fraction"] < 0.05
        ):
            failures.append(f"frame {index}: RGB is blank or nearly uniform")
        if sample["tof_valid_zones"] < 1:
            failures.append(
                f"frame {index}: independent ToF has no valid scene measurements"
            )
    if len(samples) > 1 and len({s["raw_pixel_sha256"] for s in samples}) == 1:
        failures.append("Different actual head poses produced identical RGB pixels")
    markers = samples[0]["image_evidence"]["markers"]
    if any(markers[c]["pixels"] < 20 for c in ("red", "green", "blue")):
        failures.append("Home image does not resolve all three visible axis fixtures")
    else:
        red, green, blue = (
            markers[c]["centroid_xy_px"] for c in ("red", "green", "blue")
        )
        if not green[0] < blue[0]:
            failures.append(
                "Upright image does not place left green fixture left of right blue fixture"
            )
        if not red[1] < min(green[1], blue[1]):
            failures.append(
                "Upright image does not place upper red fixture above lower fixtures"
            )
    return {
        "passed": not failures,
        "failures": failures,
        "method": "Actual rendered RGB photometry and visible color fixture centroids, nonempty independent ToF, changed images for changed actual head poses",
        "not_a_hardware_accuracy_or_timing_test": True,
    }


def draw_tof(packet, path):
    cells = packet["cells"]
    if len(cells) == 64:
        cells = [cells[i * 8 : (i + 1) * 8] for i in range(8)]
    if len(cells) != 8 or any(len(row) != 8 for row in cells):
        raise RuntimeError("Expected actual 8x8 ToF cells")
    image = Image.new("RGB", (640, 690), "#101827")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text(
        (16, 10), "8 x 8 ToF reference | metres | top row = up", fill="white", font=font
    )
    for row in range(8):
        for col in range(8):
            cell = cells[row][col]
            distance = cell.get("range_m")
            valid = distance is not None and math.isfinite(distance)
            color = (
                (
                    int(40 + min(1, distance / 2) * 80),
                    int(155 - min(1, distance / 2) * 80),
                    160,
                )
                if valid
                else (39, 45, 58)
            )
            x, y = col * 80, row * 80 + 48
            draw.rectangle((x + 2, y + 2, x + 78, y + 78), fill=color)
            label = f"{distance:.2f}" if valid else "—"
            draw.text((x + 10, y + 15), label, fill="white", font=font)
            status = "edge" if cell.get("mixed_edge") else cell.get("status", "")
            draw.text((x + 8, y + 43), status[:9], fill="white", font=font)
    image.save(path)


def preview(output, samples):
    cards = []
    for sample in samples:
        cards.append(
            "<section><h2>"
            + html.escape(sample["pose_name"])
            + '</h2><div class="views">'
            + "".join(
                '<figure><img src="'
                + sample[key]
                + '"><figcaption>'
                + label
                + "</figcaption></figure>"
                for key, label in [
                    ("rgb_raw", "GC2093 · assumed optics"),
                    ("rgb_upright", "Display · native 0°"),
                    ("tof_preview", "Independent head ToF · 8 × 8"),
                ]
            )
            + '</div><a href="'
            + sample["packet"]
            + '">Sensor packet / 参数与数据</a></section>'
        )
    text = (
        """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Microduck hardware sensor reference</title><style>body{margin:0;padding:32px;background:#0b1020;color:#e7ecfa;font:16px system-ui;max-width:1600px}h1{font-size:28px}p{max-width:980px;line-height:1.7}a{color:#77d7ff}.views{display:grid;grid-template-columns:2fr 1fr 1.4fr;gap:20px;align-items:start}figure{margin:0}img{width:100%;height:auto;max-height:540px;object-fit:contain;background:#080c15}figcaption{padding:10px 0;color:#a6b9d0}section{padding:24px 0;border-top:1px solid #26334b}@media(max-width:800px){.views{grid-template-columns:1fr}body{padding:18px}}</style><h1>Microduck · 真实头部绑定的传感器参考</h1><p>以下均为 Newton 中实际头部姿态驱动的 RTX 输出：相机使用显式标注的 Beta 仿真假设内参/畸变（不复用 alpha 标定），ToF 使用独立光心和 45° × 45°、8 × 8 分区。相机随实际头部旋转，没有世界稳定云台。GC2093 镜头尚未标定；本页是静态姿态采样演示，不是 30/15 Hz 性能测试，也不是自主射门验收。</p><p><a href="manifest.json">完整参数、版本与限制 / Manifest</a></p>"""
        + "".join(cards)
        + "</html>"
    )
    (output / "index.html").write_text(text)


def main():
    start = time.monotonic()
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True)
    sim_cfg = deepcopy(MicroDuckVelocityFlatEnvCfg().sim)
    sim_cfg.device = args.device
    sim_cfg.render_interval = 4
    sim = SimulationContext(sim_cfg)
    robot_cfg = deepcopy(MICRODUCK_BACKLASH_CFG)
    robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    if args.robot_usd:
        robot_cfg.spawn.usd_path = args.robot_usd

    @configclass
    class SensorSceneCfg(InteractiveSceneCfg):
        robot = robot_cfg
        ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(
                color=(0.12, 0.23, 0.16),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0
                ),
            ),
        )
        light = AssetBaseCfg(
            prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1800.0)
        )

    scene = InteractiveScene(SensorSceneCfg(num_envs=1, env_spacing=2.0))
    robot = scene["robot"]
    # Static visible objects are scene fixtures, never synthetic sensor outputs.
    sphere = sim_utils.SphereCfg(
        radius=0.035,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.92, 0.92, 0.88), roughness=0.8
        ),
    )
    sphere.func("/World/Football", sphere, translation=(0.25, 0, 0.035))
    for name, size, pos, color in [
        ("GoalLeft", (0.025, 0.025, 0.19), (0.85, 0.1325, 0.095), (0.9, 0.9, 0.85)),
        ("GoalRight", (0.025, 0.025, 0.19), (0.85, -0.1325, 0.095), (0.9, 0.9, 0.85)),
        ("GoalBar", (0.025, 0.29, 0.025), (0.85, 0, 0.1925), (0.9, 0.9, 0.85)),
        ("LeftGreen", (0.06, 0.06, 0.12), (0.35, 0.13, 0.06), (0.05, 0.9, 0.2)),
        ("RightBlue", (0.06, 0.06, 0.12), (0.35, -0.13, 0.06), (0.1, 0.3, 0.9)),
        ("UpperRed", (0.08, 0.08, 0.08), (0.6, 0, 0.42), (0.95, 0.08, 0.07)),
    ]:
        shape = sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color, roughness=0.8
            ),
        )
        shape.func("/World/" + name, shape, translation=pos)
    profile = BetaCameraProfile(
        width_px=args.width,
        height_px=args.height,
        assumed_horizontal_fov_deg=args.assumed_hfov,
        distortion_k1_k2_p1_p2_k3=tuple(args.assumed_distortion),
    )
    rig = IsaacBetaSensors(
        sim,
        robot,
        mount_profile=args.mount_profile,
        camera_profile=profile,
        seed=args.seed,
        noise_std_m=args.noise_std_m,
        dropout_probability=args.dropout_probability,
    )
    sim.reset()
    robot.reset()
    rig.reset()
    names = list(robot.joint_names)
    home = getattr(
        robot.data.default_joint_pos, "torch", robot.data.default_joint_pos
    ).clone()
    poses = [
        ("home", {}),
        ("head left", {"head_yaw": 0.35}),
        ("head right", {"head_yaw": -0.35}),
        ("head down", {"head_pitch": 0.65}),
        ("head roll", {"head_roll": 0.20}),
        ("head up", {"head_pitch": 0.10}),
    ]
    samples = []
    capture_origin_step = sim.get_physics_step_count()
    for index in range(args.frames):
        pose_name, overrides = poses[index % len(poses)]
        target = home.clone()
        for joint, value in overrides.items():
            target[0, names.index(joint)] = value
        root = getattr(
            robot.data.default_root_pose, "torch", robot.data.default_root_pose
        ).clone()
        root[:, :3] = torch.tensor([[-0.3, 0.0, 0.125]], device=sim.device)
        root[:, 3:] = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=sim.device)
        robot.write_root_pose_to_sim_index(root_pose=root)
        robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros_like(
                getattr(
                    robot.data.default_root_vel, "torch", robot.data.default_root_vel
                )
            )
        )
        robot.write_joint_position_to_sim_index(position=target)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(target))
        robot.reset()
        robot.set_joint_position_target_index(target=target)
        target_step = capture_origin_step + math.ceil(
            (index + 1) / 15.0 / 0.005 - 1e-10
        )
        while sim.get_physics_step_count() < target_step:
            robot.write_data_to_sim()
            sim.step(render=False)
            robot.update(0.005)
            rig.update_imu(timestamp_s=sim.get_physics_step_count() * 0.005)
        capture = rig.capture(
            timestamp_s=sim.get_physics_step_count() * 0.005, render_subframes=4
        )
        stem = f"{index:03d}"
        raw = f"{stem}-raw.png"
        upright = f"{stem}-upright.png"
        tof_png = f"{stem}-tof.png"
        packet = f"{stem}-packet.json"
        depth = f"{stem}-tof-rtx-range.npy"
        Image.fromarray(capture["rgb_raw"]).save(output / raw)
        Image.fromarray(capture["rgb_upright"]).save(output / upright)
        np.save(output / depth, capture["tof_source_ray_range"].astype(np.float32))
        draw_tof(capture["tof"], output / tof_png)
        data = {
            k: v
            for k, v in capture.items()
            if k not in ("rgb_raw", "rgb_upright", "tof_source_ray_range")
        }
        data.update(
            pose_name=pose_name,
            commanded_head_configuration=overrides,
            actual_joint_positions=dict(
                zip(names, array(robot.data.joint_pos)[0].tolist())
            ),
            purpose="static posed-sensor optics/extrinsics diagnostic, not locomotion or autonomous task execution",
            rgb_frame_scope="selected actual RTX snapshot, no measured 30 fps claim",
        )
        save_json(output / packet, data)
        samples.append(
            {
                "pose_name": pose_name,
                "rgb_raw": raw,
                "rgb_upright": upright,
                "tof_preview": tof_png,
                "packet": packet,
                "tof_rtx_depth": depth,
                "raw_pixel_sha256": hashlib.sha256(
                    capture["rgb_raw"].tobytes()
                ).hexdigest(),
                "tof_pixel_sha256": hashlib.sha256(
                    capture["tof_source_ray_range"].tobytes()
                ).hexdigest(),
                "image_evidence": image_evidence(capture["rgb_upright"]),
                "tof_valid_zones": sum(
                    cell["status"] == "valid"
                    for row in capture["tof"]["cells"]
                    for cell in row
                ),
            }
        )
    visual_validation = validate_samples(samples)
    manifest = {
        "status": (
            "captured" if visual_validation["passed"] else "failed_visual_validation"
        ),
        "visual_validation": visual_validation,
        "kind": "beta_newton_rtx_static_pose_snapshots",
        "hardwareProfileId": PROFILE_ID,
        "asset_usd": robot_cfg.spawn.usd_path,
        "physics": backend_evidence(sim),
        "sensor_model": rig.metadata(),
        "frames": len(samples),
        "samples": samples,
        "wall_seconds": time.monotonic() - start,
        "hardware_rates_are_reference_not_measured": True,
        "head_motion_mode": "explicit independent static joint configurations followed by actual Newton physics; no authored image rotation substitutes for head FK",
        "capture_schedule": "15 Hz simulation-time ToF snapshots quantized to 200 Hz physics; one selected RGB frame per snapshot, not a 30 fps wall-clock benchmark",
        "renderer_truth_labels_used": False,
        "autonomous_football_validated": False,
        "preview": "index.html",
        "source_sha256": {
            name: hashlib.sha256(
                (Path(__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in (
                "isaac_beta_sensors_smoke.py",
                "isaac_beta_sensor_rig.py",
                "beta_sensor_profile.py",
                "beta_tof_geometry.py",
                "beta_head_imu.py",
                "beta_mounts.py",
            )
        },
    }
    save_json(output / "manifest.json", manifest)
    preview(output, samples)
    if not visual_validation["passed"]:
        raise RuntimeError(
            "Real RTX sensor visual validation failed: "
            + "; ".join(visual_validation["failures"])
        )
    print("COURSE_BETA_SENSORS_COMPLETE=" + str(output / "manifest.json"), flush=True)
    sim.stop()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        runtime_context.__exit__(*sys.exc_info())
