#!/usr/bin/env python3
"""Collect one FoV batch of real RTX images; labels live in teacher-only files.

Only initialization writes truth. The camera uses the actual simulated head
pose. This is a detector dataset, not an autonomous-control rollout.
"""
from __future__ import annotations
import argparse, hashlib, json, math, sys, time, traceback
from pathlib import Path

P = Path(__file__).resolve().parent
M = P.parent
for folder in (P, M / "football", M / "sensors", M / "gait"):
    sys.path.insert(0, str(folder))


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prestartup(env, env_ids, *, width, height, hfov, seed):
    from beta_sensor_profile import BetaCameraProfile
    from isaac_beta_sensor_rig import IsaacBetaSensors
    from isaac_execute import configure_visual_assets

    configure_visual_assets(env)
    # Reserve enough actual RTX overscan for every declared Brown-Conrady
    # experiment in this FoV batch. Later remaps do not invent image content.
    profile = BetaCameraProfile(
        width_px=width,
        height_px=height,
        assumed_horizontal_fov_deg=hfov,
        distortion_k1_k2_p1_p2_k3=(-0.10, 0.01, 0.0, 0.0, 0.0),
    )
    env.beta_rig = IsaacBetaSensors(
        env.sim,
        env.scene["robot"],
        mount_profile="rl-jaw-soft-20261012",
        camera_profile=profile,
        seed=seed,
    )
    # Teacher-only depth is separate from the public RGB/ToF control packet.
    # Sensor initialization happens after this prestartup callback.
    env.beta_rig.rgb_camera.cfg.data_types = ["rgb", "distance_to_camera"]


def configure_optics(rig, coefficients):
    import cv2, numpy as np
    from dataclasses import replace

    profile = replace(rig.profile, distortion_k1_k2_p1_p2_k3=tuple(coefficients))
    xx, yy = np.meshgrid(np.arange(rig.width), np.arange(rig.height))
    points = np.stack([xx, yy], axis=-1).astype(float)
    K = np.asarray(profile.K)
    D = np.asarray(coefficients)
    xy = cv2.undistortPoints(points.reshape(-1, 1, 2), K, D).reshape(
        rig.height, rig.width, 2
    )
    mx = (xy[:, :, 0] * rig.render_focal_px + rig.render_width / 2).astype(np.float32)
    my = (xy[:, :, 1] * rig.render_focal_px + rig.render_height / 2).astype(np.float32)
    if (
        mx.min() < 1
        or my.min() < 1
        or mx.max() > rig.render_width - 2
        or my.max() > rig.render_height - 2
    ):
        raise ValueError("Declared lens exceeds actual rendered overscan")
    rig.profile = profile
    rig.k = K
    rig.d = D
    rig.map_x = mx
    rig.map_y = my


def arguments():
    from isaaclab.app import add_launcher_args

    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--plan", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--hfov", type=int, choices=[60, 70, 80, 90], required=True)
    a.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Explicit incomplete development subset; cannot assemble the full dataset",
    )
    a.add_argument("--render-subframes", type=int, default=3)
    add_launcher_args(a)
    args = a.parse_args()
    if args.output.exists():
        a.error("Keep every collection result; choose a new directory")
    args.device = args.device or "cuda:0"
    args.enable_cameras = True
    return args


def run(args, cfg, rows, plan):
    import cv2, numpy as np, torch, gymnasium as gym
    from PIL import Image
    from pxr import UsdLux
    from isaac_beta_sensor_rig import array
    from beta_head_imu import qmul
    from head_adapter import HeadAdapter
    from teacher_labels import labels

    output = args.output.resolve()
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "labels").mkdir()
    (output / "teacher").mkdir()
    started = time.monotonic()
    env = None
    results = []
    manifest = {
        "schema": "microduck.beta-detector-rtx-chunk.v1",
        "runtime_id": plan["runtime_id"],
        "hardware_profile_id": plan["hardware_profile_id"],
        "plan_sha256": digest(args.plan),
        "hfov": args.hfov,
        "renderer": "actual Isaac RTX",
        "training_truth_used_for_labels_only": True,
        "autonomous_control": False,
        "status": "running",
        "images": results,
    }
    save(output / "manifest.json", manifest)
    try:
        env = gym.make("IsaacContrib-Velocity-Flat-MicroDuck", cfg=cfg).unwrapped
        robot = env.scene["robot"]
        ball = env.scene["ball"]
        rig = env.beta_rig
        sim = env.sim
        sim.backend_manager.set_decimation(1)
        env._physics_handles_decimation = False
        if sim.backend_manager.backend_name != "newton":
            raise ValueError("Newton backend required")
        env.reset()
        rig.reset()
        home = getattr(
            robot.data.default_joint_pos, "torch", robot.data.default_joint_pos
        ).clone()
        names = list(robot.joint_names)
        adapter = HeadAdapter()
        lamp = UsdLux.DomeLight(sim.stage.GetPrimAtPath("/World/skyLight"))
        clock = 0
        for request in rows:
            rng = np.random.default_rng(request["scene_seed"])
            viewrng = np.random.default_rng(request["view_seed"])
            bx = float(rng.uniform(-0.08, 0.12))
            by = float(rng.uniform(-0.08, 0.08))
            root_y = float(rng.uniform(-0.10, 0.10))
            mode = request["view_mode"]
            root_x = {"far": -0.76, "mid": -0.42, "near": bx - 0.20, "search": -0.62}[
                mode
            ] + float(viewrng.uniform(-0.04, 0.04))
            yaw = request["root_yaw_rad"]
            roll = request["root_roll_rad"]
            pitch = request["root_pitch_rad"]
            qr = (math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2))
            qp = (0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2))
            qy = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
            quat = qmul(qmul(qy, qp), qr)
            aim_ball = mode in ("mid", "near") or (
                mode == "far" and request["view_seed"] % 2 == 0
            )
            target = (
                np.array([bx, by, 0.035]) if aim_ball else np.array([0.93, 0.0, 0.12])
            )
            dx = target[0] - root_x
            dy = target[1] - root_y
            distance = max(0.08, math.hypot(dx, dy))
            gaze = [
                math.atan2(dy, dx) - yaw + request["head_yaw_jitter_rad"],
                math.atan2(0.205 - target[2], distance)
                + request["head_pitch_jitter_rad"],
            ]
            if mode == "search":
                gaze[0] += float(viewrng.uniform(-0.55, 0.55))
            head = adapter.solve(gaze)
            joint = home.clone()
            for name, value in zip(head["joint_names"], head["desired_head_joint_rad"]):
                joint[0, names.index(name)] = value
            # Each sample begins with an explicitly declared initialization;
            # this is permitted for data collection, never an approach shortcut.
            robot.write_root_pose_to_sim_index(
                root_pose=torch.tensor(
                    [[root_x, root_y, 0.125, *quat]],
                    device=env.device,
                    dtype=torch.float32,
                )
            )
            robot.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=env.device)
            )
            robot.write_joint_position_to_sim_index(position=joint)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(joint))
            robot.reset()
            robot.set_joint_position_target_index(target=joint)
            ball.write_root_pose_to_sim_index(
                root_pose=torch.tensor(
                    [[bx, by, 0.035, 0, 0, 0, 1]],
                    device=env.device,
                    dtype=torch.float32,
                )
            )
            ball.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((1, 6), device=env.device)
            )
            ball.reset()
            lamp.GetIntensityAttr().Set(request["light_intensity"])
            configure_optics(rig, request["distortion"])
            env.scene.write_data_to_sim()
            sim.step(render=False)
            env.scene.update(0.005)
            clock += 1
            capture = rig.capture(
                timestamp_s=clock * 0.005, render_subframes=args.render_subframes
            )
            rgb = capture["rgb_raw"]
            raw_depth = array(rig.rgb_camera.data.output["distance_to_camera"])[
                0
            ].squeeze()
            teacher_depth = cv2.remap(
                raw_depth,
                rig.map_x,
                rig.map_y,
                cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=float("inf"),
            )
            camera = capture["simulated_mount_poses"][rig.paths[0]]
            actual_ball = array(ball.data.root_pos_w)[0].tolist()
            boxes, diagnostics = labels(rig.profile, camera, teacher_depth, actual_ball)
            if rgb.std() < 3 or float((rgb.max(axis=2) > 8).mean()) < 0.05:
                raise RuntimeError(
                    "Actual RTX image is blank; retain failure instead of synthesizing pixels"
                )
            name = request["id"]
            image = output / "images" / f"{name}.png"
            label = output / "labels" / f"{name}.txt"
            Image.fromarray(rgb).save(image)
            label.write_text(
                "".join(
                    str(b["class_id"])
                    + " "
                    + " ".join(f"{v:.8f}" for v in b["yolo_xywh"])
                    + "\n"
                    for b in boxes
                )
            )
            teacher = {
                "request": request,
                "camera": camera,
                "profile": rig.profile.to_dict(),
                "actual_ball_world_m": actual_ball,
                "actual_root_world_m": array(robot.data.root_pos_w)[0].tolist(),
                "actual_root_quaternion_xyzw": array(robot.data.root_quat_w)[
                    0
                ].tolist(),
                "actual_joint_positions_rad": array(robot.data.joint_pos)[0].tolist(),
                "labels": boxes,
                "visibility_diagnostics": diagnostics,
                "label_method": "ray/sphere and ray/goalbox projections cross-checked against actual separate RTX camera depth; visible bounding rectangles",
                "geometry_depth_tolerance_m": 0.015,
                "teacher_only": True,
                "never_controller_input": True,
            }
            save(output / "teacher" / f"{name}.json", teacher)
            result = {
                "id": name,
                "split": request["split"],
                "scene_seed": request["scene_seed"],
                "view_seed": request["view_seed"],
                "image": str(image.relative_to(output)),
                "label": str(label.relative_to(output)),
                "image_sha256": digest(image),
                "label_sha256": digest(label),
                "teacher": f"teacher/{name}.json",
                "teacher_sha256": digest(output / "teacher" / f"{name}.json"),
                "class_ids": [b["class_id"] for b in boxes],
                "actual_rtx": True,
                "horizontal_fov_deg": args.hfov,
                "distortion": request["distortion"],
            }
            results.append(result)
            manifest["images"] = results
            save(output / "manifest.json", manifest)
            print(
                json.dumps(
                    {
                        "captured": len(results),
                        "planned": len(rows),
                        "id": name,
                        "classes": result["class_ids"],
                        "wall_s": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        manifest.update(
            status="complete",
            sensor_model=rig.metadata(),
            visual_geometry=env.course_visual_geometry,
            scope="Real static-pose training data; no locomotion/goal-success claim",
            wall_seconds=time.monotonic() - started,
        )
    except BaseException as error:
        manifest.update(
            status="failed",
            error=repr(error),
            traceback=traceback.format_exc(),
            wall_seconds=time.monotonic() - started,
        )
        raise
    finally:
        manifest["images"] = results
        manifest["source_sha256"] = {p.name: digest(p) for p in P.glob("*.py")}
        save(output / "manifest.json", manifest)
        if env is not None:
            env.close()


def main():
    args = arguments()
    plan = json.loads(args.plan.read_text())
    from plan_dataset import validate_plan

    validate_plan(plan)
    rows = [r for r in plan["rows"] if r["horizontal_fov_deg"] == args.hfov]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No requested rows")
    from isaaclab.app import launch_simulation

    # Launch Kit before any task/config/pxr import, matching the actual passing
    # sensor smoke. This prevents loading incompatible pip USD libraries first.
    with launch_simulation(None, {**vars(args), "require_kit": True}):
        import isaaclab_tasks
        from isaaclab.managers import EventTermCfg
        from isaac_execute import build_cfg

        # Reuse native BAM and the current Beta football scene; data collection
        # never constructs or executes an autonomous controller.
        setup = argparse.Namespace(
            device=args.device,
            seed=plan["seed"],
            seconds=5.0,
            randomized=False,
            root_y=0.0,
            ball_xy=[0.0, 0.0],
            layout="empty",
            width=960,
            height=540,
            assumed_hfov=args.hfov,
            robot_usd=M / "football/sources/microduck_walk_backlash.usd",
        )
        cfg, _ = build_cfg(setup)
        cfg.events.beta_sensors = EventTermCfg(
            func=prestartup,
            mode="prestartup",
            params={
                "width": 960,
                "height": 540,
                "hfov": args.hfov,
                "seed": plan["seed"],
            },
        )
        run(args, cfg, rows, plan)


if __name__ == "__main__":
    main()
