#!/usr/bin/env python3
"""Explicit terminal entry points for the Microduck student labs."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ".local/runtime.json"


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def config():
    if not CONFIG.exists():
        raise RuntimeError("Run python3 scripts/workshop.py setup first.")
    return json.loads(CONFIG.read_text())


def environment(c):
    env = os.environ.copy()
    env.update(ACCEPT_EULA="Y", OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1")
    env["PATH"] = str(Path(c["ffmpeg"]).parent) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONPATH", None)
    lib = Path("/lib/aarch64-linux-gnu/libgomp.so.1")
    if lib.exists():
        env["LD_PRELOAD"] = str(lib)
    return env


def verify():
    d = json.loads((ROOT / "models/manifest.json").read_text())
    for item in d["files"]:
        f = ROOT / "models" / item["file"]
        if not f.is_file() or sha(f) != item["sha256"]:
            raise ValueError("Model identity mismatch: " + item["file"])
        print("OK", item["file"], item["bytes"], "bytes")
    extra = json.loads((ROOT / "models/indoor-detector/deployment.json").read_text())
    for key in ("weights", "onnx"):
        f = ROOT / "models/indoor-detector" / extra[key]
        if not f.is_file() or sha(f) != extra[key + "_sha256"]:
            raise ValueError("Indoor model identity mismatch: " + f.name)
        print("OK", "indoor-detector/" + f.name, f.stat().st_size, "bytes")
    return d


def run(argv, c):
    print("Launching:", " ".join(str(x) for x in argv), flush=True)
    return subprocess.call([str(x) for x in argv], cwd=ROOT, env=environment(c))


def fresh(p):
    p = p.resolve()
    if p.exists():
        raise FileExistsError(
            "Keep previous results. Choose a new output directory: " + str(p)
        )
    if p.is_relative_to(ROOT / "models"):
        raise ValueError("Student outputs must not replace fixed models")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("verify-models")
    sub.add_parser("doctor")
    setup = sub.add_parser("setup")
    setup.add_argument("--isaaclab", type=Path)
    setup.add_argument("--detector-python", type=Path)
    setup.add_argument("--ffmpeg", type=Path)
    setup.add_argument("--nav2-image")
    ins = sub.add_parser("inspect-model")
    ins.add_argument("--model", type=Path, required=True)
    walk = sub.add_parser("walk")
    walk.add_argument("--model", type=Path, required=True)
    walk.add_argument("--velocity", nargs=3, type=float, default=[0.3, 0.0, 0.0])
    walk.add_argument("--seconds", type=float, default=12)
    walk.add_argument("--seed", type=int, default=42)
    walk.add_argument("--output", type=Path, required=True)
    walk.add_argument("--headless", action="store_true")
    manual = sub.add_parser("manual")
    manual.add_argument("--seconds", type=float, default=600)
    manual.add_argument("--style", choices=list("ABCD"), default="A")
    manual.add_argument(
        "--locomotion", choices=["walking", "rollers"], default="walking"
    )
    manual.add_argument("--output", type=Path, required=True)
    terrain = sub.add_parser("terrain")
    terrain.add_argument("--model", type=Path, required=True)
    terrain.add_argument("--slope", type=float, default=5)
    terrain.add_argument("--num-envs", type=int, default=5)
    terrain.add_argument("--seconds", type=float, default=20)
    terrain.add_argument("--seed", type=int, default=42001)
    terrain.add_argument("--output", type=Path, required=True)
    terrain.add_argument("--headless", action="store_true")
    terrain.add_argument("--record", action="store_true")
    terrain.add_argument(
        "--locomotion", choices=["walking", "rollers"], default="walking"
    )
    slope_train = sub.add_parser("train-terrain")
    slope_train.add_argument("--yaw-precision", action="store_true")
    slope_train.add_argument("--resume", type=Path)
    slope_train.add_argument("--refine-heading", action="store_true")
    slope_train.add_argument("--model", type=Path, required=True)
    slope_train.add_argument("--num-envs", type=int, default=256)
    slope_train.add_argument("--iterations", type=int, default=1000)
    slope_train.add_argument("--seed", type=int, default=43001)
    slope_train.add_argument("--output", type=Path, required=True)
    train = sub.add_parser("train-walk")
    train.add_argument("--num-envs", type=int, default=64)
    train.add_argument("--iterations", type=int, default=20)
    train.add_argument("--seed", type=int, default=101)
    train.add_argument("--output", type=Path, required=True)
    tb = sub.add_parser("tensorboard")
    tb.add_argument("--logdir", type=Path, required=True)
    tb.add_argument("--port", type=int, default=6006)
    det = sub.add_parser("detect")
    det.add_argument("--model", type=Path, required=True)
    det.add_argument("--source", type=Path, required=True)
    det.add_argument("--output", type=Path, required=True)
    td = sub.add_parser("train-detector")
    td.add_argument("--data", type=Path, required=True)
    td.add_argument("--epochs", type=int, default=5)
    td.add_argument("--batch", type=int, default=8)
    td.add_argument("--seed", type=int, default=303)
    td.add_argument("--output", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--ball", nargs=2, type=float, required=True)
    plan.add_argument("--goal", nargs=2, type=float, required=True)
    plan.add_argument("--distance", type=float, default=0.18)
    plan.add_argument("--lateral", type=float, default=0.035)
    plan.add_argument("--output", type=Path, required=True)
    fb = sub.add_parser("football")
    fb.add_argument("--seed", type=int, default=983102)
    fb.add_argument("--locomotion", choices=["walking", "rollers"], default="walking")
    fb.add_argument("--scene", choices=["legacy", "indoor"], default="indoor")
    fb.add_argument("--style", choices=list("ABCD"), default="A")
    fb.add_argument("--record-hd", action=argparse.BooleanOptionalAction, default=True)
    fb.add_argument("--root-y", type=float)
    fb.add_argument("--ball-xy", type=float, nargs=2)
    fb.add_argument("--seconds", type=float, default=180)
    fb.add_argument("--output", type=Path, required=True)
    fb.add_argument("--headless", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    if a.command == "verify-models":
        verify()
        return 0
    if a.command == "setup":
        from setup_runtime import setup_runtime

        setup_runtime(a)
        return 0
    if a.command == "plan":
        if (
            not all(math.isfinite(x) for x in [*a.ball, *a.goal, a.distance, a.lateral])
            or a.distance <= 0
        ):
            ap.error("Use finite coordinates and positive distance")
        delta = [a.goal[i] - a.ball[i] for i in range(2)]
        length = math.hypot(*delta)
        if length < 1e-6:
            ap.error("Ball and goal must differ")
        direction = [x / length for x in delta]
        normal = [-direction[1], direction[0]]
        target = [
            a.ball[i] - a.distance * direction[i] + a.lateral * normal[i]
            for i in range(2)
        ]
        out = fresh(a.output)
        out.mkdir()
        result = dict(
            ball=a.ball,
            goal=a.goal,
            target=target,
            heading_rad=math.atan2(direction[1], direction[0]),
            scope="known-point geometric exercise; not physical navigation",
        )
        (out / "plan.json").write_text(json.dumps(result, indent=2) + "\n")

        def xy(p):
            return 80 + (p[0] + 0.3) * 430, 220 - p[1] * 430

        bx, by = xy(a.ball)
        gx, gy = xy(a.goal)
        tx, ty = xy(target)
        (out / "plan.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 700 400"><rect width="700" height="400" fill="#f3f7ef"/><path d="M{tx},{ty} L{bx},{by} L{gx},{gy}" stroke="#76b900" stroke-width="3" fill="none"/><circle cx="{bx}" cy="{by}" r="9" fill="#ef8b30"/><circle cx="{tx}" cy="{ty}" r="8" fill="#202d23"/><rect x="{gx-3}" y="{gy-40}" width="6" height="80" fill="#76b900"/><g font-family="sans-serif" font-size="16" fill="#263127"><text x="{tx}" y="{ty+35}">Target</text><text x="{bx}" y="{by-22}">Ball</text><text x="{gx-20}" y="{gy+65}">Goal</text></g></svg>'
        )
        print(json.dumps(result, indent=2))
        print("Diagram:", out / "plan.svg")
        return 0
    if a.command == "report":
        for path in [
            a.run / "result.json",
            a.run / "independent-result.json",
            a.run / "manifest.json",
            a.run / "report.json",
            a.run / "training-ledger.json",
        ]:
            if path.is_file():
                data = json.loads(path.read_text())
                print(
                    json.dumps(
                        {
                            k: data[k]
                            for k in [
                                "status",
                                "state",
                                "seed",
                                "task_seconds",
                                "task_result",
                                "autonomous_task_passed",
                                "real_wheel_ball_contact",
                                "goal_time_s",
                                "continuous_final_recovery_s",
                                "terminal_reason",
                                "scorer",
                                "course_evaluation",
                                "model_sha256",
                                "episodes",
                                "training_wall_seconds",
                                "passed",
                                "fell",
                                "reached_landing",
                                "upright_velocity_rmse",
                                "simulated_seconds",
                            ]
                            if k in data
                        },
                        indent=2,
                    )
                )
                return 0
        raise FileNotFoundError(
            "No result manifest yet. Inspect the run log or wait for the run to finish."
        )
    c = config()
    py = c["python"]
    m = Path(c["materialized"])
    if a.command == "doctor":
        verify()
        probe = "import importlib.metadata as m,json,torch;print(json.dumps({'cuda':torch.cuda.is_available(),'versions':{k:m.version(k)for k in ['isaacsim','newton','warp-lang','torch','rsl-rl-lib','onnx','onnxruntime']}}))"
        data = json.loads(
            subprocess.check_output([py, "-c", probe], env=environment(c), text=True)
        )
        data["display"] = bool(os.environ.get("DISPLAY"))
        data["ffmpeg"] = Path(c["ffmpeg"]).is_file()
        data["detector_python"] = Path(c["detector_python"]).is_file()
        image = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", c["nav2_image"]],
            capture_output=True,
            text=True,
        )
        data["nav2_image"] = (
            image.returncode == 0 and image.stdout.strip() == c["nav2_image"]
        )
        print(json.dumps(data, indent=2))
        return (
            0
            if all(
                [
                    data["cuda"],
                    data["ffmpeg"],
                    data["detector_python"],
                    data["nav2_image"],
                ]
            )
            else 1
        )
    if a.command == "inspect-model":
        return run([py, ROOT / "lab/model_info.py", "--model", a.model.resolve()], c)
    if a.command == "tensorboard":
        return run(
            [
                py,
                "-m",
                "tensorboard.main",
                "--logdir",
                a.logdir.resolve(),
                "--port",
                a.port,
                "--host",
                "127.0.0.1",
            ],
            c,
        )
    if a.command in ("walk", "football") and not (
        math.isfinite(a.seconds) and 0 < a.seconds <= 180
    ):
        ap.error("seconds must be in (0,180]")
    if a.command == "walk":
        if not all(math.isfinite(x) for x in a.velocity):
            ap.error("Velocity must be finite")
        out = fresh(a.output)
        cmd = [
            py,
            ROOT / "lab/walk.py",
            "--runtime",
            m,
            "--model",
            a.model.resolve(),
            "--seconds",
            a.seconds,
            "--seed",
            a.seed,
            "--output",
            out,
            "--velocity",
            *a.velocity,
        ]
        if a.headless:
            cmd += ["--headless"]
        return run(cmd, c)
    if a.command == "manual":
        if not math.isfinite(a.seconds) or not 1 <= a.seconds <= 3600:
            ap.error("seconds must be in [1,3600]")
        verify()
        return run(
            [
                py,
                ROOT / "lab/manual.py",
                "--locomotion",
                a.locomotion,
                "--style",
                a.style,
                "--runtime",
                m,
                "--models",
                ROOT / "models",
                "--output",
                fresh(a.output),
                "--seconds",
                a.seconds,
            ],
            c,
        )
    if a.command == "terrain":
        cmd = [
            py,
            ROOT / "lab/terrain.py",
            "--locomotion",
            a.locomotion,
            "--runtime",
            m,
            "--model",
            a.model.resolve(),
            "--slope",
            a.slope,
            "--num-envs",
            a.num_envs,
            "--seconds",
            a.seconds,
            "--seed",
            a.seed,
            "--output",
            fresh(a.output),
        ]
        if a.headless:
            cmd += ["--headless"]
        if a.record:
            cmd += ["--record"]
        return run(cmd, c)
    if a.command == "train-terrain":
        if a.num_envs < 1 or a.iterations < 1:
            ap.error("Use positive training counts")
        cmd = [
            py,
            ROOT / "lab/train_terrain.py",
            "--runtime",
            m,
            "--model",
            a.model.resolve(),
            "--num-envs",
            a.num_envs,
            "--iterations",
            a.iterations,
            "--seed",
            a.seed,
            "--output",
            fresh(a.output),
        ]
        if a.resume:
            cmd += ["--resume", a.resume.resolve()]
        if a.refine_heading:
            cmd += ["--refine-heading"]
        if a.yaw_precision:
            cmd += ["--yaw-precision"]
        return run(cmd, c)
    if a.command == "train-walk":
        if a.num_envs < 1 or a.iterations < 1:
            ap.error("Use positive training counts")
        return run(
            [
                py,
                ROOT / "lab/train_walk.py",
                "--runtime",
                m,
                "--num-envs",
                a.num_envs,
                "--iterations",
                a.iterations,
                "--seed",
                a.seed,
                "--output",
                fresh(a.output),
            ],
            c,
        )
    if a.command in ("detect", "train-detector"):
        out = fresh(a.output)
        cmd = [
            c["detector_python"],
            ROOT / "lab/detector.py",
            a.command,
            "--output",
            out,
        ]
        cmd += (
            ["--model", a.model.resolve(), "--source", a.source.resolve()]
            if a.command == "detect"
            else [
                "--data",
                a.data.resolve(),
                "--epochs",
                a.epochs,
                "--batch",
                a.batch,
                "--seed",
                a.seed,
            ]
        )
        return run(cmd, c)
    if a.command == "football":
        verify()
        out = fresh(a.output)
        if a.locomotion == "rollers":
            if a.scene != "indoor":
                ap.error("Roller shooting requires the indoor scene")
            cmd = [
                py,
                ROOT / "lab/manual.py",
                "--runtime",
                m,
                "--models",
                ROOT / "models",
                "--output",
                out,
                "--locomotion",
                "rollers",
                "--autonomous",
                "--detector-deployment",
                ROOT / "models/scene-detector/deployment.json",
                "--seconds",
                a.seconds,
                "--seed",
                a.seed,
                "--style",
                a.style,
                "--root-x",
                -0.30,
                "--root-y",
                a.root_y if a.root_y is not None else 0.045,
                "--ball-x",
                (a.ball_xy or [0.0, 0.0])[0],
                "--ball-y",
                (a.ball_xy or [0.0, 0.0])[1],
                "--shot-push-seconds",
                2.1,
            ]
            if a.headless:
                cmd += ["--headless-check"]
            if a.record_hd:
                cmd += ["--record"]
            result = run(cmd, c)
            if result:
                return result
            return run(
                [
                    py,
                    ROOT / "lab/evaluate_rollers.py",
                    "--run",
                    out,
                    "--output",
                    out / "independent-result.json",
                ],
                c,
            )
        cmd = [
            py,
            m / "evaluation/football/execute_media.py",
            "--model",
            m / "models/reference/velstand.onnx",
            "--output",
            out,
            "--development",
            "--seconds",
            a.seconds,
            "--seed",
            a.seed,
            "--body-fusion-tau-s",
            "2",
            "--body-added-delay-s",
            "0",
            "--policy-imu",
            "body50",
            "--detector-deployment",
            m.parents[1]
            / "configuration/core/beta-detector/football_detector/deployment.json",
            "--detector-backend",
            "torch",
            "--detector-device",
            "cuda:0",
            "--shared-imu-readback",
            "--packed-physics-readback",
            "--root-x",
            "-0.66",
            "--recovery-approval",
            m / "evaluation/football/recovery-approval.json",
        ]
        if a.scene == "indoor":
            manifest = ROOT / "models/scene-detector/deployment.json"
            deployment = json.loads(manifest.read_text())
            for key in ("weights", "onnx"):
                artifact = manifest.parent / deployment[key]
                if (
                    hashlib.sha256(artifact.read_bytes()).hexdigest()
                    != deployment[key + "_sha256"]
                ):
                    raise ValueError("Indoor detector artifact changed")
                deployment[key] = str(artifact)
            local_deployment = ROOT / ".local/scene-detector-deployment.json"
            local_deployment.write_text(json.dumps(deployment, indent=2) + "\n")
            cmd[1] = ROOT / "lab/football_scene.py"
            cmd += [
                "--runtime",
                m,
                "--style",
                a.style,
                "--render-subframes",
                "4",
                "--root-y",
                a.root_y if a.root_y is not None else 0.04,
                "--ball-xy",
                *(a.ball_xy if a.ball_xy is not None else [0.02, 0.0]),
            ]
            cmd[cmd.index("--root-x") + 1] = "-0.55"
            cmd[cmd.index("--detector-deployment") + 1] = local_deployment
            if a.record_hd:
                cmd += ["--record-hd"]
        if not a.headless:
            cmd += ["--visualizer", "kit"]
        result = run(cmd, c)
        if result:
            return result
        return run(
            [
                py,
                ROOT / "lab/evaluate_task.py",
                "--runtime",
                m,
                "--run",
                out,
                *(["--scene-contract"] if a.scene == "indoor" else []),
            ],
            c,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        RuntimeError,
        ValueError,
        FileNotFoundError,
        FileExistsError,
        subprocess.CalledProcessError,
    ) as error:
        print("Error:", error, file=sys.stderr)
        raise SystemExit(1)
