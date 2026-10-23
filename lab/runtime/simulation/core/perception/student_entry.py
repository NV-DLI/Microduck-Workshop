#!/usr/bin/env python3
"""Fixed course actions launched by the current Isaac Python, with owned outputs.

Detector training uses only the pinned separate overlay. Capturing three camera
poses is clearly distinct from using the supplied384-image detector dataset.
"""
import argparse
import hashlib
import json
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
M = ROOT.parent
C = M.parents[1]
L = M / "runtime/IsaacLab-5088b326c7fc81caa7899395b0176ad08b89e9e1"
SOURCE = C / "configuration/core/beta-detector/dataset"
SOURCE_SHA = "b19d1d6fce2a9d2fced110bb10a7e1ce7d792ff04131b9e4781ebcf3b1a97f2e"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


METRIC_LABELS = {
    "capturedPoses": {"zh": "本次拍摄姿态数", "en": "Camera poses captured"},
    "trained": {"zh": "是否重新训练", "en": "Training performed"},
    "epochs": {"zh": "训练轮数", "en": "Training epochs"},
    "optimizerUpdates": {"zh": "参数更新次数", "en": "Optimizer updates"},
    "trainingWallSeconds": {
        "zh": "实际训练时长（秒）",
        "en": "Training wall time (seconds)",
    },
    "mAP50": {"zh": "测试集检测准确度 mAP50", "en": "Test detection mAP50"},
    "mAP50to95": {"zh": "测试集定位精度 mAP50–95", "en": "Test localization mAP50–95"},
}


def detector_artifacts(training, parity, job):
    artifacts = []
    for source, filename, zh, en in [
        (
            training / "training-report.json",
            "training-report.json",
            "本次训练与测试报告",
            "Current training and test report",
        ),
        (
            training / "deployment.json",
            "deployment.json",
            "本次模型导出记录",
            "Current model export record",
        ),
        (
            parity / "report.json",
            "export-parity.json",
            "导出一致性检查",
            "Export parity check",
        ),
    ]:
        target = job / filename
        if target.exists():
            raise FileExistsError("Keep existing artifact")
        json.loads(source.read_text())
        shutil.copy2(source, target)
        artifacts.append(
            {
                "file": filename,
                "label": {"zh": zh, "en": en},
                "kind": "json",
                "sha256": sha(target),
            }
        )
    return artifacts


def sensor_artifacts(destination, job, manifest):
    artifacts = []
    for index, sample in enumerate(manifest["samples"]):
        for key, view, label in [
            ("rgb_raw", "raw", "RGB"),
            ("rgb_upright", "upright", "Upright RGB"),
            ("tof_preview", "tof", "8×8 ToF"),
        ]:
            source = (destination / sample[key]).resolve()
            if not source.is_relative_to(destination.resolve()) or not source.is_file():
                raise ValueError("Unsafe or missing sensor artifact")
            filename = f"sensor-{index}-{view}.png"
            target = job / filename
            if target.exists():
                raise FileExistsError("Keep existing artifact")
            shutil.copy2(source, target)
            artifacts.append(
                {
                    "file": filename,
                    "label": {
                        "zh": f"姿态{index+1} · {label}",
                        "en": f"Pose{index+1} · {label}",
                    },
                    "kind": "image",
                    "sensorView": view,
                    "sampleIndex": index,
                    "sha256": sha(target),
                }
            )
    shutil.copy2(destination / "manifest.json", job / "sensor-manifest.json")
    artifacts.append(
        {
            "file": "sensor-manifest.json",
            "label": {"zh": "本次采集记录", "en": "Current capture record"},
            "kind": "json",
            "sha256": sha(job / "sensor-manifest.json"),
        }
    )
    return artifacts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action", choices=["hardware-sensors", "capture-sensors", "detector-train"]
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Only2training epochs, never a learned detector claim",
    )
    p.add_argument("--seed", type=int, default=978301)
    a = p.parse_args()
    result = a.output.resolve()
    job = result.parent
    if result.exists():
        raise FileExistsError("Keep each student attempt")
    job.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(
        schema="microduck.student-sensor-detector-lab.v1",
        action=a.action,
        runtime_id="newton-bam-20261019",
        hardware_profile_id="beta-gc2093-l8cx-v1",
        status="running",
        qualification_claimed=False,
        physical_episode=False,
        robot_task_success_claimed=False,
        commands=[],
        summary={},
        metricLabels=METRIC_LABELS,
    )
    env = dict(os.environ)
    env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    env["LD_PRELOAD"] = "/lib/aarch64-linux-gnu/libgomp.so.1"
    env["PATH"] = str(M / "runtime/tools/bin") + os.pathsep + env.get("PATH", "")

    def run(argv):
        report["commands"].append(argv)
        result.write_text(json.dumps(report, indent=2) + "\n")
        subprocess.run(argv, check=True, env=env, cwd=job)

    try:
        if a.action in ("hardware-sensors", "capture-sensors"):
            destination = job / "sensor-capture"
            run(
                [
                    str(L / ".venv/bin/python"),
                    str(M / "sensors/isaac_beta_sensors_smoke.py"),
                    "--output",
                    str(destination),
                    "--frames",
                    "3",
                    "--smoke",
                    "--mount-profile",
                    "rl-jaw-soft-20261012",
                    "--width",
                    "960",
                    "--height",
                    "540",
                    "--seed",
                    str(a.seed),
                    "--robot-usd",
                    str(M / "visuals/assets/microduck_white_orange.usda"),
                ]
            )
            manifest = json.loads((destination / "manifest.json").read_text())
            report.update(
                status="passed",
                student_capture_performed=True,
                training_performed=False,
                capture_scope="Three actual static camera poses; not a384frame training capture",
                sensor_manifest=str(destination / "manifest.json"),
                sensor_manifest_sha256=sha(destination / "manifest.json"),
                preview=str(destination / "index.html"),
                artifacts=sensor_artifacts(destination, job, manifest),
                metrics={"capturedPoses": 3, "trained": False},
                summary={
                    "zh": "已在当前学生目录重新拍摄3个头部姿态的RGB/ToF。完整检测训练使用另行标注的课程提供384幅数据。",
                    "en": "Captured3real head poses into this student run. Detector training separately uses the disclosed supplied384-image dataset.",
                },
            )
        else:
            detector_python = ROOT / ".venv/bin/python"
            lineage = ROOT / ".venv/environment-lineage.json"
            versions = json.loads(
                subprocess.check_output(
                    [
                        str(detector_python),
                        "-c",
                        "import sys,json,importlib.metadata as m;print(json.dumps({'prefix':sys.prefix,'ultralytics':m.version('ultralytics'),'torch':m.version('torch')}))",
                    ],
                    text=True,
                )
            )
            if (
                Path(versions["prefix"]).resolve() != (ROOT / ".venv").resolve()
                or versions["ultralytics"] != "8.4.182"
            ):
                raise RuntimeError("Pinned separate detector overlay required")
            if sha(SOURCE / "manifest.json") != SOURCE_SHA:
                raise ValueError("Provided dataset identity changed")
            report["detector_environment"] = {
                **versions,
                "python": str(detector_python),
                "lineage_sha256": sha(lineage),
                "requirements_sha256": sha(ROOT / ".venv/requirements-resolved.txt"),
            }
            dataset = job / "provided-dataset"
            training = job / "detector-training"
            run(
                [
                    str(detector_python),
                    str(ROOT / "prepare_student_dataset.py"),
                    "--source",
                    str(SOURCE),
                    "--output",
                    str(dataset),
                ]
            )
            argv = [
                str(detector_python),
                str(ROOT / "train_detector.py"),
                "--dataset",
                str(dataset),
                "--output",
                str(training),
                "--epochs",
                "120",
                "--batch",
                "16",
                "--seed",
                str(a.seed),
                "--device",
                "0",
            ]
            if a.smoke:
                argv.append("--smoke")
            run(argv)
            parity = job / "export-parity"
            argv = [
                str(detector_python),
                str(ROOT / "validate_export.py"),
                "--deployment",
                str(training / "deployment.json"),
                "--dataset",
                str(dataset),
                "--output",
                str(parity),
                "--images",
                "8",
            ]
            if a.smoke:
                argv.append("--allow-smoke")
            run(argv)
            tr = json.loads((training / "training-report.json").read_text())
            report.update(
                status="passed",
                student_capture_performed=False,
                training_performed=True,
                student_weights_trained_from="random initialization, no teacher weights",
                smoke=a.smoke,
                training_report=str(training / "training-report.json"),
                training_report_sha256=sha(training / "training-report.json"),
                deployment=str(training / "deployment.json"),
                deployment_sha256=sha(training / "deployment.json"),
                parity_report=str(parity / "report.json"),
                artifacts=detector_artifacts(training, parity, job),
                metrics={
                    "epochs": tr["epochs_completed"],
                    "optimizerUpdates": tr["optimizer_updates"],
                    "trainingWallSeconds": tr["training_wall_seconds"],
                    "mAP50": tr["heldout_test_metrics"].get("metrics/mAP50(B)"),
                    "mAP50to95": tr["heldout_test_metrics"].get("metrics/mAP50-95(B)"),
                },
                summary={
                    "zh": (
                        "已完成2轮随机初始化训练与导出一致性检查，仅验证流程，尚不代表检测准确。"
                        if a.smoke
                        else "已用课程提供的真实图像重新训练检测器，并检查独立测试集与ONNX导出。检测成绩不等于射门成功率。"
                    ),
                    "en": (
                        "Completed2epochs and export parity as a pipeline check only."
                        if a.smoke
                        else "Trained new weights from supplied real images, then evaluated held-out scenes and ONNX parity. Detection scores are not football success rates."
                    ),
                },
            )
    except BaseException as error:
        report.update(status="failed", error=repr(error))
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        result.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "output": str(result),
                    "qualification_claimed": False,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
