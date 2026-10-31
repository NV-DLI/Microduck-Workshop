"""Check the in-container ROS 2 Jazzy Nav2 install before any ROS node is created.

Same checks as lab/scene_nav2/image_preflight.py (GPU hidden, message type support,
direct package versions, native planner libraries load), adapted to an apt install
in the course container: direct packages must equal the lock's upstream versions on
every architecture; identical package identities are recorded (true on the lock's
arm64). Run with the system python3 after sourcing /opt/ros/jazzy/setup.bash.
"""
import argparse
import ctypes
import hashlib
import inspect
import json
import os
import platform
import subprocess
import traceback
from pathlib import Path

RECORD = Path("/usr/local/share/microduck-nav2/install-record.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(lock_path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("NVIDIA_VISIBLE_DEVICES") != "void":
        raise RuntimeError("CPU preflight requires GPUs hidden")
    from nav2_msgs.action import ComputePathToPose
    from lifecycle_msgs.srv import ChangeState
    from nav_msgs.msg import OccupancyGrid, Odometry
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.type_support import check_for_type_support
    from tf2_ros import StaticTransformBroadcaster

    for cls in [ComputePathToPose.Goal, ComputePathToPose.Result, ChangeState.Request,
                OccupancyGrid, Odometry, PoseStamped, TransformStamped]:
        check_for_type_support(cls)
    lock = json.loads(Path(lock_path).read_text())
    raw = subprocess.check_output(
        ["dpkg-query", "-W", "-f=${binary:Package}\t${Version}\n", *lock["direct_packages"]], text=True)
    installed = dict(line.split("\t", 1) for line in raw.strip().splitlines())
    upstream = {n: v.rsplit("-", 1)[0] for n, v in installed.items()}
    if upstream != {n: v.rsplit("-", 1)[0] for n, v in lock["direct_packages"].items()}:
        raise RuntimeError("ROS package upstream versions differ from the lock")
    checked = {}
    for path in ["/opt/ros/jazzy/lib/nav2_planner/planner_server", "/opt/ros/jazzy/lib/libnav2_navfn_planner.so"]:
        deps = subprocess.check_output(["ldd", path], text=True, stderr=subprocess.STDOUT)
        if "not found" in deps:
            raise RuntimeError("Missing native dependency: " + deps)
        checked[path] = {"sha256": sha(path)}
    ctypes.CDLL("/opt/ros/jazzy/lib/libnav2_navfn_planner.so")
    return {
        "status": "passed",
        "backend": "local ROS 2 Jazzy in the course container",
        "ROS_nodes_created": False,
        "rclpy_init_called": False,
        "GPU": False,
        "architecture": platform.machine(),
        "direct_package_versions": installed,
        "identical_to_lock": installed == lock["direct_packages"],
        "lock_architecture": lock["architecture"],
        "lock_sha256": sha(lock_path),
        "install_record_sha256": sha(RECORD) if RECORD.is_file() else None,
        "native_libraries": checked,
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
        "discovery_range": os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE"),
        "no_path_constants": {k: getattr(ComputePathToPose.Result, k)
                              for k in ["NO_VALID_PATH", "GOAL_OCCUPIED", "START_OCCUPIED"]},
        "executor_shutdown": str(inspect.signature(MultiThreadedExecutor.shutdown)),
        "static_tf_broadcaster": str(inspect.signature(StaticTransformBroadcaster.__init__)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--lock", type=Path, required=True)
    a = ap.parse_args()
    try:
        result = check(a.lock)
    except BaseException as exc:
        result = {"status": "failed", "ROS_nodes_created": False, "GPU": False,
                  "error": repr(exc), "traceback": traceback.format_exc()}
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
