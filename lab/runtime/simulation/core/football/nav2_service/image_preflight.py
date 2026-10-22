"""Validate the actual isolated image before creating any ROS node."""
import argparse
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import traceback


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('NVIDIA_VISIBLE_DEVICES')!='void':
        raise RuntimeError('CPU image preflight requires GPUs hidden')
    from nav2_msgs.action import ComputePathToPose
    from lifecycle_msgs.srv import ChangeState
    from nav_msgs.msg import OccupancyGrid,Odometry
    from geometry_msgs.msg import PoseStamped,TransformStamped
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.type_support import check_for_type_support
    from tf2_ros import StaticTransformBroadcaster
    for cls in [ComputePathToPose.Goal,ComputePathToPose.Result,ChangeState.Request,OccupancyGrid,Odometry,PoseStamped,TransformStamped]:
        check_for_type_support(cls)
    lock=json.loads(Path(__file__).with_name('image-lock.json').read_text())
    requests=list(lock['direct_packages'])
    raw=subprocess.check_output(['dpkg-query','-W','-f=${binary:Package}\t${Version}\n',*requests],text=True)
    installed=dict(line.split('\t',1) for line in raw.strip().splitlines())
    if installed!=lock['direct_packages']:raise RuntimeError('Pinned ROS package versions differ')
    checked={}
    for path in ['/opt/ros/jazzy/lib/nav2_planner/planner_server','/opt/ros/jazzy/lib/libnav2_navfn_planner.so']:
        deps=subprocess.check_output(['ldd',path],text=True,stderr=subprocess.STDOUT)
        if 'not found' in deps:raise RuntimeError('Missing native dependency: '+deps)
        checked[path]={'sha256':sha(path),'ldd':deps}
    ctypes.CDLL('/opt/ros/jazzy/lib/libnav2_navfn_planner.so')
    return {'status':'passed','ROS_nodes_created':False,'rclpy_init_called':False,'GPU':False,
            'architecture':platform.machine(),'os_release':Path('/etc/os-release').read_text(),
            'direct_package_versions':installed,'native_libraries':checked,
            'image_id_binding':lock['image_id'],'image_lock_sha256':sha(Path(__file__).with_name('image-lock.json')),
            'dpkg_inventory_sha256':sha('/usr/local/share/nav2-cpu/dpkg.tsv'),
            'deb_archive_manifest_sha256':sha('/usr/local/share/nav2-cpu/deb-sha256.txt'),
            'action_goal_fields':ComputePathToPose.Goal.get_fields_and_field_types(),
            'action_result_fields':ComputePathToPose.Result.get_fields_and_field_types(),
            'lifecycle_request_fields':ChangeState.Request.get_fields_and_field_types(),
            'no_path_constants':{k:getattr(ComputePathToPose.Result,k) for k in ['NO_VALID_PATH','GOAL_OCCUPIED','START_OCCUPIED']},
            'executor_shutdown':str(inspect.signature(MultiThreadedExecutor.shutdown)),
            'static_tf_broadcaster':str(inspect.signature(StaticTransformBroadcaster.__init__))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    try:result=check()
    except BaseException as exc:result={'status':'failed','ROS_nodes_created':False,'GPU':False,'error':repr(exc),'traceback':traceback.format_exc()}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
