"""Geometry: Microduck simulation and control."""
import hashlib
import importlib.util
import math
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).resolve().parent.parent/'ground_geometry/geometry.py'
SOURCE_SHA256='08f1ee1183c101953b5d408de76ba63501e661a1b9008ac5574443f48b4df8fe'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA256:
    raise RuntimeError('Frozen ground_geometry geometry changed')
spec=importlib.util.spec_from_file_location('frozen_ground_geometry_geometry',SOURCE)
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)


def _time(value):
    if isinstance(value,bool):raise ValueError('Boolean is not a timestamp')
    value=float(value)
    if not math.isfinite(value)or value<0:raise ValueError('Finite nonnegative timestamp required')
    return value


def refine_grounded_complete_ball(perception, *, tof_frame, now_s):
    """Refine one validated capture, using its original exact ToF packet."""
    rejected={'kind':'grounded_complete_YOLO_sphere_v2','accepted':False,
              'reason':'invalid_same_capture_input','truth_used':False,
              'kick_authorized':False,'task_or_hardware_validated':False}
    try:
        now=_time(now_s)
        if perception['schema']!='microduck.beta-geometry.v1'or perception['truth_used']is not False:
            return {**rejected,'reason':'original_public_perception_required'}
        ball=perception['ball']
        if not isinstance(ball,dict)or ball.get('accepted')is not True:
            return {**rejected,'reason':'original_complete_YOLO_geometry_required'}
        original=ball['geometry'];pose=perception['pose'];ground=perception['ground'];floor=perception['floor_observation']
        times=[_time(v)for v in(perception['timestamp_s'],original['capture_timestamp_s'],pose['timestamp_s'],tof_frame['timestamp_s'])]
        if max(times)-min(times)>1e-6:
            return {**rejected,'reason':'mixed_capture_timestamps'}
        if any(not 0<=now-t<=.2 for t in times):
            return {**rejected,'reason':'stale_or_future_capture'}
        frame=pose['reference_frame']
        if not isinstance(frame,str)or not frame or ground['reference_frame']!=frame or original['reference_frame']!=frame:
            return {**rejected,'reason':'mixed_reference_frame'}
        if not isinstance(floor,dict)or floor.get('accepted')is not True or floor.get('truth_used')is not False:
            return {**rejected,'reason':'accepted_actual_ToF_floor_required'}
        if tof_frame.get('frame_dropped',False) or len(tof_frame['cells'])!=8 or any(len(row)!=8 for row in tof_frame['cells']):
            return {**rejected,'reason':'same_capture_actual64cell_ToF_required'}
        # Parent fit_virtual_floor supplies these exact plane coefficients to
        # BetaPerception.pose. A different plane must not masquerade as that fit.
        n=np.asarray(ground['normal'],float);f=np.asarray(floor['normal_body'],float)
        offsets=[float(ground['offset_m']),float(floor['offset_m'])]
        if n.shape!=(3,)or f.shape!=(3,)or not np.isfinite(n).all()or not np.isfinite(f).all()or not all(math.isfinite(x)for x in offsets):
            return {**rejected,'reason':'invalid_floor_plane'}
        if not np.array_equal(n,f)or offsets[0]!=offsets[1]:
            return {**rejected,'reason':'ground_not_same_observed_floor'}
        uncertainty=float(original['estimated_range_uncertainty_m'])
        if not math.isfinite(uncertainty)or uncertainty<0:
            return {**rejected,'reason':'invalid_parent_uncertainty'}
        # No geometry or parent uncertainty change: the only new behavior is
        # admission validation above. v1's accepted metadata is retained exactly.
        return parent.refine_grounded_complete_ball(original,pose,ground,floor,now_s=now)
    except (KeyError,TypeError,ValueError,OverflowError):
        return rejected
