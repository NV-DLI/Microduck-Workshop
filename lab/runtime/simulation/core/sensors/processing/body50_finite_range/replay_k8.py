"""Offline sensor-hypothesis comparison on unchanged recorded physical states.

Common explicit zero Gaussian draws isolate the reset rule; the original GPU
random stream was not recorded. This is not bit-exact historical packet replay,
new closed-loop physics, or evidence of a successful task after its old end.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
M=HERE.parents[2]
O=M.parents[1]/'configuration/core'
sys.path.insert(0,str(M/'evaluation/body50_training'))
from physics_producer import Body50PhysicsProducer
from vector_body_imu import rotate,inverse
from adapter import install_before_first_tick

D=O/'kick-v8-parent-region-heldout32-s979801'
state=np.load(D/'physical-training-trace.npz')['state']
actual=np.load(D/'body50-delivered-packets.npz')
conditions=np.load(D/'evaluation-conditions.npz')
manifest=json.loads((D/'manifest.json').read_text())
n=state.shape[1]
producers=[Body50PhysicsProducer(n),Body50PhysicsProducer(n)]
install_before_first_tick(producers[1])
noise=torch.zeros(n,6,dtype=torch.float64)
enc=torch.zeros(n,14,dtype=torch.float64)
rows=[]
previous_episode=None
for step,row in enumerate(state):
    data=torch.from_numpy(row).to(torch.float64)
    if previous_episode is not None:
        ids=torch.where(data[:,82]!=previous_episode)[0]
        if len(ids):
            for p in producers:p.reset(ids)
    previous_episode=data[:,82].clone()
    q=data[:,3:7]
    packets=[p.update(trunk_position_world_m=data[:,:3],trunk_quaternion_xyzw=q,
        trunk_angular_velocity_world_rad_s=rotate(q,data[:,10:13]),
        effective_encoder_q_rad=enc,effective_encoder_qd_rad_s=enc,
        noise_standard_normal=noise)for p in producers]
    gravity_truth=rotate(inverse(q),torch.tensor([0.,0.,-1.],dtype=torch.float64))
    out={}
    for label,p in zip(('old','continuous'),packets):
        g=p['gravity_body'];dot=(g*gravity_truth).sum(-1)/torch.linalg.vector_norm(g,dim=-1).clamp_min(1e-30)
        out[label+'_valid']=p['orientation_valid'].numpy().copy()
        out[label+'_gravity_error_deg']=torch.rad2deg(torch.acos(dot.clamp(-1,1))).numpy()
        out[label+'_clipped']=p['gyro_clipped'].numpy().copy()
    out['measurement_seq_same']=torch.eq(packets[0]['seq'],packets[1]['seq']).numpy()
    out['gyro_counts_same']=(packets[0]['gyro_counts']==packets[1]['gyro_counts']).all(-1).numpy()
    rows.append(out)
arrays={key:np.stack([r[key]for r in rows])for key in rows[0]}
np.savez_compressed(O/'body50-continuous-range-k8-replay-v1.npz',**arrays)
worlds=[]
for j in range(n):
    # physical_step is one-based, preserve the original terminal bound.
    tick=int(conditions['finish_tick'][j]);ix=np.flatnonzero(state[:,j,88]<=tick)
    last=int(ix[-1])if len(ix)else 0
    task=state[:last+1,j,83]>.5
    errors=arrays['continuous_gravity_error_deg'][:last+1,j][task&arrays['continuous_valid'][:last+1,j]]
    worlds.append({'world':j,'original_finish_tick':tick,'original_finish_reason_code':int(conditions['finish_reason'][j]),
        'original_delivered_valid_at_finish':bool(actual['orientation_valid'][last,j]),
        'original_delivered_clip_at_finish':actual['gyro_clipped'][last,j].tolist(),
        'reconstructed_old_valid_at_finish':bool(arrays['old_valid'][last,j]),
        'continuous_valid_at_original_finish':bool(arrays['continuous_valid'][last,j]),
        'continuous_gyro_clipped_at_original_finish':arrays['continuous_clipped'][last,j].tolist(),
        'task_gravity_error_deg_p95':float(np.quantile(errors,.95))if len(errors)else None,
        'task_gravity_error_deg_max':float(np.max(errors))if len(errors)else None})
report={'schema':'microduck.body50-continuous-range-k8-offline.v1','scope':__doc__,
    'source_trajectory':str(D),'source_trajectory_unchanged':True,'physical_ticks':len(state),'worlds':worlds,
    'comparison_checks':{'same_bus_sequences':bool(arrays['measurement_seq_same'].all()),'same_quantized_gyro_counts':bool(arrays['gyro_counts_same'].all()),'all_values_finite':bool(all(np.isfinite(v).all()for v in arrays.values()))},
    'explicit_limitations':['Float32 saved kinematics and common zero Gaussian draws; not original GPU noise realization.','No true attitude enters fusion state, only raw sensor synthesis; true gravity is used for this offline error score.','Original finish ticks retained. No later movement is counted as task success.','Continuous output can be inaccurate after saturation; readiness does not certify accuracy.','ST proprietary SFLP implementation and actual board saturation recovery are not calibrated.'],
    'finish_reason_codes':manifest['finish_reason_codes'],'files':{}}
for p in [D/'physical-training-trace.npz',D/'body50-delivered-packets.npz',D/'evaluation-conditions.npz',D/'manifest.json',HERE/'finite_range_core.py',HERE/'adapter.py',HERE/'replay_k8.py',O/'body50-continuous-range-k8-replay-v1.npz']:
    report['files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
out=O/'body50-continuous-range-k8-replay-v1.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out)
print(report['comparison_checks'])
print('valid at original end:',sum(w['reconstructed_old_valid_at_finish']for w in worlds),'->',sum(w['continuous_valid_at_original_finish']for w in worlds))
