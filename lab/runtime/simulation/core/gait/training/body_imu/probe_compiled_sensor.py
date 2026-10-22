"""Bounded16-env CUDA comparison; sensor computation only, no Kit/physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch

M=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(M/'evaluation/body50_training'))
from physics_producer import Body50PhysicsProducer
from compiled_sensor import install

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if a.output.exists():raise FileExistsError('Preserve GPU compilation evidence')
n=16; eager=Body50PhysicsProducer(n,device='cuda',seed=8741)
parameters=Body50PhysicsProducer(n,device='cuda',seed=8741)
generator=torch.Generator(device='cuda').manual_seed(8742)
noise=torch.randn((450,n,6),device='cuda',dtype=torch.float64,generator=generator)
position=torch.tensor([[0.,0.,.125]]*n,device='cuda',dtype=torch.float64)
q=torch.tensor([[0.,0.,0.,1.]]*n,device='cuda',dtype=torch.float64)
w=torch.zeros(n,3,device='cuda',dtype=torch.float64)
eq=torch.zeros(n,14,device='cuda',dtype=torch.float64)

def args(i):
    return {'trunk_position_world_m':position,'trunk_quaternion_xyzw':q,
            'trunk_angular_velocity_world_rad_s':w,
            'effective_encoder_q_rad':eq,'effective_encoder_qd_rad_s':eq,
            'noise_standard_normal':noise[i]}

for i in range(200):
    eager.update(**args(i)); parameters.update(**args(i))
torch.cuda.synchronize()
adapter=install(parameters,backend='inductor')
compiled_start=time.monotonic()
expected=eager.update(**args(200));actual=parameters.update(**args(200))
torch.cuda.synchronize()
compile_seconds=time.monotonic()-compiled_start
errors={key:torch.zeros((),device='cuda',dtype=torch.float64) for key in expected}
exact={key:torch.ones((),device='cuda',dtype=torch.bool) for key in expected}
for i in range(201,301):
    if i==260:
        ids=torch.tensor([3],device='cuda');eager.reset(ids);parameters.reset(ids)
    w[:,0]=.1
    eq[:]=.0007*(i-200)
    expected=eager.update(**args(i));actual=parameters.update(**args(i))
    for key,val in expected.items():
        exact[key]&=torch.eq(val,actual[key]).all()
        errors[key]=torch.maximum(errors[key],(val.double()-actual[key].double()).abs().max())
torch.cuda.synchronize()
exact={k:bool(v) for k,v in exact.items()}; errors={k:float(v) for k,v in errors.items()}

timings={}
for name,producer in [('eager',eager),('compiled',parameters)]:
    torch.cuda.synchronize();wall=time.monotonic()
    begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    begin.record()
    for i in range(301,401):producer.update(**args(i))
    end.record();torch.cuda.synchronize()
    timings[name]={'ticks':100,'wall_seconds':time.monotonic()-wall,'cuda_elapsed_ms':begin.elapsed_time(end)}
wire=('gyro_counts','quaternion_xyz_half_bits','orientation_valid','delivered_mask','seq','sample_seq','timestamp_s','read_timestamp_s','encoder_read_timestamp_s','encoder_q_rad','encoder_qd_rad_s','gyro_clipped','accel_clipped')
checks={'wire_flags_timestamps_encoder_exact':all(exact[k] for k in wire),
        'all_decoded_float_errors_at_most_1e_10':max(errors.values())<=1e-10}
report={'schema':'microduck.body50-compiled-producer-cuda.v1','state':'passed'if all(checks.values())else'failed',
    'checks':checks,'envs':n,'warmup_eager_ticks':200,'compared_compiled_ticks':100,
    'partial_reset_tick':260,'first_compile_and_call_seconds':compile_seconds,
    'packet_fields_exact':exact,'packet_field_max_absolute_difference':errors,
    'timing_shared_gpu':True,'timing':timings,'adapter':adapter.report(),
    'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest()for f in [Path(__file__),Path(__file__).parent/'compiled_sensor.py',M/'evaluation/body50_training/physics_producer.py',M/'evaluation/body50_training/vector_body_imu.py']},
    'task_success_claimed':False,'timing_scope':'Sensor-only synthetic-input computation under shared GPU load, not end-to-end training throughput'}
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
if not all(checks.values()):raise SystemExit(1)
