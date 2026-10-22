"""Independent CPU check: gyro dynamics do not pass through a tau-second pose filter."""
from pathlib import Path
import hashlib
import json
import math
import sys
import torch

M = Path(__file__).resolve().parents[2]
O = M.parents[1] / 'outputs/migration-20261019'
sys.path.insert(0, str(M / 'acceptance/body50_training'))
from vector_body_imu import VectorBodyImu, VectorBodyProfile

def main():
    dtype = torch.float64
    profiles = [.5, 2., 10.]
    sensors = [VectorBodyImu(1, profile=VectorBodyProfile(orientation_correction_tau_s=tau,
        gyro_bias_rad_s=(0., 0., 0.), gyro_noise_std_rad_s=0., accel_noise_std_m_s2=0.)) for tau in profiles]
    start = 1.5
    speed = 1.
    rows = []
    # Tiny initial yaw makes the exact-identity all-zero quaternion sentinel
    # distinction observable without injecting initial attitude into the filter.
    # A nonzero gyro during warmup naturally creates live quaternion bytes.
    for tick in range(501):
        t = tick * .005
        moving = t > start
        theta = max(0., t - start) * speed
        q = torch.tensor([[0., math.sin(theta/2), 0., math.cos(theta/2)]], dtype=dtype)
        w = torch.tensor([[0., speed if moving else 0., .002 if not moving else 0.]], dtype=dtype)
        # During rotation, 2g specific force disables only the accel correction.
        # This is a filter unit test, not a simulated robot or task acceptance.
        acceleration = torch.tensor([[0., 0., 9.80665 if moving else 0.]], dtype=dtype)
        packets = [sensor.poll(t, q_world_from_trunk_xyzw=q,
            angular_velocity_world_rad_s=w, linear_acceleration_world_m_s2=acceleration,
            noise_standard_normal=torch.zeros((1,6), dtype=dtype)) for sensor in sensors]
        if moving and bool(packets[0]['delivered_mask'][0]):
            pitches = [float(2*torch.atan2(packet['quat_wxyz'][0,2], packet['quat_wxyz'][0,0])) for packet in packets]
            rows.append({'t': t, 'ideal_pitch': theta, 'fused_pitch': pitches,
                'valid': [bool(packet['orientation_valid'][0]) for packet in packets]})
    differences = [max(row['fused_pitch'])-min(row['fused_pitch']) for row in rows]
    errors = [abs(row['fused_pitch'][1]-row['ideal_pitch']) for row in rows]
    checks = {'all_dynamic_packets_valid': all(all(row['valid']) for row in rows),
        'tau_does_not_lowpass_gyro_motion': max(differences)<1e-12,
        'gyro_integrator_tracks_one_radian_motion': max(errors)<.02,
        'final_pitch_near_one_radian': abs(rows[-1]['fused_pitch'][1]-1.)<.02}
    sources = [Path(__file__), M/'acceptance/body50_training/vector_body_imu.py',
               M/'gait/experiments/body50_v9/compiled_sensor.py', M/'sensors/beta_head_imu.py',
               M/'sensors/sources/body-imu/duck-control__src__imu.rs']
    report = {'schema':'microduck.body50-fusion-tau-independent-review.v1', 'checks':checks,
        'all_passed':all(checks.values()), 'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'tau_values_s':profiles, 'max_fused_pitch_difference_between_tau_rad':max(differences),
        'maximum_fused_vs_input_dynamic_pitch_error_rad':max(errors), 'last_sample':rows[-1],
        'algorithm':'omega=quantized noisy local gyro+(1/tau)*cross(normalized measured specific force,predicted up); integrate quaternion increment each sensor sample. Accel correction gain is zero outside8.3..11.3m/s².',
        'truth_boundary':'Input world quaternion rotates private simulated angular velocity/specific force into sensor axes; it is not used as the fused attitude or a pose low-pass target.',
        'official_source_scope':'Official Rust decodes chipSFLP quaternion and applies median3. It does not publish tau2 or an equivalent proprietary fusion algorithm. tau2 is an explicitly assumed simulation sensitivity, not an official hardware constant.',
        'not_measured_hardware_fidelity':True, 'production_source_changed':False}
    out=O/'body50-fusion-tau-independent-review-v1.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'checks':checks,'max_error_rad':max(errors),'tau_difference_rad':max(differences),'output':str(out)}))

if __name__=='__main__':main()
