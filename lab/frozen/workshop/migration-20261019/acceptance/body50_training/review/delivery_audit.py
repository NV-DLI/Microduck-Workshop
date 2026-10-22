"""Independent CPU review of delivery isolation and scalar input guards."""
import hashlib
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from vector_body_imu import VectorBodyImu


def state(sensor, t, *, omega=0.):
    return sensor.poll(t, q_world_from_trunk_xyzw=[[0., 0., 0., 1.]],
                       angular_velocity_world_rad_s=[[omega, 0., 0.]],
                       linear_acceleration_world_m_s2=[[0., 0., 0.]],
                       noise_standard_normal=torch.zeros((1, 6)))


def run():
    sensor = VectorBodyImu(1)
    for tick in range(201):
        delivered = state(sensor, tick * .005)
    before = {k: v.clone() for k, v in delivered.items() if isinstance(v, torch.Tensor)}
    pending = state(sensor, 1.01, omega=14.)
    checks = {
        'warmup_ready': bool(before['orientation_valid'][0]),
        'sample_between_reads_not_delivered': not bool(pending['delivered_mask'][0]),
        'gyro_value_held_until_delivery': torch.equal(before['gyro_body_rad_s'], pending['gyro_body_rad_s']),
        'gravity_held_until_delivery': torch.equal(before['gravity_body'], pending['gravity_body']),
        'gyro_clipping_flag_held_until_delivery': torch.equal(before['gyro_clipped'], pending['gyro_clipped']),
        'accel_clipping_flag_held_until_delivery': torch.equal(before['accel_clipped'], pending['accel_clipped']),
        'timestamp_held_until_delivery': torch.equal(before['timestamp_s'], pending['timestamp_s']),
    }
    invalid_input_checks = {}
    for name, t in [('negative_time', -.1), ('backwards_time', .5), ('nonfinite_time', float('nan'))]:
        try:
            state(sensor, t)
        except ValueError:
            invalid_input_checks[name] = True
        else:
            invalid_input_checks[name] = False
    source = HERE.parent / 'vector_body_imu.py'
    return {
        'schema': 'microduck.body50.independent-cpu-review.v1',
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'delivery_checks': checks, 'scalar_input_guard_parity': invalid_input_checks,
        'diagnostic': {'before_gyro_clipped': before['gyro_clipped'].tolist(),
                       'undelivered_gyro_clipped': pending['gyro_clipped'].tolist()},
        'scope': 'CPU-only targeted review; no physics or performance qualification',
    }


if __name__ == '__main__':
    result = run()
    output = Path(sys.argv[1])
    if output.exists():
        raise FileExistsError('Preserve review evidence')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
