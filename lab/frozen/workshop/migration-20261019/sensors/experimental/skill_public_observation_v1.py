"""Actor boundary consuming delivered public BODY50/encoder packets only.

This module has no simulator import, environment handle, pose, velocity, or
contact input. Invalid rows are sentinels: callers must exclude them from both
teacher/learner inference, recurrent-state updates, normalizers and PPO samples.
"""
import math
import torch

PUBLIC_PACKET_FIELDS = frozenset({
    'delivered_mask', 'orientation_valid', 'gyro_body_rad_s', 'gravity_body',
    'quat_wxyz', 'timestamp_s', 'read_timestamp_s', 'measurement_age_s', 'seq',
    'sample_seq', 'live_quaternion_reads', 'gyro_counts', 'quaternion_xyz_half_bits',
    'gyro_clipped', 'accel_clipped', 'encoder_q_rad', 'encoder_qd_rad_s',
    'encoder_read_timestamp_s',
})


def source61(packet, *, home14, previous_raw14, public_commands13):
    if set(packet)-PUBLIC_PACKET_FIELDS:
        raise ValueError('Unknown/private sensor fields at actor boundary')
    n = len(previous_raw14)
    if previous_raw14.shape != (n, 14) or public_commands13.shape != (n, 13):
        raise ValueError('Explicit14raw history and13public commands required')
    if home14.shape not in ((14,), (1, 14), (n, 14)):
        raise ValueError('Explicit native14 joint home required')
    for key, width in [('gyro_body_rad_s', 3), ('gravity_body', 3), ('encoder_q_rad', 14), ('encoder_qd_rad_s', 14)]:
        if packet[key].shape != (n, width):
            raise ValueError('Wrong delivered field shape: '+key)
    for key in ('orientation_valid', 'measurement_age_s', 'encoder_read_timestamp_s', 'read_timestamp_s'):
        if packet[key].shape != (n,):
            raise ValueError('Wrong delivered metadata shape: '+key)
    dtype = previous_raw14.dtype
    values = [packet['gyro_body_rad_s'].to(dtype), packet['gravity_body'].to(dtype),
              packet['encoder_q_rad'].to(dtype)-home14, packet['encoder_qd_rad_s'].to(dtype),
              previous_raw14, public_commands13]
    result = torch.cat(values, -1)
    age = packet['measurement_age_s']
    same_read = (packet['encoder_read_timestamp_s']-packet['read_timestamp_s']).abs() <= 1e-9
    ready = packet['orientation_valid'].bool() & (age >= -1e-9) & (age <= .08) & same_read
    ready &= torch.isfinite(result).all(-1) & torch.isfinite(age)
    # Deliberately no normalization of median3 gravity, ideal IMU substitute,
    # synthesized velocity, prediction from undelivered samples, or clip reset.
    return torch.where(ready[:, None], result, torch.zeros_like(result)), ready


def jump63(packet, *, home14, previous_raw14, public_commands13, public_trigger_elapsed_s):
    source, ready = source61(packet, home14=home14, previous_raw14=previous_raw14,
                             public_commands13=public_commands13)
    if public_trigger_elapsed_s.shape != ready.shape:
        raise ValueError('Each world needs its explicit public trigger clock')
    ready &= torch.isfinite(public_trigger_elapsed_s) & (public_trigger_elapsed_s >= 0)
    phase = public_trigger_elapsed_s.to(source.dtype)*(2*math.pi/1.2)
    out = torch.cat((source, phase.sin()[:, None], phase.cos()[:, None]), -1)
    return torch.where(ready[:, None], out, torch.zeros_like(out)), ready
