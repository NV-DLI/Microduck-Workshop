#!/usr/bin/env python3
"""Reproducible CPU-only correction of Beta body-vs-head sensor source scope."""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from beta_body_imu import BetaBodyImu, BetaBodyImuProfile
from beta_sensor_profile import CAMERA_PROFILE, HEAD_IMU_PROFILE, OFFICIAL_COMMIT, PROFILE_ID
from beta_tof_geometry import TOF_PROFILE

S = Path(__file__).resolve().parent
M = S.parent
C = M.parents[1]
O = C / 'configuration/core'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=S / 'reports/camera-profile.json')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous_path = S / 'reports/sensor-profile.json'
    previous = json.loads(previous_path.read_text())
    checks, sources, texts = [], [], {}

    def check(name, passed, **evidence):
        checks.append({'name': name, 'passed': bool(passed), **evidence})

    def lines(key, patterns):
        return {'file': key, 'matched_lines': sorted({i + 1 for i, line in enumerate(texts[key].splitlines())
                for pattern in patterns if pattern in line})}

    for item in previous['official_sources']:
        path = Path(item['path'])
        actual = sha(path)
        sources.append({**item, 'actual_sha256': actual,
                        'hash_matches': actual == item['expected_sha256']})
    for item in json.loads((S / 'sources/body-imu/manifest.json').read_text()):
        path = S / 'sources/body-imu' / item['path']
        actual = sha(path)
        sources.append({**item, 'path': str(path), 'expected_sha256': item['sha256'],
                        'actual_sha256': actual, 'hash_matches': actual == item['sha256']})
        texts[item['upstream_path']] = path.read_text()
    check('all_pinned_official_sources_match_recorded_sha256', all(x['hash_matches'] for x in sources),
          source_count=len(sources))
    check('single_pinned_official_runtime_revision', previous['official_runtime_commit'] == OFFICIAL_COMMIT
          and all(OFFICIAL_COMMIT in x['url'] for x in sources if '/pollen-robotics/microduck/' in x['url']),
          commit=OFFICIAL_COMMIT)
    norm = lambda value: json.loads(json.dumps(value))
    for name, value in [('camera', CAMERA_PROFILE.to_dict()), ('tof', TOF_PROFILE.to_dict()),
                        ('head_imu', HEAD_IMU_PROFILE.to_dict())]:
        check(name + '_configuration_still_matches_pinned_v2_source_audit',
              norm(value) == previous['implemented_profiles'][name],
              inherited_source_audit=str(previous_path), inherited_source_sha256=sha(previous_path))
    imu, bus, obs, runtime = [texts[x] for x in ('duck-control/src/imu.rs', 'duck-control/src/bus.rs',
                                               'duck-control/src/obs.rs', 'robotd/src/main.rs')]
    body = BetaBodyImuProfile().to_dict()
    check('body_lsm6dsv16x_is_combined_dynamixel_sync_read',
          all(x in imu for x in ('imu_to_dxl', 'LSM6DSV16X', 'same `sync_read` as the servos', 'address 124',
                                'IMU_BLOCK_LEN: usize = 12')),
          source=lines('duck-control/src/imu.rs', ['imu_to_dxl', 'same `sync_read`', 'address 124', 'IMU_BLOCK_LEN:']))
    check('beta_body_axes_identity_not_alpha_mount',
          'BETA_MOUNT: [f64; 4] = [1.0, 0.0, 0.0, 0.0]' in imu
          and body['sensor_to_trunk_quaternion_wxyz'] == [1., 0., 0., 0.],
          source=lines('duck-control/src/imu.rs', ['power board lies flat', 'BETA_MOUNT:']))
    check('body50hz_is_distinct_from_head60hz15hz_fifo',
          '25 reads is half a second at 50 Hz' in bus and body['control_read_hz'] == 50
          and HEAD_IMU_PROFILE.simulated_sample_hz == 60 and HEAD_IMU_PROFILE.fifo_read_hz == 15,
          source=lines('duck-control/src/bus.rs', ['25 reads is half a second']))
    check('body_decoder_quantization_and_quaternion_contract',
          all(x in imu for x in ('0.0175 *', 'i16::from_le_bytes', 'half(packed[0])', 'norm_sq <= 1.02'))
          and body['gyro_lsb_deg_s'] == .0175,
          source=lines('duck-control/src/imu.rs', ['GYRO_RAD_PER_LSB:', 'half(packed[0])', 'norm_sq <= 1.02']))
    check('median3_and25_live_read_ready_gate_are_retained',
          all(x in imu for x in ('self.quat_samples >= 25', 'packed != [0, 0, 0]',
                                'gyro: median3_each', 'gravity: median3_each')),
          source=lines('duck-control/src/imu.rs', ['quat_samples >= 25', 'packed !=', 'gyro: median3_each', 'gravity: median3_each']))
    check('native_policy_consumes_body_gyro_gravity_and_previous_raw_action',
          all(x in obs for x in ('fill(gyro, imu.gyro)', 'fill(gravity, imu.gravity)', '*previous_action = *last_action')),
          source=lines('duck-control/src/obs.rs', ['fill(gyro, imu.gyro)', 'fill(gravity, imu.gravity)', '*previous_action =']))
    read_loop = runtime[runtime.index('let mut sensors_read_ns = 0u64;'):]
    order = [read_loop.index(x) for x in ('ticker.tick().await', 'let fresh = match safety.read()',
                                        'controller.step(sensors,', 'match safety.apply(targets,')]
    check('read_policy_apply_are_in_the_same_control_loop', order == sorted(order)
          and 'Stamped at the read, not at publish' in runtime,
          source=lines('robotd/src/main.rs', ['Stamped at the read', 'ticker.tick().await',
                       'let fresh = match safety.read()', 'controller.step(sensors,', 'match safety.apply(targets,']))
    check('same_tick_default_and_added_delay_are_explicit_unmeasured_assumptions',
          body['pipeline_delay_s'] == .0 and body['pipeline_delay_is_measured'] is False
          and 'no fixed post-read20ms queue' in body['pipeline_delay_origin'])
    check('body_internal_rate_noise_position_and_firmware_are_unmeasured',
          all(body[x] is False for x in ('internal_sample_rate_is_measured', 'noise_is_measured',
                                        'sensor_position_is_measured', 'sflp_firmware_reproduced')))

    # The rate-dependent initialization must work while polled on the real200Hz
    # scheduler, not merely when a decoder receives already-valid quaternions.
    sensor = BetaBodyImu(seed=38112)
    packets = []
    for tick in range(401):
        packets.extend(sensor.poll(tick * .005, q_world_from_trunk_xyzw=(0., 0., 0., 1.),
                       angular_velocity_world_rad_s=(0., 0., 0.), linear_acceleration_world_m_s2=(0., 0., 0.)))
    valid = [p for p in packets if p['orientation_valid']]
    initialization = sensor.metadata()['fusion_initialization']
    check('real200hz_poll_initializes120hz_fusion_then25_read_gate', len(valid) > 0
          and valid[0]['delivered_timestamp_s'] > .6 and all(p['quat'] is None for p in packets if not p['orientation_valid']),
          simulated_seconds=2., delivered_packets=len(packets), valid_packets=len(valid),
          first_valid_timestamp_s=valid[0]['delivered_timestamp_s'] if valid else None,
          initialization=initialization)

    test = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(S / 'tests'),
                           '-p', 'test_*.py', '-v'], text=True, capture_output=True, cwd=S)
    log = args.output.with_suffix('.cpu-tests.log')
    log.write_text(test.stdout + test.stderr)
    count = re.search(r'Ran (\d+) tests', test.stderr)
    check('current_camera_tof_head_body_cpu_regressions', test.returncode == 0,
          test_count=int(count.group(1)) if count else None, log=str(log), log_sha256=sha(log))

    probe_path = O / 'runtime-smoke-backlash-1024.json'
    probe = json.loads(probe_path.read_text())
    check('actual_native_newton28dof14servo61obs_interface', probe['native_bam_verified'] and probe['finite']
          and len(probe['joint_names']) == 28 and probe['observations']['policy'] == [1024, 61],
          report=str(probe_path), report_sha256=sha(probe_path))
    passed = all(x['passed'] for x in checks)
    timing = {
        'official_order': ['50Hz ticker', 'combined fresh IMU+servo bus read', 'median3 IMU decoding',
                           'observation construction', 'inference', 'target application'],
        'fixed_post_read20ms_queue_established_by_source': False,
        'one_tick_old_comment_is_not_latency_measurement': True,
        'median3_phase': 'For monotonic scalar inputs, median3 equals the preceding50Hz sample; approximately20ms lag is already present.',
        'v9_historical_added_delay_s': .020,
        'recommended_explicit_added_delay_sensitivity_s': [0., .005, .010, .020],
        'zero_added_delay_means_measured_zero_hardware_latency': False,
        'remaining_unknowns': ['Actual body firmware SFLP internal rate and fusion phase',
                               'Per-unit bus read time, inference time and target-write time'],
    }
    report = {
        'schema': 'microduck.hardware-profile-source-audit.v3',
        'checked_at': datetime.now(timezone.utc).isoformat(), 'hardware_profile_id': PROFILE_ID,
        'state': 'passed' if passed else 'failed', 'source_scope_passed': passed,
        'recommended_gate_scope': 'hardware_source_and_simulation_interface_only',
        'physical_hardware_calibrated': False, 'body_sensor_closed_loop_success_claimed': False,
        'football_success_claimed': False, 'release_qualification_claimed': False,
        'gpu_jobs_started_by_this_audit': False, 'official_runtime_commit': OFFICIAL_COMMIT,
        'supersedes_interpretation': {'report': str(previous_path), 'sha256': sha(previous_path),
            'correction': 'The50Hz power-board IMU is the native locomotion source. Head60Hz/15HzFIFO is a separate stream; v2 head-to-body trace remains a historical diagnostic, not the official control path.'},
        'checks': checks, 'official_sources': sources,
        'implementation_sha256': {str(p.relative_to(M)): sha(p) for p in sorted(S.glob('*.py'))},
        'implemented_profiles': {'camera': CAMERA_PROFILE.to_dict(), 'tof': TOF_PROFILE.to_dict(),
                                'head_imu': HEAD_IMU_PROFILE.to_dict(), 'body_imu': body},
        'interface': {**previous['interface'], 'imu': 'Body trunkFLU gyro/gravity via noisy virtual sensor, fusion, official binary encoding, median3 and50Hz delivery; head FIFO stays separate'},
        'timing_interpretation': timing,
        'unmeasured_assumptions': previous['unmeasured_assumptions'] + body['assumptions'],
        'limits': previous['limits'] + ['Body simulation is a configurable sensor-driven SFLP approximation; no proprietary ST firmware or physical bus timing has been reproduced.'],
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'state': report['state'], 'checks': len(checks),
                      'failed': [x['name'] for x in checks if not x['passed']], 'report': str(args.output)}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
