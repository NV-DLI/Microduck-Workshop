#!/usr/bin/env python3
"""Frozen physical criteria plus uninterrupted, actually delivered BODY50 control.

Sensor outages remain in the original cohort. They fail deployment even if the
uncontrolled physical trajectory happens to satisfy the old motion criteria.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import audit_navigation_v3 as physical
from audit_navigation_v4 import sha
from navigation_metrics_v2 import wilson


def validate_body_arrays(data, *, nominal):
    obs, control = data['actor_observation'], data['control']
    count, worlds = control.shape[:2]
    startup = data['initial_startup_completed_mask']
    if startup.shape != (worlds,) or startup.dtype != np.bool_:
        raise ValueError('Every original world must have a physical startup outcome')
    ready = data['body50_orientation_valid'] & startup[None]
    inferred = data['actor_inferred_mask']
    if obs.shape != (count, worlds, 61) or not np.isfinite(obs).all():
        raise ValueError('Every finite actor row must remain in the record')
    if ready.shape != (count, worlds) or not np.array_equal(ready, inferred):
        raise ValueError('Inference mask must equal actual sensor readiness')
    if np.any(obs[~ready] != 0) or np.any(control[:, :, :14][~ready] != 0):
        raise ValueError('Unready rows must be unused zero placeholders')
    previous = np.concatenate((data['initial_previous_raw'][None], control[:-1, :, :14]))
    previous = np.where(ready[..., None], previous, 0.)
    if not np.allclose(obs[:, :, 34:48], previous, atol=1e-6, rtol=0):
        raise ValueError('Wrong preceding raw action history')
    imu_t, encoder_t = data['body50_read_timestamp_s'], data['body50_encoder_read_timestamp_s']
    if imu_t.shape != (count, worlds) or not np.array_equal(imu_t, encoder_t):
        raise ValueError('IMU and encoder timestamps must come from the same read')
    if not np.isfinite(imu_t).all() or np.any(np.diff(imu_t, axis=0) < 0):
        raise ValueError('Sensor timestamps must be finite and monotonic')
    if nominal:
        public = np.concatenate((data['body50_gyro_body_rad_s'], data['body50_gravity_body'],
            data['body50_encoder_q_rad']-data['native_home14'][None], data['body50_encoder_qd_rad_s']), -1)
        if not np.allclose(obs[:, :, :34][ready], public[ready], atol=2e-6, rtol=0):
            raise ValueError('Actor input differs from delivered public sensor values')
    wrench = data['external_wrench_peak']
    if wrench.shape != (count*4, worlds) or not np.isfinite(wrench).all() or np.any(wrench != 0):
        raise ValueError('Actual task external wrench must be zero at every physical tick')
    return {'actor_rows':count*worlds, 'ready_inference_rows':int(ready.sum()),
        'unready_placeholder_rows':int((~ready).sum()), 'zero_task_wrench':True,
        'same_read_encoders_verified':True, 'raw_history_verified':True,
        'nominal_public_sensor_values_verified':nominal,
        'unready_control_steps_per_world':(~ready).sum(0).tolist(),
        'startup_completed_mask':startup.tolist()}


def apply_sensor_outcomes(episodes, counts):
    if len(episodes) != len(counts):
        raise ValueError('All original worlds must have a sensor outcome')
    result = copy.deepcopy(episodes)
    for row, count in zip(result, counts, strict=True):
        row['physical_criteria_passed'] = row['passed']
        row['sensor_unready_control_steps'] = int(count)
        row['uninterrupted_sensor_control'] = count == 0
        if count:
            row['failure_reasons'].append('sensor_readiness_outage')
        row['passed'] = row['passed'] and count == 0
    return result


def audit_part(path, *, allow_development=False):
    r = physical.audit_part(path, allow_development=allow_development)
    contract = {'actor_dimensions':61, 'body_tau_s':2., 'same_read_encoders':True,
        'readiness_required':True, 'extra_observation_dr':r['condition']=='full_official_DR',
        'initialization_max_seconds':8., 'actor_world_state':False}
    if r.get('body50_observation_contract') != contract:
        raise ValueError('Missing exact BODY50 contract')
    for source, expected in r['body50_sources'].items():
        if sha(source) != expected:
            raise ValueError('Frozen BODY50 source changed')
    init = r['physical_initialization']
    if init['automatic_resets'] != 0 or not init['task_measurement_excludes_initialization']:
        raise ValueError('Initialization cannot select new worlds or count as task')
    if init['physical_ticks'] > 1600 or len(init['completed_mask']) != r['batch_size']:
        raise ValueError('Every original world must have a startup outcome within8s')
    if sum(init['completed_mask']) != init['task_environments'] or not init['failed_worlds_retained']:
        raise ValueError('Startup failures must remain in the original cohort')
    if r['body50_task_readback']['actual_task_external_wrench_peak'] != 0:
        raise ValueError('Actual task support detected')
    with np.load(Path(path).parent/'raw-trace.npz', allow_pickle=False) as data:
        evidence = validate_body_arrays(data, nominal=r['condition']=='nominal')
    expected_startup = [init['completed_mask'][i] for i in r['batch_environment_indices']]
    if evidence['startup_completed_mask'] != expected_startup:
        raise ValueError('Per-part startup trace differs from original cohort')
    r['body50_input_audit'] = evidence
    r['episodes'] = apply_sensor_outcomes(r['episodes'], evidence['unready_control_steps_per_world'])
    r['audit_success_count'] = sum(row['passed'] for row in r['episodes'])
    return r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reports', nargs='+', type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--development', action='store_true')
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Keep every earlier audit')
    # This also checks unique cohorts, model identity and complete frozen groups.
    result = physical.audit(a.reports, allow_development=a.development)
    parts = [audit_part(path, allow_development=a.development) for path in a.reports]
    result['physical_criteria_successes'] = result['successes']
    for group in result['groups']:
        matches = [r for r in parts if (r['family'],r['case'],r['seed']) == (group['family'],group['case'],group['seed'])]
        group['successes'] = sum(r['audit_success_count'] for r in matches)
        group['passed'] = len(matches)==2 and group['successes']>=61 and all(r['expected_episodes']==32 and r['audit_success_count']>=31 for r in matches)
    result['successes'] = sum(r['audit_success_count'] for r in parts)
    result['wilson95_lower'] = wilson(result['successes'], result['trials'])
    result['passed'] = result['passed'] and all(g['passed'] for g in result['groups'])
    result['parts'] = [{'path':r['report_path'],'sha256':r['report_sha256'],
        'successes':r['audit_success_count'],'trials':r['expected_episodes'],
        'body50_input_audit':r['body50_input_audit'],'episodes':r['episodes']} for r in parts]
    result['schema'] = 'microduck.navigation-v2-body50-audit.v2'
    result['auditor_sources'] = {str(Path(file).resolve()):sha(file) for file in (__file__, physical.__file__)}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('passed','trials','successes','development_only')}))


if __name__ == '__main__':
    main()
