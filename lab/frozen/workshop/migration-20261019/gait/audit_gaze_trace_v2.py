"""Recompute gaze development measurements from the recorded physical trace."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(path):
    r = json.loads(path.read_text())
    trace = path.parent / 'raw-trace.npz'
    if r['status'] != 'recorded' or sha(trace) != r['trace_sha256']:
        raise ValueError('Completed, unchanged physical recording required')
    if sha(r['model']) != r['model_sha256']:
        raise ValueError('Model changed')
    if not all(sha(p) == h for p, h in r['source_sha256'].items()):
        raise ValueError('Collector or trained adapter source changed')
    if any(r[k] for k in ('qualification_claimed', 'automatic_resets', 'external_support', 'state_projection')):
        raise ValueError('Expected unsupported development evidence')
    with np.load(trace, allow_pickle=False) as f:
        s, c, t = f['state'], f['control'], f['time_s']
        ids, pids = f['servo_ids'], f['passive_ids']
        limits, output = f['servo_limits'][:, ids], f['output_limits']
        if s.shape != (4000, len(r['episodes']), 41) or c.shape != (1000, len(r['episodes']), 35):
            raise ValueError('Missing episode or 200 Hz physical sample')
        if not np.allclose(t, np.arange(1, 4001) / 200, atol=1e-12, rtol=0):
            raise ValueError('Wrong physical timebase')
        if not all(np.isfinite(f[k]).all() for k in f.files):
            raise ValueError('Nonfinite physical evidence')
        if len(f['bam']) != len(s):
            raise ValueError('BAM physical telemetry is incomplete')
        q = s[:, :, 9:37]
        servo = q[:, :, ids]
        physical = servo + q[:, :, pids]
        sv = np.maximum(limits[None, :, :, 0] - servo, servo - limits[None, :, :, 1]).clip(0).max((0, 2))
        pv = np.maximum(output[None, None, :, 0] - physical, physical - output[None, None, :, 1]).clip(0).max((0, 2))
        # Accumulate physical angular energy in float64; do not loosen comparison tolerances.
        camera = np.linalg.norm(s[1600:, :, 37:40].astype(np.float64), axis=-1)
        rms = np.sqrt(np.mean(camera * camera, axis=0))
        p95 = np.percentile(camera, 95, axis=0)
        for i, e in enumerate(r['episodes']):
            for field, value in [('servo_peak_violation_rad', sv[i]), ('physical_output_peak_violation_rad', pv[i]),
                                 ('camera_angular_rms_rad_s_after8s', rms[i]), ('camera_angular_p95_rad_s_after8s', p95[i])]:
                if not np.isclose(e[field], value, atol=1e-7, rtol=1e-6):
                    raise ValueError('Metric does not match raw trace: ' + field)
            if bool(s[-1, i, 40]) != e['fell']:
                raise ValueError('Reported fall differs from raw history')
    groups = []
    for case in r['cases']:
        entries = [e for e in r['episodes'] if e['case'] == case['name']]
        groups.append({'case': case['name'], 'trials': len(entries), 'survivors': sum(not e['fell'] for e in entries),
                       'mechanical_passes': sum(max(e['servo_peak_violation_rad'], e['physical_output_peak_violation_rad']) <= .001 for e in entries),
                       'camera_rms_mean_rad_s': float(np.mean([e['camera_angular_rms_rad_s_after8s'] for e in entries])),
                       'head_rmse_mean_rad': float(np.mean([e['head_tracking_rmse_rad_after8s'] for e in entries]))})
    return {'schema': 'microduck.gaze-physical-trace-audit.v1', 'evidence_integrity_passed': True,
            'report': str(path.resolve()), 'report_sha256': sha(path), 'trace_sha256': sha(trace),
            'model_sha256': r['model_sha256'], 'groups': groups, 'development_only': True,
            'qualification_claimed': False, 'sensor_scope': r['sensor_scope'],
            'limitation': 'Physical camera motion and mechanical bounds independently recomputed. Head tracking is reported from the collector. No deployed Beta sensor or hardware qualification.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report', type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous audits')
    result = audit(a.report)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'evidence_integrity_passed': True, 'groups': len(result['groups']), 'qualification_claimed': False}))
