#!/usr/bin/env python3
"""Recompute both61D ONNX actors and every supervisor weight from real traces."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'acceptance'))
import audit_navigation_v5 as physical
from batch_policy import BatchPolicy


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('batch_report', type=Path)
    p.add_argument('--physical-audit', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError('Keep every earlier result')
    batch = json.loads(a.batch_report.read_text())
    base = json.loads(a.physical_audit.read_text())
    assert base['development_only'] and batch['development_only']
    contract = batch['dual_policy']
    assert contract['transition_seconds'] == .4
    assert sha(contract['stand_model']) == contract['stand_sha256']
    torch.set_num_threads(2)
    models = [BatchPolicy(path, sha(path), 'cpu') for path in
              (contract['walk_model'], contract['stand_model'])]
    results = []
    for report_path in batch['part_reports']:
        r = physical.audit_part(Path(report_path), allow_development=True)
        if r['dual_policy'] != contract: raise ValueError('Mixed supervisor versions')
        if not any(x['path'] == str(Path(report_path).resolve()) and
                   x['sha256'] == sha(report_path) for x in base['parts']):
            raise ValueError('Physical audit does not cover every original part')
        with np.load(Path(report_path).parent/'raw-trace.npz', allow_pickle=False) as d:
            obs, ready = d['actor_observation'], d['actor_inferred_mask']
            weights, walk, stand = d['dual_weight'], d['dual_walk_raw'], d['dual_stand_raw']
            worlds = obs.shape[1]
            alpha = np.zeros(worlds, dtype=np.float32)
            for step in range(len(obs)):
                if not np.allclose(obs[step,ready[step],48:51], d['control'][step,ready[step],70:73], atol=1e-7, rtol=0):
                    raise ValueError('Policy and supervisor must receive the requested public command')
                ids = ready[step]
                target = (np.max(np.abs(obs[step,ids,48:51]), axis=-1) < 1e-6).astype(np.float32)
                alpha[ids] += np.clip(target-alpha[ids], -.05, .05)
                if not np.allclose(alpha, weights[step], atol=1e-7, rtol=0):
                    raise ValueError('Supervisor weight differs from public-command state machine')
            combined = walk + weights[...,None]*(stand-walk)
            if not np.allclose(combined, d['control'][...,:14], atol=2e-6, rtol=0):
                raise ValueError('Actual mixed request differs from recorded component actions')
            errors = []
            for model, values in zip(models, [walk, stand], strict=True):
                if np.any(values[~ready] != 0): raise ValueError('Unready sentinel was inferred')
                x, expected = obs[ready], values[ready]
                maximum = 0.
                with torch.inference_mode():
                    for i in range(0, len(x), 1024):
                        actual = model(torch.from_numpy(x[i:i+1024])).numpy()
                        maximum = max(maximum, float(np.abs(actual-expected[i:i+1024]).max()))
                if maximum > 1e-5: raise ValueError(f'Actual component actor mismatch: {maximum}')
                errors.append(maximum)
            results.append({'case':r['case'], 'original_worlds':worlds,
                            'control_steps':len(obs), 'ready_actor_rows':int(ready.sum()),
                            'actor_cpu_max_error':errors, 'trace_sha256':r['trace_sha256']})
    result = {'schema':'microduck.dual-policy-actual-audit.v12', 'evidence_integrity_passed':True,
              'qualification_claimed':False, 'development_only':True,
              'physical_audit':str(a.physical_audit), 'physical_audit_sha256':sha(a.physical_audit),
              'strict_successes':base['successes'], 'trials':base['trials'], 'parts':results,
              'both_actors_receive_same_raw_history':True, 'public_command_only_selection':True,
              'source_sha256':{str(x):sha(x) for x in [Path(__file__),HERE/'dual_policy_v12.py',HERE/'batch_policy.py']}}
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['evidence_integrity_passed','strict_successes','trials']}))


if __name__ == '__main__': main()
