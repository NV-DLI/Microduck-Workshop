#!/usr/bin/env python3
"""Run frozen candidate cohorts; a successful execution is not qualification."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

M = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--recipe-dir', type=Path, required=True)
    p.add_argument('--recipe-version', choices=['v11'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    collectors = {family: M/'gait'/f'evaluate_{name}_{a.recipe_version}.py'
                  for family, name in [('fixed', 'navigation_batch'), ('gaze', 'gaze')]}
    auditors = {'fixed': M/'acceptance/audit_navigation_v5.py',
                'gaze': M/'gait/audit_gaze_body50_v2.py'}
    sources = [Path(__file__), M/'gait/audit_gaze_trace_v2.py', *collectors.values(), *auditors.values(), a.model,
               a.model.with_suffix('.json')]
    snapshot = a.model.parent/'training-ledger.snapshot.json'
    meta = json.loads(a.model.with_suffix('.json').read_text())
    if not snapshot.is_file() or sha(snapshot) != meta['training_ledger_snapshot_sha256']:
        raise ValueError('An immutable training snapshot must accompany the export')
    sources.append(snapshot)
    ledger = {'schema': 'microduck.candidate-development-execution.v1', 'state': 'running',
              'qualification_claimed': False, 'condition': 'nominal', 'envs_per_case': 2,
              'model': str(a.model.resolve()), 'model_sha256': sha(a.model),
              'recipe_version': a.recipe_version,
              'source_sha256': {str(path.resolve()): sha(path) for path in sources},
              'runs': []}

    def save():
        tmp = a.output/'execution.tmp'
        tmp.write_text(json.dumps(ledger, indent=2)+'\n')
        tmp.replace(a.output/'execution.json')

    save()
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES',
               LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1')
    for family, seed in [('fixed', 978931), ('gaze', 978951)]:
        if any(sha(path) != h for path, h in ledger['source_sha256'].items()):
            raise ValueError('A frozen source or model changed between cohorts')
        output = a.output/family
        cmd = [sys.executable, str(collectors[family]), '--model', str(a.model),
               '--recipe-dir', str(a.recipe_dir), '--output', str(output), '--seed', str(seed)]
        if family == 'fixed':
            cmd += ['--adapter', 'trained-head-reference-v8', '--model-line', 'from-scratch',
                    '--family', 'fixed', '--development', '--num-envs', '2']
        else:
            cmd += ['--envs-per-case', '2']
        item = {'family': family, 'seed': seed, 'command': cmd,
                'started_at_utc': datetime.now(timezone.utc).isoformat()}
        ledger['runs'].append(item)
        save()
        print(json.dumps({'starting': str(output)}), flush=True)
        with (a.output/(family+'.log')).open('x') as stream:
            result = subprocess.run(cmd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        item['collector_exit_code'] = result.returncode
        report = output/'report.json'
        if report.is_file():
            data = json.loads(report.read_text())
            item['report_sha256'] = sha(report)
            if result.returncode == 0:
                audit = output/'independent-audit.json'
                if family == 'fixed':
                    audit_cmd = [sys.executable, str(auditors[family]), *data['part_reports'],
                                 '--output', str(audit), '--development']
                else:
                    audit_cmd = [sys.executable, str(auditors[family]), str(report), '--output', str(audit)]
                checked = subprocess.run(audit_cmd, capture_output=True, text=True)
                item.update(audit_exit_code=checked.returncode,
                            audit_output=checked.stdout+checked.stderr)
                if audit.is_file():
                    item['audit_sha256'] = sha(audit)
                    if family == 'fixed':
                        summary = json.loads(audit.read_text())
                        item.update(successes=summary['successes'], trials=summary['trials'])
        item['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        if result.returncode or item.get('audit_exit_code') != 0:
            ledger['state'] = 'execution-failed'
            save()
            print(json.dumps(item), flush=True)
            raise RuntimeError('Failed original cohort preserved; diagnose before further tests')
        save()
        print(json.dumps(item), flush=True)
    ledger['state'] = 'development-recorded'
    save()


if __name__ == '__main__':
    main()
