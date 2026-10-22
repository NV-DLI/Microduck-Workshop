#!/usr/bin/env python3
"""Run complete, disjoint-seed development cohorts without selecting survivors."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--recipe-dir', required=True, type=Path)
    parser.add_argument('--initial', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--candidate-label', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source = root/'gait/evaluate_navigation_batch_v9.py'
    auditor = root/'acceptance/audit_navigation_v5.py'
    sources = [Path(__file__), source, auditor, root/'acceptance/navigation_metrics_v2.py']
    ledger = {'schema':'microduck.paired-development-execution.v1',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':{str(p):digest(p) for p in sources},
        'candidate_label':args.candidate_label,
        'model_sha256':{str(p):digest(p) for p in (args.initial,args.candidate)},
        'conditions':'nominal; no formal or full-DR claim',
        'episodes_per_case':2,'qualification_claimed':False,'runs':[]}
    def save():
        temp=args.output/'execution.tmp'
        temp.write_text(json.dumps(ledger,indent=2)+'\n')
        temp.replace(args.output/'execution.json')
    save()
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1')
    for family,seed in [('fixed',978931),('stop',978941)]:
        for label,model in [('initial',args.initial),(args.candidate_label,args.candidate)]:
            if any(digest(p)!=h for p,h in ledger['source_sha256'].items()):
                raise RuntimeError('Do not change a collector between paired runs')
            output=args.output/(family+'-'+label)
            cmd=[sys.executable,str(source),'--model',str(model),'--recipe-dir',str(args.recipe_dir),
                '--output',str(output),'--adapter','trained-head-reference-v8','--model-line','from-scratch',
                '--family',family,'--development','--seed',str(seed),'--num-envs','2']
            item={'label':label,'family':family,'seed':seed,'command':cmd,
                'started_at_utc':datetime.now(timezone.utc).isoformat()}
            ledger['runs'].append(item);save()
            print(json.dumps({'starting':str(output)}),flush=True)
            with (args.output/(family+'-'+label+'.log')).open('x') as stream:
                result=subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT)
            item['collector_exit_code']=result.returncode
            report=output/'report.json'
            if report.is_file():
                item['report_sha256']=digest(report)
                data=json.loads(report.read_text())
                if data.get('status')=='recorded':
                    audit=output/'independent-audit.json'
                    audit_cmd=[sys.executable,str(auditor),*data['part_reports'],'--output',str(audit),'--development']
                    audited=subprocess.run(audit_cmd,capture_output=True,text=True)
                    item.update(audit_exit_code=audited.returncode,audit_output=audited.stdout+audited.stderr)
                    if audit.is_file():
                        checked=json.loads(audit.read_text())
                        item.update(audit_sha256=digest(audit),successes=checked['successes'],trials=checked['trials'])
            item['finished_at_utc']=datetime.now(timezone.utc).isoformat();save()
            print(json.dumps(item),flush=True)
            if result.returncode or item.get('audit_exit_code')!=0:
                raise RuntimeError('Keep this failed execution and diagnose before subsequent runs')
    ledger['state']='development-recorded';save()


if __name__=='__main__':
    main()
