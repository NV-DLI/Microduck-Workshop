#!/usr/bin/env python3
"""Bind a verified export to the exact saved training record, without running Kit."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('export_report', type=Path)
    parser.add_argument('--ledger', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.export_report.read_text())
    ledger_bytes = args.ledger.read_bytes()
    ledger = json.loads(ledger_bytes)
    model = Path(report['output'])
    checkpoint = Path(report['checkpoint'])
    if not report['passed'] or report['parity_cases'] != 32:
        raise ValueError('Require the strict32 native/export comparison')
    if sha(model) != report['sha256'] or sha(checkpoint) != report['checkpoint_sha256']:
        raise ValueError('The verified checkpoint or export changed')
    if (Path(ledger['latest_checkpoint']).resolve() != checkpoint.resolve()
            or ledger['latest_checkpoint_sha256'] != report['checkpoint_sha256']):
        raise ValueError('The ledger must still describe this exact saved checkpoint')
    snapshot = args.export_report.parent / 'training-ledger.snapshot.json'
    if snapshot.exists() or 'training_ledger_snapshot_sha256' in report:
        raise FileExistsError('Never overwrite frozen export evidence')
    with snapshot.open('xb') as stream:
        stream.write(ledger_bytes)
    report.update(
        training_ledger_snapshot=str(snapshot.resolve()),
        training_ledger_snapshot_sha256=sha(snapshot),
        training_ledger_at_capture=str(args.ledger.resolve()),
        completed_updates_at_export=ledger['iterations_this_run'],
        qualification_claimed=False,
    )
    temporary = args.export_report.with_suffix('.json.tmp')
    with temporary.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.export_report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
