#!/usr/bin/env python3
"""Root-only helper: write the v121 recovery approval manifest, then freeze release.json.

Run ONLY after Stage A (seed 981131, 8 worlds, vx -.35, 1.6 s, offsets +.10/+.20, 4 reps)
has an eligible offset and Stage B (>=2 new near-ball fixtures with EXACTLY the
selected helper/duration/offset/vx) passed. Nothing here runs physics; it binds
hashes of evidence root already holds and copies root's per-world gate results.

  1) approve: python freeze_v121.py approve --stage-a-protocol P --stage-a-evidence E.. \
        --stage-a-results stage_a_results.json --stage-b-protocol P --stage-b-evidence E1 E2.. \
        --stage-b-tests stage_b_tests.json --output <INSTALL_DIR>/recovery-approval-v2.json
     stage_a_results.json = {"0.10": {"worlds": [4 x {criterion: bool}]}, "0.20": {...}}
       (criteria: recovery_config.STAGE_A_CRITERIA_KEYS; eligibility and the smaller
        eligible offset are DERIVED here, never typed in)
     stage_b_tests.json = [{"seed", "fixture", "pre_kick_ball_contact_observed", "fell",
        "max_foot_ball_force_n", "ball_displacement_m", "audit_passed"}, ...] (>=2)
  2) release: python freeze_v121.py release --approval <INSTALL_DIR>/recovery-approval-v2.json

--test-only exists for CPU tests; such manifests/releases are refused in production.
"""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D))
from v121_paths import INSTALL_DIR, BASE_DIR  # noqa: E402
from recovery_config import SCHEMA, STAGE_A_SPEC, STAGE_A_CRITERIA_KEYS, candidate_hashes, verify_approval, sha  # noqa: E402
from retreat_recovery import STAGE_A_VX, STAGE_A_DURATION_S, STAGE_A_OFFSETS_RAD, POLICY_DT_S  # noqa: E402

RELEASE_SCHEMA = 'microduck.v121-retreat-recovery-release.v1'


def evidence(paths):
    out = []
    for p in paths:
        p = Path(p).resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))
        out.append({'path': str(p), 'sha256': sha(p)})
    return out


def approve(*, candidate_dir, stage_a_protocol, stage_a_evidence, stage_a_results, stage_b_protocol, stage_b_evidence,
            stage_b_tests, output, test_only=False, note=''):
    candidate_dir = Path(candidate_dir).resolve(); output = Path(output)
    if not test_only and candidate_dir != INSTALL_DIR.resolve():
        raise ValueError('Production approval binds only ' + str(INSTALL_DIR))
    if output.exists():
        raise FileExistsError('Never overwrite an approval manifest: ' + str(output))
    helper = sha(candidate_dir/'heading_feedback.py')
    offsets = {}
    for offset in STAGE_A_OFFSETS_RAD:
        worlds = stage_a_results['%.2f' % offset]['worlds']
        offsets['%.2f' % offset] = {'worlds': worlds,
                                    'eligible': bool(len(worlds) == STAGE_A_SPEC['repetitions_per_offset']
                                                     and all(all(w.get(k) is True for k in STAGE_A_CRITERIA_KEYS) for w in worlds))}
    eligible = [float(k) for k, v in offsets.items() if v['eligible']]
    if not eligible:
        raise ValueError('Stage A has no eligible offset; v121 cannot be approved')
    selected = min(eligible)
    motion = {'helper_sha256': helper, 'duration_s': STAGE_A_DURATION_S, 'heading_offset_rad': selected, 'vx': STAGE_A_VX}
    tests = [{**t, **motion} for t in stage_b_tests]
    manifest = {
        'schema': SCHEMA, 'test_only': bool(test_only), 'approved_by': 'root' if not test_only else 'cpu_test_fixture',
        'approved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'physical_calibration_complete': not test_only, 'candidate_dir': str(candidate_dir),
        'candidate_source_sha256': candidate_hashes(candidate_dir),
        'parameters': {'retreat_vx': STAGE_A_VX, 'duration_s': STAGE_A_DURATION_S, 'move_steps': round(STAGE_A_DURATION_S/POLICY_DT_S),
                       'heading_offset_rad': selected, 'helper_sha256': helper},
        'stage_a': {'passed': True, 'spec': STAGE_A_SPEC, 'protocol': evidence([stage_a_protocol])[0], 'evidence': evidence(stage_a_evidence),
                    'offset_results': offsets, 'selected_heading_offset_rad': selected, **motion, 'test_only': bool(test_only)},
        'stage_b': {'passed': True, 'protocol': evidence([stage_b_protocol])[0], 'evidence': evidence(stage_b_evidence),
                    'tests': tests, **motion, 'test_only': bool(test_only)},
        'note': note, 'qualification_claimed': False, 'success_rate_claimed': False}
    output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    try:
        verify_approval(output, candidate_dir=candidate_dir, allow_test_only=test_only)
    except Exception:
        output.rename(output.with_name(output.name + '.rejected'))
        raise
    return output


def release(*, candidate_dir, approval, test_only=False):
    candidate_dir = Path(candidate_dir).resolve()
    if not test_only and candidate_dir != INSTALL_DIR.resolve():
        raise ValueError('Production release freezes only ' + str(INSTALL_DIR))
    verified = verify_approval(approval, candidate_dir=candidate_dir, allow_test_only=test_only)
    target = candidate_dir / 'release.json'
    if target.exists():
        raise FileExistsError('Never overwrite a frozen release')
    files = sorted(p for p in candidate_dir.iterdir() if p.is_file() and p.name != 'release.json')
    base = BASE_DIR / 'release.json'
    value = {'schema': RELEASE_SCHEMA, 'test_only': bool(test_only),
             'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'base_release': {'path': str(base), 'sha256': sha(base)},
             'recovery_approval': {'path': str(Path(approval).resolve()), 'sha256': verified['manifest_sha256']},
             'motion': verified['stage_a_binding']['motion'],
             'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
             'qualification_claimed': False}
    if Path(approval).resolve() not in files:
        value['source_sha256'][str(Path(approval).resolve())] = verified['manifest_sha256']
    target.write_text(json.dumps(value, indent=2) + '\n')
    return target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('approve')
    a.add_argument('--candidate-dir', type=Path, default=INSTALL_DIR)
    a.add_argument('--stage-a-protocol', type=Path, required=True)
    a.add_argument('--stage-a-evidence', type=Path, nargs='+', required=True)
    a.add_argument('--stage-a-results', type=Path, required=True)
    a.add_argument('--stage-b-protocol', type=Path, required=True)
    a.add_argument('--stage-b-evidence', type=Path, nargs='+', required=True)
    a.add_argument('--stage-b-tests', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--note', default='')
    a.add_argument('--test-only', action='store_true')
    r = sub.add_parser('release')
    r.add_argument('--candidate-dir', type=Path, default=INSTALL_DIR)
    r.add_argument('--approval', type=Path, required=True)
    r.add_argument('--test-only', action='store_true')
    args = p.parse_args(argv)
    if args.command == 'approve':
        out = approve(candidate_dir=args.candidate_dir, stage_a_protocol=args.stage_a_protocol, stage_a_evidence=args.stage_a_evidence,
                      stage_a_results=json.loads(args.stage_a_results.read_text()), stage_b_protocol=args.stage_b_protocol,
                      stage_b_evidence=args.stage_b_evidence, stage_b_tests=json.loads(args.stage_b_tests.read_text()),
                      output=args.output, test_only=args.test_only, note=args.note)
    else:
        out = release(candidate_dir=args.candidate_dir, approval=args.approval, test_only=args.test_only)
    print(json.dumps({'written': str(out), 'sha256': sha(out)}))


if __name__ == '__main__':
    main()
