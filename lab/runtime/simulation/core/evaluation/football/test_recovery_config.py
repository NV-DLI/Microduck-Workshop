"""Test recovery config: Microduck simulation and control."""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recovery_config as RC  # noqa: E402
import verify_release as FZ  # noqa: E402

PASS = {k: True for k in RC.STAGE_A_CRITERIA_KEYS}
FAIL1 = {**PASS, 'yaw_stop_plus_1s_le_0p25': False}
TESTS = [{'seed': 1, 'fixture': 'nearball-a', 'pre_kick_ball_contact_observed': False, 'max_foot_ball_force_n': 0., 'ball_displacement_m': .001, 'fell': False, 'audit_passed': True},
         {'seed': 2, 'fixture': 'nearball-b', 'pre_kick_ball_contact_observed': False, 'max_foot_ball_force_n': 0., 'ball_displacement_m': .002, 'fell': False, 'audit_passed': True}]


def scratch():
    base = os.environ.get('V121_TEST_SCRATCH')
    if base:
        Path(base).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='v121-config-', dir=base))


def results(a=(PASS,)*4, b=(PASS,)*4):
    return {'0.10': {'worlds': list(a)}, '0.20': {'worlds': list(b)}}


class Gate(unittest.TestCase):
    def setUp(self):
        self.d = scratch(); self.ev = []
        for name in ('protocolA.json', 'a1.json', 'protocolB.json', 'b1.json', 'b2.json'):
            p = self.d/name; p.write_text(json.dumps({'SYNTHETIC_TEST_ONLY': name})); self.ev.append(p)
        self.n = 0

    def manifest(self, *, stage_a_results=None, stage_b_tests=None, **kw):
        self.n += 1
        args = dict(parameters_dir=HERE, stage_a_protocol=self.ev[0], stage_a_evidence=[self.ev[1]], stage_a_results=stage_a_results or results(),
                    stage_b_protocol=self.ev[2], stage_b_evidence=self.ev[3:], stage_b_tests=stage_b_tests or TESTS,
                    output=self.d/('approval%d.json' % self.n), test_only=True)
        args.update(kw)
        return FZ.approve(**args)

    def edit(self, path, change):
        m = json.loads(path.read_text()); change(m); out = path.with_name('edited-%d-' % self.n+path.name); self.n += 1
        out.write_text(json.dumps(m)); return out

    def verify(self, path, **kw):
        return RC.verify_approval(path, parameters_dir=HERE, allow_test_only=kw.get('test', True))

    def test_missing_and_test_only_refused_in_production(self):
        with self.assertRaises(RC.ApprovalError):
            RC.verify_approval(None, parameters_dir=HERE)
        path = self.manifest(); ok = self.verify(path)
        self.assertTrue(ok['test_only']); self.assertFalse(ok['parameters'].calibrated)
        with self.assertRaises(RC.ApprovalError) as e:
            self.verify(path, test=False)
        self.assertEqual(str(e.exception), 'test_only_manifest_rejected_in_production')

    def test_old_v120_constant_motion_schema_refused(self):
        path = self.edit(self.manifest(), lambda m: m.update(schema='microduck.football-retreat-recovery-approval.v1'))
        with self.assertRaises(RC.ApprovalError) as e:
            self.verify(path)
        self.assertIn('v120_constant_motion', str(e.exception))

    def test_forged_production_with_test_evidence_refused(self):
        path = self.edit(self.manifest(), lambda m: m.update(test_only=False, approved_by='root', physical_calibration_complete=True))
        with self.assertRaises(RC.ApprovalError) as e:
            self.verify(path, test=False)
        self.assertIn('unqualified_or_test_only', str(e.exception))

    def test_selection_is_derived_smaller_eligible_offset(self):
        self.assertEqual(self.verify(self.manifest())['parameters'].heading_offset_rad, .10)
        p = self.verify(self.manifest(stage_a_results=results(a=(PASS, PASS, FAIL1, PASS))))['parameters']
        self.assertEqual(p.heading_offset_rad, .20)
        with self.assertRaises(ValueError):
            self.manifest(stage_a_results=results(a=(PASS, FAIL1, PASS, PASS), b=(FAIL1, PASS, PASS, PASS)))

    def test_generic_passed_bool_or_forged_selection_refused(self):
        path = self.manifest()
        for change in (lambda m: m['stage_a'].update(selected_heading_offset_rad=.20),
                       lambda m: m['stage_a'].pop('offset_results'),
                       lambda m: m['stage_a']['offset_results']['0.10'].update(eligible=False),
                       lambda m: m['stage_a']['offset_results']['0.20']['worlds'].pop(),
                       lambda m: m['stage_a'].update(spec={**m['stage_a']['spec'], 'seed': 981128}),
                       lambda m: m['stage_a']['spec'].update(duration_s=1.8),
                       lambda m: m['stage_a'].update(passed=False), lambda m: m['stage_b'].update(passed=False)):
            with self.subTest(), self.assertRaises(RC.ApprovalError):
                self.verify(self.edit(path, change))

    def test_motion_binding_helper_duration_offset_vx(self):
        path = self.manifest()
        for change in (lambda m: m['stage_a'].update(helper_sha256='0'*64), lambda m: m['stage_a'].update(duration_s=1.8),
                       lambda m: m['stage_a'].update(heading_offset_rad=.20), lambda m: m['stage_a'].update(vx=-.40),
                       lambda m: m['stage_b'].update(heading_offset_rad=.20), lambda m: m['stage_b'].update(helper_sha256='1'*64),
                       lambda m: m['stage_b']['tests'][0].update(duration_s=1.8), lambda m: m['stage_b']['tests'][1].update(heading_offset_rad=.20),
                       lambda m: m['stage_b']['tests'][0].update(audit_passed=False),
                       lambda m: m['parameters'].update(heading_offset_rad=.20), lambda m: m['parameters'].update(move_steps=90),
                       lambda m: m['parameters'].update(retreat_vx=-.40), lambda m: m['parameters'].update(helper_sha256='2'*64)):
            with self.subTest(), self.assertRaises(RC.ApprovalError):
                self.verify(self.edit(path, change))

    def test_stage_b_requires_two_clean_distinct_new_fixtures(self):
        for tests in (TESTS[:1], [TESTS[0], {**TESTS[1], 'pre_kick_ball_contact_observed': True}],
                      [TESTS[0], {**TESTS[1], 'max_foot_ball_force_n': .01}], [TESTS[0], {**TESTS[1], 'ball_displacement_m': .005}],
                      [TESTS[0], {**TESTS[1], 'fell': True}], [TESTS[0], {**TESTS[1], 'seed': 1}], [TESTS[0], {**TESTS[1], 'fixture': 'nearball-a'}]):
            with self.subTest(tests=tests), self.assertRaises(RC.ApprovalError):
                self.manifest(stage_b_tests=tests)

    def test_source_helper_and_evidence_hash_binding(self):
        path = self.manifest()
        for name in ('heading_feedback.py', 'retreat_recovery.py', 'executor.py'):
            bad = self.edit(path, lambda m, n=name: m['parameters_source_sha256'].update({n: '0'*64}))
            with self.subTest(name=name), self.assertRaises(RC.ApprovalError) as e:
                self.verify(bad)
            self.assertIn('parameters_source_changed_since_approval', str(e.exception))
        self.ev[3].write_text('changed after approval')
        with self.assertRaises(RC.ApprovalError) as e:
            self.verify(path)
        self.assertIn('stage_b_evidence_hash_mismatch', str(e.exception))

    def test_other_parameters_dir_and_production_freeze_location(self):
        with self.assertRaises(RC.ApprovalError):
            self.verify(self.edit(self.manifest(), lambda m: m.update(parameters_dir=str(self.d))))
        with self.assertRaises(ValueError):
            self.manifest(test_only=False,parameters_dir=self.d)  # Explicit non-install path, also when running tests after installation.
        with self.assertRaises(ValueError):
            FZ.release(parameters_dir=HERE, approval=self.manifest(), test_only=False)

    def test_never_overwrite(self):
        path = self.manifest()
        with self.assertRaises(FileExistsError):
            self.manifest(output=path)


if __name__ == '__main__':
    unittest.main()
