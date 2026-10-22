"""CPU tests for the independent v121 recovery auditor: positive and strong negative fixtures.

Fixture runs come from recovery_fixture (real handoff/memory/supervisor/helper, toy kinematics).
Negative fixtures mutate archived traces as a faulty controller would have written them.
Scratch output goes to $V121_TEST_SCRATCH (or a system temp dir if unset)."""
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recovery_fixture as F  # noqa: E402
import audit_retreat_recovery_v2 as A  # noqa: E402

_CACHE = {}


def scratch():
    base = os.environ.get('V121_TEST_SCRATCH')
    if base:
        Path(base).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='v121-audit-', dir=base))


def episode_files(*faults, **kw):
    k = (faults, tuple(sorted(kw.items())))
    if k not in _CACHE:
        d = F.Episode(faults, **kw).run().write(scratch()/'run')
        _CACHE[k] = {p.name: p.read_text() for p in d.iterdir()}
    return _CACHE[k]


def audit_files(files, *, allow_test_only=True):
    d = scratch()/'run'; d.mkdir()
    for name, text in files.items():
        (d/name).write_text(text)
    out = d.parent/'audit.json'
    argv = [str(d), '--controller', str(HERE), '--output', str(out), '--fixture-leg', str(HERE/'fixture_leg.py')]
    if allow_test_only:
        argv.append('--allow-test-only-approval')
    code = A.main(argv)
    result = json.loads(out.read_text())
    return code, result, sorted(k for k, c in result['checks'].items() if not c['passed'])


def load(files, name):
    return [json.loads(x) for x in files[name].splitlines() if x.strip()]


def dump(files, name, rows):
    out = dict(files); out[name] = ''.join(json.dumps(r)+'\n' for r in rows); return out


def mirror(files, times, change):
    """Apply change(policy_row, recovery_row) at the given policy times (both traces)."""
    pol = load(files, 'policy-trace.jsonl'); rec = load(files, 'recovery-trace.jsonl'); times = set(times)
    rp = {r['timestamp_s']: r for r in rec if r['kind'] == 'policy'}
    for r in pol:
        if r['timestamp_s'] in times:
            change(r, rp.get(r['timestamp_s']))
    return dump(dump(files, 'policy-trace.jsonl', pol), 'recovery-trace.jsonl', rec)


def set_cmd(v):
    def change(p, r):
        p['actual_velocity_command'] = list(v)
        if r is not None:
            r['actual_velocity_command'] = list(v)
    return change


class Positive(unittest.TestCase):
    def test_nominal_passes_with_independent_wz(self):
        code, result, failed = audit_files(episode_files())
        self.assertEqual(failed, []); self.assertEqual(code, 0)
        seg = result['summary']['segments'][0]
        self.assertEqual(seg['motion_count'], 80); self.assertEqual(seg['nonzero_ticks'], 80)
        self.assertEqual(result['checks']['wz_independent_rederivation']['count'], 80)
        self.assertTrue(result['summary']['physical_stopping'][0]['evaluation_only_privileged_physics'])

    def test_repeated_recoveries_pass(self):
        code, result, failed = audit_files(episode_files(cycles=3))
        self.assertEqual(failed, []); self.assertEqual([s['motion_count'] for s in result['summary']['segments']], [80, 80, 80])

    def test_fail_closed_controller_runs_pass(self):
        for fault in ('helper_read_gap', 'yaw_drift_big', 'overshoot', 'short_travel', 'dead_zone', 'stale_capture',
                      'guard_unsafe_in_retreat', 'odometry_reset', 'no_reacquire', 'unsafe_reacquire', 'entry_head_nonneutral'):
            with self.subTest(fault=fault):
                self.assertEqual(audit_files(episode_files(fault))[2], [], fault)

    def test_production_mode_refuses_test_only_approval(self):
        code, result, failed = audit_files(episode_files(), allow_test_only=False)
        self.assertIn('approval_present_and_production', failed)


class Negative(unittest.TestCase):
    def setUp(self):
        self.base = episode_files()
        rec = load(self.base, 'recovery-trace.jsonl')
        self.mv = [r for r in rec if r['kind'] == 'policy' and any(r['actual_velocity_command'])]
        self.t = [r['timestamp_s'] for r in self.mv]
        self.after = [r['timestamp_s'] for r in rec if r['kind'] == 'policy' and r['timestamp_s'] > self.t[-1]]

    def expect(self, files, *checks):
        code, result, failed = audit_files(files)
        self.assertEqual(code, 1)
        for c in checks:
            self.assertIn(c, failed)
        return failed

    def test_command_bias_on_wz(self):
        files = mirror(self.base, self.t[10:12], lambda p, r: (p['actual_velocity_command'].__setitem__(2, p['actual_velocity_command'][2]+.01),
                                                                r['actual_velocity_command'].__setitem__(2, r['actual_velocity_command'][2]+.01)))
        self.expect(files, 'wz_independent_rederivation')

    def test_blanket_wz_or_wrong_vx_vy(self):
        self.expect(mirror(self.base, self.t[20:21], set_cmd([-.35, 0., .5])), 'wz_independent_rederivation')
        self.expect(mirror(self.base, self.t[20:21], set_cmd([-.35, 0., 0.])), 'wz_independent_rederivation')
        self.expect(mirror(self.base, self.t[20:21], set_cmd([-.40, 0., self.mv[20]['actual_velocity_command'][2]])), 'vx_exact_vy_zero_on_motion_ticks')
        self.expect(mirror(self.base, self.t[20:21], set_cmd([-.35, .02, self.mv[20]['actual_velocity_command'][2]])), 'vx_exact_vy_zero_on_motion_ticks')

    def test_wrong_duration_extra_ticks_after_stop(self):
        files = mirror(self.base, self.after[:5], set_cmd([-.35, 0., .1]))
        self.expect(files, 'no_post_stop_authority', 'no_command_outside_motion_window')

    def test_wrong_duration_short_window_then_settle(self):
        rec = load(self.base, 'recovery-trace.jsonl')
        cut = set(self.t[70:])
        for r in rec:
            if r['kind'] == 'policy' and r['timestamp_s'] in cut:
                r['actual_velocity_command'] = [0., 0., 0.]; r['recovery_state'] = 'SETTLE'
        pol = load(self.base, 'policy-trace.jsonl')
        for r in pol:
            if r['timestamp_s'] in cut:
                r['actual_velocity_command'] = [0., 0., 0.]
        self.expect(dump(dump(self.base, 'recovery-trace.jsonl', rec), 'policy-trace.jsonl', pol), 'wz_independent_rederivation')

    def test_forged_offset_target(self):
        m = json.loads(self.base['manifest.json']); m['retreat_recovery_approval']['parameters']['heading_offset_rad'] = .20
        files = dict(self.base); files['manifest.json'] = json.dumps(m)
        self.expect(files, 'wz_independent_rederivation')
        m['retreat_recovery_approval']['parameters']['heading_offset_rad'] = .15
        files['manifest.json'] = json.dumps(m)
        self.expect(files, 'approved_fixed_motion_parameters')
        m['retreat_recovery_approval']['parameters'].update(heading_offset_rad=.10, helper_sha256='0'*64)
        files['manifest.json'] = json.dumps(m)
        self.expect(files, 'approved_fixed_motion_parameters')

    def test_clock_and_sequence_faults_must_latch(self):
        pol = load(self.base, 'policy-trace.jsonl')
        for r in pol:
            if r['timestamp_s'] == self.t[30]:
                r['IMU_prediction']['seq'] -= 3
        self.expect(dump(self.base, 'policy-trace.jsonl', pol), 'latched_feedback_fault_zero_and_terminal')
        pol = load(self.base, 'policy-trace.jsonl')
        for r in pol:
            if r['timestamp_s'] == self.t[30]:
                r['IMU_prediction']['read_timestamp_s'] -= .1; r['IMU_prediction']['encoder_read_timestamp_s'] -= .1
        self.expect(dump(self.base, 'policy-trace.jsonl', pol), 'latched_feedback_fault_zero_and_terminal')
        pol = load(self.base, 'policy-trace.jsonl')
        for r in pol:
            if r['timestamp_s'] == self.t[30]:
                r['IMU_prediction']['gyro_body_rad_s'][2] += .3
        self.expect(dump(self.base, 'policy-trace.jsonl', pol), 'helper_inputs_are_current_raw_packet')

    def test_early_stop_pretending_complete(self):
        rec = load(self.base, 'recovery-trace.jsonl'); pol = load(self.base, 'policy-trace.jsonl')
        cut = set(self.t[60:])
        for rows_, key in ((rec, 'actual_velocity_command'), (pol, 'actual_velocity_command')):
            for r in rows_:
                if r.get('kind', 'policy') == 'policy' and r['timestamp_s'] in cut:
                    r[key] = [0., 0., 0.]
                    if 'recovery_state' in r:
                        r['recovery_state'] = 'SETTLE'
        self.expect(dump(dump(self.base, 'recovery-trace.jsonl', rec), 'policy-trace.jsonl', pol),
                    'wz_independent_rederivation', 'early_stop_is_terminal_not_motion_complete')

    def test_index_and_window_tampering(self):
        rec = load(self.base, 'recovery-trace.jsonl')
        for r in rec:
            if r['kind'] == 'policy' and r['timestamp_s'] == self.t[5]:
                r['helper']['index'] = 7
        self.expect(dump(self.base, 'recovery-trace.jsonl', rec), 'helper_index_contiguous')
        rec = load(self.base, 'recovery-trace.jsonl')
        for r in rec:
            if r['kind'] == 'policy' and r['timestamp_s'] == self.t[0]:
                r['recovery']['retreat_start_s'] = self.t[0]-.04
        self.expect(dump(self.base, 'recovery-trace.jsonl', rec), 'motion_window_starts_first_tick_after_authorization')

    def test_repeated_recovery_helper_not_fresh(self):
        files = episode_files(cycles=3)
        rec = load(files, 'recovery-trace.jsonl')
        second = [r for r in rec if r['kind'] == 'policy' and r.get('helper') and r['helper']['index'] == 0][1]['timestamp_s']
        # A controller that reused the first helper state would not start at h=h_lp=+offset.
        files = mirror(files, [second], lambda p, r: (p['actual_velocity_command'].__setitem__(2, 0.02), r['actual_velocity_command'].__setitem__(2, 0.02)))
        self.expect(files, 'wz_independent_rederivation')

    def test_missed_safety_stop(self):
        files = episode_files('overshoot')
        rec = load(files, 'recovery-trace.jsonl')
        fail_t = next(r['timestamp_s'] for r in rec if r['kind'] == 'policy' and r['handoff_state'] == 'FAILED')
        pol = load(files, 'policy-trace.jsonl')
        for r in pol:
            if r['timestamp_s'] == fail_t:
                r['actual_velocity_command'] = [-.35, 0., .1]; r['public_handoff']['state'] = 'RECOVERY'
        for r in rec:
            if r['kind'] == 'policy' and r['timestamp_s'] == fail_t:
                r.update(actual_velocity_command=[-.35, 0., .1], handoff_state='RECOVERY', recovery_state='RETREAT', decision='retreat_command')
        self.expect(dump(dump(files, 'policy-trace.jsonl', pol), 'recovery-trace.jsonl', rec), 'safety_stop_due_zero_and_terminal')

    def test_inherited_v120_negatives_still_detected(self):
        rec = load(self.base, 'recovery-trace.jsonl')
        caps = [r for r in rec if r['kind'] == 'capture' and r['recovery_before'] == 'RETREAT']
        drop = {c['timestamp_s'] for c in caps[5:9]}
        self.expect(dump(self.base, 'recovery-trace.jsonl', [r for r in rec if not (r['kind'] == 'capture' and r['timestamp_s'] in drop)]),
                    'fresh_guarded_capture_before_command')
        rec = load(self.base, 'recovery-trace.jsonl')
        for r in rec:
            if r['kind'] == 'capture' and r['timestamp_s'] == caps[3]['timestamp_s']:
                r['guard_safe'] = False
        self.expect(dump(self.base, 'recovery-trace.jsonl', rec), 'retreat_capture_unsafe_withdraws')
        rec = load(self.base, 'recovery-trace.jsonl')
        for r in rec:
            if r['kind'] == 'capture' and r['timestamp_s'] == caps[2]['timestamp_s']:
                r['rear_path']['target_map_xy'][0] -= .1
        self.expect(dump(self.base, 'recovery-trace.jsonl', rec), 'rear_path_fixed_target')
        reads = load(self.base, 'public-memory-readback.jsonl')
        for r in reads:
            q = (r.get('readback') or {}).get('actual_packet')
            if q is not None and r['timestamp_s'] >= self.t[20]:
                q['seq'] += 1
        self.expect(dump(self.base, 'public-memory-readback.jsonl', reads), 'odometry_continuity_since_entry')
        rec = load(self.base, 'recovery-trace.jsonl')
        for r in rec:
            if r['kind'] == 'capture' and r['released']:
                r['guard_safe'] = False
        self.expect(dump(self.base, 'recovery-trace.jsonl', rec), 'release_current_complete_guarded_in_domain')
        rec = load(self.base, 'recovery-trace.jsonl')
        for r in rec:
            for e in r.get('events', []):
                if e['event'] == 'recovery_entry':
                    e['recoveries_before'] = 3
        self.expect(dump(self.base, 'recovery-trace.jsonl', rec), 'entry_whitelist_budget_reserve_memory_clock')
        settle_t = next(r['timestamp_s'] for r in load(self.base, 'recovery-trace.jsonl') if r['kind'] == 'policy' and r['recovery_state'] == 'SETTLE')
        self.expect(mirror(self.base, [settle_t], lambda p, r: p.update(kick_requested=True)), 'no_kick_during_recovery')
        ctrl = load(self.base, 'control.jsonl'); ctrl[len(ctrl)//2]['final_approach']['turn_attempts'] = 0
        self.expect(dump(self.base, 'control.jsonl', ctrl), 'cumulative_counters_never_reset')
        self.expect(mirror(self.base, self.t[8:9], lambda p, r: (p.update(actual_head_command_rad=[0., .02, 0., 0.]),
                                                                  r.update(actual_head_command=[0., .02, 0., 0.]))), 'neutral_sent_head_while_moving')

    def test_physical_stage_a_window_violation(self):
        phys = load(self.base, 'physics-trace.jsonl')
        reacq = next(r['timestamp_s'] for r in load(self.base, 'recovery-trace.jsonl') if r['kind'] == 'policy' and r['recovery_state'] == 'REACQUIRE')
        for r in phys:
            if r['timestamp_s'] >= reacq-.3:
                r['root_position_world_m'][0] -= .06
        self.expect(dump(self.base, 'physics-trace.jsonl', phys), 'physical_stage_a_displacement_yaw_drift', 'public_odometry_vs_physical_stop')


if __name__ == '__main__':
    unittest.main()
