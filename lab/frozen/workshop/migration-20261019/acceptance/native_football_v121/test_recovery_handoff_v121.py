"""CPU tests: v121 owner lifecycle through the real HeadHandoffV121, real pinned public
memory, real RetreatRecovery + approved-hash heading-feedback helper, real
ObservationHeadFeedback (recovery_fixture). Synthetic packets/toy kinematics only;
no physics, contact, calibration or success evidence."""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import recovery_fixture as F  # noqa: E402
from recovery_handoff import HeadHandoffV121, recovery_reacquire_head_allowed, NEUTRAL_OWNED  # noqa: E402
from retreat_recovery import RecoveryParams, MIN_BACKWARD_M, MAX_BACKWARD_M  # noqa: E402
from fixture_leg import FixtureLeg  # noqa: E402

_CACHE = {}


def run(*faults, **kw):
    k = (faults, tuple(sorted(kw.items())))
    if k not in _CACHE:
        _CACHE[k] = F.Episode(faults, **kw).run()
    return _CACHE[k]


def rows(e, kind='policy'):
    return [r for r in e.files['recovery-trace'] if r['kind'] == kind]


def moving(e):
    return [r for r in rows(e) if any(r['actual_velocity_command'])]


class Lifecycle(unittest.TestCase):
    def test_whitelisted_rejection_runs_exact_80_tick_window_then_returns_to_approach(self):
        e = run(); h = e.handoff
        self.assertEqual(e.terminal, 'fixture_stop_after_release'); self.assertEqual(h.state, 'APPROACH')
        self.assertEqual([x[0] for x in h.recovery.history], ['HOLD', 'AUTHORIZE', 'RETREAT', 'STOP', 'REACQUIRE', 'RELEASED'])
        stop = h.recovery.history[3]
        self.assertEqual(stop[1:2]+stop[3:4], ('fixed_duration_complete', 80)); self.assertGreaterEqual(stop[4], MIN_BACKWARD_M)
        mv = moving(e)
        self.assertEqual(len(mv), 80); self.assertEqual([r['helper']['index'] for r in mv], list(range(80)))
        times = [r['timestamp_s'] for r in mv]
        self.assertTrue(all(abs(b-a-.02) < 1e-9 for a, b in zip(times, times[1:])))
        self.assertEqual(mv[0]['recovery']['retreat_start_s'], times[0])      # start synchronized to first motion tick
        nxt = next(r for r in rows(e) if r['timestamp_s'] > times[-1])
        self.assertEqual(nxt['actual_velocity_command'], [0., 0., 0.]); self.assertEqual(nxt['recovery_state'], 'SETTLE')
        self.assertFalse(any(any(r['actual_velocity_command']) for r in rows(e) if r['timestamp_s'] > times[-1]))
        self.assertEqual(h.attempts, 1); self.assertFalse(h.kick_ever_started); self.assertIsNone(h.kick_start)

    def test_motion_is_exact_helper_command_vx_vy_and_bounded_wz(self):
        e = run()
        for r in moving(e):
            v = r['actual_velocity_command']
            self.assertEqual(v[:2], [-.35, 0.]); self.assertLessEqual(abs(v[2]), .5)
            self.assertEqual(r['expected_helper_command'], v); self.assertEqual(r['helper']['diag']['fault_code'], [0])
            self.assertTrue(all(abs(x) <= 1e-9 for x in r['actual_head_command']))
        self.assertGreater(moving(e)[0]['actual_velocity_command'][2], 0.)    # +offset start turns toward +heading

    def test_hold_and_capture_ticks_never_count_as_motion(self):
        e = run()
        for r in rows(e):
            if r['recovery_state'] in ('HOLD', 'AUTHORIZE'):
                self.assertFalse(r['motion_tick']); self.assertIsNone(r['helper'])
        self.assertTrue(all(c['kick_authorized'] is False for c in rows(e, 'capture')))

    def test_fresh_helper_per_recovery_and_cumulative_budgets(self):
        e = run(cycles=3); h = e.handoff
        self.assertEqual(h.recovery.recoveries, 3); self.assertEqual(len(h.rejections), 4)
        self.assertEqual(h.rejections[-1]['admission'], 'recovery_budget_exhausted'); self.assertEqual(h.attempts, 4)
        idx = [r['helper']['index'] for r in moving(e)]
        self.assertEqual(idx, list(range(80))*3)
        firsts = [r for r in moving(e) if r['helper']['index'] == 0]
        self.assertEqual(len({r['actual_velocity_command'][2] for r in firsts}), 1)   # same fresh start each time
        turns = [c['approach_counters']['final_turns'] for c in rows(e, 'capture')]
        self.assertEqual(turns, sorted(turns))

    def test_memory_anchor_and_release(self):
        e = run()
        caps = rows(e, 'capture')
        for c in caps:
            if c['recovery_before'] in NEUTRAL_OWNED:
                self.assertFalse((c['anchor_attempt'] or {}).get('anchored'))
        rel = [c for c in caps if c['released']]
        self.assertEqual(len(rel), 1); c = rel[0]
        self.assertTrue(c['guard_safe'] and c['complete'] and c['memory_anchored_now'])
        self.assertTrue(.20 <= c['memory']['point_level_m'][0] <= .38)

    def test_fixed_plan_path_to_fixed_endpoint(self):
        e = run()
        paths = [c['rear_path'] for c in rows(e, 'capture') if c['recovery_before'] in ('AUTHORIZE', 'RETREAT')]
        self.assertTrue(paths)
        self.assertTrue(all(p['distance_m'] == .28 and p['allowed'] and p['layout'] == 'empty' for p in paths))
        self.assertEqual(len({tuple(p['target_map_xy']) for p in paths}), 1)
        self.assertFalse(any(p['rear_sensor_coverage'] for p in paths))


class EarlyStopsAreTerminal(unittest.TestCase):
    def assertTerminal(self, e, reason, *, max_ticks=80):
        h = e.handoff
        self.assertEqual(h.state, 'FAILED'); self.assertEqual(h.reason, 'recovery_failed:'+reason)
        self.assertNotIn('SETTLE', [x[0] for x in h.recovery.history]); self.assertNotIn('REACQUIRE', [x[0] for x in h.recovery.history])
        self.assertLessEqual(len(moving(e)), max_ticks)
        self.assertEqual(e.files['policy-trace'][-1]['actual_velocity_command'], [0., 0., 0.])
        self.assertEqual(h.rejections[0]['original_reason'], 'propagated_point_outside_training_box')

    def test_helper_fault_latches_zero(self):
        # index 20 re-observes the unchanged packet (LP-only, still commanded); the next new read is
        # .04 s later -> fault 6 latched at index 21 with zero command.
        e = run('helper_read_gap'); self.assertTerminal(e, 'retreat_helper_fault:6', max_ticks=21)
        self.assertEqual([r['helper']['index'] for r in moving(e)], list(range(21)))

    def test_yaw_drift(self):
        e = run('yaw_drift_big'); self.assertTerminal(e, 'retreat_drift_limit', max_ticks=79)

    def test_public_backward_hard_bound(self):
        e = run('overshoot'); self.assertTerminal(e, 'retreat_public_backward_exceeds_0p24', max_ticks=79)
        last = [r for r in rows(e) if r['retreat_relative'] and r['retreat_relative']['retreat_dx_m'] is not None][-1]
        self.assertGreater(-last['retreat_relative']['retreat_dx_m'], MAX_BACKWARD_M)

    def test_short_travel_at_normal_stop(self):
        e = run('short_travel'); self.assertTerminal(e, 'retreat_public_backward_below_0p10')
        self.assertEqual(len(moving(e)), 80)

    def test_onset_deadline_1s(self):
        e = run('dead_zone'); self.assertTerminal(e, 'retreat_dead_zone_no_onset', max_ticks=50)
        self.assertEqual(len(moving(e)), 50)

    def test_guard_capture_odometry(self):
        for fault, reason in (('stale_capture', 'retreat_capture_watchdog'), ('guard_unsafe_in_retreat', 'retreat_current_body_guard_rejected'),
                              ('odometry_reset', 'recovery_public_odometry_invalid')):
            with self.subTest(fault=fault):
                self.assertTerminal(run(fault), reason, max_ticks=79)

    def test_reacquire_and_layout_and_time(self):
        self.assertEqual(run('no_reacquire').handoff.reason, 'recovery_failed:recovery_reacquire_deadline')
        self.assertEqual(run(layout='one-obstacle').handoff.reason, 'recovery_failed:retreat_rear_path_not_authorized:rear_path_inputs')
        self.assertFalse(moving(run(layout='one-obstacle')))
        e = run(task_origin=-140.)
        self.assertEqual(e.handoff.reason, 'propagated_point_outside_training_box')
        self.assertEqual(e.handoff.rejections[0]['admission'], 'insufficient_original_task_time')

    def test_hold_waits_for_actual_sent_neutral_head(self):
        e = run('entry_head_nonneutral')
        hold = [r for r in rows(e) if r['recovery_state'] == 'HOLD']
        self.assertGreater(max(abs(r['actual_head_command'][1]) for r in hold), .05)
        self.assertEqual(e.handoff.recovery.state, 'RELEASED')


class Units(unittest.TestCase):
    def handoff(self):
        return HeadHandoffV121(FixtureLeg(), F.NAMES, params=RecoveryParams.provisional(), task_limit_s=180.)

    def retreating(self, wz=.12):
        h = self.handoff(); h.state = 'RECOVERY'; h.recovery.state = 'RETREAT'
        h.last_recovery_out = {'timestamp_s': 1., 'velocity': [-.35, 0., wz], 'motion_tick': True, 'decision': 'retreat_command', 'helper': {'index': 3}}
        return h

    def after(self, h, velocity, sent=(0., 0., 0., 0.), withdrawal=False):
        return h.recovery_after_head(now_s=1., task_s=1., requested_velocity=list(h.last_recovery_out['velocity']), velocity=velocity,
                                     requested_head=[0.]*4, sent_head=list(sent), head_feedback={'body_zero_for_withdrawal': withdrawal},
                                     head_permission=None, packet=None)

    def test_only_independently_computed_expected_helper_command(self):
        h = self.retreating(); v, row = self.after(h, [-.35, 0., .12])
        self.assertEqual(v, [-.35, 0., .12]); self.assertEqual(h.state, 'RECOVERY'); self.assertIsNone(row['withdrawn'])
        for bad in ([-.35, 0., .13], [-.35, 0., -.12], [-.35, .01, .12], [-.40, 0., .12], [-.35, 0., 0.]):
            h = self.retreating(); v, row = self.after(h, bad)
            self.assertEqual(v, [0., 0., 0.]); self.assertEqual(h.state, 'FAILED'); self.assertEqual(row['withdrawn'], 'unapproved_recovery_command')

    def test_stale_or_non_motion_expected_is_refused(self):
        h = self.retreating(); h.last_recovery_out['timestamp_s'] = .98
        v, row = self.after(h, [-.35, 0., .12]); self.assertEqual(v, [0., 0., 0.]); self.assertEqual(h.state, 'FAILED')
        h = self.retreating(); h.last_recovery_out['motion_tick'] = False
        v, row = self.after(h, [-.35, 0., .12]); self.assertEqual(h.state, 'FAILED')
        h = self.retreating(wz=.6); v, row = self.after(h, [-.35, 0., .6]); self.assertEqual(h.state, 'FAILED')

    def test_head_ownership_and_feedback_withdrawal_fail_closed(self):
        h = self.retreating(); v, row = self.after(h, [-.35, 0., .12], sent=(.01, 0., 0., 0.))
        self.assertEqual(row['withdrawn'], 'retreat_withdrawn_actual_sent_head_not_neutral'); self.assertEqual(v, [0., 0., 0.])
        h = self.retreating(); v, row = self.after(h, [-.35, 0., .12], withdrawal=True)
        self.assertEqual(row['withdrawn'], 'retreat_withdrawn_by_head_feedback')

    def test_non_whitelisted_and_post_kick(self):
        h = self.handoff(); h.head = np.zeros(4); h.reason = 'headreturn_deadline'; h.state = 'FAILED'; h._consider_recovery(1., 10.)
        self.assertEqual(h.rejections[-1]['admission'], 'reason_not_recoverable')
        h = self.handoff(); h.kick_ever_started = True; h.state = 'FAILED'; h.head = np.zeros(4)
        h.reason = 'goal_heading_outside_development_gate'; h._consider_recovery(1., 10.)
        self.assertEqual(h.rejections[-1]['admission'], 'never_after_kick_onset')

    def test_reacquire_head_permission(self):
        good = {'public_attention_owner': {'owner': 'BALL_REACQUIRE'}, 'velocity_body_xyw': [0., 0., 0.], 'kick_request': False,
                'public_attention_head_permission': {'accepted': True, 'owner': 'BALL_REACQUIRE', 'capture_s': 2.,
                                                     'body_authorized': False, 'kick_authorized': False}}
        self.assertTrue(recovery_reacquire_head_allowed(now_s=2.02, capture_s=2., control=good)['allowed'])
        c = copy.deepcopy(good); c['public_attention_owner']['owner'] = 'NONE'
        self.assertFalse(recovery_reacquire_head_allowed(now_s=2.02, capture_s=2., control=c)['allowed'])


if __name__ == '__main__':
    unittest.main()
