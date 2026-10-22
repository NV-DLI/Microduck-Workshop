"""CPU unit tests of the v121 supervisor. Covers every topic of root's v120 prototype tests
(retained unmodified in ../v120-root-prototype-tests-retained; they assert the abandoned
constant-motion/distance-stop semantics) under the fixed 80-tick helper window."""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
M = Path('@COURSE@/workshop/migration-20261019')
for p in (HERE, M/'football'):
    sys.path.insert(0, str(p))
import retreat_recovery as rr  # noqa: E402

V117_MEMORY = {'memory_available': True, 'ball_capture_timestamp_s': 12.1, 'sensor_read_timestamp_s': 15.82,
               'point_level_m': [0.09028235245258838, -0.05894066203680877, -0.08633258107262796]}


def packet(t, seq, *, gyro=(0., 0., 0.), qd=0., gravity=(0., 0., -1.), valid=True):
    return {'orientation_valid': valid, 'seq': seq, 'timestamp_s': t, 'read_timestamp_s': t, 'encoder_read_timestamp_s': t,
            'gyro_body_rad_s': list(gyro), 'gravity_body': list(gravity), 'encoder_qd_rad_s': [qd]*14}


def path(t, target=(-.27, 0.), **kw):
    return {'allowed': True, 'capture_s': t, 'layout': 'empty', 'map_request': {}, 'target_map_xy': list(target), 'distance_m': .28, **kw}


class Driver:
    """50 Hz ticks with delivered packets; captures every 4th tick when requested."""
    def __init__(self, sup, t0=15.82):
        self.s = sup; self.t = t0; self.seq = 100; self.odom = np.zeros(3); self.out = None

    def tick(self, n=1, *, vx_per_s=None, yaw_rate=0., **pk):
        for _ in range(n):
            self.t = round(self.t+.02, 6); self.seq += 1
            pk.setdefault('gyro', (0., 0., yaw_rate))
            self.out = self.s.policy(now_s=self.t, packet=packet(self.t, self.seq, **pk),
                                     odom_xyyaw=self.odom.tolist(), actual_head=[0.]*4, odom_valid=True)
            if vx_per_s is not None and self.out['velocity'][0] < 0:
                self.odom[0] += vx_per_s*.02
            self.odom[2] += yaw_rate*.02
        return self.out

    def cap(self, **kw):
        kw.setdefault('guard_safe', True); kw.setdefault('rear_path', path(self.t))
        return self.s.capture(now_s=self.t, capture_s=self.t, odom_xyyaw=self.odom.tolist(), **kw)

    def retreat(self, ticks, *, vx_per_s=-.11, cap_every=4, **kw):
        for i in range(ticks):
            self.tick(vx_per_s=vx_per_s, **kw)
            if self.s.state != 'RETREAT':
                break
            if i % cap_every == cap_every-1:
                self.cap()
        return self.out


def start(task_s=13.485, sup=None):
    sup = sup or rr.RetreatRecovery()
    ok, why = sup.on_handoff_rejection(now_s=15.82, task_s=task_s, reason='propagated_point_outside_training_box', kick_started=False, memory=V117_MEMORY)
    return sup, ok, why


def authorized():
    sup, ok, _ = start(); d = Driver(sup); d.tick(26)
    assert sup.state == 'AUTHORIZE', sup.state
    assert d.cap() == 'RETREAT'
    return sup, d


class Entry(unittest.TestCase):
    def test_entry_fixed_plan_distance(self):
        sup, ok, why = start(); self.assertTrue(ok, why); self.assertEqual(sup.state, 'HOLD'); self.assertEqual(sup.distance, .28)

    def test_only_listed_prekick_reasons_after_kick_budget_time_memory(self):
        for reason in ('headreturn_ball_memory_expired', 'headreturn_deadline', 'kick_requires_body_guard_safe', 'fresh_goal_fix_required_after_headreturn'):
            self.assertEqual(rr.RetreatRecovery().on_handoff_rejection(now_s=15.82, task_s=10., reason=reason, kick_started=False, memory=V117_MEMORY)[1], 'reason_not_recoverable')
        s = rr.RetreatRecovery()
        self.assertEqual(s.on_handoff_rejection(now_s=15.82, task_s=10., reason='goal_heading_outside_development_gate', kick_started=True, memory=V117_MEMORY)[1], 'never_after_kick_onset')
        self.assertEqual(s.on_handoff_rejection(now_s=160., task_s=151., reason='goal_heading_outside_development_gate', kick_started=False, memory=V117_MEMORY)[1], 'insufficient_original_task_time')
        for k in ('ball_capture_timestamp_s', 'sensor_read_timestamp_s'):
            m = dict(V117_MEMORY); del m[k]
            self.assertFalse(rr.RetreatRecovery().on_handoff_rejection(now_s=15.82, task_s=10., reason='propagated_point_outside_training_box', kick_started=False, memory=m)[0])
        for p in ([.2, 0., 0.], [.09, .2, 0.], [-.09, -.05, 0.]):
            self.assertFalse(rr.RetreatRecovery().on_handoff_rejection(now_s=15.82, task_s=10., reason='propagated_point_outside_training_box', kick_started=False, memory=dict(V117_MEMORY, point_level_m=p))[0])

    def test_budget_of_three(self):
        s = rr.RetreatRecovery()
        for _ in range(3):
            self.assertTrue(start(sup=s)[1]); s.state = 'RELEASED'
        self.assertEqual(start(sup=s)[2], 'recovery_budget_exhausted')


class Motion(unittest.TestCase):
    def test_no_motion_until_quiet_and_hold_deadline(self):
        sup, ok, _ = start(); d = Driver(sup); out = d.tick(30, gyro=(.2, 0., 0.))
        self.assertEqual(sup.state, 'HOLD'); self.assertEqual(out['velocity'], [0., 0., 0.])
        d.tick(130, gyro=(.2, 0., 0.)); self.assertEqual(sup.reason, 'recovery_hold_deadline')

    def test_authorize_requires_same_capture_guard_path_plan_and_timeout(self):
        for kw, reason in ((dict(guard_safe=False), 'retreat_requires_body_guard_safe_capture'),
                           (dict(rear_path=None), 'retreat_rear_path_not_authorized:None'),
                           (dict(rear_path={'allowed': False, 'reason': 'unknown_in_envelope'}), 'retreat_rear_path_not_authorized:unknown_in_envelope')):
            sup, ok, _ = start(); d = Driver(sup); d.tick(26)
            if kw.get('rear_path', 0) is None:
                kw['rear_path'] = path(d.t-.0667)
            d.cap(**kw); self.assertEqual(sup.state, 'FAILED')
        sup, ok, _ = start(); d = Driver(sup); d.tick(26); d.cap(rear_path=path(d.t, distance_m=.17)); self.assertEqual(sup.state, 'FAILED')
        sup, ok, _ = start(); d = Driver(sup); d.tick(26); d.tick(30); self.assertEqual(sup.reason, 'retreat_authorization_capture_timeout')

    def test_exactly_80_helper_ticks_then_zero_and_settle(self):
        sup, d = authorized(); cmds = []
        for i in range(120):
            out = d.tick(vx_per_s=-.11)
            cmds.append(out['velocity'])
            if sup.state != 'RETREAT':
                break
            if i % 4 == 3:
                d.cap()
        nonzero = [c for c in cmds if any(c)]
        self.assertEqual(len(nonzero), 80); self.assertEqual(cmds[80], [0., 0., 0.]); self.assertEqual(sup.state, 'SETTLE')
        self.assertTrue(all(c[0] == -.35 and c[1] == 0. and abs(c[2]) <= .5 for c in nonzero))
        self.assertAlmostEqual(nonzero[0][2], 2.5*.10, places=12)          # h=h_lp=+.10, rate 0 at start
        self.assertEqual(sup.motion_index, 80)
        d.tick(26); self.assertEqual(sup.state, 'REACQUIRE')

    def test_capture_does_not_count_or_advance_clock(self):
        sup, d = authorized(); d.cap(); d.cap()
        self.assertEqual(sup.motion_index, 0); self.assertIsNone(sup.retreat_start)
        d.tick(vx_per_s=-.11); self.assertEqual(sup.motion_index, 1); self.assertEqual(sup.retreat_start, d.t)

    def test_terminal_early_stops(self):
        cases = [(dict(vx_per_s=0.), 'retreat_dead_zone_no_onset', 50),
                 (dict(vx_per_s=-.25), 'retreat_public_backward_exceeds_0p24', 79),
                 (dict(vx_per_s=-.05), 'retreat_public_backward_below_0p10', 80),
                 (dict(vx_per_s=-.11, yaw_rate=-.4), 'retreat_drift_limit', 79)]
        for kw, reason, ticks in cases:
            with self.subTest(reason=reason):
                sup, d = authorized(); n = 0
                for i in range(120):
                    out = d.tick(**kw); n += bool(any(out['velocity']))
                    if sup.state != 'RETREAT':
                        break
                    if i % 4 == 3:
                        d.cap()
                self.assertEqual(sup.reason, reason); self.assertEqual(out['velocity'], [0., 0., 0.]); self.assertLessEqual(n, ticks)
                self.assertNotIn('SETTLE', [h[0] for h in sup.history])

    def test_forward_motion_and_odometry_invalid(self):
        sup, d = authorized(); d.tick(vx_per_s=-.11); d.odom[0] = .02; d.tick(); self.assertEqual(sup.reason, 'retreat_forward_motion_toward_ball')
        sup, d = authorized(); d.tick(vx_per_s=-.11)
        out = sup.policy(now_s=d.t+.02, packet=packet(d.t+.02, d.seq+1), odom_xyyaw=[0, 0, 0], actual_head=[0]*4, odom_valid=False)
        self.assertEqual(sup.reason, 'recovery_public_odometry_invalid'); self.assertEqual(out['velocity'], [0., 0., 0.])

    def test_helper_fault_is_latched_terminal(self):
        sup, d = authorized(); d.tick(3, vx_per_s=-.11)
        t = round(d.t+.02, 6); p = packet(t, d.seq+1); p['read_timestamp_s'] = p['encoder_read_timestamp_s'] = t+.001   # future read
        out = sup.policy(now_s=t, packet=p, odom_xyyaw=d.odom.tolist(), actual_head=[0]*4, odom_valid=True)
        self.assertTrue(sup.reason.startswith('recovery_public_sensor_unavailable') or sup.reason.startswith('retreat_helper_fault'))
        self.assertEqual(out['velocity'], [0., 0., 0.])
        sup, d = authorized(); d.tick(3, vx_per_s=-.11)
        t = round(d.t+.02, 6); p = packet(t, d.seq-5)   # sequence going backwards: helper fault 4
        out = sup.policy(now_s=t, packet=p, odom_xyyaw=d.odom.tolist(), actual_head=[0]*4, odom_valid=True)
        self.assertEqual(sup.reason, 'retreat_helper_fault:4'); self.assertEqual(out['velocity'], [0., 0., 0.])

    def test_watchdogs_guard_path_head_packet_clock(self):
        sup, d = authorized(); out = d.retreat(11, cap_every=1000)
        self.assertEqual(sup.reason, 'retreat_capture_watchdog'); self.assertEqual(out['velocity'], [0., 0., 0.])
        sup, d = authorized(); d.tick(vx_per_s=-.11); d.cap(guard_safe=False); self.assertEqual(sup.reason, 'retreat_current_body_guard_rejected')
        sup, d = authorized(); d.tick(vx_per_s=-.11); d.cap(rear_path=path(d.t, target=(-.3, .1))); self.assertEqual(sup.reason, 'retreat_rear_path_recheck_failed')
        sup, d = authorized(); out = sup.policy(now_s=d.t+.02, packet=packet(d.t+.02, d.seq+1), odom_xyyaw=[0, 0, 0], actual_head=[.02, 0, 0, 0], odom_valid=True)
        self.assertEqual(sup.reason, 'retreat_actual_head_not_neutral')
        sup, d = authorized(); p = packet(d.t+.02, d.seq+1); del p['gyro_body_rad_s']
        sup.policy(now_s=d.t+.02, packet=p, odom_xyyaw=[0, 0, 0], actual_head=[0]*4, odom_valid=True)
        self.assertEqual(sup.reason, 'recovery_public_sensor_unavailable')
        sup, d = authorized(); sup.policy(now_s=d.t+.05, packet=packet(d.t+.05, d.seq+1), odom_xyyaw=[0, 0, 0], actual_head=[0]*4, odom_valid=True)
        self.assertEqual(sup.reason, 'recovery_policy_clock_gap')
        sup, d = authorized(); sup.task_origin = d.t-180.; d.tick(); self.assertEqual(sup.reason, 'original_180s_limit')
        sup, d = authorized(); d.tick(gravity=(0., .6, -.8)); self.assertEqual(sup.reason, 'recovery_tilt_limit')

    def test_reacquire_release_rules(self):
        sup, d = authorized(); d.retreat(81)
        self.assertEqual(sup.state, 'SETTLE'); d.tick(26); self.assertEqual(sup.state, 'REACQUIRE')
        self.assertFalse(d.tick()['head_neutral'])
        fb = lambda x, **k: {'complete': True, 'memory_anchored_now': True, 'capture_s': d.t, 'point_level_m': [x, -.04, -.086], **k}
        self.assertEqual(d.cap(fresh_ball=fb(.15)), 'REACQUIRE')
        self.assertEqual(d.cap(fresh_ball=fb(.26, memory_anchored_now=False)), 'REACQUIRE')
        self.assertEqual(d.cap(fresh_ball=fb(.26), guard_safe=False), 'REACQUIRE')
        self.assertEqual(d.cap(fresh_ball=fb(.26)), 'RELEASED')
        self.assertFalse(d.tick()['kick_authorized'])
        sup, d = authorized(); d.retreat(81); d.tick(26); sup.reacquire_start = d.t-8.; d.tick()
        self.assertEqual(sup.reason, 'recovery_reacquire_deadline')

    def test_settle_public_travel_bounded_by_plan(self):
        sup, d = authorized(); d.retreat(81); self.assertEqual(sup.state, 'SETTLE')
        d.odom[0] = -.29; d.tick(); self.assertEqual(sup.reason, 'settle_public_travel_outside_planned_path')


class RearPath(unittest.TestCase):
    def tof(self, t, xy):
        rays = []
        for row in range(8):
            for col in range(8):
                a, e = math.radians(-19.7+col*5.625), math.radians(19.7-row*5.625)
                v = np.array([math.cos(e)*math.cos(a), math.cos(e)*math.sin(a), math.sin(e)]); rays.append((v/np.linalg.norm(v)).tolist())
        return {'capture_s': t, 'fk_capture_s': t, 'provenance': 'simulation_approximation', 'status': [None]*64, 'ranges_m': [None]*64,
                'isaac_valid_mask': [False]*64, 'near_m': .1, 'far_m': 4., 'unit_rays_sensor': rays,
                'sensor_translation_odom': [xy[0], xy[1], .2], 'sensor_quaternion_odom_xyzw': [0., 0., 0., 1.]}

    def p(self, t, pose, layout='empty', goal=None):
        return rr.rear_path_authority(now_s=t, pose_map_xyyaw=pose, goal_fix_s=t if goal is None else goal, distance_m=rr.PLAN_DISTANCE_M,
                                      tof_sample=self.tof(t, pose[:2]), layout=layout, image_sha256='0'*64)

    def test_v117_failure_pose_plan_prior_not_sensed(self):
        r = self.p(15.8, [-0.018582903114977494, 0.053301156041499737, 0.022864512722746654])
        self.assertTrue(r['allowed'], r); self.assertFalse(r['rear_sensor_coverage']); self.assertEqual(r['distance_m'], .28)
        self.assertIn('not_traversed_history', r['plan']); self.assertIn('case', r['map_request'])

    def test_boundary_obstacle_stale_goal(self):
        self.assertFalse(self.p(1., [-.75, 0., 0.])['allowed'])
        self.assertEqual(self.p(1., [-.05, .03, 0.], layout='one-obstacle')['reason'], 'rear_path_inputs')
        self.assertEqual(self.p(20., [0., 0., 0.], goal=11.)['reason'], 'rear_path_inputs')


if __name__ == '__main__':
    unittest.main()
