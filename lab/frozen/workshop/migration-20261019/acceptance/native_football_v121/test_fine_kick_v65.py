"""CPU tests: ownership, invalid/stale/TooClose inputs, STEP-end routing into the
UNCHANGED HeadHandoff kick gates. Real frozen v64 classes; only the memory
adapter is replaced by a deterministic public-memory double."""
import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fine_kick_v65 as F  # noqa: E402
from neutral_fine_step import NeutralFineStep  # noqa: E402


class FakeMemory:
    def __init__(self, point, anchor, ttl=4.5): self.point, self.anchor, self.ttl = point, anchor, ttl
    def status(self, now_s):
        age = now_s-self.anchor; ok = 0 <= age <= self.ttl
        return {'memory_available': ok, 'point_level_m': list(self.point) if ok else None, 'ball_capture_timestamp_s': self.anchor, 'ball_age_s': age}


class FakeAdapter:
    def __init__(self, mem): self.memory = mem; self.last_read_s = None
    def consume(self, packet, *, now_s): return {'integrated': packet is not None}


def handoff(point=(.095, -.035, -.08), anchor=0.):
    h = F.HeadHandoffV65.__new__(F.HeadHandoffV65)
    h.adapter = FakeAdapter(FakeMemory(point, anchor))
    h.state = 'APPROACH'; h.started = None; h.last_t = None; h.head = None; h.quiet_start = None; h.kick_start = None
    h.reason = None; h.attempts = 0; h.latest_readback = None; h.last_goal_heading = None; h.last_goal_s = None
    h.initial_capture_s = None; h.latest_capture = None; h.routed_from_fine = False
    return h


def packet(t, quiet=True):
    return {'orientation_valid': True, 'read_timestamp_s': t, 'timestamp_s': t, 'gravity_body': [0., 0., -1.],
            'gyro_body_rad_s': [0., 0., 0.] if quiet else [.5, 0., 0.], 'encoder_qd_rad_s': [0.]*14}


def fine_at_step_end(t=3.):
    f = F.FineStepV65(); f.state = 'SETTLE'; f.step_end = t; f.started = 0.; f.anchor_s = 0.; f.attempts = 1; return f


def run_handoff(h, t0, t1, *, goal_heading=.02, capture_guard=True, quiet=True, gap_at=None):
    t = t0
    while t <= t1+1e-9:
        if abs((t*15) - round(t*15)) < 1e-6 or True:      # capture each tick (simplified 50 Hz public capture stream)
            h.note_capture(capture_s=t, guard_safe=capture_guard, fresh_floor=True)
            h.capture(now_s=t, geometry=None, head_offsets=[0, 0, 0, 0], candidate=False, goal_heading=goal_heading, goal_fix_s=t)
        h.policy(now_s=t, packet=packet(t, quiet))
        if h.state in ('KICK', 'FAILED'):
            return t
        t = round(t+(.1 if gap_at is not None and abs(t-gap_at) < 1e-9 else .02), 6)
    return t


class Entry(unittest.TestCase):
    def base(self, **over):
        a = dict(t=10., handoff_state='APPROACH', fine_active=False, fine_state='IDLE',
                 final_diag={'phase': 'OBSERVE', 'measured_quiet': False}, velocity_body_xyw=[0., 0., 0.], complete=True, guard_safe=True,
                 memory={'memory_available': True, 'ball_capture_timestamp_s': 10., 'point_level_m': [.138, -.035, -.08]},
                 goal_heading=0., goal_fix_s=9., actual_head=[-.5, .5, 0., 0.])
        a.update(over); return a

    def test_accepts_and_original_capture_enters_return(self):
        a = self.base(); ok, why = F.standoff_entry(**a); self.assertTrue(ok, why)
        st = NeutralFineStep().capture(t=a['t'], guard_safe=True, complete=True, trigger=True, actual_head=a['actual_head'], memory=a['memory'],
                                       goal_heading=a['goal_heading'], goal_fix_s=a['goal_fix_s'])
        self.assertEqual(st['state'], 'RETURN'); self.assertEqual(st['velocity'], [0., 0., 0.])

    def test_rejections(self):
        m = self.base()['memory']
        cases = [(dict(final_diag={'phase': 'TURN'}), 'final_phase_observe'), (dict(velocity_body_xyw=[.2, 0, 0]), 'exact_zero_body'),
                 (dict(handoff_state='HEAD_RETURN'), 'handoff_approach'), (dict(fine_active=True), 'fine_idle'),
                 (dict(complete=False), 'complete_fresh_ball'), (dict(guard_safe=False), 'body_guard_safe'),
                 (dict(memory=dict(m, ball_capture_timestamp_s=9.935)), 'memory_anchored_now'), (dict(memory=None), 'memory_anchored_now'),
                 (dict(memory=dict(m, point_level_m=[math.nan, 0, 0])), 'nonfinite_input'), (dict(goal_heading=math.nan), 'nonfinite_input'),
                 (dict(memory=dict(m, point_level_m=[.129, -.035, -.08])), 'original_domain'), (dict(goal_fix_s=1.9), 'goal_fix_age_0_to_8s'),
                 (dict(memory=dict(m,point_level_m=[.20,.01,-.08])), 'waypoint_bearing_le_0p15'), (dict(goal_heading=.066), 'goal_heading_le_public_margin'),
                 (dict(actual_head=[0, 0, 0, .32]), 'native_head_in_bounds'),
                 (dict(memory=dict(m, point_level_m=[.16, -.035, -.08]),actual_head=[-.8,.8,0,0]), 'not_in_model_dead_zone')]
        for over, name in cases:
            ok, why = F.standoff_entry(**self.base(**over)); self.assertFalse(ok); self.assertEqual(why, 'rejected:'+name, over)

    def test_never_triggers_a_failing_original_entry(self):
        rng = np.random.default_rng(65); hits = 0
        for _ in range(2000):
            a = self.base(); p = list(a['memory']['point_level_m']); p[0] += rng.uniform(-.02, .09); p[1] += rng.uniform(-.08, .1)
            a['memory'] = dict(a['memory'], point_level_m=p); a['goal_heading'] = rng.uniform(-.04, .04); a['goal_fix_s'] = a['t']-rng.uniform(-.5, 9.)
            ok, _ = F.standoff_entry(**a)
            if ok:
                hits += 1
                st = NeutralFineStep().capture(t=a['t'], guard_safe=True, complete=True, trigger=True, actual_head=a['actual_head'],
                                               memory=a['memory'], goal_heading=a['goal_heading'], goal_fix_s=a['goal_fix_s'])
                self.assertEqual(st['state'], 'RETURN')
        self.assertGreater(hits, 30)


class Routing(unittest.TestCase):
    def test_route_transfers_single_ownership_with_zero_body(self):
        f, h = fine_at_step_end(), handoff(); h.note_capture(capture_s=2.95, guard_safe=True, fresh_floor=True)
        d = F.route_at_step_end(now_s=3., fine=f, handoff=h, memory=h.adapter.memory.status(3.), goal_heading=.02, goal_fix_s=2.9)
        self.assertEqual(d, 'route_to_kick_handoff')
        self.assertFalse(f.active); self.assertEqual(f.reason, 'released_to_kick_handoff')
        st = h.status(3.); self.assertEqual(st['state'], 'HEAD_RETURN'); self.assertTrue(st['override_body_zero']); self.assertFalse(st['kick_runtime_authorized'])
        self.assertEqual(st['head_offsets'], [0., 0., 0., 0.])
        with self.assertRaises(ValueError): h.enter_from_fine(now_s=3.02, goal_heading=0., goal_fix_s=3.)

    def test_route_keeps_existing_path_when_not_clear(self):
        cases = [(dict(capture_s=2.7), 'keep:stale_capture'), (dict(guard_safe=False), 'keep:body_guard_unsafe'),
                 (dict(fresh_floor=False), 'keep:body_guard_unsafe'), (dict(point=(.15, -.03, -.08)), 'keep:observable_reacquire'),
                 (dict(point=(math.nan, 0., 0.)), 'keep:memory_invalid'), (dict(anchor=-1.6), 'keep:memory_invalid'),
                 (dict(goal_heading=math.nan), 'keep:goal_heading_nonfinite')]
        for over, exp in cases:
            f = fine_at_step_end(); h = handoff(point=over.get('point', (.095, -.035, -.08)), anchor=over.get('anchor', 0.))
            h.note_capture(capture_s=over.get('capture_s', 2.95), guard_safe=over.get('guard_safe', True), fresh_floor=over.get('fresh_floor', True))
            d = F.route_at_step_end(now_s=3., fine=f, handoff=h, memory=h.adapter.memory.status(3.), goal_heading=over.get('goal_heading', .02), goal_fix_s=2.9)
            self.assertEqual(d, exp, over); self.assertEqual(f.state, 'SETTLE'); self.assertEqual(h.state, 'APPROACH')
        f = fine_at_step_end(); f.state = 'STEP'; h = handoff(); h.note_capture(capture_s=2.95, guard_safe=True, fresh_floor=True)
        self.assertEqual(F.route_at_step_end(now_s=3., fine=f, handoff=h, memory=h.adapter.memory.status(3.), goal_heading=0., goal_fix_s=2.9),
                         'keep:not_at_step_end_or_handoff_owned')


class ExistingKickGates(unittest.TestCase):
    def routed(self, **kw):
        f, h = fine_at_step_end(), handoff(**kw); h.note_capture(capture_s=2.98, guard_safe=True, fresh_floor=True)
        self.assertEqual(F.route_at_step_end(now_s=3., fine=f, handoff=h, memory=h.adapter.memory.status(3.), goal_heading=.02, goal_fix_s=2.98), 'route_to_kick_handoff')
        return h

    def test_kick_only_after_all_existing_gates(self):
        h = self.routed(); t = run_handoff(h, 3.02, 4.4)
        self.assertEqual(h.state, 'KICK'); self.assertGreaterEqual(t-3.02, .5-1e-6); self.assertTrue(h.status(t)['kick_runtime_authorized'])

    def test_each_existing_gate_fails_closed(self):
        cases = [(dict(point=(.11, -.035, -.08)), {}, 'propagated_point_outside_training_box'),
                 (dict(point=(.095, -.01, -.08)), {}, 'propagated_point_outside_training_box'),
                 ({}, dict(goal_heading=.081), 'goal_heading_outside_development_gate'),
                 (dict(anchor=-1.2), {}, 'headreturn_ball_memory_expired'),
                 ({}, dict(gap_at=3.2), 'headreturn_policy_clock_gap')]
        for hk, rk, reason in cases:
            h = self.routed(**hk); run_handoff(h, 3.02, 4.4, **rk)
            self.assertEqual((h.state, h.reason), ('FAILED', reason), (hk, rk))

    def test_stale_goal_never_kicks(self):
        h = self.routed(); h.last_goal_s = None
        t = 3.02
        while t < 4.4 and h.state == 'HEAD_RETURN':
            h.note_capture(capture_s=t, guard_safe=True, fresh_floor=True); h.policy(now_s=t, packet=packet(t)); t = round(t+.02, 6)
        self.assertEqual((h.state, h.reason), ('FAILED', 'fresh_goal_fix_required_after_headreturn'))

    def test_not_quiet_never_kicks_and_hits_deadline(self):
        h = self.routed(anchor=-0.); run_handoff(h, 3.02, 7.6, quiet=False)
        self.assertEqual(h.state, 'FAILED'); self.assertIn(h.reason, ('headreturn_deadline', 'headreturn_ball_memory_expired'))

    def test_tooclose_or_stale_capture_at_kick_fails_closed(self):
        h = self.routed(); run_handoff(h, 3.02, 4.4, capture_guard=False)
        self.assertEqual((h.state, h.reason), ('FAILED', 'kick_requires_body_guard_safe'))
        h = self.routed(); t = 3.02
        while t < 4.4 and h.state == 'HEAD_RETURN':
            h.capture(now_s=t, geometry=None, head_offsets=[0, 0, 0, 0], candidate=False, goal_heading=.02, goal_fix_s=t)
            h.policy(now_s=t, packet=packet(t)); t = round(t+.02, 6)                       # no new public capture after routing
        self.assertEqual((h.state, h.reason), ('FAILED', 'kick_requires_fresh_public_capture'))

    def test_original_approach_path_unchanged_until_kick_check(self):
        h = handoff(); self.assertEqual(h.policy(now_s=1., packet=packet(1.))['state'], 'APPROACH')


if __name__ == '__main__':
    unittest.main()
