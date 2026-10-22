import math
import copy
import unittest

import numpy as np

from command_navigation_native import NativePhaseCommands
from range_progress import RangeProgress


class ProgressContract(unittest.TestCase):
    def test_invalid_direct_inputs_reject_without_state_mutation(self):
        progress = RangeProgress()
        progress.unproven = 2
        progress.begin(0., .32, .02, .4)
        original = copy.deepcopy(progress.__dict__)
        for call in [lambda: progress.choose(float('nan')),
                     lambda: progress.choose(float('inf')),
                     lambda: progress.begin(1., .32, -.01, .6),
                     lambda: progress.settled(float('nan'), .32, .02),
                     lambda: progress.end(float('nan'), True)]:
            with self.assertRaises(ValueError):
                call()
            self.assertEqual(progress.__dict__, original)

    def call(self, control, tick, radius=.32, uncertainty=.02, **override):
        args = dict(timestamp_s=tick / 15, authorized=True, requested=[.06, 0, 0],
                    point_level=[radius, 0, 0], range_uncertainty_m=uncertainty,
                    gyro_level_z=0., gyro_body_rad_s=[0., 0., 0.],
                    near_goal_heading=0., near_position_error=[.2, 0.])
        args.update(override)
        return control.update(**args)

    def test_two_complete_quiet_cycles_allow_only_declared_longer_probe(self):
        control = NativePhaseCommands()
        onsets = []
        previous = 'OBSERVE'
        for tick in range(65):
            command, info = self.call(control, tick)
            if info['phase'] == 'ADVANCE' and previous != 'ADVANCE':
                onsets.append((tick, info['selected_duration_s'], command[0]))
            previous = info['phase']
            self.assertIn(command, ([0, 0, 0], [.2, 0, 0], [.3, 0, 0]))
        self.assertEqual([r[1] for r in onsets[:3]], [.4, .4, .4])
        self.assertEqual([r[2] for r in onsets[:3]], [.2, .2, .3])
        self.assertGreater(control.progress.probe_pulses, 0)
        self.assertFalse(control.progress.last_comparison['confirmed_minimum_progress'])

    def test_yaw_rotation_is_not_radial_progress(self):
        progress = RangeProgress()
        point = np.array([.3, -.08])
        rotated = np.array([[math.cos(.7), -math.sin(.7)], [math.sin(.7), math.cos(.7)]]) @ point
        progress.begin(0., np.linalg.norm(point), .02, .4)
        progress.end(.4, True)
        progress.settled(.9, np.linalg.norm(rotated), .02)
        self.assertAlmostEqual(progress.last_comparison['nominal_radial_approach_m'], 0., places=14)
        self.assertFalse(progress.last_comparison['confirmed_minimum_progress'])

    def test_wide_uncertainty_does_not_assert_zero_motion(self):
        progress = RangeProgress()
        progress.begin(0, .32, .025, .4)
        progress.end(.4, True)
        progress.settled(1, .30, .025)
        self.assertEqual(progress.last_comparison['classification'], 'insufficient_evidence_of_minimum_progress')
        self.assertLess(progress.last_comparison['radial_approach_interval_m'][0], 0)
        self.assertGreater(progress.last_comparison['radial_approach_interval_m'][1], .02)

    def test_confirmed_progress_resets_streak(self):
        progress = RangeProgress()
        progress.unproven = 2
        progress.begin(0, .32, .005, .4)
        progress.end(.4, True)
        progress.settled(1, .29, .005)
        self.assertTrue(progress.last_comparison['confirmed_minimum_progress'])
        self.assertEqual(progress.unproven, 0)

    def test_short_direction_interruption_and_stale_comparison_cannot_escalate(self):
        for complete, end in [(False, 1.), (True, 4.)]:
            progress = RangeProgress()
            progress.unproven = 1
            progress.begin(0, .32, .02, .4)
            progress.end(.2, complete)
            progress.settled(end, .32, .02)
            self.assertEqual(progress.unproven, 0)

    def test_reserve_and_total_budget_survive_phase_reset(self):
        control = NativePhaseCommands()
        control.progress.unproven = 2
        self.assertEqual(control.progress.choose(.279)[0], .4)
        self.assertEqual(control.progress.choose(.28)[0], .4)
        control.progress.probe_pulses = 3
        self.assertIsNone(control.progress.choose(.30)[0])
        control.reset()
        self.assertEqual(control.progress.probe_pulses, 3)

    def test_longer_active_phase_stops_at_reserve(self):
        control = NativePhaseCommands()
        control.phase = 'ADVANCE'; control.started = 0.; control.last_t = 0.
        control.active_duration = .4; control.active_forward_magnitude = .3; control.active_progress_probe = True
        command, info = self.call(control, 1, radius=.259, uncertainty=.02)
        self.assertEqual(command, [0, 0, 0])
        self.assertEqual(info['reason'], 'progress_probe_distance_reserve')

    def test_guard_preempts_longer_probe_and_clears_pending_evidence(self):
        for fault in [dict(authorized=False), dict(requested=[0, 0, 0]),
                      dict(point_level=None), dict(near_goal_heading=None),
                      dict(gyro_body_rad_s=[float('nan'), 0, 0])]:
            control = NativePhaseCommands()
            control.phase = 'ADVANCE'; control.started = 0.; control.last_t = 0.
            control.active_duration = .4; control.active_forward_magnitude = .3; control.active_progress_probe = True
            control.progress.begin(0, .32, .02, .4, probe=True)
            command, info = self.call(control, 1, **fault)
            self.assertEqual(command, [0, 0, 0])
            self.assertIsNone(control.progress.anchor)
            self.assertEqual(control.progress.probe_pulses, 1)

    def test_probe_flag_cannot_leak_to_new_far_forward(self):
        control = NativePhaseCommands()
        control.active_progress_probe = True
        for tick in range(7):
            command, info = self.call(control, tick, radius=.6)
        self.assertEqual(command, [.3, 0, 0])
        self.assertFalse(info['active_progress_probe'])
        self.assertEqual(control.progress.probe_pulses, 0)

    def test_nonquiet_windows_never_compare_or_start(self):
        control = NativePhaseCommands()
        control.progress.unproven = 2
        for tick in range(30):
            command, _ = self.call(control, tick, gyro_body_rad_s=[.3, 0, 0])
            self.assertEqual(command, [0, 0, 0])
        self.assertEqual(control.progress.probe_pulses, 0)


if __name__ == '__main__':
    unittest.main()
