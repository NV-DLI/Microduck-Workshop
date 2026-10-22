import unittest
import numpy as np
from command_navigation_native import NativePhaseCommands
from turn_prediction import near_turn_stop_hint


class TurnPredictiveGuards(unittest.TestCase):
    def call(self, obj, t, bearing=.3, rate=0., **kw):
        args = dict(timestamp_s=t, authorized=True, requested=[.06, 0., .1],
                    point_level=[.3, 0., -.09], range_uncertainty_m=.015,
                    gyro_level_z=rate, gyro_body_rad_s=[0., 0., rate],
                    near_goal_heading=bearing, near_position_error=[.2, .04])
        args.update(kw)
        return obj.update(**args)

    def start(self, bearing):
        obj = NativePhaseCommands()
        for t in np.arange(7)/15:
            result = self.call(obj, float(t), bearing)
        return obj, result

    def test_small_error_uses_existing_forward_angle_domain(self):
        for b in (-.15, -.125, .1, .15):
            _, result = self.start(b)
            self.assertEqual(result[0], [.2, 0., 0.])

    def test_predictive_stop_only_after_actual_toward_rotation(self):
        for sign in (-1., 1.):
            obj, result = self.start(.3*sign)
            self.assertEqual(result[0], [0., 0., .8*sign])
            result = self.call(obj, .47, .2*sign, 1.0*sign)
            self.assertEqual(result[0], [0., 0., 0.])
            self.assertEqual(result[1]['reason'], 'near_turn_predictive_stop')
            self.assertTrue(result[1]['turn_stop_prediction']['stop_hint'])
            # Fresh quiet is required again; early stop never chains to motion.
            self.assertEqual(self.call(obj, .54, 0., 0.)[0], [0., 0., 0.])

    def test_opposite_or_quiet_rate_cannot_trigger_hint(self):
        for rate in (0., .12, -.5):
            obj, _ = self.start(.3)
            result = self.call(obj, .47, .2, rate)
            self.assertEqual(result[0], [0., 0., .8])
            self.assertFalse(result[1]['turn_stop_prediction']['stop_hint'])

    def test_every_guard_precedes_prediction(self):
        for fault in ({'authorized': False}, {'point_level': None},
                      {'point_level': [.13, 0., 0.], 'range_uncertainty_m': .02},
                      {'near_goal_heading': None}, {'gyro_level_z': float('nan')}):
            obj, _ = self.start(.3)
            result = self.call(obj, .47, .2, 1., **fault)
            self.assertEqual(result[0], [0., 0., 0.])
            self.assertNotEqual(result[1]['reason'], 'near_turn_predictive_stop')

    def test_pure_hint_invalid_input_never_mutates_controller(self):
        for vals in ((float('nan'), 1., 1.), (.1, float('inf'), 1.), (.1, 1., 0.)):
            with self.assertRaises(ValueError):
                near_turn_stop_hint(*vals)


if __name__ == '__main__':
    unittest.main()
