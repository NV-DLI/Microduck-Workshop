import unittest
import torch
from skill_public_observation_v1 import source61, jump63


class PublicObservationTests(unittest.TestCase):
    def case(self):
        n = 3
        packet = dict(orientation_valid=torch.ones(n, dtype=torch.bool),
            gyro_body_rad_s=torch.ones(n, 3, dtype=torch.float64),
            gravity_body=torch.tensor([[.03, .04, -.99]]*n, dtype=torch.float64),
            encoder_q_rad=torch.arange(n*14).reshape(n, 14).double()*.01,
            encoder_qd_rad_s=torch.ones(n, 14, dtype=torch.float64)*.123,
            measurement_age_s=torch.tensor([0., .005, .02], dtype=torch.float64),
            encoder_read_timestamp_s=torch.ones(n, dtype=torch.float64),
            read_timestamp_s=torch.ones(n, dtype=torch.float64),
            gyro_clipped=torch.ones(n, 3, dtype=torch.bool))
        return packet, dict(home14=torch.zeros(14), previous_raw14=torch.ones(n, 14)*2.,
                            public_commands13=torch.zeros(n, 13))

    def test_exact_public_fields_no_gravity_normalization_or_clip_reset(self):
        p, args = self.case()
        out, ready = source61(p, **args)
        self.assertTrue(bool(ready.all()))
        self.assertEqual(out.shape, (3, 61))
        torch.testing.assert_close(out[:, 3:6], p['gravity_body'].float(), rtol=0, atol=0)
        torch.testing.assert_close(out[:, 6:20], p['encoder_q_rad'].float(), rtol=0, atol=0)
        torch.testing.assert_close(out[:, 34:48], args['previous_raw14'], rtol=0, atol=0)

    def test_invalid_age_or_unsynchronized_encoder_is_not_actor_input(self):
        p, args = self.case(); p['measurement_age_s'][0] = .081
        p['encoder_read_timestamp_s'][1] += .02; p['orientation_valid'][2] = False
        out, ready = source61(p, **args)
        self.assertFalse(bool(ready.any())); self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_private_field_rejected_and_nan_not_hidden(self):
        p, args = self.case(); p['root_linear_velocity'] = torch.zeros(3, 3)
        with self.assertRaises(ValueError): source61(p, **args)
        p.pop('root_linear_velocity'); p['gyro_body_rad_s'][1, 0] = float('nan')
        out, ready = source61(p, **args)
        self.assertEqual(ready.tolist(), [True, False, True]); self.assertTrue(bool(torch.isfinite(out).all()))

    def test_jump_has_only_public_timer_no_root_velocity_or_flight_columns(self):
        p, args = self.case()
        out, ready = jump63(p, **args, public_trigger_elapsed_s=torch.tensor([0., .3, .6]))
        self.assertEqual(out.shape, (3, 63)); self.assertTrue(bool(ready.all()))
        torch.testing.assert_close(out[:, 61:], torch.tensor([[0., 1.], [1., 0.], [0., -1.]]), atol=2e-7, rtol=0.)
        with self.assertRaises(ValueError): jump63(p, **args, public_trigger_elapsed_s=torch.zeros(3, 1))


if __name__ == '__main__': unittest.main()
