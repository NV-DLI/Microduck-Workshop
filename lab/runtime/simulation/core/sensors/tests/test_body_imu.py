import math
import struct
import unittest
from types import SimpleNamespace

import numpy as np

from beta_body_imu import BetaBodyImu, BetaBodyImuProfile, BodySflpDecoder
from isaac_beta_body_imu import IsaacBodyImu


def block(gyro=(0, 0, 0), xyz=(.1, 0., 0.)):
    return struct.pack('<3h3e', *gyro, *xyz)


class BodyDecoderTests(unittest.TestCase):
    def test_ready_requires25_live_reads_and_zero_is_not_identity_measurement(self):
        decoder = BodySflpDecoder()
        for _ in range(40): decoder.decode(bytes(12))
        self.assertFalse(decoder.ready)
        for _ in range(24): decoder.decode(block())
        self.assertFalse(decoder.ready)
        sample = decoder.decode(block())
        self.assertTrue(sample['decoder_ready'])
        previous = sample['quat']
        self.assertEqual(decoder.decode(bytes(12))['quat'], previous)

    def test_signed_gyro_identity_beta_axes_and_median3(self):
        decoder = BodySflpDecoder()
        expected = [-1000 * decoder.GYRO_LSB, 2000 * decoder.GYRO_LSB, 0.]
        for _ in range(3): sample = decoder.decode(block((-1000, 2000, 0)))
        np.testing.assert_allclose(sample['gyro'], expected, atol=1e-12)
        spike = decoder.decode(block((20000, -20000, 5000)))
        np.testing.assert_allclose(spike['gyro'], expected, atol=1e-12)

    def test_half_quaternion_reconstruction_and_invalid_hold(self):
        decoder = BodySflpDecoder()
        for _ in range(3): sample = decoder.decode(block(xyz=(.5, 0., 0.)))
        np.testing.assert_allclose(sample['quat'], [math.sqrt(.75), .5, 0., 0.], atol=1e-12)
        np.testing.assert_allclose(sample['gravity'], [0., -math.sqrt(.75), -.5], atol=1e-12)
        bad = decoder.decode(block(xyz=(1., 1., 1.)))
        self.assertEqual(sample['quat'], bad['quat'])


class BodySensorTests(unittest.TestCase):
    def collect(self, sensor, seconds=1.2, yaw=0., roll=.15, acceleration=(0., 0., 0.)):
        # Rz(yaw)*Rx(roll), simulator quaternion used only to synthesize raw IMU.
        q = (math.cos(yaw/2)*math.sin(roll/2), math.sin(yaw/2)*math.sin(roll/2),
             math.sin(yaw/2)*math.cos(roll/2), math.cos(yaw/2)*math.cos(roll/2))
        result = []
        for tick in range(round(seconds/.005)+1):
            result.extend(sensor.poll(tick*.005, q_world_from_trunk_xyzw=q,
                angular_velocity_world_rad_s=(0., 0., 0.), linear_acceleration_world_m_s2=acceleration))
        return result

    def test50hz_delivery_explicit20ms_age_and_no_world_yaw_copy(self):
        historical = BetaBodyImuProfile(pipeline_delay_s=.020)
        first = self.collect(BetaBodyImu(seed=4, profile=historical))
        yaw_changed = self.collect(BetaBodyImu(seed=4, profile=historical), yaw=1.2)
        self.assertEqual(len(first), 60)
        self.assertEqual(first, yaw_changed)
        self.assertTrue(first[-1]['orientation_valid'])
        self.assertTrue(all(abs(x['delivered_timestamp_s']-x['read_timestamp_s']-.02)<1e-9 for x in first))
        # The board's120Hz latest sample can predate the50Hz register read.
        self.assertTrue(all(.02-1e-9 <= x['sample_age_at_delivery_s'] <= .025+1e-9 for x in first))
        valid = [x for x in first if x['orientation_valid']]
        self.assertGreater(valid[0]['delivered_timestamp_s'], .6)
        np.testing.assert_allclose(valid[-1]['gravity'], [0., -math.sin(.15), -math.cos(.15)], atol=.003)

    def test_default_has_no_added_queue_and_still_waits_for_real_fusion(self):
        sensor = BetaBodyImu(seed=4)
        packets = self.collect(sensor)
        self.assertEqual(sensor.profile.pipeline_delay_s, 0.)
        self.assertEqual(len(packets), 61)
        self.assertTrue(all(x['delivered_timestamp_s'] == x['read_timestamp_s'] for x in packets))
        self.assertTrue(all(0. <= x['sample_age_at_delivery_s'] <= .005+1e-9 for x in packets))
        valid = [p for p in packets if p['orientation_valid']]
        self.assertGreater(valid[0]['delivered_timestamp_s'], .6)
        self.assertTrue(all(p['quat'] is None for p in packets if not p['orientation_valid']))

    def test_delay_sensitivity_retains_identical_raw_measurements_and_median(self):
        baseline = self.collect(BetaBodyImu(seed=8))
        for delay in (.005, .010, .020):
            rows = self.collect(BetaBodyImu(seed=8, profile=BetaBodyImuProfile(pipeline_delay_s=delay)))
            for delayed, raw in zip(rows, baseline):
                self.assertAlmostEqual(delayed['delivered_timestamp_s']-raw['delivered_timestamp_s'], delay)
                for key in ('raw_block_hex', 'gyro', 'gravity', 'quat', 'orientation_valid',
                            'read_timestamp_s', 'median_window_acquisition_timestamps_s'):
                    self.assertEqual(delayed[key], raw[key])

    def test_freefall_never_becomes_valid(self):
        packets = self.collect(BetaBodyImu(seed=4), acceleration=(0., 0., -9.80665))
        self.assertFalse(any(p['orientation_valid'] for p in packets))
        self.assertTrue(all(p['quat'] is None and p['gravity'] is None for p in packets))

    def test_latest_is_only_delivered_and_reset_clears_it(self):
        sensor = BetaBodyImu(profile=BetaBodyImuProfile(pipeline_delay_s=.020))
        sensor.poll(0., q_world_from_trunk_xyzw=(0., 0., 0., 1.),
                    angular_velocity_world_rad_s=(0., 0., 0.), linear_acceleration_world_m_s2=(0., 0., 0.))
        self.assertIsNone(sensor.latest)
        self.collect(sensor)
        self.assertIsNotNone(sensor.latest)
        sensor.reset()
        self.assertIsNone(sensor.latest)

    def test_wrapper_keeps_world_data_inside_sensor_boundary(self):
        robot = SimpleNamespace(body_names=['trunk_base'], data=SimpleNamespace(
            body_pos_w=np.zeros((1,1,3)), body_quat_w=np.array([[[0.,0.,0.,1.]]]),
            body_ang_vel_w=np.zeros((1,1,3))))
        sensor = IsaacBodyImu(robot)
        self.assertEqual(sensor.update(.005), [])
        self.assertEqual(sensor.update(.010), [])
        packets = []
        for tick in range(3, 251): packets.extend(sensor.update(tick*.005))
        self.assertTrue(packets[-1]['orientation_valid'])
        self.assertFalse(any('root' in key or 'world_position' in key for key in packets[-1]))
        self.assertFalse(sensor.metadata()['world_kinematics_exposed_in_packet'])


if __name__ == '__main__':
    unittest.main()
