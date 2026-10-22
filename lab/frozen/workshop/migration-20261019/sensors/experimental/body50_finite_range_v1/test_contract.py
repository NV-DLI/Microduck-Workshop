import hashlib
import json
import math
from pathlib import Path
import sys
import unittest
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2] / 'acceptance/body50_training'
sys.path.insert(0, str(BASE))
from vector_body_imu import VectorBodyImu as Original, VectorBodyProfile
from finite_range_core import VectorBodyImu as Candidate


class Contract(unittest.TestCase):
    def pair(self):
        return Original(3), Candidate(3)

    def tick(self, s, t, w=None, q=None):
        return s.poll(t, q_world_from_trunk_xyzw=torch.tensor([[0.,0.,0.,1.]]*3) if q is None else q,
                      angular_velocity_world_rad_s=torch.zeros(3,3) if w is None else w,
                      linear_acceleration_world_m_s2=torch.zeros(3,3),
                      noise_standard_normal=torch.zeros(3,6))

    def warm(self, pair):
        for i in range(1,201):
            packets = [self.tick(s,i*.005) for s in pair]
        self.assertTrue(packets[0]['orientation_valid'].all())
        return packets

    def test_only_one_executable_change(self):
        c=json.loads((HERE/'change-contract.json').read_text())
        b=Path(c['parent']).read_text()
        self.assertEqual(hashlib.sha256(b.encode()).hexdigest(),c['parent_sha256'])
        copied=(HERE/'finite_range_core.py').read_text().splitlines(True)[3:]
        self.assertEqual(''.join(copied),b.replace('lost=acquire&(gap|gc.any(-1))','lost=acquire&gap'))

    def test_no_clip_packets_bit_exact(self):
        pair=self.pair()
        for i in range(1,301):
            packets=[self.tick(s,i*.005,w=torch.full((3,3),.01)) for s in pair]
            for name in packets[0]:
                self.assertTrue(torch.equal(packets[0][name],packets[1][name]),name)

    def test_clip_integrates_measured_rate_without_false_readiness(self):
        pair=self.pair(); self.warm(pair)
        w=torch.zeros(3,3); w[0,0]=20.; w[1,2]=-20.
        for i in range(201,241):
            packets=[self.tick(s,i*.005,w=w) for s in pair]
        self.assertFalse(packets[0]['orientation_valid'][:2].any())
        self.assertTrue(packets[1]['orientation_valid'].all())
        self.assertTrue(packets[1]['gyro_clipped'][:2].any(-1).all())
        self.assertLessEqual(float(packets[1]['gyro_body_rad_s'].abs().max()),math.radians(500)+math.radians(.0175))
        # Constant truth orientation was supplied only to synthesize raw sensors.
        # Candidate accumulated measured angular motion, so it cannot be truth-fed.
        self.assertGreater(float(pair[1].orientation[0,0].abs()),.5)
        self.assertTrue(torch.equal(packets[0]['gyro_counts'],packets[1]['gyro_counts']))
        self.assertTrue(torch.equal(packets[0]['seq'],packets[1]['seq']))

    def test_initial_saturation_cannot_initialize(self):
        s=Candidate(3)
        for i in range(1,301):p=self.tick(s,i*.005,w=torch.full((3,3),20.))
        self.assertFalse(p['orientation_valid'].any())

    def test_gap_and_independent_reset_preserved(self):
        pair=self.pair();self.warm(pair)
        for s in pair:
            p=self.tick(s,1.2)
            self.assertFalse(p['orientation_valid'].any())
        s=Candidate(3)
        self.warm([s,s])
        s.reset(torch.tensor([1]))
        self.assertFalse(s.initialized[1]);self.assertTrue(s.initialized[[0,2]].all())


if __name__=='__main__':unittest.main()
