import sys
import unittest
from pathlib import Path
import numpy as np
import torch

M = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(M/'football'), str(M/'sensors')]
from startup_support import StartupSupport
from startup_fixture import StartupFixture


class FixtureTests(unittest.TestCase):
    def test_matches_single_robot_fixture_and_task_zero(self):
        ref = torch.tensor([[0., 0., .125, 0., 0., 0., 1.]], dtype=torch.float32)
        fixture = StartupFixture(1, 'cpu')
        fixture.reset(None, reference_pose_xyzw=ref, actual_mass_kg=torch.tensor([.74]))
        scalar = StartupSupport(ref[0].tolist(), .74)
        for tick in range(701):
            t = tick*.005
            pos = ref[:, :3] + torch.tensor([[.001*np.sin(t), .001*np.cos(t), -.002]])
            angle = .02*np.sin(t)
            q = torch.tensor([[0., np.sin(angle/2), 0., np.cos(angle/2)]])
            v = torch.tensor([[.001, -.002, .003]])
            w = torch.tensor([[.04, -.03, .02]])
            valid = t >= .8
            f, torque = fixture.update(torch.tensor([t]), position=pos, quaternion=q, linear_velocity=v,
                                      angular_velocity=w, orientation_valid=torch.tensor([valid]))
            r = scalar.update(t, position_world=pos[0].tolist(), quaternion_xyzw=q[0].tolist(),
                              velocity_world=v[0].tolist(), angular_velocity_world=w[0].tolist(), imu_orientation_valid=valid)
            np.testing.assert_allclose(f.numpy()[0], r['force_world_n'], atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(torque.numpy()[0], r['torque_world_nm'], atol=2e-6, rtol=2e-6)
            self.assertEqual(bool(fixture.task_active[0]), r['score_enabled'])
        self.assertTrue(torch.equal(f, torch.zeros_like(f)))

    def test_reset_isolation_and_timeout(self):
        f = StartupFixture(2, 'cpu')
        ref = torch.tensor([[0., 0., .125, 0., 0., 0., 1.]]*2)
        f.reset(None, reference_pose_xyzw=ref, actual_mass_kg=torch.tensor([.7,.8]))
        zeros = torch.zeros((2, 3))
        for t in [0., .8, 1.3, 2.3, 5.]:
            f.update(torch.tensor([t,t]), position=ref[:, :3], quaternion=ref[:, 3:], linear_velocity=zeros,
                     angular_velocity=zeros, orientation_valid=torch.tensor([t>=.8,False]))
        self.assertEqual(f.task_active.tolist(), [True,False])
        self.assertEqual(f.failed.tolist(), [False,True])
        before = f.task_started[0].clone()
        f.reset(torch.tensor([1]), reference_pose_xyzw=ref, actual_mass_kg=torch.tensor([.7,.8]))
        self.assertEqual(f.task_started[0], before)
        self.assertEqual(f.failed.tolist(), [False,False])


if __name__ == '__main__':
    unittest.main()
