"""Regression checks for low-speed commands and reward degeneracies."""
import unittest
from types import SimpleNamespace
import torch
from precision_recipe_v4 import AxisCommand, COMMAND_CASES, planar_score, yaw_score

class PrecisionTest(unittest.TestCase):
    def test_command_space_has_no_hidden_fast_bucket(self):
        torch.manual_seed(970161)
        term=object.__new__(AxisCommand)
        term._env=SimpleNamespace(num_envs=2048,device='cpu')
        term.vel_command_b=torch.zeros(2048,3)
        term.is_standing_env=torch.zeros(2048,dtype=torch.bool)
        term.is_heading_env=torch.ones(2048,dtype=torch.bool)
        term._resample_command(slice(None))
        expected=torch.tensor(COMMAND_CASES)
        self.assertEqual(len(torch.unique(term.vel_command_b,dim=0)),len(expected))
        self.assertTrue(((term.vel_command_b[:,None]==expected[None]).all(-1).any(-1)).all())
        self.assertTrue(torch.equal(term.is_standing_env,term.vel_command_b.abs().sum(-1)==0))
        self.assertFalse(term.is_heading_env.any())

    def test_tracking_beats_stationary_at_every_moving_command(self):
        command=torch.tensor(COMMAND_CASES)
        correct=planar_score(command,command)+yaw_score(command,command)
        zero=planar_score(torch.zeros_like(command),command)+yaw_score(torch.zeros_like(command),command)
        self.assertTrue((correct[1:]>zero[1:]+6.).all())
        self.assertTrue(torch.allclose(correct,torch.full_like(correct,24.)))

    def test_wrong_direction_and_overspeed_are_worse(self):
        command=torch.tensor(COMMAND_CASES)
        def score(v): return planar_score(v,command)+yaw_score(v,command)
        self.assertTrue((score(command)[1:]>score(-command)[1:]).all())
        self.assertTrue((score(command)[1:]>score(command*3)[1:]).all())

    def test_zero_command_prefers_no_drift(self):
        zero=torch.zeros(4,3)
        drift=torch.tensor([[.025,0,0],[0,.025,0],[0,0,.2],[.1,.1,.4]])
        self.assertTrue(((planar_score(drift,zero)+yaw_score(drift,zero))<24.).all())

    def test_reward_stays_bounded_for_large_physical_velocity(self):
        command=torch.tensor(COMMAND_CASES)
        velocity=torch.randn_like(command)*1e5
        score=planar_score(velocity,command)+yaw_score(velocity,command)
        self.assertTrue(torch.isfinite(score).all())
        self.assertTrue(((score>=0)&(score<=24)).all())

if __name__=='__main__':unittest.main()
