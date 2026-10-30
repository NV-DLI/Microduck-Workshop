import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.recovery import ReleaseBrakeExpert


class RecoveryGate(unittest.TestCase):
    def setUp(self):
        self.observation = dict(
            proposal=[0.6, 0, 0, 0],
            robot_position=[-0.1, 0, 0.12],
            ball_position=[0.15, 0, 0.05],
            ball_velocity=[0.5, 0, 0],
            forward_velocity=0.2,
            gravity_z=-0.99,
            angular_speed=0.5,
            timestamp_s=6.2,
        )

    def test_goal_directed_release_latches_braking(self):
        expert = ReleaseBrakeExpert()
        self.assertTrue(expert.update(**self.observation))
        self.observation["proposal"] = [0, 0, 0, 0]
        self.observation["ball_velocity"] = [0, 0, 0]
        self.assertTrue(expert.update(**self.observation))

    def test_rejects_states_outside_recovery_scope(self):
        for key, value in [
            ("proposal", [0, 0, 0, 0]),
            ("ball_position", [0.05, 0, 0.05]),
            ("ball_velocity", [0.5, 0.5, 0]),
            ("ball_velocity", [-0.5, 0, 0]),
            ("robot_position", [0.5, 0, 0.12]),
            ("gravity_z", -0.5),
            ("angular_speed", 3),
            ("forward_velocity", -0.1),
        ]:
            with self.subTest(key=key, value=value):
                obs = copy.deepcopy(self.observation)
                obs[key] = value
                self.assertFalse(ReleaseBrakeExpert().update(**obs))

    def test_nonfinite_state_rejected(self):
        self.observation["forward_velocity"] = float("nan")
        with self.assertRaises(ValueError):
            ReleaseBrakeExpert().update(**self.observation)


if __name__ == "__main__":
    unittest.main()
