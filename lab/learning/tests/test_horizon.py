"""The native denoising length must match the training packer, before cropping."""

from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.horizon import configure_prediction_horizon


class NativePredictionHorizon(unittest.TestCase):
    def setUp(self):
        cfg = NS(action_horizon=40)
        self.policy = NS(
            config=NS(chunk_size=16, n_action_steps=4),
            _groot_model=NS(config=cfg, action_head=NS(config=cfg, action_horizon=40)),
        )
        self.pre = NS(steps=[NS(), NS(action_horizon=16)])

    def test_native_fields_match_training_before_inference(self):
        result = configure_prediction_horizon(self.policy, self.pre)
        self.assertEqual(result["native_prediction_horizon"], 16)
        self.assertEqual(result["original_native_horizon"], 40)
        self.assertEqual(result["execution_horizon"], 4)
        self.assertEqual(self.policy._groot_model.action_head.config.action_horizon, 16)
        self.assertEqual(self.policy._groot_model.action_head.action_horizon, 16)

    def test_legacy_comparison_is_explicit(self):
        result = configure_prediction_horizon(self.policy, self.pre, "checkpoint")
        self.assertEqual(result["native_prediction_horizon"], 40)

    def test_processor_disagreement_rejected(self):
        self.pre.steps[1].action_horizon = 40
        with self.assertRaises(ValueError):
            configure_prediction_horizon(self.policy, self.pre)

    def test_execution_beyond_trained_chunk_rejected(self):
        self.policy.config.n_action_steps = 20
        with self.assertRaises(ValueError):
            configure_prediction_horizon(self.policy, self.pre)


if __name__ == "__main__":
    unittest.main()
