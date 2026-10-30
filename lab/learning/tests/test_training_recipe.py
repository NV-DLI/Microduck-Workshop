"""A saved training recipe must survive fresh model construction."""

import json
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from learning.training_recipe import checkpoint_dropout, configure_training


class TrainingRecipePersistence(unittest.TestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "Requires the learning environment"
    )
    def test_recorded_zero_dropout_survives_fresh_head_defaults(self):
        import torch

        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.state_dropout_prob = 0.2
                self.layer = torch.nn.Sequential(
                    torch.nn.Linear(2, 2), torch.nn.Dropout(0.2)
                )

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "training-recipe.json").write_text(
                json.dumps({"head_dropout": "disabled"})
            )
            head = Head()
            policy = SimpleNamespace(
                _groot_model=SimpleNamespace(action_head=head),
                config=SimpleNamespace(
                    tune_projector=True,
                    tune_diffusion_model=False,
                    tune_vlln=False,
                    optimizer_lr=0.001,
                ),
            )
            recipe = configure_training(policy, checkpoint, "checkpoint", "checkpoint")
            self.assertEqual(recipe["head_dropout"], "disabled")
            self.assertEqual(head.state_dropout_prob, 0)
            self.assertEqual(head.layer[1].p, 0)
            self.assertTrue(all(p.requires_grad for p in head.parameters()))

    def test_older_experiment_preserves_recorded_regularization(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp)
            (checkpoint / "cached-training-experiment.json").write_text(
                json.dumps({"regularization_changes": [{"old": 0.2, "new": 0.0}]})
            )
            self.assertEqual(checkpoint_dropout(checkpoint), "disabled")

    def test_legacy_checkpoint_keeps_base_regularization(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(checkpoint_dropout(tmp), "base")


if __name__ == "__main__":
    unittest.main()
