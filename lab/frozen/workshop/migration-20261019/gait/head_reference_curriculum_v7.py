"""Explicit, gradual change of action mapping for a new training lineage."""
import torch
from head_reference_v7 import DirectHeadReferenceAction
from precision_recipe_v3 import BoundedTrainingAction


class ProgressiveHeadResidualAction(DirectHeadReferenceAction):
    residual_limit_rad = .15

    @property
    def reference_blend(self):
        # 24 rollout steps/update; the new mapping is complete after300 updates.
        return min(1., .25 + .75 * self._env.common_step_counter / (24*300))

    def process_actions(self, actions):
        combined = actions.clone()
        reference = self._env.command_manager.get_command('head_pose')
        residual = self.residual_limit_rad * torch.tanh(actions[:, self.head_channels])
        blend = self.reference_blend
        combined[:, self.head_channels] = (1-blend)*actions[:,self.head_channels] + blend*(reference+residual)
        self.reference_combined_action = combined
        BoundedTrainingAction.process_actions(self, combined)


class FinalHeadResidualAction(ProgressiveHeadResidualAction):
    @property
    def reference_blend(self):
        return 1.
