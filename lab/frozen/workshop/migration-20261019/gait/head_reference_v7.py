"""Development head-reference action mapping; not compatible with a v6 bundle.

Leg targets retain the learned policy. Head targets start from the requested
encoder offsets, optionally with a bounded learned residual. Physical BAM,
backlash, encoder correction, target margins and slew remain active. This is
not image stabilization and never overwrites a measured joint or robot state.
"""
import torch
from precision_recipe_v3 import BoundedTrainingAction
from bounded_actions import JOINT_NAMES


class DirectHeadReferenceAction(BoundedTrainingAction):
    residual_limit_rad = 0.

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if list(self._joint_names) != JOINT_NAMES or not torch.all(torch.as_tensor(self._scale) == 1.):
            raise ValueError('Head reference requires the explicit14-channel unit-scale action order')
        self.head_channels = [JOINT_NAMES.index(name) for name in ('neck_pitch','head_pitch','head_yaw','head_roll')]

    def process_actions(self, actions):
        combined = actions.clone()
        reference = self._env.command_manager.get_command('head_pose')
        combined[:, self.head_channels] = reference + self.residual_limit_rad * torch.tanh(actions[:, self.head_channels])
        self.reference_combined_action = combined
        super().process_actions(combined)


class ResidualHeadReferenceAction(DirectHeadReferenceAction):
    residual_limit_rad = .15
