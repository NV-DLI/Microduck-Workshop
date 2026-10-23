"""Head reference: Microduck simulation and control."""

import torch
from cycle_tracking import BoundedTrainingAction
from bounded_actions import JOINT_NAMES


class DirectHeadReferenceAction(BoundedTrainingAction):
    residual_limit_rad = 0.0

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if list(self._joint_names) != JOINT_NAMES or not torch.all(
            torch.as_tensor(self._scale) == 1.0
        ):
            raise ValueError(
                "Head reference requires the explicit14-channel unit-scale action order"
            )
        self.head_channels = [
            JOINT_NAMES.index(name)
            for name in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")
        ]

    def process_actions(self, actions):
        combined = actions.clone()
        reference = self._env.command_manager.get_command("head_pose")
        combined[:, self.head_channels] = (
            reference
            + self.residual_limit_rad * torch.tanh(actions[:, self.head_channels])
        )
        self.reference_combined_action = combined
        super().process_actions(combined)


class ResidualHeadReferenceAction(DirectHeadReferenceAction):
    residual_limit_rad = 0.15
