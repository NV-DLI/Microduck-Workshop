"""Bound the native roll command without changing the original IK ray solve.

Only public requested gaze and named model FK are used. This adapter does not
change measured joints, raw actions, observation history, or motor physics.
"""

import math
import numpy as np
from head_adapter import HeadAdapter as OriginalHeadAdapter, HOME, rotation_error

SOURCE_HEAD_COMMAND_BOUNDS = np.array([1.1, 1.1, 1.4, 0.31])


class HeadCommandDomainAdapter(OriginalHeadAdapter):
    def __init__(self):
        super().__init__()
        self.last_accepted_command_q = HOME.copy()

    def solve(self, yaw_pitch, *, max_iterations=40):
        result = super().solve(yaw_pitch, max_iterations=max_iterations)
        original_q = np.asarray(result["desired_head_joint_rad"], dtype=float)
        q = original_q.copy()
        q[3] = np.clip(q[3] - HOME[3], -0.31, 0.31) + HOME[3]
        yaw, pitch = result["gaze_body_yaw_pitch_rad"]
        z = np.array(
            [
                math.cos(yaw) * math.cos(pitch),
                math.sin(yaw) * math.cos(pitch),
                -math.sin(pitch),
            ]
        )
        x = np.cross(z, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        desired = np.column_stack((x, np.cross(z, x), z))
        actual = self.rotation(q)
        pointing = float(
            np.arctan2(np.linalg.norm(np.cross(z, actual[:, 2])), z @ actual[:, 2])
        )
        full = float(np.linalg.norm(rotation_error(desired, actual)))
        native = q - HOME
        excess = float(
            np.maximum(np.abs(native) - SOURCE_HEAD_COMMAND_BOUNDS, 0.0).max()
        )
        physical_excess = float(
            np.maximum(self.limits[:, 0] + 0.02 - q, q - (self.limits[:, 1] - 0.02))
            .clip(0)
            .max()
        )
        accepted = bool(pointing < 0.03 and excess <= 1e-9 and physical_excess <= 1e-9)
        result.update(
            original_orientation_residual_rad=result["orientation_residual_rad"],
            original_IK_desired_head_joint_rad=original_q.tolist(),
            requested_pointing_residual_rad=pointing,
            requested_full_orientation_residual_rad=full,
            source_command_excess_rad=excess,
            head_roll_command_capped=bool(q[3] != original_q[3]),
            evaluation_contract="pointing error < .03rad and native command box plus existing physical IK margins",
            source_head_command_bounds_rad=SOURCE_HEAD_COMMAND_BOUNDS.tolist(),
            world_stabilized=False,
            accepted=accepted,
        )
        # Preserve original private IK iteration state, so the first three
        # requested joints retain exactly the prior solver's sequence.
        # A rejected request cannot leak an out-of-domain head command through
        # the entry's safe-zero body-motion branch.
        if accepted:
            self.last_accepted_command_q = q.copy()
        else:
            q = self.last_accepted_command_q.copy()
            result["rejected_request_held_previous_accepted_head_command"] = True
        result.update(
            desired_head_joint_rad=q.tolist(),
            head_pose_native_offsets_rad=(q - HOME).tolist(),
            orientation_residual_rad=float(
                np.linalg.norm(rotation_error(desired, self.rotation(q)))
            ),
        )
        return result
