"""A deliberately narrow privileged expert: stop after a goal-directed release.

This does not navigate to a misplaced ball or recover a fall. Qualification
requires actual closed-loop recovery tests, not just these trigger checks.
"""

import math


class ReleaseBrakeExpert:
    def __init__(self):
        self.active = False
        self.trigger = None

    def update(
        self,
        *,
        proposal,
        robot_position,
        ball_position,
        ball_velocity,
        forward_velocity,
        gravity_z,
        angular_speed,
        timestamp_s
    ):
        values = [
            *proposal,
            *robot_position,
            *ball_position,
            *ball_velocity,
            forward_velocity,
            gravity_z,
            angular_speed,
            timestamp_s,
        ]
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Non-finite recovery observation")
        if self.active:
            return True
        bx, by, bz = ball_position
        vx, vy, _ = ball_velocity
        # Same 10 cm ball / 38.8 cm opening as the fixed scene. Conservative
        # straight-line projection is only a gate, never a success score.
        projected_y = by + (0.68 - bx) * vy / vx if vx > 0 else math.inf
        eligible = (
            proposal[0] > 0.05
            and 0.10 < bx < 0.68
            and vx > 0.4
            and abs(projected_y) < 0.13
            and 0.04 <= bz <= 0.10
            and robot_position[0] < 0.40
            and 0.04 < forward_velocity < 0.60
            and gravity_z < -0.94
            and angular_speed < 2.0
        )
        if eligible:
            self.active = True
            self.trigger = dict(
                timestamp_s=timestamp_s,
                projected_ball_y=projected_y,
                reason="goal_directed_ball_released_while_policy_still_propels",
            )
        return self.active
