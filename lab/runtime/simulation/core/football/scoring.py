"""Scoring is separate from the executor; simulator truth is allowed here."""

import math

GOAL_SCORING_VERSION = "whole_ball_v2_ground_tolerance_1mm"
GROUND_PENETRATION_TOLERANCE_M = 0.001


def goal_crossing(
    previous,
    current,
    goal_x=0.93,
    goal_y=0.0,
    width=0.24,
    height=0.18,
    radius=0.035,
    ground_penetration_tolerance=GROUND_PENETRATION_TOLERANCE_M,
):
    """Score when the trailing ball surface passes the forward goal plane.

    Interpolate the centre at x=goal_x+radius. Side-post and crossbar clearance
    retain their full-radius margins. Only the floor permits up to 1 mm of
    explicit solver contact penetration: a resting Newton ball in this course
    has centre z=0.0346328 m for r=0.035 m. Larger/deep penetrations are rejected.
    Record GOAL_SCORING_VERSION and the tolerance alongside measured outcomes.
    """
    try:
        if len(previous) != 3 or len(current) != 3:
            return False
        vals = (
            *previous,
            *current,
            goal_x,
            goal_y,
            width,
            height,
            radius,
            ground_penetration_tolerance,
        )
        if not all(math.isfinite(v) for v in vals):
            return False
    except (TypeError, ValueError):
        return False
    if radius <= 0 or width < 2 * radius or height < 2 * radius:
        return False
    if not 0 <= ground_penetration_tolerance <= GROUND_PENETRATION_TOLERANCE_M:
        return False
    crossing_x = goal_x + radius
    dx = current[0] - previous[0]
    if not previous[0] < crossing_x <= current[0] or dx <= 0:
        return False
    t = (crossing_x - previous[0]) / dx
    y = previous[1] + t * (current[1] - previous[1])
    z = previous[2] + t * (current[2] - previous[2])
    return (
        abs(y - goal_y) <= width / 2 - radius
        and radius - ground_penetration_tolerance <= z <= height - radius
    )


def wilson(successes, total, z=1.96):
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Invalid trial counts")
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return centre - half, centre + half
