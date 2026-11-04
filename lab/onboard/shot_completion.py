"""Episode termination only; privileged ball motion never controls the robot."""
import math


class ShotCompletion:
    """Finish after a completed kick, physical kick contact and sustained rest.

    Goal/miss is scored separately. A stopped ball before contact, a missed kick,
    or a navigation rejection cannot finish the episode. Time is simulation time.
    """
    speed_limit_m_s = .02
    rest_duration_s = .5

    def __init__(self, limit_s):
        if not math.isfinite(limit_s) or limit_s < 0:
            raise ValueError("Nonnegative finite duration required; zero means manual reset only")
        self.limit_s = float(limit_s)
        self.previous = None
        self.rest_since = None
        self.contact = False
        self.kick_completed = False
        self.reason = None
        self.speed_m_s = None

    def update(self, *, now_s, task_s, ball_xyz, kick_completed, kick_contact):
        xyz = tuple(float(v) for v in ball_xyz)
        if len(xyz) != 3 or not all(math.isfinite(v) for v in (*xyz, now_s, task_s)):
            raise ValueError("Finite ball samples and simulation clock required")
        if task_s < 0 or self.previous is not None and now_s <= self.previous[0]:
            raise ValueError("Monotonic simulation clock required")
        self.contact |= bool(kick_contact)
        self.kick_completed |= bool(kick_completed)
        self.speed_m_s = None if self.previous is None else math.dist(xyz, self.previous[1]) / (now_s - self.previous[0])
        gap_ok = self.previous is not None and now_s - self.previous[0] <= .025
        self.previous = (now_s, xyz)
        if (self.kick_completed and self.contact and gap_ok
                and self.speed_m_s <= self.speed_limit_m_s):
            if self.rest_since is None:
                self.rest_since = now_s
        else:
            self.rest_since = None
        # A task cannot acquire extra time through any local recovery/retry.
        if self.limit_s > 0 and task_s >= self.limit_s - 1e-9:
            self.reason = "task_duration_limit"
        elif self.rest_since is not None and now_s - self.rest_since >= self.rest_duration_s - 1e-9:
            self.reason = "shot_complete_ball_settled"
        return self.reason

    def status(self):
        return dict(protocol="settled-shot-v2", kick_action_complete="actual zero kick weight after the 0.14 s hold and 0.12 s blend", kick_completed=self.kick_completed, physical_kick_contact=self.contact,
                    ball_speed_m_s=self.speed_m_s, rest_since_s=self.rest_since,
                    speed_limit_m_s=self.speed_limit_m_s, rest_duration_s=self.rest_duration_s,
                    task_limit_s=self.limit_s, terminal_reason=self.reason,
                    scope="Episode termination only; not a motion/planning input")
