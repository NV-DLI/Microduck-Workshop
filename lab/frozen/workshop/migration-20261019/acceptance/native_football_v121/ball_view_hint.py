"""Last complete-ball gaze command, for head attention only.

This stores a previously issued gaze, not a new observation or a metric pose.
The caller retains all head permissions, source-domain checks and deadlines.
"""
import math
import numpy as np


class BallViewHint:
    def __init__(self):
        self.capture_s = None
        self.gaze = None
        self.returning = False
        self.return_started_s = None
        self.reason = 'no_complete_capture'

    def observe(self, *, now_s, capture_s, gaze, complete):
        # Invalid/partial/old/future captures cannot erase a valid hint.
        if not complete:
            return False
        try:
            t, capture = float(now_s), float(capture_s)
            q = np.asarray(gaze, float)
        except (TypeError, ValueError):
            return False
        if (not math.isfinite(t) or not math.isfinite(capture) or t < 0
                or capture < 0 or abs(t-capture) > 1e-9
                or q.shape != (2,) or not np.isfinite(q).all()
                or np.any(q < [-1.6, 0.]) or np.any(q > [1.6, 1.4])
                or self.capture_s is not None and capture <= self.capture_s):
            return False
        self.capture_s, self.gaze = capture, q.copy()
        self.reason = 'stored_complete_capture_gaze_command'
        return True

    def begin_return(self, now_s):
        if (not math.isfinite(now_s) or now_s < 0 or self.gaze is None
                or self.capture_s > now_s):
            return False
        self.returning = True
        self.return_started_s = now_s
        self.reason = 'return_to_last_complete_view_only'
        return True

    def target(self, *, now_s, current):
        if not self.returning:
            return None
        q = np.asarray(current, float)
        if (not math.isfinite(now_s) or now_s < self.return_started_s
                or q.shape != (2,) or not np.isfinite(q).all()):
            return None
        if np.max(abs(q-self.gaze)) <= .03:
            self.returning = False
            self.reason = 'gaze_command_arrived_resume_original_search'
            return None
        return self.gaze.copy()

    def cancel_return(self):
        self.returning = False
        self.return_started_s = None

    def diagnostic(self, now_s):
        return {
            'capture_timestamp_s': self.capture_s,
            'original_capture_age_s': None if self.capture_s is None else now_s-self.capture_s,
            'gaze_command_rad': None if self.gaze is None else self.gaze.tolist(),
            'returning': self.returning,
            'return_started_s': self.return_started_s,
            'reason': self.reason,
            'metric_position': None,
            'refreshes_observation_time': False,
            'body_authorized': False,
            'kick_authorized': False,
            'world_stabilized': False,
        }
