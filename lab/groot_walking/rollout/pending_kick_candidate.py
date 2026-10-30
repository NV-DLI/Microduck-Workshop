"""Uninstalled proposal: retain one kick intent while returning to measured quiet.

This changes command acceptance, not the learned choice of where to kick.
It has no camera geometry, ball pose, teacher controller or scoring input.
"""
import math
from learning.walking import KickLatch, bound_request


class PendingKickLatch(KickLatch):
    def __init__(self, timeout_s=4.0):
        super().__init__()
        if not math.isfinite(timeout_s) or timeout_s <= .5:
            raise ValueError('Positive timeout beyond the unchanged quiet gate required')
        self.timeout_s = timeout_s
        self.pending_since = None
        self.failure = None

    def update(self, now, request, **measurements):
        if not math.isfinite(now) or now < 0 or (
            self.last_t is not None and now <= self.last_t
        ):
            raise ValueError('Finite increasing skill clock required')
        request = bound_request(request)
        if self.started is None and self.failure is None:
            if self.pending_since is None and request[7] >= .5:
                self.pending_since = now
            if self.pending_since is not None:
                if now - self.pending_since >= self.timeout_s - 1e-8:
                    self.failure = 'kick_preparation_timeout'
                else:
                    request = [0.] * 7 + [1.]
        if self.failure is not None:
            # Keep validating the original measurements and never authorize a
            # late kick or a second attempt after a preparation timeout.
            super().update(now, [0.] * 8, **measurements)
            return [0.] * 8
        return super().update(now, request, **measurements)
