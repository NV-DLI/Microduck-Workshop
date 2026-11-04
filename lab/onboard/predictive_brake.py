"""Empirical stopping-distance trigger using public same-anchor ball history.

The constants are experimental, not a calibrated physical brake model. Only the
existing ramp trigger changes; action limits, ramp duration and guards remain.
"""
from collections import deque
import math


class ClosingSpeed:
    def __init__(self):
        self.samples = deque()
        self.anchor = None

    def update(self, t, x, anchor):
        if not all(math.isfinite(v) for v in (t, x, anchor)):
            self.samples.clear()
            return dict(threshold_m=.125, valid=False, reason='nonfinite')
        if self.anchor != anchor or (self.samples and not 0 < t-self.samples[-1][0] <= .025):
            self.samples.clear()
        self.anchor = anchor
        self.samples.append((t, x))
        while self.samples and t-self.samples[0][0] > .24+1e-9:
            self.samples.popleft()
        span = t-self.samples[0][0]
        if len(self.samples) < 8 or span < .16-1e-9:
            return dict(threshold_m=.125, valid=False, reason='insufficient_same_anchor_history',
                        sample_count=len(self.samples), span_s=span)
        mean_t = sum(a for a, _ in self.samples)/len(self.samples)
        mean_x = sum(b for _, b in self.samples)/len(self.samples)
        slope = sum((a-mean_t)*(b-mean_x) for a,b in self.samples)/sum((a-mean_t)**2 for a,_ in self.samples)
        speed = max(0., min(.5, -slope))
        threshold = max(.125, min(.17, .11+.28*speed))
        return dict(threshold_m=threshold, valid=True, closing_speed_m_s=speed,
                    slope_m_s=slope, sample_count=len(self.samples), span_s=span,
                    target_x_m=.11, effective_stopping_s=.28, anchor_s=anchor,
                    source='public same-anchor ball memory', truth_used=False)
