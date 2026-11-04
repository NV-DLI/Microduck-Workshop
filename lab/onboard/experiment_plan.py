"""Explicit, frozen initial conditions for private paired experiments."""
import hashlib
import json
import math
from pathlib import Path


def load_plan(path):
    raw = Path(path).read_bytes()
    plan = json.loads(raw)
    if plan.get('schema') != 'onboard-paired-experiment-v1':
        raise ValueError('Unsupported experiment plan')
    if plan.get('planner', 'direct') != 'direct':
        raise ValueError('Only the direct planner is supported')
    if plan.get('braking', 'predictive') != 'predictive':
        raise ValueError('Only predictive braking is supported')
    plan['planner'] = 'direct'
    plan['braking'] = 'predictive'
    rows = plan.get('episodes')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 40:
        raise ValueError('Plan must have 1..40 episodes')
    ids = set()
    for row in rows:
        seed, offset, yaw = row['seed'], row['offset_xy_m'], row['yaw_deg']
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError('Invalid episode seed')
        if not isinstance(row['case_id'], str) or row['case_id'] in ids:
            raise ValueError('Duplicate or invalid case id')
        ids.add(row['case_id'])
        if len(offset) != 2 or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v) and abs(v) <= .10+1e-12 for v in offset):
            raise ValueError('Offsets must stay within +/-10 cm')
        if not isinstance(yaw, (int, float)) or not math.isfinite(yaw) or abs(yaw) > 30:
            raise ValueError('Yaw must stay within +/-30 degrees')
        if not 0 < row['range_xy_m'] <= .10 or not 0 < row['range_yaw_deg'] <= 30:
            raise ValueError('Invalid distribution bounds')
        if max(map(abs, offset)) > row['range_xy_m']+1e-12 or abs(yaw) > row['range_yaw_deg']+1e-12:
            raise ValueError('Actual initial condition outside declared distribution')
    return dict(document=plan, sha256=hashlib.sha256(raw).hexdigest())
