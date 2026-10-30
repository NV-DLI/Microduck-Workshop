"""Sensor-only walking observations and requests for the preserved motor skills.

GR00T predicts body/head requests and a right-kick request. The 50 Hz walking
actor and four-second kick/recovery skill still produce the 14 motor targets.
"""
import math

ACTION_NAMES = [
    'forward_request_native', 'lateral_request_native', 'yaw_request_native',
    'neck_pitch_request_rad', 'head_pitch_request_rad', 'head_yaw_request_rad',
    'head_roll_request_rad', 'right_kick_request',
]
LOWER = [-.35, -.2, -.8, -1.1, -1.1, -1.4, -.31, 0.]
UPPER = [.35, .2, .8, 1.1, 1.1, 1.4, .31, 1.]
TASK = 'Walk to the football, kick it into the goal with the right foot, and recover to standing.'


def finite(values, size):
    result = [float(v) for v in values]
    if len(result) != size or not all(math.isfinite(v) for v in result):
        raise ValueError(f'Expected {size} finite values')
    return result


def bound_request(request):
    return [max(lo, min(hi, v)) for v, lo, hi in zip(finite(request, 8), LOWER, UPPER)]


def validate_demonstrated_request(request):
    values = finite(request, 8)
    if any(v < lo-1e-7 or v > hi+1e-7 for v,lo,hi in zip(values,LOWER,UPPER)):
        raise ValueError('Recorded command is outside the walking contract; do not relabel by clipping')
    return values


def request_group(chunk):
    values = [finite(row, 8) for row in chunk]
    if any(row[7] >= .5 for row in values):
        return 'kick_and_recover'
    if any(max(abs(v) for v in row[:3]) > .05 for row in values):
        return 'approach_and_align'
    if any(max(abs(v) for v in row[3:7]) > .1 for row in values):
        return 'observe'
    return 'stand'


def pack_state(body61, previous_request, *, skill_phase_s=None):
    # Elements 48:61 contain the CURRENT motor commands: including them would
    # leak the demonstrated action. Only proprioception and previous raw motor
    # actions from 0:48 are inputs. Previous high-level request is explicitly lagged.
    body = finite(body61, 61)
    previous = finite(previous_request, 8)
    if skill_phase_s is not None and not 0 <= skill_phase_s <= 4:
        raise ValueError('Invalid local skill clock')
    return body[:48] + previous + [
        float(skill_phase_s is not None),
        0. if skill_phase_s is None else skill_phase_s / 4.,
    ]


def contract(joint_names):
    if len(joint_names) != 14 or len(set(joint_names)) != 14:
        raise ValueError('Fourteen ordered motor joints required')
    state_names = (
        ['gyro/'+a for a in 'xyz'] + ['gravity/'+a for a in 'xyz']
        + ['q_relative/'+n for n in joint_names]
        + ['qd/'+n for n in joint_names]
        + ['previous_motor/'+n for n in joint_names]
        + ['previous_request/'+n for n in ACTION_NAMES]
        + ['kick_skill_busy', 'kick_skill_phase']
    )
    return dict(version=1, locomotion='walking', state_names=state_names,
                action_names=ACTION_NAMES, sample_hz=12.5, actor_hz=50,
                request_lower=LOWER, request_upper=UPPER,
                motor_execution='preserved walking actor and native K10 kick/recovery',
                kick_request_threshold=.5, kick_duration_s=4.,
                observation='causal RGB, BODY50 IMU, encoders, previous commands, local skill clock')


class KickLatch:
    """A request starts one bounded motor skill; it cannot restart mid-kick."""
    def __init__(self):
        self.started = None
        self.completed = False
        self.quiet_since = None
        self.last_t = None

    def phase(self, now):
        if not math.isfinite(now) or now < 0:
            raise ValueError('Finite nonnegative skill clock required')
        if self.started is None or self.completed:
            return None
        return max(0., min(4., now-self.started))

    def update(self, now, request, *, gyro, gravity, qd, previous_head):
        if not math.isfinite(now) or now < 0 or (self.last_t is not None and now <= self.last_t):
            raise ValueError('Finite increasing skill clock required')
        self.last_t = now
        request = bound_request(request)
        gyro, gravity = finite(gyro, 3), finite(gravity, 3)
        qd, previous_head = finite(qd, 14), finite(previous_head, 4)
        if self.started is not None:
            if now-self.started >= 4.-1e-8:
                self.completed = True
            return [0.]*7 + [float(not self.completed)]
        quiet = (sum(v*v for v in gyro) <= .1**2
                 and sum(v*v for v in qd)/14 <= .25**2
                 and gravity[2] < -.94 and max(abs(v) for v in previous_head) <= 1e-9)
        self.quiet_since = (now if self.quiet_since is None else self.quiet_since) if quiet else None
        if request[7] >= .5:
            # Stop/return the head before entering the unchanged kick skill.
            if self.quiet_since is not None and now-self.quiet_since >= .5-1e-8:
                self.started = now
                return [0.]*7+[1.]
            return [0.]*8
        return request[:7]+[0.]
