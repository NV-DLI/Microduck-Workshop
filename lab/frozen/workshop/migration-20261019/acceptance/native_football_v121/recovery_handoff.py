"""v121 owner lifecycle: original HeadHandoff + bounded pre-kick retreat RECOVERY.

v121: the retreat body command is the exact fixed-duration public-heading-feedback
helper command computed by RetreatRecovery on this tick; recovery_after_head accepts
only that independently computed expected command (never any other vx/vy/wz).

HeadHandoffV121 subclasses the unchanged v65 handoff. Only when the ORIGINAL
HEAD_RETURN gate rejects with a whitelisted reason, before any K10 onset, and
RetreatRecovery admits it, the handoff enters state 'RECOVERY' instead of the
terminal 'FAILED'. Otherwise the original failure stands. In RECOVERY:
  * every 50 Hz same-read packet still integrates the unchanged public memory;
    the anchor is never refreshed except by a current complete grounded
    same-capture ball in REACQUIRE (no TTL extension, no reset);
  * relative odometry is the memory pose; any memory invalidation/reset after
    entry (gap, sequence gap, invalid read) permanently invalidates it;
  * body velocity comes only from RetreatRecovery (zero or the approved command);
  * the head is commanded neutral except in REACQUIRE, where the original
    attention owner drives it under its current head-only permission;
  * no kick authority: on release the handoff returns to APPROACH, preserving
    attempts, rejection history and kick-ever-started; original gates re-run.
"""
import copy
import math
import numpy as np

from fine_kick_v65 import HeadHandoffV65, CAPTURE_MAX_AGE_S
from retreat_recovery import RetreatRecovery, RECOVERABLE_REASONS

NEUTRAL_OWNED = ('HOLD', 'AUTHORIZE', 'RETREAT', 'SETTLE')
ZERO = [0., 0., 0.]


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def complete_same_capture(geometry, now_s):
    """Executor's original 'complete' predicate plus the grounded sphere flags."""
    g = geometry or {}
    ball = g.get('ball') or {}
    geo = ball.get('geometry') or {}
    return bool(g.get('timestamp_s') == now_s and ball.get('accepted') and (g.get('floor_observation') or {}).get('accepted')
                and geo.get('accepted') and geo.get('observed_boundary_required'))


def geometry_summary(geometry, now_s):
    g = geometry or {}
    ball = g.get('ball') or {}
    geo = ball.get('geometry') or {}
    return {'geometry_timestamp_s': g.get('timestamp_s'), 'same_capture': g.get('timestamp_s') == now_s,
            'ball_accepted': bool(ball.get('accepted')), 'floor_accepted': bool((g.get('floor_observation') or {}).get('accepted')),
            'grounded_refinement_accepted': (g.get('grounded_refinement') or {}).get('accepted'),
            'geometry_accepted': bool(geo.get('accepted')), 'observed_boundary_required': bool(geo.get('observed_boundary_required')),
            'point_reference_m': geo.get('point_reference_m'), 'complete_grounded_same_capture': complete_same_capture(geometry, now_s)}


def recovery_reacquire_head_allowed(*, now_s, capture_s, control):
    """Original BALL_REACQUIRE/GOAL_REFRESH current head-only permission, no body/kick."""
    c = control or {}
    permission = c.get('public_attention_head_permission') or {}
    owner = (c.get('public_attention_owner') or {}).get('owner')
    checks = {
        'fresh_capture': _num(now_s) and _num(capture_s) and 0 <= now_s-capture_s <= CAPTURE_MAX_AGE_S,
        'attention_owner': owner in ('GOAL_REFRESH', 'BALL_REACQUIRE'),
        'permission_current': permission.get('accepted') is True and permission.get('owner') == owner and permission.get('capture_s') == capture_s,
        'no_body_or_kick': permission.get('body_authorized') is False and permission.get('kick_authorized') is False
                           and c.get('velocity_body_xyw') == ZERO and c.get('kick_request') is False,
    }
    return {'allowed': all(checks.values()), 'checks': checks, 'owner': owner, 'capture_s': capture_s,
            'source': 'original attention owner head-only permission'}


def rear_path_from_controller(controller, frame, proprio, R, height, *, distance_m, fixed_target):
    """Current ToF hazard + original surveyed-prior 10 cm envelope to a fixed rear target."""
    from retreat_recovery import rear_path_authority
    t = proprio['timestamp_s']
    base = {'allowed': False, 'capture_s': t, 'layout': controller.config.layout, 'truth_used': False}
    pose = controller.pose
    if pose is None or not np.isfinite(pose).all():
        return {**base, 'reason': 'rear_path_pose_unavailable'}
    try:
        sample, hazard = controller._tof_sample(frame, proprio, R, height, None)
    except (ValueError, KeyError) as error:
        return {**base, 'reason': 'rear_path_tof_invalid:'+str(error)}
    if hazard:
        return {**base, 'reason': 'current_ToF_hazard'}
    try:
        return rear_path_authority(now_s=t, pose_map_xyyaw=np.asarray(pose, float).tolist(), goal_fix_s=controller.goal_time,
                                   distance_m=distance_m, tof_sample=sample, layout=controller.config.layout,
                                   image_sha256=controller.image_sha, fixed_target_map_xy=fixed_target)
    except (ValueError, KeyError, TypeError) as error:
        # Original map contract refused the snapshot (stale ToF, grid bounds...): no authority.
        return {**base, 'reason': 'rear_path_contract_error:'+str(error)}


def json_safe(value):
    """Archive rows must survive allow_nan=False; non-finite numbers become explicit strings."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return 'nonfinite:'+repr(value)
    return value


class HeadHandoffV121(HeadHandoffV65):
    def __init__(self, leg_fk, joint_names, *, params, task_limit_s):
        super().__init__(leg_fk, joint_names)
        self.recovery = RetreatRecovery(params, task_limit_s=task_limit_s)
        self.rejections = []          # every original HEAD_RETURN rejection, admitted or not
        self.recovery_events = []     # append-only lifecycle events
        self.kick_ever_started = False
        self.odometry_epoch = 0
        self.odometry_invalidations = []
        self.recovery_origin_epoch = None
        self.last_recovery_out = None
        self.last_recovery_odometry = None
        self.pending_events = []
        memory = self.adapter.memory
        original = memory.invalidate

        def counted_invalidate(reason, _original=original):
            # Any reset (gap, sequence gap, invalid/duplicate read, clock) ends
            # relative-odometry continuity for the active recovery permanently.
            self.odometry_epoch += 1
            self.odometry_invalidations.append({'reason': reason, 'epoch': self.odometry_epoch,
                                                'last_read_s': self.adapter.last_read_s})
            return _original(reason)
        memory.invalidate = counted_invalidate

    # -- ownership ---------------------------------------------------------
    @property
    def recovery_owns_body(self):
        return self.state == 'RECOVERY'

    @property
    def recovery_owns_head(self):
        return self.state == 'RECOVERY' and self.recovery.state in NEUTRAL_OWNED

    def _event(self, now_s, kind, **fields):
        event = {'timestamp_s': now_s, 'event': kind, **fields}
        self.recovery_events.append(event); self.pending_events.append(event)

    def recovery_odometry(self, now_s):
        m = self.adapter.memory
        read_s = None if m.previous is None else float(m.previous[0])
        continuous = self.recovery_origin_epoch is not None and self.odometry_epoch == self.recovery_origin_epoch
        fresh = read_s is not None and _num(now_s) and 0 <= now_s-read_s <= .02000001
        pose = None
        if continuous and fresh:
            p = m.predicted_pose(now_s)
            if np.isfinite(p).all():
                pose = [float(p[0]), float(p[1]), float(p[3])]
        return {'valid': pose is not None, 'pose_xyyaw': pose, 'epoch': self.odometry_epoch,
                'origin_epoch': self.recovery_origin_epoch, 'continuous_since_entry': continuous,
                'read_timestamp_s': read_s, 'fresh_read': fresh, 'source': 'public same-read BallMemory pose (relative, level frame)',
                'truth_used': False}

    # -- original policy with recovery admission ------------------------------
    def policy(self, *, now_s, packet, actual_sent_head=None, task_s=None):
        if self.state == 'RECOVERY':
            self.read(packet, now_s=now_s)    # unchanged same-read integration; never anchors
            odom = self.recovery_odometry(now_s)
            self.last_recovery_odometry = odom
            self.last_recovery_out = {**self.recovery.policy(now_s=now_s, packet=packet, odom_xyyaw=odom['pose_xyyaw'],
                                                           actual_head=actual_sent_head, odom_valid=odom['valid']),
                                      'timestamp_s': now_s}
            self._after_recovery(now_s)
            return self.status(now_s)
        before = self.state
        out = super().policy(now_s=now_s, packet=packet)
        if self.kick_start is not None:
            self.kick_ever_started = True
        if before == 'HEAD_RETURN' and self.state == 'FAILED':
            self._consider_recovery(now_s, task_s)
            out = self.status(now_s)
        return out

    def _consider_recovery(self, now_s, task_s):
        memory = self.adapter.memory.status(now_s)
        head = None if self.head is None else np.asarray(self.head, float)
        record = {'timestamp_s': now_s, 'task_s': task_s, 'original_reason': self.reason, 'attempt': self.attempts,
                  'kick_ever_started': self.kick_ever_started or self.kick_start is not None,
                  'memory': copy.deepcopy(memory), 'handoff_head': None if head is None else head.tolist(),
                  'recoveries_before': self.recovery.recoveries, 'whitelisted': self.reason in RECOVERABLE_REASONS}
        if head is None or head.shape != (4,) or not np.isfinite(head).all() or np.max(np.abs(head)) > 1e-9:
            ok, why = False, 'handoff_head_not_neutral_at_rejection'
        else:
            ok, why = self.recovery.on_handoff_rejection(now_s=now_s, task_s=task_s, reason=self.reason,
                                                        kick_started=record['kick_ever_started'], memory=memory)
        record.update(admitted=ok, admission=why)
        self.rejections.append(record)
        if not ok:
            self._event(now_s, 'original_rejection_terminal', **record)
            return
        self.state = 'RECOVERY'; self.head = np.zeros(4); self.quiet_start = None
        self.recovery_origin_epoch = self.odometry_epoch
        self.last_recovery_out = None
        self._event(now_s, 'recovery_entry', **record, plan_distance_m=self.recovery.distance,
                    odometry_origin_epoch=self.recovery_origin_epoch)

    def _after_recovery(self, now_s):
        r = self.recovery
        if r.state == 'FAILED':
            self.state = 'FAILED'; self.reason = 'recovery_failed:'+str(r.reason)
            self._event(now_s, 'recovery_failed', reason=r.reason, status=r.status())
        elif r.state == 'RELEASED':
            # Return to the original APPROACH owner. Attempts, rejection history
            # and kick-ever-started are preserved; no kick authority is granted.
            self.state = 'APPROACH'; self.reason = None; self.head = None; self.started = None; self.last_t = None
            self.quiet_start = None; self.initial_capture_s = None; self.routed_from_fine = False
            self.recovery_origin_epoch = None
            self._event(now_s, 'recovery_released_to_original_approach', attempts=self.attempts,
                        recoveries=r.recoveries, rejections=len(self.rejections))

    # -- captures ----------------------------------------------------------------
    def capture(self, *, now_s, geometry, head_offsets, candidate, goal_heading, goal_fix_s):
        if self.state != 'RECOVERY':
            return super().capture(now_s=now_s, geometry=geometry, head_offsets=head_offsets, candidate=candidate,
                                   goal_heading=goal_heading, goal_fix_s=goal_fix_s)
        if _num(goal_heading) and _num(goal_fix_s) and 0 <= now_s-goal_fix_s <= .5:
            self.last_goal_heading = float(goal_heading); self.last_goal_s = float(goal_fix_s)
        self.last_anchor_attempt = {'timestamp_s': now_s, 'anchored': False, 'reason': 'not_reacquire'}
        if self.recovery.state != 'REACQUIRE':
            return self.status(now_s)
        if not complete_same_capture(geometry, now_s):
            self.last_anchor_attempt['reason'] = 'no_current_complete_grounded_ball'
            return self.status(now_s)
        read_s = self.adapter.last_read_s
        if read_s is None or not 0 <= now_s-read_s <= .02000001:
            self.last_anchor_attempt['reason'] = 'no_same_read_packet'
            return self.status(now_s)
        g = geometry['ball']['geometry']
        try:
            self.adapter.memory.observe_complete_ball(timestamp_s=now_s, point_body_m=g['point_reference_m'],
                estimated_range_uncertainty_m=g['estimated_range_uncertainty_m'], complete=bool(g['observed_boundary_required']),
                geometry_accepted=bool(g['accepted']), capture_read_age_s=now_s-read_s)
            self.last_anchor_attempt = {'timestamp_s': now_s, 'anchored': True, 'reason': 'reacquire_current_complete_grounded_ball'}
        except (ValueError, KeyError, TypeError) as error:
            self.last_anchor_attempt['reason'] = 'anchor_rejected:'+str(error)
        return self.status(now_s)

    def recovery_capture(self, *, now_s, task_s, guard_safe, complete, rear_path, geometry=None, control=None, controller_pose=None, goal_fix_s=None,
                         approach_counters=None):
        """Call once per actual capture after guards; returns the archive row (or None)."""
        if self.state != 'RECOVERY':
            return None
        before = self.recovery.state
        memory = self.adapter.memory.status(now_s)
        anchored_now = bool(memory['memory_available'] and memory['ball_capture_timestamp_s'] == now_s)
        fresh_ball = {'complete': complete is True, 'capture_s': now_s, 'memory_anchored_now': anchored_now,
                      'point_level_m': memory['point_level_m'] if anchored_now else None}
        odom = self.recovery_odometry(now_s)
        self.recovery.capture(now_s=now_s, capture_s=now_s, guard_safe=guard_safe, rear_path=rear_path,
                              odom_xyyaw=odom['pose_xyyaw'], fresh_ball=fresh_ball, task_s=task_s,
                              odom_continuous=odom['continuous_since_entry'])
        after = self.recovery.state
        self._after_recovery(now_s)
        c = control or {}
        row = {'kind': 'capture', 'timestamp_s': now_s, 'task_s': task_s, 'recovery_before': before, 'recovery_after': after,
               'handoff_state': self.state, 'decision': self.recovery.decision, 'guard_safe': guard_safe is True,
               'complete': complete is True, 'geometry': geometry_summary(geometry, now_s), 'anchor_attempt': getattr(self, 'last_anchor_attempt', None),
               'memory': memory, 'memory_anchored_now': anchored_now, 'odometry': odom, 'rear_path': rear_path,
               'controller_pose_map_xyyaw': controller_pose, 'goal_fix_s': goal_fix_s, 'approach_counters': approach_counters,
               'attempts': self.attempts,
               'attention_owner': c.get('public_attention_owner'), 'attention_head_permission': c.get('public_attention_head_permission'),
               'released': after == 'RELEASED', 'withdrawn_from_retreat': before == 'RETREAT' and after != 'RETREAT',
               'recovery': self.recovery.status(), 'events': self._take_events(), 'kick_authorized': False, 'truth_used': False}
        return json_safe(row)

    def recovery_capture_unavailable(self, *, now_s, task_s, reason):
        """Capture without a usable navigation evaluation: no guard, no path, no anchor."""
        if self.state != 'RECOVERY':
            return None
        return self.recovery_capture(now_s=now_s, task_s=task_s, guard_safe=False, complete=False,
                                     rear_path={'allowed': False, 'capture_s': now_s, 'reason': reason, 'truth_used': False})

    # -- 50 Hz command composition ---------------------------------------------------
    def recovery_velocity(self, now_s):
        """None when recovery does not own the body; otherwise this tick's requested command."""
        if self.state != 'RECOVERY':
            return None
        out = self.last_recovery_out
        if out is None or out.get('timestamp_s') != now_s:
            return list(ZERO)
        return list(out['velocity'])

    def recovery_after_head(self, *, now_s, task_s, requested_velocity, velocity, requested_head, sent_head,
                            head_feedback, head_permission, packet):
        """Final withdrawal check after head feedback; returns (velocity, archive row or None)."""
        active = requested_velocity is not None or self.pending_events
        if not active:
            return velocity, None
        sent = np.asarray(sent_head, float)
        v = list(map(float, velocity))
        withdrawn = None
        out = self.last_recovery_out or {}
        current = out.get('timestamp_s') == now_s
        expected = list(map(float, out['velocity'])) if current else list(ZERO)
        p = self.recovery.params
        expected_ok = bool(current and out.get('motion_tick') and out.get('decision') == 'retreat_command'
                           and expected[0] == p.retreat_vx and expected[1] == 0. and abs(expected[2]) <= .5)
        if self.state == 'RECOVERY' and (any(v) or any(expected)):
            if not expected_ok or v != expected:
                withdrawn = 'unapproved_recovery_command'
            elif sent.shape != (4,) or not np.isfinite(sent).all() or np.max(np.abs(sent)) > 1e-9:
                withdrawn = 'retreat_withdrawn_actual_sent_head_not_neutral'
            elif (head_feedback or {}).get('body_zero_for_withdrawal'):
                withdrawn = 'retreat_withdrawn_by_head_feedback'
            if withdrawn:
                self.recovery.fail(withdrawn); self._after_recovery(now_s); v = list(ZERO)
        row = {'kind': 'policy', 'timestamp_s': now_s, 'task_s': task_s, 'handoff_state': self.state,
               'handoff_reason': self.reason, 'recovery_state': self.recovery.state, 'decision': self.recovery.decision,
               'requested_velocity': requested_velocity, 'expected_helper_command': expected if current else None,
               'actual_velocity_command': v, 'withdrawn': withdrawn,
               'helper': out.get('helper') if current else None, 'motion_tick': bool(current and out.get('motion_tick')),
               'requested_head_nominal': list(map(float, requested_head)), 'actual_head_command': sent.tolist(),
               'head_permission': head_permission, 'head_feedback_withdrawal': (head_feedback or {}).get('body_zero_for_withdrawal'),
               'odometry': self.last_recovery_odometry if out.get('timestamp_s') == now_s else None,
               'retreat_relative': {k: out.get(k) for k in ('retreat_dx_m', 'retreat_dy_m', 'retreat_dyaw_rad', 'retreat_elapsed_s')} if out.get('timestamp_s') == now_s else None,
               'packet': None if packet is None else {k: packet.get(k) for k in ('seq', 'timestamp_s', 'read_timestamp_s', 'encoder_read_timestamp_s',
                                                                                  'orientation_valid', 'gyro_body_rad_s', 'gravity_body')},
               'recovery': self.recovery.status(), 'attempts': self.attempts, 'kick_ever_started': self.kick_ever_started,
               'events': self._take_events(), 'kick_authorized': False, 'truth_used': False}
        return v, json_safe(row)

    def _take_events(self):
        events, self.pending_events = self.pending_events, []
        return events

    # -- status ------------------------------------------------------------------------
    def status(self, now_s):
        out = super().status(now_s)
        if self.state == 'RECOVERY':
            out['head_offsets'] = [0., 0., 0., 0.] if self.recovery.state in NEUTRAL_OWNED else None
        out['recovery'] = {'state': self.recovery.state, 'reason': self.recovery.reason, 'decision': self.recovery.decision,
                           'recoveries': self.recovery.recoveries, 'rejections': len(self.rejections),
                           'kick_ever_started': self.kick_ever_started, 'owns_body': self.recovery_owns_body,
                           'owns_head': self.recovery_owns_head, 'odometry_epoch': self.odometry_epoch,
                           'provisional_uncalibrated_motion': not self.recovery.params.calibrated, 'kick_authorized': False}
        return out

    def recovery_report(self):
        return json_safe({'recovery': self.recovery.status(full=True), 'rejections': copy.deepcopy(self.rejections),
                'events': copy.deepcopy(self.recovery_events), 'odometry_invalidations': copy.deepcopy(self.odometry_invalidations),
                'attempts': self.attempts, 'kick_ever_started': self.kick_ever_started, 'kick_authorized': False, 'truth_used': False})
