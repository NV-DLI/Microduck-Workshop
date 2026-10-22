"""v121 bounded pre-kick retreat/reacquire supervisor (pure logic, no kick authority).

v121 replaces v120's constant-command variable-duration retreat (its open-loop
Stage A failed and that approval must never be used) by an EXACT fixed-duration
public-heading-feedback helper (heading_feedback.BackwardHeadingFeedback):
  * a fresh helper per recovery, start_step 0, move_steps = duration*50 (1.6 s -> 80);
  * called only on real 50 Hz RETREAT policy ticks with the current delivered packet;
  * exactly 80 active ticks with the helper command [vx, 0, wz], then zero; no
    post-stop authority; the stored retreat start time is the first motion tick;
  * helper fault, or any distance/yaw/lateral/forward/guard/head/odometry/onset stop
    before the 80 ticks complete, is a TERMINAL failure (never "motion complete");
  * at the normal stop, public backward travel < .10 m fails; otherwise SETTLE.
Public same-read odometry only; no simulator truth. Motion parameters come only
from a verified root approval binding Stage A and Stage B to the exact helper,
duration, offset and vx; RecoveryParams.provisional() is CPU-test-only.

Lifecycle (one episode, original task clock, scorer untouched):
  IDLE -> HOLD -> AUTHORIZE -> RETREAT(80 helper ticks) -> SETTLE -> REACQUIRE -> RELEASED
  any violation -> FAILED (terminal, body zero).
"""
from dataclasses import dataclass, asdict
import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import numpy as np

HELPER_PATH = Path(__file__).resolve().with_name('heading_feedback.py')
RECOVERABLE_REASONS = frozenset({'propagated_point_outside_training_box', 'goal_heading_outside_development_gate'})
MAX_RECOVERIES = 3
TASK_LIMIT_S = 180.
RESERVE_S = 30.
POLICY_DT_S = .02
STAGE_A_VX = -.35
STAGE_A_DURATION_S = 1.6
STAGE_A_OFFSETS_RAD = (.10, .20)
PLAN_DISTANCE_M = .28                # newly declared fixed rear plan: .24 max travel + original .04 post-stop
MAX_BACKWARD_M = .24                 # hard public bound while moving
MIN_BACKWARD_M = .10                 # public minimum at the normal stop
FORWARD_LIMIT_M = .01
ONSET_DEADLINE_S, ONSET_MIN_M = 1.0, .01
YAW_LIMIT_RAD, LATERAL_LIMIT_M = .25, .05
QUIET_S, GYRO_QUIET, QD_RMS_QUIET, TILT_LIMIT = .5, .1, .25, .35
CAPTURE_MAX_AGE_S, POLICY_GAP_S, PACKET_MAX_AGE_S = .2, .025, .08
AUTHORIZE_TIMEOUT_S = .5
HOLD_DEADLINE_S = SETTLE_DEADLINE_S = 3.
REACQUIRE_DEADLINE_S = 8.
RELEASE_X, RELEASE_ABS_Y = (.20, .38), .10
ENTRY_X, ENTRY_ABS_Y = (.06, .14), .10
MEMORY_TTL_S = 4.5
REAR_PATH_RADIUS_M = .10
GOAL_FIX_MAX_AGE_S = 8.
WZ_MAX = .5


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class RecoveryParams:
    """Fixed motion parameters; production values only from a verified root approval."""
    retreat_vx: float
    duration_s: float
    move_steps: int
    heading_offset_rad: float
    helper_sha256: str
    calibrated: bool
    source: str

    @classmethod
    def provisional(cls, heading_offset_rad=.10):
        return cls(STAGE_A_VX, STAGE_A_DURATION_S, round(STAGE_A_DURATION_S/POLICY_DT_S), float(heading_offset_rad),
                   sha256(HELPER_PATH), False, 'PROVISIONAL_uncalibrated_cpu_test_only')

    @property
    def command_form(self):
        return (float(self.retreat_vx), 0., 'helper_wz')

    def as_dict(self):
        return asdict(self)


def load_helper_module(expected_sha256):
    actual = sha256(HELPER_PATH)
    if actual != expected_sha256:
        raise ValueError('heading feedback helper differs from the approved helper hash')
    spec = importlib.util.spec_from_file_location('v121_recovery_heading_feedback_'+actual[:16], HELPER_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def make_helper(params):
    """Fresh single-world helper: start_step 0, move_steps from the approved duration."""
    module = load_helper_module(params.helper_sha256)
    cls = module.BackwardHeadingFeedback
    names = set(inspect.signature(cls.__init__).parameters)
    kwargs = dict(vx=np.array([params.retreat_vx]), start_step=np.array([0]), move_steps=np.array([params.move_steps]))
    offset_name = next((n for n in ('desired_heading', 'desired_heading_rad', 'heading_offset', 'heading_offset_rad', 'target_heading') if n in names), None)
    if offset_name is None:
        raise ValueError('approved helper has no explicit heading offset input')
    kwargs[offset_name] = np.array([params.heading_offset_rad])
    return cls(**kwargs), offset_name


def _finite(*values):
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)


def rear_path_authority(*, now_s, pose_map_xyyaw, goal_fix_s, distance_m, tof_sample, layout, image_sha256, fixed_target_map_xy=None):
    """Original map contract and 10 cm envelope; rear cells FREE only via the surveyed empty prior."""
    from nav2_bridge import make_request
    from nav2_service.map_contract import snapshot, check_path
    pose = np.asarray(pose_map_xyyaw, float)
    result = dict(capture_s=now_s, allowed=False, layout=layout, truth_used=False, rear_sensor_coverage=False,
                  rear_provenance='original_surveyed_empty_field_prior_not_rear_sensing',
                  plan='fixed_declared_plan_distance_not_traversed_history',
                  pose_map_xyyaw=pose.tolist() if pose.shape == (3,) else None, goal_fix_s=goal_fix_s,
                  distance_m=distance_m, radius_m=REAR_PATH_RADIUS_M)
    if layout != 'empty' or pose.shape != (3,) or not np.isfinite(pose).all() or not _finite(now_s, goal_fix_s, distance_m) or not 0 <= now_s-goal_fix_s <= GOAL_FIX_MAX_AGE_S:
        result['reason'] = 'rear_path_inputs'
        return result
    target = pose[:2]-distance_m*np.array([math.cos(pose[2]), math.sin(pose[2])])
    if fixed_target_map_xy is not None:
        target = np.asarray(fixed_target_map_xy, float)
        if target.shape != (2,) or not np.isfinite(target).all():
            result['reason'] = 'fixed_rear_target_invalid'
            return result
    case, prior = make_request(time_s=now_s, pose_map_xyyaw=pose.tolist(), velocity_body_xyw=[0., 0., 0.], goal_map_xy=target.tolist(),
                               tof_sample=tof_sample, layout=layout, image_sha256=image_sha256, localization_capture_s=goal_fix_s)
    gate = check_path([pose[:2].tolist(), target.tolist()], snapshot(case, prior), radius=REAR_PATH_RADIUS_M)
    result.update(allowed=bool(gate['accepted']), reason=gate['reason'], envelope=gate, target_map_xy=target.tolist(),
                  fixed_target_used=fixed_target_map_xy is not None, source_capture_s=tof_sample['capture_s'],
                  map_request={'case': case, 'provenance': prior})
    return result


def helper_inputs(packet):
    """Current delivered packet -> helper's public inputs (no truth, no prediction)."""
    p = packet if isinstance(packet, dict) else {}

    def arr(key, n):
        try:
            a = np.asarray(p.get(key), float)
        except (TypeError, ValueError):
            a = np.full(n, np.nan)
        return a if a.shape == (n,) else np.full(n, np.nan)
    seq = p.get('seq')
    clock = lambda k: float(p[k]) if _finite(p.get(k)) else math.nan
    return dict(gyro_body=arr('gyro_body_rad_s', 3)[None], gravity_body=arr('gravity_body', 3)[None],
                seq=np.array([seq if isinstance(seq, int) and not isinstance(seq, bool) else -1]),
                sample_s=np.array([clock('timestamp_s')]), read_s=np.array([clock('read_timestamp_s')]),
                encoder_read_s=np.array([clock('encoder_read_timestamp_s')]), valid=np.array([p.get('orientation_valid') is True]))


class RetreatRecovery:
    def __init__(self, params=None, *, task_limit_s=TASK_LIMIT_S):
        if params is None:
            params = RecoveryParams.provisional()
        if not isinstance(params, RecoveryParams):
            raise TypeError('RecoveryParams required')
        if not (params.retreat_vx == STAGE_A_VX and params.duration_s == STAGE_A_DURATION_S
                and params.move_steps == round(params.duration_s/POLICY_DT_S) and params.heading_offset_rad in STAGE_A_OFFSETS_RAD):
            raise ValueError('only the Stage A fixed motion (vx -.35, 1.6 s / 80 ticks, offset .10 or .20) is admissible')
        load_helper_module(params.helper_sha256)
        if not _finite(task_limit_s) or not 0 < task_limit_s <= TASK_LIMIT_S:
            raise ValueError('Task limit must be within the original 180 s')
        self.params = params; self.task_limit_s = float(task_limit_s)
        self.state = 'IDLE'; self.reason = None; self.recoveries = 0; self.history = []
        self.decision = 'idle'
        self._clear()

    def _clear(self):
        self.started = None; self.last_t = None; self.quiet_start = None; self.distance = None
        self.last_capture = None; self.task_origin = None; self.fixed_target = None
        self.origin = None; self.retreat_start = None; self.settle_start = None; self.reacquire_start = None; self.authorize_start = None
        self.authorized_at = None; self.last_quiet = False; self.last_head_neutral = False
        self.helper = None; self.helper_offset_input = None; self.motion_index = 0; self.helper_diag = None; self.last_helper_inputs = None
        self.stop_rel = None

    def fail(self, reason):
        self.state = 'FAILED'; self.reason = reason; self.decision = 'fail:'+reason; self.history.append(('FAILED', reason))

    # -- entry -----------------------------------------------------------
    def on_handoff_rejection(self, *, now_s, task_s, reason, kick_started, memory):
        if self.state not in ('IDLE', 'RELEASED'):
            return False, 'recovery_already_active'
        if kick_started:
            return False, 'never_after_kick_onset'
        if reason not in RECOVERABLE_REASONS:
            return False, 'reason_not_recoverable'
        if self.recoveries >= MAX_RECOVERIES:
            return False, 'recovery_budget_exhausted'
        if not _finite(now_s, task_s) or not 0 <= task_s <= now_s or self.task_limit_s-task_s < RESERVE_S:
            return False, 'insufficient_original_task_time'
        if not (isinstance(memory, dict) and memory.get('memory_available') is True):
            return False, 'public_memory_unavailable_at_rejection'
        if not (_finite(memory.get('ball_capture_timestamp_s'), memory.get('sensor_read_timestamp_s'))
                and 0 <= now_s-memory['ball_capture_timestamp_s'] <= MEMORY_TTL_S
                and 0 <= now_s-memory['sensor_read_timestamp_s'] <= .02000001):
            return False, 'invalid_entry_memory_clock'
        try:
            p = np.asarray(memory['point_level_m'], float)
            if p.shape != (3,) or not np.isfinite(p).all() or not (ENTRY_X[0] < p[0] <= ENTRY_X[1] and abs(p[1]) <= ENTRY_ABS_Y):
                return False, 'recovery_entry_outside_near_domain'
        except (ValueError, KeyError, TypeError):
            return False, 'public_memory_point_invalid'
        self._clear()
        self.task_origin = now_s-task_s
        self.distance = PLAN_DISTANCE_M
        self.recoveries += 1; self.state = 'HOLD'; self.reason = None; self.decision = 'entry_hold'
        self.started = self.last_t = now_s; self.quiet_start = None
        self.history.append(('HOLD', reason, now_s, self.distance))
        return True, 'recovery_hold'

    # -- 50 Hz policy tick -------------------------------------------------
    def _packet_ok(self, now_s, packet):
        try:
            return bool(isinstance(packet, dict) and packet.get('orientation_valid') is True
                        and _finite(packet.get('read_timestamp_s'), packet.get('timestamp_s'))
                        and 0 <= now_s-packet['read_timestamp_s'] <= PACKET_MAX_AGE_S
                        and 0 <= now_s-packet['timestamp_s'] <= PACKET_MAX_AGE_S
                        and all(np.asarray(packet[k]).shape == (n,) and np.isfinite(np.asarray(packet[k], float)).all()
                                for k, n in [('gravity_body', 3), ('gyro_body_rad_s', 3), ('encoder_qd_rad_s', 14)]))
        except (KeyError, TypeError, ValueError):
            return False

    def _quiet(self, packet):
        g = np.asarray(packet['gravity_body'], float); w = np.asarray(packet['gyro_body_rad_s'], float); qd = np.asarray(packet['encoder_qd_rad_s'], float)
        tilt = math.atan2(float(np.linalg.norm(g[:2])), -float(g[2]))
        return bool(np.linalg.norm(w) <= GYRO_QUIET and math.sqrt(float(np.mean(qd**2))) <= QD_RMS_QUIET and tilt <= TILT_LIMIT), tilt

    def _done(self, out):
        out['state'] = self.state; out['decision'] = self.decision
        return out

    def policy(self, *, now_s, packet, odom_xyyaw, actual_head, odom_valid):
        """Body velocity for this tick: zero, or the helper command on a RETREAT motion tick."""
        out = dict(state=self.state, velocity=[0., 0., 0.], head_neutral=True, kick_authorized=False, truth_used=False,
                   motion_tick=False, helper=None)
        if self.state in ('IDLE', 'RELEASED', 'FAILED'):
            self.decision = 'inactive:'+self.state.lower()
            return self._done(out)
        if not _finite(now_s) or now_s-self.task_origin >= self.task_limit_s:
            self.fail('original_180s_limit'); return self._done(out)
        if self.state == 'HOLD' and now_s-self.started > HOLD_DEADLINE_S:
            self.fail('recovery_hold_deadline'); return self._done(out)
        if self.state == 'REACQUIRE' and now_s-self.reacquire_start > REACQUIRE_DEADLINE_S:
            self.fail('recovery_reacquire_deadline'); return self._done(out)
        if self.state == 'RETREAT' and (self.last_capture is None or not 0 <= now_s-self.last_capture <= CAPTURE_MAX_AGE_S):
            self.fail('retreat_capture_watchdog'); return self._done(out)
        if odom_valid is not True:
            self.fail('recovery_public_odometry_invalid'); return self._done(out)
        try:
            head = np.asarray(actual_head, float)
        except (TypeError, ValueError):
            head = np.full(1, np.nan)
        if head.shape != (4,) or not np.isfinite(head).all():
            self.fail('recovery_actual_head_invalid'); return self._done(out)
        if self.state == 'AUTHORIZE' and now_s-self.authorize_start > AUTHORIZE_TIMEOUT_S:
            self.fail('retreat_authorization_capture_timeout'); return self._done(out)
        if not 0 <= now_s-self.last_t <= POLICY_GAP_S:
            self.fail('recovery_policy_clock_gap'); return self._done(out)
        self.last_t = now_s
        if not self._packet_ok(now_s, packet):
            self.fail('recovery_public_sensor_unavailable'); return self._done(out)
        quiet, tilt = self._quiet(packet)
        neutral = bool(max(abs(head)) <= 1e-9)
        self.last_quiet, self.last_head_neutral = quiet, neutral
        quiet = quiet and neutral
        if self.state == 'RETREAT' and not neutral:
            self.fail('retreat_actual_head_not_neutral'); return self._done(out)
        if tilt > TILT_LIMIT:
            self.fail('recovery_tilt_limit'); return self._done(out)
        if self.state == 'AUTHORIZE':
            if not quiet:
                self.fail('recovery_authorize_quiet_or_neutral_lost'); return self._done(out)
            self.decision = 'authorize_wait_same_capture_rear_path'
            return self._done(out)
        if self.state == 'REACQUIRE':
            out['head_neutral'] = False
            self.decision = 'reacquire_wait_current_complete_ball'
            return self._done(out)
        if self.state in ('HOLD', 'SETTLE'):
            if self.state == 'SETTLE':
                rel = self._relative_or_none(odom_xyyaw)
                if rel is None:
                    self.fail('recovery_public_odometry_invalid'); return self._done(out)
                if -rel[0] > PLAN_DISTANCE_M or rel[0] > FORWARD_LIMIT_M:
                    self.fail('settle_public_travel_outside_planned_path'); return self._done(out)
            self.quiet_start = (self.quiet_start if self.quiet_start is not None else now_s) if quiet else None
            self.decision = self.state.lower()+('_quiet' if quiet else '_wait_quiet_neutral')
            if self.quiet_start is not None and now_s-self.quiet_start >= QUIET_S-1e-8:
                self.state = 'AUTHORIZE' if self.state == 'HOLD' else 'REACQUIRE'
                if self.state == 'REACQUIRE':
                    self.reacquire_start = now_s
                else:
                    self.authorize_start = now_s
                self.quiet_start = None
                self.decision = 'enter_'+self.state.lower()
                self.history.append((self.state, now_s))
            elif self.state == 'SETTLE' and now_s-self.settle_start > SETTLE_DEADLINE_S:
                self.fail('recovery_settle_deadline')
            out['head_neutral'] = self.state != 'REACQUIRE'
            return self._done(out)
        return self._retreat_tick(now_s, packet, odom_xyyaw, out)

    def _relative_or_none(self, odom_xyyaw):
        try:
            o = np.asarray(odom_xyyaw, float)
        except (TypeError, ValueError):
            return None
        if o.shape != (3,) or not np.isfinite(o).all() or self.origin is None:
            return None
        return self._relative(o)

    def _retreat_tick(self, now_s, packet, odom_xyyaw, out):
        p = self.params
        if self.motion_index == 0 and self.retreat_start is None:
            self.retreat_start = now_s          # synchronized to the first motion tick
        rel = self._relative_or_none(odom_xyyaw)
        if rel is None:
            self.fail('recovery_public_odometry_invalid'); return self._done(out)
        dx, dy, dyaw = rel
        elapsed = now_s-self.retreat_start
        out.update(retreat_dx_m=dx, retreat_dy_m=dy, retreat_dyaw_rad=dyaw, retreat_elapsed_s=elapsed)
        # Every safety stop before the fixed window completes is terminal.
        if dx > FORWARD_LIMIT_M:
            self.fail('retreat_forward_motion_toward_ball'); return self._done(out)
        if -dx > MAX_BACKWARD_M:
            self.fail('retreat_public_backward_exceeds_0p24'); return self._done(out)
        if abs(dyaw) > YAW_LIMIT_RAD or abs(dy) > LATERAL_LIMIT_M:
            self.fail('retreat_drift_limit'); return self._done(out)
        if elapsed >= ONSET_DEADLINE_S-1e-9 and -dx < ONSET_MIN_M:
            self.fail('retreat_dead_zone_no_onset'); return self._done(out)
        if self.motion_index >= p.move_steps:
            # Normal stop after exactly move_steps helper ticks: zero, no post-stop authority.
            self.stop_rel = rel
            self.history.append(('STOP', 'fixed_duration_complete', now_s, self.motion_index, -dx))
            if -dx < MIN_BACKWARD_M:
                self.fail('retreat_public_backward_below_0p10'); return self._done(out)
            self.state = 'SETTLE'; self.settle_start = now_s; self.quiet_start = None
            self.decision = 'stop:fixed_duration_complete'
            return self._done(out)
        inputs = helper_inputs(packet)
        vel, diag = self.helper.update(step=self.motion_index, now_s=now_s, **inputs)
        self.last_helper_inputs = {k: np.asarray(v).tolist() for k, v in inputs.items()}
        self.helper_diag = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in diag.items()}
        index = self.motion_index
        self.motion_index += 1
        out.update(motion_tick=True, helper={'index': index, 'inputs': self.last_helper_inputs, 'diag': self.helper_diag,
                                             'offset_input': self.helper_offset_input})
        fault = int(np.asarray(diag['fault_code']).reshape(-1)[0])
        v = np.asarray(vel, float).reshape(-1, 3)[0]
        if fault != 0 or not np.isfinite(v).all():
            self.fail('retreat_helper_fault:'+str(fault)); return self._done(out)
        if not (v[0] == p.retreat_vx and v[1] == 0. and abs(v[2]) <= WZ_MAX):
            self.fail('retreat_helper_command_outside_approved_form'); return self._done(out)
        out['velocity'] = [float(v[0]), 0., float(v[2])]
        self.decision = 'retreat_command'
        return self._done(out)

    def _relative(self, o):
        d = o[:2]-self.origin[:2]; c, s = math.cos(self.origin[2]), math.sin(self.origin[2])
        dyaw = math.atan2(math.sin(o[2]-self.origin[2]), math.cos(o[2]-self.origin[2]))
        return float(c*d[0]+s*d[1]), float(-s*d[0]+c*d[1]), dyaw

    # -- 15 Hz capture ------------------------------------------------------
    def capture(self, *, now_s, capture_s, guard_safe, rear_path, odom_xyyaw, fresh_ball=None, task_s=None, odom_continuous=True):
        fresh = _finite(now_s, capture_s) and 0 <= now_s-capture_s <= CAPTURE_MAX_AGE_S
        path_ok = bool(isinstance(rear_path, dict) and rear_path.get('allowed') is True and rear_path.get('capture_s') == capture_s and rear_path.get('layout') == 'empty'
                       and rear_path.get('map_request') is not None and rear_path.get('distance_m') == PLAN_DISTANCE_M
                       and (self.fixed_target is None or rear_path.get('target_map_xy') == self.fixed_target))
        if self.state in ('HOLD', 'SETTLE', 'IDLE', 'RELEASED', 'FAILED'):
            self.decision = 'capture_ignored_in_'+self.state.lower()
            return self.state
        if task_s is not None and (not _finite(task_s) or task_s >= self.task_limit_s):
            self.fail('original_180s_limit'); return self.state
        if self.state == 'AUTHORIZE':
            if not fresh:
                self.decision = 'authorize_capture_not_fresh'
                return self.state
            if guard_safe is not True:
                self.fail('retreat_requires_body_guard_safe_capture'); return self.state
            if not path_ok:
                self.fail('retreat_rear_path_not_authorized:'+str((rear_path or {}).get('reason'))); return self.state
            if not (self.last_quiet and self.last_head_neutral):
                self.fail('recovery_authorize_quiet_or_neutral_lost'); return self.state
            try:
                o = np.asarray(odom_xyyaw, float)
            except (TypeError, ValueError):
                o = np.full(1, np.nan)
            if o.shape != (3,) or not np.isfinite(o).all():
                if odom_continuous is True:
                    self.decision = 'authorize_wait_same_read_odometry'
                    return self.state
                self.fail('recovery_public_odometry_invalid'); return self.state
            target = np.asarray(rear_path.get('target_map_xy'), float)
            if target.shape != (2,) or not np.isfinite(target).all():
                self.fail('fixed_rear_target_invalid'); return self.state
            try:
                self.helper, self.helper_offset_input = make_helper(self.params)   # fresh helper per recovery
            except (ValueError, TypeError, AttributeError) as error:
                self.fail('retreat_helper_unavailable:'+str(error)); return self.state
            self.motion_index = 0; self.retreat_start = None
            self.fixed_target = target.tolist(); self.last_capture = capture_s
            self.origin = o.copy(); self.authorized_at = now_s; self.state = 'RETREAT'
            self.decision = 'retreat_authorized_same_capture'
            self.history.append(('RETREAT', now_s, rear_path.get('target_map_xy')))
            return self.state
        if self.state == 'RETREAT':
            if not fresh:
                self.fail('retreat_capture_not_fresh')
            elif guard_safe is not True:
                self.fail('retreat_current_body_guard_rejected')
            elif not path_ok:
                self.fail('retreat_rear_path_recheck_failed')
            else:
                self.last_capture = capture_s; self.decision = 'retreat_capture_rechecked'
            return self.state
        if self.state == 'REACQUIRE':
            if now_s-self.reacquire_start > REACQUIRE_DEADLINE_S:
                self.fail('recovery_reacquire_deadline'); return self.state
            fb = fresh_ball or {}
            p = fb.get('point_level_m')
            try:
                pv = np.asarray(p, float) if p is not None else np.full(1, np.nan)
            except (TypeError, ValueError):
                pv = np.full(1, np.nan)
            ok = bool(fresh and guard_safe is True and fb.get('complete') is True and fb.get('memory_anchored_now') is True and fb.get('capture_s') == capture_s
                      and pv.shape == (3,) and np.isfinite(pv).all()
                      and RELEASE_X[0] <= pv[0] <= RELEASE_X[1] and abs(pv[1]) <= RELEASE_ABS_Y)
            if ok:
                self.state = 'RELEASED'; self.decision = 'release_current_complete_anchor'
                self.history.append(('RELEASED', now_s, pv.tolist())); self._clear()
            else:
                self.decision = 'reacquire_capture_not_releasable'
            return self.state
        return self.state

    def status(self, *, full=False):
        out = dict(state=self.state, reason=self.reason, decision=self.decision, recoveries=self.recoveries, max_recoveries=MAX_RECOVERIES,
                   plan_distance_m=self.distance, fixed_target_map_xy=self.fixed_target, last_path_capture_s=self.last_capture,
                   origin_odom_xyyaw=None if self.origin is None else self.origin.tolist(), authorized_at_s=self.authorized_at,
                   retreat_start_s=self.retreat_start, motion_ticks=self.motion_index,
                   settle_start_s=self.settle_start, reacquire_start_s=self.reacquire_start, authorize_start_s=self.authorize_start,
                   hold_start_s=self.started, task_origin_s=self.task_origin, task_limit_s=self.task_limit_s,
                   stop_relative_dx_dy_dyaw=None if self.stop_rel is None else list(self.stop_rel),
                   parameters=self.params.as_dict(), provisional_uncalibrated_motion=not self.params.calibrated, kick_authorized=False,
                   memory_ttl_changed=False, scorer_or_simulator_reset=False, truth_used=False)
        if full:
            out['history'] = [list(x) for x in self.history]
        return out
