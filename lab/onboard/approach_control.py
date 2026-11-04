"""Continuous straight approach with public feedback, predictive braking and short reverse turns."""
import hashlib
import inspect
import math
import textwrap
import numpy as np

PARAMETERS = dict(lookahead_m=.18, coarse_duration_s=30., coarse_stop_m=.115,
                  track_duration_s=30., prealign_rad=math.radians(10),
                  goal_age_s=2., memory_age_s=6.5, parking_angle_rad=math.radians(20),
                  parking_vx=-.20, parking_wz=.5, parking_duration_s=.8,
                  transfer_x_m=[.13,.28], transfer_y_m=[-.10,.08])
_installed = None
_policy_source = None
_base_sources = {}

def install_duration_support(navigator_class):
    base = next(c for c in navigator_class.__mro__
                if 'step' in c.__dict__ and c.__name__ == 'BetaNavigator')
    if base in _base_sources:
        return
    source = textwrap.dedent(inspect.getsource(base.step))
    source = replace(source, 'if t - self.first_time > self.config.duration_s:',
                     'if self.config.duration_s > 0 and t - self.first_time > self.config.duration_s:')
    _base_sources[base] = source
    base.step = recompile(base.step, source)


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise RuntimeError('Navigation source contract changed: ' + old)
    return source.replace(old, new)


def recompile(function, source):
    scope = dict(function.__globals__)
    exec(compile(source, '<onboard-straight-approach>', 'exec'), scope)
    return scope[function.__name__]


def fine_policy_source(source):
    if _installed is None:
        return source
    source = replace(source, 'p[0] <= 0.36 or t - self.step_start >= 2.0 - 1e-9',
                     f'p[0] <= {PARAMETERS["coarse_stop_m"]} or t - self.step_start >= {PARAMETERS["coarse_duration_s"]} - 1e-9')
    source = replace(source, 't - self.step_start >= 2.3 - 1e-9', f't - self.step_start >= {PARAMETERS["track_duration_s"]} - 1e-9')
    source = replace(source, 'abs(self.goal_heading) > 0.08 and not self.coarse_track',
                     'abs(self.goal_heading) > 0.17453292519943295 and not self.coarse_track')
    source = replace(source, '0 <= t - self.anchor_s <= 4.5', '0 <= t - self.anchor_s <= 6.5')
    source = replace(source, '0 <= t - self.goal_s <= 0.5', '0 <= t - self.goal_s <= (8.0 if getattr(self, "parking_arc", False) else 2.0)', 4)
    source = replace(source, 'if p[0] <= 0.125:', 'if p[0] <= getattr(self, "brake_threshold_m", 0.125):')
    return source


def install(navigator_class):
    global _installed
    if _installed is not None:
        return _installed
    base = next(c for c in navigator_class.__mro__ if c in _base_sources)
    source = _base_sources[base]
    changed = replace(source, 'math.dist(p, self.pose[:2]) > 0.10',
                      'math.dist(p, self.pose[:2]) > 0.18')
    base.step = recompile(base.step, changed)
    import coarse_track
    source = textwrap.dedent(inspect.getsource(coarse_track.path_authority))
    # The map clears only the coarse prefix. The ball itself is an obstacle in
    # this map, so extending its body envelope into the kick region rejects the
    # whole route. Continuous mode hands over to the existing fine domain there.
    changed_path = replace(source, 'ball - 0.36 * unit', 'ball - 0.28 * unit')
    coarse_track.path_authority = recompile(coarse_track.path_authority, changed_path)
    _installed = dict(name='straight-continuous', parameters=PARAMETERS,
        coarse_map_endpoint_m=.28, continuous_fine_transfer=True,
        predictive_braking=True,
        nav_step_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        coarse_path_sha256=hashlib.sha256(changed_path.encode()).hexdigest(),
        qualification_claimed=False, hidden_state_used=False)
    # This budget only admits an existing recovery. It never extends task time.
    import retreat_recovery
    retreat_recovery.RESERVE_S = 5.
    _installed['legacy_recovery_reserve_s'] = 5.
    return _installed


def transfer_continuous_track(fine, *, t, memory, heading, goal_s, guard):
    """Transfer an active step into the original fine domain without a pause.

    No anchor, clock, head, filter or physical state is reset. The coarse map
    grants no authority inside this domain; current fine guards still apply.
    A boundary outside the near controller's domain stops for observation.
    """
    if not fine.coarse_track or fine.state != 'TRACK':
        return None
    p = np.asarray(memory.get('point_level_m'), float)
    if p.shape != (3,) or not np.isfinite(p).all() or p[0] > .28:
        return None
    checks = dict(current_guard=bool(guard), memory=memory.get('memory_available') is True,
        near_domain=bool(.13 <= p[0] <= .28 and np.linalg.norm(p[:2]) <= .4
                         and PARAMETERS["transfer_y_m"][0] <= p[1] <= PARAMETERS["transfer_y_m"][1]),
        heading=math.isfinite(heading) and abs(heading) <= math.radians(10),
        goal=math.isfinite(goal_s) and 0 <= t-goal_s <= 2.)
    allowed = all(checks.values())
    if allowed:
        fine.coarse_track = False
    else:
        fine.state = 'SETTLE'
        fine.step_end = t
        fine.quiet_start = None
        fine.track_velocity = [0., 0., 0.]
        fine.reason = 'continuous_near_domain_requires_observation'
    return dict(capture_s=t, transferred=allowed, checks=checks, point_level_m=p.tolist(),
                step_start_s=fine.step_start, anchor_s=fine.anchor_s,
                route='coarse_map_prefix_to_existing_fine_domain', truth_used=False)


def kick_box(point):
    p = np.asarray(point, float)
    return bool(p.shape == (3,) and np.isfinite(p).all()
                and .070 <= p[0] <= .125 and -.065 <= p[1] <= -.005)


def remembered_kick_gate(*, t, memory, heading, goal_s, guard, moving, kicked):
    """A lost image alone is insufficient: require a recent, continuous estimate."""
    checks = dict(not_kicked=not kicked, stopped=not moving, current_floor_guard=bool(guard),
        continuous_memory=memory.get('memory_available') is True,
        in_kick_region=kick_box(memory.get('point_level_m')),
        finite_heading=math.isfinite(heading) and abs(heading) <= math.radians(15),
        recent_goal=math.isfinite(goal_s) and 0 <= t-goal_s <= 2.,
        recent_ball=isinstance(memory.get('ball_age_s'), (int, float))
            and 0 <= memory['ball_age_s'] <= 6.5,
        initial_range_uncertainty=isinstance(memory.get('original_range_uncertainty_m'), (int, float))
            and 0 <= memory['original_range_uncertainty_m'] <= .03)
    return dict(allowed=all(checks.values()), checks=checks,
                point_level_m=memory.get('point_level_m'), ball_age_s=memory.get('ball_age_s'),
                source='last observed ball propagated by continuous encoder/IMU odometry',
                truth_used=False, calibrated_uncertainty=False)


def install_fine_wrappers(cls):
    """Replace in-place large turns with short rear-path-authorized arcs."""
    if getattr(cls, '_parking_wrapped', False):
        return
    original_capture, original_policy, original_status = cls.capture, cls.policy, cls.status
    original_track = cls.track_filtered
    def track_filtered(self, p, heading, dt, packet):
        from predictive_brake import ClosingSpeed
        if not hasattr(self, 'closing_speed'):
            self.closing_speed = ClosingSpeed()
        self.braking_trace = self.closing_speed.update(packet['control_timestamp_s'], float(p[0]), self.anchor_s)
        self.brake_threshold_m = self.braking_trace['threshold_m']
        return original_track(self, p, heading, dt, packet)
    def capture(self, **kw):
        entering = self.state == 'IDLE' and kw['trigger']
        result = original_capture(self, **kw)
        if entering and self.state == 'RETURN':
            self.parking_arc = bool(kw['motion_kind'] == 'TURN'
                and abs(kw['goal_heading']) >= PARAMETERS['parking_angle_rad']
                and getattr(self, 'parking_path', {}).get('allowed'))
            if self.parking_arc:
                self.return_target = np.zeros(4)
        return self.status(kw['t'], kw['memory'])
    def policy(self, **kw):
        t = kw['t']
        if getattr(self, 'parking_arc', False) and self.state == 'TURN':
            path = getattr(self, 'parking_path', {})
            if not path.get('allowed') or not 0 <= t-path.get('capture_s', -math.inf) <= .2:
                self.state = 'SETTLE'; self.step_end = t; self.quiet_start = None
                self.reason = 'parking_path_requires_reobservation'
        result = original_policy(self, **kw)
        if getattr(self, 'parking_arc', False) and self.state == 'TURN':
            path = getattr(self, 'parking_path', {})
            if not path.get('allowed') or not 0 <= t-path.get('capture_s', -math.inf) <= .2:
                self.state = 'SETTLE'; self.step_end = t; self.quiet_start = None
                self.reason = 'parking_path_requires_reobservation'
                return self.status(t, kw['memory'])
        return result
    def status(self, t, memory):
        result = original_status(self, t, memory)
        parking = bool(getattr(self, 'parking_arc', False))
        if parking and self.state == 'TURN':
            result['velocity'] = [PARAMETERS['parking_vx'], 0., PARAMETERS['parking_wz']*self.turn_sign]
        result['maximum_memory_age_s'] = PARAMETERS['memory_age_s']
        if self.motion_kind == 'TRACK':
            result['maximum_step_s'] = PARAMETERS['coarse_duration_s'] if self.coarse_track else PARAMETERS['track_duration_s']
        result['parking_arc'] = parking
        result['parking_path'] = getattr(self, 'parking_path', None)
        result['predictive_braking'] = getattr(self, 'braking_trace', None)
        return result
    cls.capture, cls.policy, cls.status, cls.track_filtered = capture, policy, status, track_filtered
    cls._parking_wrapped = True
