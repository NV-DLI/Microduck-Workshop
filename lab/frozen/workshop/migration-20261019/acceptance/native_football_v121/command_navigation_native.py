"""Public visual-bearing phases with actual delivered-gyro stopping evidence.

Far approach and a separate bounded near-observation alignment are armed. Every original sensor/navigation hold remains
authoritative. No state, achieved velocity or policy action is overwritten.
"""
import math
from collections import deque
import numpy as np
from angular_navigation_native import AngularBallNavigator
from perception import level_rotation
from range_progress import RangeProgress
from turn_prediction import near_turn_stop_hint


class NativePhaseCommands:
    forward_s = .6
    turn_min_s = .4
    turn_max_s = 1.2
    near_turn_max_s = 2.4
    near_turn_magnitude = .8
    forward_magnitude = .3
    near_forward_magnitude = .2
    observe_min_s = .4
    observe_warning_s = 2.
    near_hold_m = .12
    near_entry_m = .34

    def __init__(self):
        self.progress = RangeProgress()
        self.reset()

    def reset(self):
        self.progress.invalidate()
        self.phase = 'OBSERVE'
        self.started = None
        self.last_t = None
        self.turn_sign = 0.
        self.turn_integral = 0.
        self.active_turn_max_s = self.turn_max_s
        self.active_turn_magnitude = .6
        self.samples = deque()
        self.active_duration = self.forward_s
        self.active_forward_magnitude = self.forward_magnitude
        self.active_progress_probe = False
        self.strafe_sign = 0.

    def _hold(self, t, why, clear=True):
        if clear:
            self.phase = 'OBSERVE'
            self.started = t
            self.samples.clear()
        return [0., 0., 0.], {'phase': self.phase, 'reason': why,
                               'guard_zero_preserved': True, 'truth_used': False}

    def _base_update(self, *, timestamp_s, authorized, requested, point_level,
               range_uncertainty_m, gyro_level_z, gyro_body_rad_s, near_goal_heading=None, near_position_error=None):
        t = float(timestamp_s)
        u = np.asarray(requested, float)
        p = np.asarray(point_level, float) if point_level is not None else np.full(3, np.nan)
        measured_gyro = np.asarray(gyro_body_rad_s, float)
        good = (measured_gyro.shape == (3,) and np.isfinite(measured_gyro).all() and math.isfinite(t) and t >= 0 and u.shape == (3,)
                and p.shape == (3,) and np.isfinite(u).all() and np.isfinite(p).all()
                and math.isfinite(range_uncertainty_m) and range_uncertainty_m >= 0
                and math.isfinite(gyro_level_z))
        if not good:
            self.reset()
            return [0., 0., 0.], {'phase': 'OBSERVE', 'reason': 'invalid_public_input',
                                  'guard_zero_preserved': True, 'truth_used': False}
        dt = 0. if self.last_t is None else t-self.last_t
        if self.last_t is not None and not 0 < dt <= .2:
            self.reset(); self.last_t = t
            return self._hold(t, 'nonmonotone_or_gap')
        self.last_t = t
        if not authorized or not np.any(u):
            return self._hold(t, 'upstream_hold')
        distance = float(np.linalg.norm(p[:2]))
        visual_bearing = float(math.atan2(p[1], p[0]))
        near = distance-range_uncertainty_m <= self.near_entry_m
        error = None
        if near:
            error = np.asarray(near_position_error, float) if near_position_error is not None else np.full(2, np.nan)
            if near_goal_heading is None or not math.isfinite(near_goal_heading) or error.shape != (2,) or not np.isfinite(error).all():
                return self._hold(t, 'fresh_near_goal_heading_and_position_required')
        bearing = float(near_goal_heading) if near else visual_bearing
        if p[0] <= 0 or distance-range_uncertainty_m <= self.near_hold_m:
            return self._hold(t, 'near_or_behind_ball_requires_separate_validation')
        if self.started is None:
            self.started = t
        self.samples.append((t, gyro_level_z, p[:2].copy(), float(measured_gyro @ measured_gyro)))
        while self.samples and t-self.samples[0][0] > .45:
            self.samples.popleft()
        diagnostic = {'phase': self.phase, 'bearing_rad': bearing,
                      'range_lower_m': distance-range_uncertainty_m, 'near_mode': near,
                      'visual_ball_bearing_rad': visual_bearing,
                      'body_heading_owner': 'public_preparation_waypoint' if near else 'public_ball_bearing',
                      'public_gyro_integral_rad': self.turn_integral,
                      'truth_used': False, 'achieved_velocity_calibrated': False}
        if self.phase == 'TURN':
            self.turn_integral += gyro_level_z*dt
            hint = near_turn_stop_hint(bearing, gyro_level_z, self.turn_sign) if near else None
            diagnostic['turn_stop_prediction'] = hint
            crossed = bearing*self.turn_sign <= 0
            done = t-self.started >= self.turn_min_s and abs(bearing) <= .06
            predicted_stop = bool(hint and hint['stop_hint'])
            if crossed or done or predicted_stop or t-self.started >= self.active_turn_max_s:
                why = ('turn_bearing_complete' if crossed or done else
                       'near_turn_predictive_stop' if predicted_stop else 'turn_duration_cap')
                output, stopped = self._hold(t, why)
                stopped['turn_stop_prediction'] = hint
                return output, stopped
            diagnostic.update(phase='TURN', elapsed_s=t-self.started,
                              public_gyro_integral_rad=self.turn_integral, maximum_duration_s=self.active_turn_max_s)
            return [0., 0., self.active_turn_magnitude*self.turn_sign], diagnostic
        if self.phase == 'ADVANCE':
            if abs(bearing) > (.15 if near else .35) or t-self.started >= self.active_duration:
                return self._hold(t, 'advance_direction_or_duration_cap')
            diagnostic.update(phase='ADVANCE', elapsed_s=t-self.started)
            diagnostic['command_magnitude'] = self.active_forward_magnitude
            return [self.active_forward_magnitude, 0., 0.], diagnostic
        if self.phase == 'STRAFE':
            if not near or error[1]*self.strafe_sign <= 0 or abs(error[1]) <= .02 or abs(bearing) > .15 or t-self.started >= 2.:
                return self._hold(t, 'lateral_error_or_duration_cap')
            diagnostic.update(phase='STRAFE', elapsed_s=t-self.started, lateral_error_m=float(error[1]))
            return [0., .3*self.strafe_sign, 0.], diagnostic
        elapsed = t-self.started
        covered = len(self.samples) >= 6 and t-self.samples[0][0] >= .33
        gyro_mean = float(np.mean([x[1] for x in self.samples]))
        visual_motion = float(np.linalg.norm(self.samples[-1][2]-self.samples[0][2]))
        gyro_rms = float(np.sqrt(np.mean([x[3] for x in self.samples])))
        points = np.stack([x[2] for x in self.samples])
        visual_diameter = float(np.linalg.norm(points[:, None]-points[None, :], axis=-1).max())
        quiet = covered and gyro_rms <= .12 and visual_diameter <= .02
        diagnostic.update(phase='OBSERVE', elapsed_s=elapsed,
                          gyro_window_mean_rad_s=gyro_mean,
                          ball_relative_change_m=visual_motion,
                          gyro_window_3D_RMS_rad_s=gyro_rms,
                          ball_window_XY_diameter_m=visual_diameter,
                          quiet_statistics="3D gyro RMS and maximum pairwise public ball XY difference",
                          measured_quiet=quiet,
                          settling_warning=elapsed > self.observe_warning_s)
        if elapsed < self.observe_min_s or not quiet:
            return [0., 0., 0.], diagnostic
        self.started = t
        self.samples.clear()
        if abs(bearing) > (.15 if near else .18):
            self.phase = 'TURN'; self.turn_sign = math.copysign(1., bearing)
            self.active_turn_max_s = self.near_turn_max_s if near else self.turn_max_s
            self.active_turn_magnitude = self.near_turn_magnitude if near else .6
            self.turn_integral = 0.
            return [0., 0., self.active_turn_magnitude*self.turn_sign], {**diagnostic, 'phase': 'TURN', 'reason': 'fresh_visual_bearing'}
        if u[0] <= 0:
            return self._hold(t, 'upstream_path_does_not_authorize_forward')
        self.phase = 'ADVANCE'
        self.active_duration = .4 if near else self.forward_s
        self.active_forward_magnitude = self.near_forward_magnitude if near else self.forward_magnitude
        return [self.active_forward_magnitude, 0., 0.], {**diagnostic, 'phase': 'ADVANCE', 'reason': 'bearing_aligned_fresh_path', 'command_magnitude': self.active_forward_magnitude}


    def update(self, **inputs):
        previous_phase, previous_started = self.phase, self.started
        previous_duration = self.active_duration
        output, diagnostic = self._base_update(**inputs)
        t = inputs['timestamp_s']
        reason = diagnostic.get('reason')
        # The original controller evaluates every public guard first. Invalid
        # or missing observations must also invalidate pending progress.
        harmless_stop = reason == 'advance_direction_or_duration_cap'
        if reason and reason not in ('bearing_aligned_fresh_path', 'fresh_visual_bearing') and not harmless_stop:
            self.progress.invalidate()
        if harmless_stop and previous_phase == 'ADVANCE':
            p = np.asarray(inputs['point_level'], float)
            near = np.linalg.norm(p[:2]) - inputs['range_uncertainty_m'] <= self.near_entry_m
            heading = inputs.get('near_goal_heading') if near else math.atan2(p[1], p[0])
            complete = (previous_started is not None and t - previous_started >= previous_duration - 1e-9
                        and heading is not None and abs(heading) <= (.15 if near else .35))
            self.progress.end(t, complete)
        fresh_near = diagnostic.get('near_mode', False)
        if fresh_near and diagnostic.get('measured_quiet'):
            radius = float(np.linalg.norm(np.asarray(inputs['point_level'])[:2]))
            self.progress.settled(t, radius, inputs['range_uncertainty_m'])
        if self.phase == 'TURN':
            self.progress.invalidate()
        new_advance = self.phase == 'ADVANCE' and previous_phase != 'ADVANCE'
        if self.phase != 'ADVANCE' or new_advance:
            self.active_progress_probe = False
        if new_advance and fresh_near:
            duration, why = self.progress.choose(diagnostic['range_lower_m'])
            if duration is None:
                output, diagnostic = self._hold(t, why)
            else:
                self.active_duration = duration
                self.active_progress_probe = why == 'bounded_stronger_probe_after_unproven_progress'
                self.active_forward_magnitude = self.progress.probe_command_magnitude if self.active_progress_probe else self.near_forward_magnitude
                output = [self.active_forward_magnitude, 0., 0.]
                diagnostic['command_magnitude'] = self.active_forward_magnitude
                self.progress.begin(t, float(np.linalg.norm(np.asarray(inputs['point_level'])[:2])),
                                    inputs['range_uncertainty_m'], duration, probe=self.active_progress_probe)
                diagnostic['progress_duration_reason'] = why
        if self.phase == 'ADVANCE' and self.active_progress_probe:
            # This extra stop is checked every real sensor/control frame; it
            # does not replace or defer the original 12cm or heading guards.
            if diagnostic.get('range_lower_m', -math.inf) <= self.progress.active_stop_lower_m:
                self.progress.invalidate()
                output, diagnostic = self._hold(t, 'progress_probe_distance_reserve')
        diagnostic['progress_monitor'] = self.progress.status()
        if self.phase == 'ADVANCE':
            diagnostic['selected_duration_s'] = self.active_duration
            diagnostic['active_progress_probe'] = self.active_progress_probe
        return output, diagnostic


class NativeCommandNavigator(AngularBallNavigator):
    def reset(self):
        super().reset()
        self.native_commands = NativePhaseCommands()

    def step(self, rgb, proprio, tof_frame, **kwargs):
        out = super().step(rgb, proprio, tof_frame, **kwargs)
        # Keep upstream authorization but remove the obsolete trained-gait PI
        # correction from this reference's command profile. No accumulated PI
        # state can choose a native turn or prevent a visually aligned advance.
        previous = out.get('measured_yaw_feedback', {})
        requested = list(out['velocity_body_xyw'])
        requested[2] = float(previous.get('desired_yaw_rate_rad_s', 0.))
        self.yaw_feedback.reset()
        out['measured_yaw_feedback'] = {'active': False, 'reason': 'replaced_by_public_visual_phase',
                                        'old_desired_rate_diagnostic': requested[2]}
        geo = self.last_geometry
        ball = geo.get('ball') if geo else None
        fresh = bool(geo and geo['timestamp_s'] == proprio['timestamp_s']
                     and ball and ball['accepted'] and geo['floor_observation']['accepted'])
        point = None; uncertainty = 0.; omega = 0.
        if fresh:
            R = level_rotation((-np.asarray(geo['ground']['normal'])).tolist())
            point = R @ np.asarray(ball['geometry']['point_reference_m'])
            uncertainty = float(ball['geometry']['estimated_range_uncertainty_m'])
            omega = float((R @ np.asarray(proprio['gyro_body_rad_s']))[2])
        allowed = bool(out.get('motion_authorization', {}).get('allowed')
                       and fresh and not self.refresh_too_close)
        near_diag = out.get('near_ball_phase', {})
        near_heading = near_diag.get('translation_heading_error_rad') if 0 <= near_diag.get('goal_fix_age_s', -1) <= 8. else None
        near_error = near_diag.get('relative_position_error_m')
        actual, diagnostic = self.native_commands.update(
            timestamp_s=proprio['timestamp_s'], authorized=allowed, requested=requested,
            point_level=point, range_uncertainty_m=uncertainty, gyro_level_z=omega,
            gyro_body_rad_s=proprio["gyro_body_rad_s"],
            near_goal_heading=near_heading, near_position_error=near_error)
        out['velocity_body_xyw'] = actual
        out['native_command_adapter'] = diagnostic
        out['public_metric_ball'] = ({'timestamp_s': float(proprio['timestamp_s']),
            'point_level_m': point.tolist(), 'range_uncertainty_m': uncertainty,
            'horizontal_range_m': float(np.linalg.norm(point[:2])),
            'horizontal_range_lower_m': float(np.linalg.norm(point[:2]))-uncertainty,
            'source': 'same-frame accepted complete RGB sphere and accepted public floor estimate',
            'truth_used': False} if fresh else None)
        out['velocity_semantics'] = 'Published community command primitives, not achieved physical velocity'
        out['kick_request'] = False
        return out
