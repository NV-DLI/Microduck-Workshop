"""Independent public-range reconstruction; private motion only in separate scoring."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from PIL import Image

M = Path(__file__).resolve().parents[2]
O = M.parents[1]/'outputs/migration-20261019'
D = O/'football-v36-near-turn-amplitude20s'
sys.path[:0] = [str(M/'football'), str(M/'sensors')]
from perception import BetaPerception, level_rotation
from beta_sensor_profile import BetaCameraProfile


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name): return [json.loads(x) for x in (D/name).read_text().splitlines()]


def main():
    manifest = json.loads((D/'manifest.json').read_text())
    sources = manifest['native_reference_contract']['source_sha256']
    checks = {str(p): sha(p) == sources[str(p)] for p in
              [M/'football/perception.py', M/'football/ground_plane.py', M/'football/vision/beta_box_geometry.py',
               M/'acceptance/native_football_v36/command_navigation_native.py']}
    assert all(checks.values())
    camera = manifest['sensors']['camera']
    vision = BetaPerception(BetaCameraProfile(width_px=camera['width_px'], height_px=camera['height_px'],
                            assumed_horizontal_fov_deg=camera['assumed_horizontal_fov_deg']))
    sensors = {x['timestamp_s']: x for x in read('sensor-inputs.jsonl')}
    detections = {x['timestamp_s']: x for x in read('detections.jsonl')}
    controls = read('control.jsonl')
    target_reason = 'near_or_behind_ball_requires_separate_validation'
    selected = [x for x in controls if x.get('native_command_adapter', {}).get('reason') == target_reason
                or 14.19 <= x['timestamp_s'] <= 14.81]
    public = []
    for control in selected:
        t = control['timestamp_s']; sensor = sensors[t]; detection = detections[t]
        assert sha(D/sensor['rgb']) == sensor['rgb_file_sha256']
        assert detection['image_sha256'] == sensor['rgb_pixel_sha256']
        geometry = vision.estimate(np.asarray(Image.open(D/sensor['rgb']).convert('RGB')), sensor['proprio'],
                                   detection['detections'], now_s=t, tof_frame=sensor['tof_frame'])
        accepted = bool(geometry.get('ball') and geometry['ball']['accepted'] and geometry['floor_observation']['accepted'])
        row = {'timestamp_s': t, 'native_reason': control.get('native_command_adapter', {}).get('reason', 'phase_active'),
               'output_command': control['velocity_body_xyw'], 'public_geometry_accepted': accepted}
        if accepted:
            rotation = level_rotation(-np.asarray(geometry['ground']['normal']))
            point = rotation @ np.asarray(geometry['ball']['geometry']['point_reference_m'])
            uncertainty = float(geometry['ball']['geometry']['estimated_range_uncertainty_m'])
            distance = float(np.linalg.norm(point[:2]))
            logged = control.get('native_command_adapter', {}).get('range_lower_m')
            row.update(point_level_m=point.tolist(), horizontal_distance_m=distance, uncertainty_m=uncertainty,
                       range_lower_m=distance-uncertainty, behind=bool(point[0] <= 0),
                       lower_bound_le_018=distance-uncertainty <= .18,
                       logged_range_lower_abs_error_m=None if logged is None else abs(logged-(distance-uncertainty)))
        public.append(row)
    held = [r for r in public if r['native_reason'] == target_reason]
    # Only after public decision reconstruction is complete, read private scorer
    # fields to describe actual stop motion. They never enter `vision` above.
    physics = [x for x in read('physics-trace.jsonl') if x['score_enabled']]
    pt = np.array([x['timestamp_s'] for x in physics])
    pos = np.array([x['root_position_world_m'] for x in physics])
    ball = np.array([x['ball_position_world_m'] for x in physics])
    def index(t):
        i = int(np.argmin(abs(pt-t))); assert abs(pt[i]-t) < 1e-9
        return i
    stop, guard = 14.47, held[0]['timestamp_s']
    i, j = index(stop), index(guard)
    distances = np.linalg.norm(pos[:, :2]-ball[:, :2], axis=-1)
    tail = pt >= pt[-1]-1.
    vel = np.array([x['root_linear_velocity_body_m_s'] for x in physics])
    gyro = np.array([x['root_angular_velocity_body_rad_s'] for x in physics])
    checks.update(all56public_hold_frames_retained=len(held)==56,
                  all_hold_geometry_accepted=all(r['public_geometry_accepted'] for r in held),
                  all_hold_range_lower_at_or_under_018=all(r.get('lower_bound_le_018', False) for r in held),
                  no_hold_frame_ball_behind=all(not r.get('behind', True) for r in held),
                  logged_public_ranges_reproduced=all(r['logged_range_lower_abs_error_m'] < 1e-12 for r in public if r.get('logged_range_lower_abs_error_m') is not None),
                  all_commands_zero_after_last_forward=all(not any(r['velocity_body_xyw']) for r in controls if r['timestamp_s'] >= stop),
                  actual_scored_force_zero=all(not any(x['support_force_world_n']) and not any(x['support_torque_world_nm']) for x in physics),
                  no_actual_fall=not any(x['fell'] for x in physics))
    result = {
        'schema': 'microduck.v36-public-near-stop-independent-review.v1', 'checks': checks,
        'technical_reconstruction_passed': all(checks.values()), 'qualification_claimed': False,
        'public_reconstruction': {'inputs': 'Only stored RGB, detections,64zone ToF, public same-time encoder/IMU; no world pose/depth truth',
                                  'raw_timestamps_not_rounded': True, 'frames': public},
        'chronology': {'last_forward_start_s': 14.2, 'last_forward_end_s': stop, 'duration_s': .27,
                       'stop_reason': 'advance_direction_or_duration_cap', 'public_heading_at_stop_rad': .1689868772026185,
                       'near_direction_limit_rad': .15, 'near_duration_limit_s': .4,
                       'first_range_lower_018_hold_s': guard, 'hold_after_stop_s': guard-stop,
                       'interpretation': 'Direction guard ended the last forward interval. The18cm lower-bound guard appeared0.4s later and prevented restart; it did not initiate that stop.'},
        'private_motion_scoring': {'scope': 'Scorer truth only, never inputs to public reconstruction/controller',
            'stop_to_first_range_guard_root_link_xy_delta_m': (pos[j, :2]-pos[i, :2]).tolist(),
            'stop_to_first_range_guard_root_link_xy_norm_m': float(np.linalg.norm(pos[j, :2]-pos[i, :2])),
            'ball_horizontal_distance_at_stop_m': float(distances[i]),
            'ball_horizontal_distance_at_first_guard_m': float(distances[j]),
            'radial_approach_after_stop_m': float(distances[i]-distances[j]),
            'after_guard_root_link_xy_peak_drift_m': float(np.linalg.norm(pos[j:, :2]-pos[j, :2], axis=1).max()),
            'final_ball_horizontal_distance_m': float(distances[-1]),
            'last1s_root_body_COM_planar_speed_rms_m_s': float(np.sqrt(np.square(vel[tail, :2]).sum(1).mean())),
            'last1s_angular_speed_rms_rad_s': float(np.sqrt(np.square(gyro[tail]).sum(1).mean()))},
        'limitations': ['The18cm value is an existing conservative public range lower bound, not calibrated braking distance.',
                       'Public reconstructed range and physical horizontal distance use different uncertainty/reference assumptions; they are separate measurements.',
                       'Root link displacement is distinct from root-body COM velocity; no equality between their integrals is assumed.',
                       'This single run does not establish safety of a smaller stop threshold or a new kick preparation route.'],
        'evidence_sha256': {str(p): sha(p) for p in [D/'manifest.json', D/'control.jsonl', D/'sensor-inputs.jsonl',
                               D/'detections.jsonl', D/'physics-trace.jsonl', Path(__file__)]},
    }
    output = O/'football-v36-public-stop-independent-review-v1.json'
    assert not output.exists(); output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'checks': checks, 'private_motion_scoring': result['private_motion_scoring']}))


if __name__ == '__main__': main()
