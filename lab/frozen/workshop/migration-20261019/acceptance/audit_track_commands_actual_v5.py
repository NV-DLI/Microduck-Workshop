"""Re-derive actual TRACK commands and the final command ramp from public traces."""
import argparse
import hashlib
import json
import math
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lateral-target', type=float, choices=(-.035, -.028), required=True)
    parser.add_argument('--require-brake', action='store_true')
    parser.add_argument('--coarse-lateral-target', type=float, choices=(-.028,), required=True)
    parser.add_argument('--brake-duration',type=float,choices=(.16,.40),required=True)
    parser.add_argument('--brake-start-x',type=float,choices=(.125,.160),required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    reads = {round(r['timestamp_s'], 6): r['readback'].get('actual_packet')
             for r in map(json.loads, (args.run / 'public-memory-readback.jsonl').open())}
    rate_lp = 0.0
    last_time = None
    key = None
    lat_lp = heading_lp = None
    brake_start = brake_command = None
    errors = []
    brake_ticks = 0
    brake_stops = 0
    for row in map(json.loads, (args.run / 'policy-trace.jsonl').open()):
        t = row['timestamp_s']
        dt = 0.0 if last_time is None else t - last_time
        last_time = t
        fine = row['public_neutral_fine_step']
        state = fine['state']
        if state not in ('TRACK', 'MACRO'):
            if brake_start is not None:
                assert all(v == 0 for v in row['actual_velocity_command'])
                assert t - brake_start <= args.brake_duration + .02 + 1e-8
                brake_stops += 1
                brake_start = brake_command = None
            continue
        packet = reads[round(t, 6)]
        g, w = packet['gravity_body'], packet['gyro_body_rad_s']
        rate = -sum(x*y for x, y in zip(w, g)) / math.sqrt(sum(x*x for x in g))
        if state == 'MACRO':
            rate_lp += (1 - math.exp(-dt/.1)) * (rate - rate_lp)
            continue
        current = (fine['attempts'], fine['step_start_s'])
        if current != key:
            key = current
            lat_lp = heading_lp = None
            brake_start = brake_command = None
        p = row['public_handoff']['memory']['point_level_m']
        heading = fine['motion_heading_prediction_rad']
        if brake_start is not None:
            factor = max(0., 1. - (t-brake_start)/args.brake_duration)
            expected = [v*factor for v in brake_command]
            brake_ticks += 1
            assert fine['track_brake_started_s'] == brake_start
            assert fine['track_brake_initial_command'] == brake_command
        else:
            target = args.coarse_lateral_target if fine.get('coarse_track') else args.lateral_target
            lateral = -math.sin(heading)*p[0] + math.cos(heading)*p[1] - target
            if lat_lp is None:
                lat_lp, heading_lp = lateral, heading
            else:
                lat_lp += (1-math.exp(-dt/.20))*(lateral-lat_lp)
                heading_lp += (1-math.exp(-dt/.12))*(heading-heading_lp)
            rate_lp += (1-math.exp(-dt/.1))*(rate-rate_lp)
            vy = max(-.2, min(.2, .65*math.sin(heading_lp) + 10*lat_lp))
            wz = max(-.5, min(.5, 2.5*heading_lp - .25*rate_lp))
            expected = [.3 if p[0] > .17 else .25, vy, wz]
            if fine['track_brake_started_s'] is not None:
                assert .115 < p[0] <= args.brake_start_x
                assert fine['track_brake_started_s'] == t
                brake_start, brake_command = t, expected
                # Floating point re-derivation is checked numerically below.
                errors.extend(abs(x-y) for x, y in zip(expected, fine['track_brake_initial_command']))
                brake_command = fine['track_brake_initial_command']
        errors.extend(abs(x-y) for x, y in zip(expected, row['actual_velocity_command']))
    passed = bool(errors and max(errors) < 1e-9 and (not args.require_brake or brake_ticks > 0 and brake_stops > 0))
    sources = [Path(__file__), args.run/'policy-trace.jsonl', args.run/'public-memory-readback.jsonl']
    result = dict(schema='microduck.actual-track-command-audit.v5', passed=passed, brake_duration_s=args.brake_duration,brake_start_x_m=args.brake_start_x,lateral_target_m=args.lateral_target,coarse_lateral_target_m=args.coarse_lateral_target,require_brake=args.require_brake,
                  maximum_command_error=max(errors, default=None), brake_ticks=brake_ticks,
                  brake_stops=brake_stops, compared_scalars=len(errors),
                  scope='Public memory, same-read gyro/gravity and actual commands. Motion heading is separately re-derived by audit_motion_heading_reference_v2. No physics truth or model training.',
                  source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_sha256','scope')}))
    raise SystemExit(0 if passed else 1)

if __name__ == '__main__':
    main()
