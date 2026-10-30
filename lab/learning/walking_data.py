"""Export successful sensor-driven walking demonstrations, with causal RGB alignment."""
import argparse
import hashlib
import json
import math
from pathlib import Path

try:
    from .walking import ACTION_NAMES, TASK, contract, pack_state, validate_demonstrated_request
except ImportError:
    from walking import ACTION_NAMES, TASK, contract, pack_state, validate_demonstrated_request


def rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s]


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def score(directory):
    d = Path(directory)
    manifest = json.loads((d/'manifest.json').read_text())
    s = manifest['scorer']
    quiet_start = None
    physics = rows(d/'physics-trace.jsonl')
    for row in physics:
        t = row['timestamp_s']
        quiet = (sum(v*v for v in row['root_linear_velocity_body_m_s']) < .02**2
                 and sum(v*v for v in row['root_angular_velocity_body_rad_s']) < .3**2
                 and row['projected_gravity_body'][2] < -.94)
        quiet_start = (t if quiet_start is None else quiet_start) if quiet else None
    duration = 0 if quiet_start is None else physics[-1]['timestamp_s']-quiet_start
    passed = (manifest['terminal_reason']=='development_kick_window_complete'
              and s['real_foot_ball_contact'] and s['whole_ball_crossed_inside_goal']
              and not s['fell'] and not s['obstacle_collision']
              and s['mechanical_output_bounds_passed'] and duration >= 1.)
    return dict(passed=bool(passed), contact=s['real_foot_ball_contact'],
                goal=s['whole_ball_crossed_inside_goal'], fall=s['fell'],
                final_quiet_seconds=duration, seed=manifest['fixture']['seed'],
                seconds=manifest['task_seconds'], terminal=manifest['terminal_reason'])


def samples(directory, *, align_kick_onset=False):
    d = Path(directory)
    m = json.loads((d/'manifest.json').read_text())
    native = rows(d/'policy-trace.jsonl')
    camera = rows(d/'sensor-inputs.jsonl')
    startup = rows(d/'startup-support.jsonl')
    start = next(r['timestamp_s'] for r in startup if r['navigation_enabled'])
    onset = next((i for i, row in enumerate(native)
                  if row['kick_requested'] and i > 0 and not native[i-1]['kick_requested']), None)
    if align_kick_onset and onset is None:
        raise ValueError('A recorded kick onset is required for phase-aligned sampling')
    sample_phase = onset % 4 if align_kick_onset else 0
    schema = contract(m['policy_joint_names'])
    cursor = 0
    result = []
    for i, row in enumerate(native):
        t = row['timestamp_s']
        if t < start or i % 4 != sample_phase or i == 0:
            continue
        while cursor+1 < len(camera) and camera[cursor+1]['timestamp_s'] <= t+1e-9:
            cursor += 1
        image = camera[cursor]
        if not 0 <= t-image['timestamp_s'] <= .08+1e-9:
            raise ValueError('Missing fresh causal RGB')
        previous = native[i-1]
        previous_request = previous['actual_velocity_command'] + previous['actual_head_command_rad'] + [float(previous['kick_requested'])]
        phase = (previous.get('native_k10') or {}).get('phase_s')
        if phase is not None:
            phase = min(4., phase+t-previous['timestamp_s'])
        action = row['actual_velocity_command'] + row['actual_head_command_rad'] + [float(row['kick_requested'])]
        validate_demonstrated_request(action)
        validate_demonstrated_request(previous_request)
        result.append(dict(timestamp_s=t, image=image['rgb'], image_capture_s=image['timestamp_s'],
                           state=pack_state(row['observation'], previous_request, skill_phase_s=phase),
                           action=action, source_policy_index=i))
    if len(result)<16 or any(not math.isclose(b['timestamp_s']-a['timestamp_s'],.08,abs_tol=1e-8) for a,b in zip(result,result[1:])):
        raise ValueError('Incomplete 12.5 Hz demonstration')
    return schema, result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--align-kick-onset',action='store_true',
                    help='Shift the 12.5 Hz sampling phase to retain the real first kick request')
    a=ap.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    entries=json.loads(a.manifest.read_text())
    splits={e['split'] for e in entries}
    if len(splits)!=1 or not splits <= {'train','validation','test'}:
        raise ValueError('One declared split required')
    prepared=[]; seeds=set()
    for entry in entries:
        d=Path(entry['path']); s=score(d)
        if not s['passed']: raise ValueError('Unsuccessful demonstration: '+str(d))
        if s['seed'] in seeds: raise ValueError('Duplicate episode')
        seeds.add(s['seed']); schema, data=samples(d, align_kick_onset=a.align_kick_onset)
        prepared.append((d,s,schema,data))
    if not prepared or any(x[2]!=prepared[0][2] for x in prepared):
        raise ValueError('Missing or mixed walking contracts')
    import numpy as np
    from PIL import Image
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    schema=prepared[0][2]
    features={'observation.state':dict(dtype='float32',shape=(58,),names=schema['state_names']),
              'action':dict(dtype='float32',shape=(8,),names=ACTION_NAMES),
              'observation.images.front':dict(dtype='image',shape=(270,480,3),names=['height','width','channels'])}
    dataset=LeRobotDataset.create(repo_id='local/'+a.output.name,root=a.output,fps=12.5,
                                 robot_type='microduck_walking',features=features,use_videos=False)
    provenance=[]
    for d,s,_,data in prepared:
        for row in data:
            rgb=np.asarray(Image.open(d/row['image']).convert('RGB').resize((480,270)))
            dataset.add_frame({'observation.state':np.asarray(row['state'],np.float32),
                               'action':np.asarray(row['action'],np.float32),
                               'observation.images.front':rgb,'task':TASK})
        dataset.save_episode()
        provenance.append(dict(seed=s['seed'],source=str(d.resolve()),frames=len(data),
                               samples_sha256=digest(d/'policy-trace.jsonl'),
                               sensors_sha256=digest(d/'sensor-inputs.jsonl'),
                               score=s,split=next(iter(splits)),label_source='sensor_based_traditional_expert'))
        provenance[-1].update(
            sampling_phase_policy='actual_kick_onset' if a.align_kick_onset else 'original_zero',
            source_policy_index_mod4=data[0]['source_policy_index'] % 4,
            first_sample_timestamp_s=data[0]['timestamp_s'],
        )
    dataset.finalize()
    for name,value in [('control-contract',schema),('provenance',provenance),('split',dict(split=next(iter(splits))))]:
        (a.output/(name+'.json')).write_text(json.dumps(value,indent=2))


if __name__=='__main__': main()
