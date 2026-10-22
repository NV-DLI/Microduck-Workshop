"""Read-only chronology and physical stop audit; no controller edits."""
from pathlib import Path
import hashlib
import json
import numpy as np

M=Path(__file__).resolve().parents[2];O=M.parents[1]/'outputs/migration-20261019'
D=O/'football-v37-closer-observation20s'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda name:[json.loads(x) for x in(D/name).read_text().splitlines()]


def main():
    manifest=json.loads((D/'manifest.json').read_text());release=manifest['startup_support']['task_started_at_s']
    controls=[r for r in read('control.jsonl')if r['timestamp_s']>=release]
    physics=[r for r in read('physics-trace.jsonl')if r['score_enabled']]
    policy=read('policy-trace.jsonl');detection={r['timestamp_s']:r for r in read('detections.jsonl')}
    moving=[i for i,r in enumerate(controls)if any(r['velocity_body_xyw'])]
    last=moving[-1];stop=controls[last+1];t=stop['timestamp_s']
    begin=last
    while begin>0 and controls[begin-1]['velocity_body_xyw']==controls[last]['velocity_body_xyw']:begin-=1
    guard=next(r for r in controls if r.get('native_command_adapter',{}).get('reason')=='near_or_behind_ball_requires_separate_validation')
    first_crop=next(r for r in controls if r['timestamp_s']>=t and r.get('image_head_servo',{}).get('cropped'))
    first_invalid=next(r for r in controls if r['timestamp_s']>=t and r.get('native_command_adapter',{}).get('reason')=='invalid_public_input')
    post=[r for r in controls if r['timestamp_s']>=t]
    held=[r for r in controls if r.get('native_command_adapter',{}).get('reason')=='near_or_behind_ball_requires_separate_validation']
    pt=np.array([r['timestamp_s']for r in physics]);xy=np.array([r['root_position_world_m'][:2]for r in physics]);ball=np.array([r['ball_position_world_m'][:2]for r in physics])
    distance=np.linalg.norm(ball-xy,axis=1)
    def motion(start):
        i=int(np.argmin(abs(pt-start)));assert abs(pt[i]-start)<1e-9
        displacement=np.linalg.norm(xy[i:]-xy[i],axis=1);peak=i+int(displacement.argmax())
        return {'start_s':start,'initial_ball_horizontal_distance_m':float(distance[i]),
            'peak_link_xy_displacement_m':float(displacement.max()),'peak_timestamp_s':float(pt[peak]),
            'link_xy_delta_at_peak_m':(xy[peak]-xy[i]).tolist(),
            'minimum_ball_horizontal_distance_m':float(distance[i:].min()),
            'maximum_radial_approach_m':float(distance[i]-distance[i:].min()),
            'final_link_xy_delta_m':(xy[-1]-xy[i]).tolist()}
    checks={'all_trace_hashes':all(sha(D/n)==h for n,h in manifest['trace_sha256'].items()),
        'all_output_commands_zero_after_stop':all(not any(r['velocity_body_xyw'])for r in post),
        'all_output_commands_zero_after16cm_guard':all(not any(r['velocity_body_xyw'])for r in controls if r['timestamp_s']>=guard['timestamp_s']),
        'all_policy_commands_zero_from_next_control':all(not any(r['observation'][48:51])for r in policy if r['timestamp_s']>=t+.02-1e-9),
        'hold_frames_public_range_proves_lower16':all(r['public_metric_ball'] and r['public_metric_ball']['horizontal_range_lower_m']<=.16 and r['public_metric_ball']['point_level_m'][0]>0 for r in held),
        'public_range_scalar_consistent':all(abs(r['public_metric_ball']['horizontal_range_m']-r['public_metric_ball']['range_uncertainty_m']-r['public_metric_ball']['horizontal_range_lower_m'])<1e-12 for r in controls if r.get('public_metric_ball')),
        'no_kick':all(not r['kick_request']for r in controls),'no_fall':not any(r['fell']for r in physics),
        'zero_task_support':all(not any(r['support_force_world_n'])and not any(r['support_torque_world_nm'])for r in physics)}
    rows=[]
    for r in controls:
        if controls[begin]['timestamp_s']<=r['timestamp_s']<=first_invalid['timestamp_s']+.2:
            rows.append({'timestamp_s':r['timestamp_s'],'command':r['velocity_body_xyw'],'reason':r.get('native_command_adapter',{}).get('reason','phase_active'),
                'translation_heading_rad':r.get('near_ball_phase',{}).get('translation_heading_error_rad'),
                'public_metric_ball':r.get('public_metric_ball'),'cropped':r.get('image_head_servo',{}).get('cropped'),
                'ball_boxes':[x['bbox_xyxy_px']for x in detection[r['timestamp_s']]['detections']if x['class_name']=='ball']})
    result={'schema':'microduck.v37-public-stop-independent-review.v1','qualification_claimed':False,'checks':checks,
        'technical_reconstruction_passed':all(checks.values()),'chronology':{
            'last_forward_start_s':controls[begin]['timestamp_s'],'last_forward_stop_s':t,
            'last_forward_duration_s':t-controls[begin]['timestamp_s'],'stop_reason':stop['native_command_adapter']['reason'],
            'public_heading_at_stop_rad':stop['near_ball_phase']['translation_heading_error_rad'],
            'public_range_lower_at_stop_m':stop['public_metric_ball']['horizontal_range_lower_m'],
            'first16cm_hold_s':guard['timestamp_s'],'first16cm_lower_bound_m':guard['public_metric_ball']['horizontal_range_lower_m'],
            'first_subsequent_cropped_image_s':first_crop['timestamp_s'],'first_subsequent_invalid_public_input_s':first_invalid['timestamp_s'],
            'all_control_rows_after_stop':len(post),'near_hold_frames':len(held),'selected_public_frames':rows},
        'private_scoring':{'scope':'Independent scoring only; no private field enters control/public range decisions',
            'from_last_forward_stop':motion(t),'from_first16cm_hold':motion(guard['timestamp_s']),
            'final_ball_horizontal_distance_m':float(distance[-1])},
        'interpretation':['Last forward ended on the direction guard before the16cm lower-bound guard. Crop/invalid public input occurred later; neither caused the initial forward stop.',
            'All subsequent high-level commands remain zero, and actual policy command slots become zero by the next50Hz control. Remaining motion is physical response under the zero-command native policy, not continued requested forward.',
            'This audit does not identify a safe shorter braking threshold or authorize closer motion.'],
        'evidence_sha256':{str(p):sha(p)for p in [D/'manifest.json',D/'control.jsonl',D/'policy-trace.jsonl',D/'physics-trace.jsonl',D/'detections.jsonl',Path(__file__)]}}
    output=O/'football-v37-public-stop-independent-review-v1.json';assert not output.exists();output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'report':str(output),'checks':checks,'chronology':{k:v for k,v in result['chronology'].items()if k!='selected_public_frames'},'private':result['private_scoring']}))


if __name__=='__main__':main()
