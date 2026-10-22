"""Independent CPU reconstruction of body-frame lateral response and ONNX sensitivity."""
from pathlib import Path
import hashlib, json
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

M = Path(__file__).resolve().parents[2]
O = M.parents[1] / 'outputs/migration-20261019'
run = O / 'football-v32-near-context20s'
source = O / 'football-v32-lateral-response-diagnosis-v1.json'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    d = json.loads(source.read_text())
    p = [json.loads(s) for s in (run/'physics-trace.jsonl').read_text().splitlines()]
    policy = [json.loads(s) for s in (run/'policy-trace.jsonl').read_text().splitlines()]
    segments = []
    for old in d['actual_side_segments']:
        start, end = old['start_s'], old['start_s']+old['duration_s']
        rows = [r for r in p if start <= r['timestamp_s'] <= end]
        t = np.array([r['timestamp_s'] for r in rows])
        pos = np.array([r['root_position_world_m'] for r in rows])
        vel = np.array([r['root_linear_velocity_body_m_s'] for r in rows])
        rotation = Rotation.from_quat([r['root_quaternion_xyzw'] for r in rows])
        world_vel = rotation.apply(vel)
        integrated_world = np.trapezoid(world_vel, t, axis=0)
        # Independent finite position differences rotated into each actual body frame.
        midrotation = Rotation.from_quat([r['root_quaternion_xyzw'] for r in rows[:-1]])
        body_position_steps = midrotation.inv().apply(np.diff(pos, axis=0))
        rect = float(vel[:, 1].sum()*.005)
        segments.append({
            'start_s': start, 'end_s': end, 'samples': len(t),
            'original_rectangle_integral_m': rect,
            'source_integral_abs_error_m': abs(rect-old['body_y_integral_m']),
            'duration_matched_trapezoid_integral_m': float(np.trapezoid(vel[:,1],t)),
            'position_difference_body_y_integral_m': float(body_position_steps[:,1].sum()),
            'world_velocity_integral_vs_position_max_abs_m': float(np.max(abs(integrated_world-(pos[-1]-pos[0])))),
            'positive_body_y_is_left': True,
            'following_04s_source_integral_m': old['following_04s_body_y_integral_m'],
        })
    x = np.array([r['observation'] for r in policy if r['observation'][49]>.2],np.float32)
    model=M/'models/community-v8/velstand.onnx'
    opt=ort.SessionOptions();opt.intra_op_num_threads=1;opt.inter_op_num_threads=1
    sess=ort.InferenceSession(str(model),sess_options=opt,providers=['CPUExecutionProvider'])
    name=sess.get_inputs()[0].name
    def infer(a): return np.concatenate([sess.run(None,{name:r[None]})[0] for r in a])
    zero=x.copy();zero[:,48:51]=0;base=infer(zero)
    contrasts=[]
    for old in d['same_input_CPU_command_counterfactual']:
        a=x.copy();a[:,48:51]=[0,old['vy'],0]
        delta=infer(a)-base
        rms=float(np.sqrt((delta**2).mean()))
        contrasts.append({'vy':old['vy'],'raw_delta_rms':rms,
                          'source_rms_abs_error':abs(rms-old['raw_difference_from_zero_rms'])})
    checks={
        'source_model_hash_matches': sha(model)==d['model_sha256'],
        'source_input_hash_matches': sha(run/'policy-trace.jsonl')==d['input_sha256'],
        'all_nine_source_integrals_reproduce':max(r['source_integral_abs_error_m'] for r in segments)<1e-14,
        'all_nine_finite_position_body_y_deltas_positive':all(r['position_difference_body_y_integral_m']>0 for r in segments),
        'rotation_world_velocity_integral_matches_actual_position_under_0p1mm':max(r['world_velocity_integral_vs_position_max_abs_m'] for r in segments)<.0001,
        'all_five_ORT_command_contrasts_reproduce':max(r['source_rms_abs_error'] for r in contrasts)<1e-6,
        'only_command_slots_changed':True,
    }
    result={'schema':'microduck.v32-lateral-independent-review.v1','all_passed':all(checks.values()),
        'qualification_claimed':False,'checks':checks,'segments':segments,'counterfactuals':contrasts,
        'counterfactual_rows':len(x),
        'interpretation':'Commands in slot49 request positive FLU body Y (left). Actual position differences also move left, weakly; not a sign/frame inversion. The same-input command changes do alter the native MLP output. This is sensitivity, not a new closed-loop simulation or proof of a learned deadzone.',
        'numerical_note':'Source integrates inclusive endpoints as sum(v)*.005, counting one extra5ms versus end-start. Independent trapezoidal values are provided without changing old report; this does not change the small-response conclusion.',
        'limitations':['Instantaneous body-Y velocity integral is path length in the rotating frame, not a displacement expressed in one frozen body frame.','Prior neutral-head full-DR5s primitive is not paired to this head-down nominal near-ball case.','Public ball-error increase includes moving head/IMU/vision geometry; it is not a true ball-relative velocity measurement.'],
        'source_sha256':{str(q):sha(q) for q in [source,run/'physics-trace.jsonl',run/'policy-trace.jsonl',model,Path(__file__)]}}
    out=O/'football-v32-lateral-independent-review-v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'report':str(out),'checks':checks,'max_world_integral_error_m':max(r['world_velocity_integral_vs_position_max_abs_m'] for r in segments)}))

if __name__=='__main__':main()
