#!/usr/bin/env python3
"""Audit physical camera motion and actual delivered BODY50 actor observations."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from audit_gaze_trace import audit, sha

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'acceptance'))


def validate_gaze_inputs(data):
    obs, control = data['actor_observation'], data['control']
    ready, inferred = data['body50_orientation_valid'], data['actor_inferred_mask']
    if obs.shape != (*control.shape[:2],61) or not np.isfinite(obs).all():
        raise ValueError('All actor rows must remain in the record')
    if not np.array_equal(ready,inferred):
        raise ValueError('Inference must use exactly the ready sensor rows')
    if np.any(obs[~ready]!=0) or np.any(control[:,:,:14][~ready]!=0):
        raise ValueError('Unready rows must be unused zero placeholders')
    history = np.concatenate((data['initial_previous_raw'][None],control[:-1,:,:14]))
    history = np.where(ready[...,None],history,0.)
    if not np.allclose(obs[:,:,34:48],history,atol=1e-6,rtol=0):
        raise ValueError('Wrong preceding raw action history')
    public = np.concatenate((data['body50_gyro_body_rad_s'],data['body50_gravity_body'],
        data['body50_encoder_q_rad']-data['native_home14'][None],data['body50_encoder_qd_rad_s']),-1)
    if not np.allclose(obs[:,:,:34][ready],public[ready],atol=2e-6,rtol=0):
        raise ValueError('Actor input differs from actual delivered sensor packet')
    if not np.array_equal(data['body50_read_timestamp_s'],data['body50_encoder_read_timestamp_s']):
        raise ValueError('IMU/encoder timestamps must come from the same read')
    wrench = data['external_wrench_peak']
    if wrench.shape != (control.shape[0]*4,control.shape[1]) or not np.isfinite(wrench).all() or np.any(wrench!=0):
        raise ValueError('Actual task assistance detected')
    return {'ready_actor_rows':int(ready.sum()),'unready_placeholder_rows':int((~ready).sum()),
        'worlds_with_sensor_outage':int((~ready).any(0).sum()),'unready_rows_not_inferred':True,
        'all_actual_task_wrenches_zero':True,'public_packet_and_raw_history_verified':True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('report',type=Path)
    p.add_argument('--output',required=True,type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep every earlier audit')
    result = audit(args.report)
    r = json.loads(args.report.read_text())
    if r['physical_initialization']['automatic_resets'] != 0:
        raise ValueError('Initialization cannot silently select replacement worlds')
    if r['physical_initialization']['task_environments'] != len(r['episodes']):
        raise ValueError('Every original world must finish physical startup')
    if r['physical_initialization']['physics_ticks'] > 1600:
        raise ValueError('Startup exceeded the declared8s deadline')
    if r['body50_task_readback']['actual_task_external_wrench_peak'] != 0:
        raise ValueError('Task assistance detected')
    table_path = next(Path(path) for path in r['source_sha256'] if Path(path).name=='gaze-command-table-v6.json')
    table = json.loads(table_path.read_text())
    # Recorded servo order is the frozen bounded-action contract, not an inferred
    # order from USD traversal or motor-array storage.
    recipe = next(Path(path).parent for path in r['source_sha256'] if Path(path).name=='bounded_actions.py')
    sys.path.insert(0,str(recipe))
    from bounded_actions import JOINT_NAMES
    hids = [JOINT_NAMES.index(name) for name in table['head_joint_order']]
    with np.load(args.report.parent/'raw-trace.npz',allow_pickle=False) as data:
        result['body50_input_audit'] = validate_gaze_inputs(data)
        q = data['state'][:,:,9:37]
        effective = q[:,:,data['servo_ids']]+q[:,:,data['passive_ids']]
        for e in r['episodes']:
            i=e['episode_index']
            unready=int((~data['actor_inferred_mask'][:,i]).sum())
            if e['sensor_unready_control_steps']!=unready or e['uninterrupted_sensor_control']!=(unready==0):
                raise ValueError('Sensor-outage result differs from actual inference mask')
            case=next(c for c in r['cases'] if c['name']==e['case'])
            target=np.asarray(case['head_delta_target_rad'])+data['native_home14'][i,hids]
            error=effective[1600:,i][:,hids]-target
            if not np.isclose(np.sqrt(np.mean(error*error)),e['head_tracking_rmse_rad_after8s'],atol=1e-6,rtol=0):
                raise ValueError('Head tracking report differs from actual physical joints and declared target')
    for group in result['groups']:
        group['uninterrupted_sensor_control_trials']=sum(e['uninterrupted_sensor_control'] for e in r['episodes'] if e['case']==group['case'])
    result['schema']='microduck.body50-gaze-physical-audit.v1'
    result['head_tracking_independently_recomputed']=True
    result['auditor_sha256']=sha(__file__)
    result['limitation']='Development test of a virtual BODY50 pipeline and physical camera motion; no formal task or physical hardware qualification.'
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'evidence_integrity_passed':True,'groups':len(result['groups']),'qualification_claimed':False}))


if __name__=='__main__':
    main()
