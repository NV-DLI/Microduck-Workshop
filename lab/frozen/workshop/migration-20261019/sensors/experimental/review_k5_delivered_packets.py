"""Independent read-only public delivered-packet attribution; no task re-scoring."""
from pathlib import Path
import hashlib
import json
import numpy as np

C=Path('@COURSE@');M=C/'workshop/migration-20261019';O=C/'outputs/migration-20261019'
run=O/'kick-v5-target-blend100-heldout32-s979801'
m=json.loads((run/'manifest.json').read_text())
d=np.load(run/'body50-delivered-packets.npz');s=np.load(run/'physical-training-trace.npz')['state'];c=np.load(run/'evaluation-conditions.npz')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(run/'body50-delivered-packets.npz')==m['physical_trace']['delivered_body50_packets']['sha256']
assert sha(run/'physical-training-trace.npz')==m['physical_trace']['sha256']
assert sha(run/'evaluation-conditions.npz')==m['conditions']['sha256']
rows=[]
for i in np.flatnonzero(c['finish_reason']==4):
    k=int(c['finish_tick'][i])-1
    start=max(0,k-20);last=np.flatnonzero(d['orientation_valid'][:k,i]);prior=int(last[-1])if len(last)else None
    distinct=np.r_[True,np.diff(d['seq'][start:k+1,i])!=0]
    seq=d['seq'][start:k+1,i][distinct]
    sample={name:d[name][k,i].tolist()for name in d.files}
    rows.append({'world':int(i),'termination_physical_tick_one_based':k+1,
        'kick_phase_at_termination_s':float(s[k,i,81]),'before_authorized_kick':bool(s[k,i,81]<0),
        'delivered_packet_at_termination':sample,'last_valid_physics_row_zero_based':prior,
        'previous_physics_row_valid':bool(d['orientation_valid'][k-1,i]),
        'distinct_bus_reads_in_previous100ms':int(len(seq)),'bus_sequence_increased_by_one':bool(np.all(np.diff(seq)==1)),
        'same_episode_in_previous100ms':bool(np.all(s[start:k+1,i,82]==s[k,i,82])),
        'actual_body_gyro_abs_peak_previous100ms_rad_s':np.max(abs(s[start:k+1,i,10:13]),axis=0).tolist(),
        'classification':'delivered gyroZ clipped -> conservative approximate fusion validity lost; prior valid stream and continuing fresh reads, not initial warmup or transport gap',
        'classification_supported':bool(not sample['orientation_valid']and sample['gyro_clipped'][2]and prior==k-1 and sample['live_quaternion_reads']>=25 and np.all(np.diff(seq)==1))})
files=[Path(__file__),run/'manifest.json',run/'body50-delivered-packets.npz',run/'physical-training-trace.npz',run/'evaluation-conditions.npz',run/'independent-audit.json',O/'kick-v3-gyro-outage-independent-audit-v1.json',M/'sensors/sources/body-imu/duck-control__src__imu.rs',M/'acceptance/body50_training/vector_body_imu.py']
result={'schema':'microduck.kick-v5-delivered-body50-independent-review.v1','run':str(run),'all_worlds':32,
    'sensor_outage_terminal_worlds':len(rows),'all_outages_attributed_from_delivered_flags':all(r['classification_supported']for r in rows),
    'gyro_axis':'Z for all17 recorded terminal packets','physical_task_restart_or_transport_gap_claimed':False,
    'before_kick_authorization_count':sum(r['before_authorized_kick']for r in rows),
    'source_evidence':{str(p):sha(p)for p in files},'rows':rows,
    'limits':['Packets are delivered50Hz values held at200Hz. Repeated seq is not a new measurement.',
              'Absence of gyro_clipped in a held packet would not rule out an intermediate120Hz fusion clip. Here every terminal packet positively retains gyroZ clip, so no hidden-clip inference is required.',
              'Official±500deg/s range and.0175deg/s count scale match the model. Clip-triggered fusion invalidation is our conservative approximation, not a verified ST SFLP firmware response.',
              'No trajectory after termination is credited to goal or recovery; no thresholds or source models were changed.'],
    'scope':'Read-only evidence classification, not a sensor hardware accuracy or football success claim.'}
output=O/'kick-v5-body50-delivered-outage-independent-review-v1.json';output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'output':str(output),'outages':len(rows),'supported':result['all_outages_attributed_from_delivered_flags'],'before_authorization':result['before_kick_authorization_count']}))
