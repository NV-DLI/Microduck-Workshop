"""Compare retained matched startup worlds; never infer full task success from5s."""
import hashlib,json
from pathlib import Path
import numpy as np
O=Path('@COURSE@/outputs/migration-20261019')
base=O/'gait-v9-4000-dual-startup-reference5s-v1'; dual=O/'gait-dual-v12-stop-dev-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[];sources={}
for case in ['forward_positive','forward_negative','yaw_positive','yaw_negative']:
 a,b=[json.loads((root/case/'report.json').read_text()) for root in (base,dual)]
 assert a['seed']==b['seed']==978961
 assert a['physical_initialization']['physical_ticks']==b['physical_initialization']['physical_ticks']==472
 paths=[root/case/'raw-trace.npz' for root in (base,dual)]
 with np.load(paths[0],allow_pickle=False) as x,np.load(paths[1],allow_pickle=False) as y:
  assert np.array_equal(x['initial_startup_completed_mask'],y['initial_startup_completed_mask'])
  # Same warmup model, physical recipe, seed, and duration. First task action differs.
  initial_obs_diff=float(np.abs(x['actor_observation'][0]-y['actor_observation'][0]).max())
  initial_previous_diff=float(np.abs(x['initial_previous_raw']-y['initial_previous_raw']).max())
  assert initial_obs_diff<=1e-6 and initial_previous_diff<=1e-6
  for i in range(2):
   info={'case':case,'world':i,'initial_observation_max_difference':initial_obs_diff,
         'initial_previous_raw_max_difference':initial_previous_diff}
   for name,d in [('walk_only',x),('dual',y)]:
    hits=np.flatnonzero(d['state'][:,i,-1]>0)
    info[name]={'first_fall_s':float(d['time_s'][hits[0]]) if len(hits) else None,
     'initial_sensor_gravity_tilt_rad':float(np.arccos(np.clip(-d['actor_observation'][0,i,5],-1,1))),
     'duration_s':float(d['time_s'][-1])}
   records.append(info)
 for p in paths+[base/case/'report.json',dual/case/'report.json']:sources[str(p)]=sha(p)
assert [(r['case'],r['world']) for r in records if r['walk_only']['first_fall_s'] is not None]==[('yaw_positive',1)]
failed=next(r for r in records if r['walk_only']['first_fall_s'] is not None)
assert failed['dual']['first_fall_s'] is not None
for p in [base/'independent-audit.json',dual/'independent-physical-audit.json',dual/'independent-dual-audit.json',Path(__file__)]:sources[str(p)]=sha(p)
out={'schema':'microduck.dual-policy-startup-reference.v1','technical_passed':True,'qualification_claimed':False,
 'records':records,'sources':sources,
 'finding':'The same yaw_positive world1 falls within0.225s with the walk actor alone and0.220s with the dual actor. Initial61D observations and prior raw action match within1e-6; both used472 real warmup ticks. The early failure is already present in the parent startup state, so it is not evidence that the dual blend caused this fall.',
 'limits':'Reference lasts5s, dual78s. Reference cannot validate full stop/restart. All worlds retained. Dual remains0/8 strict and0/24 restart tracking; no promotion.'}
p=O/'gait-dual-v12-matched-startup-comparison-v1.json'
with p.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps({'technical_passed':True,'same_early_failure':failed,'output':str(p)},indent=2))
