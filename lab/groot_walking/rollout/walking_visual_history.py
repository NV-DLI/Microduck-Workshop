"""Causal four-view memory proposal, independent of object detectors or simulator truth.

Decision-cadence proposal; no qualified deployment uses this module yet. Both data and a future bridge must call the
same transform on every 12.5 Hz observation, before choosing an action.
"""
from collections import deque
import math
from learning.walking import finite, contract as base_contract
LAGS=(72,48,24,0)
IMAGE_KEYS=tuple(f'observation.images.frame{i:02d}' for i in range(4))
# Head encoder offsets in the base58 are ordered joint indices5:9 after gyro/gravity.
PAST_INDICES=tuple(range(6))+tuple(range(11,15))+tuple(range(48,56))


def specification(joint_names):
 base=base_contract(joint_names)
 if list(joint_names[5:9])!=["neck_pitch","head_pitch","head_yaw","head_roll"]:raise ValueError("Expected the preserved motor ordering")
 names=[n.replace('previous_request/', 'previous_decision_request/') for n in base['state_names']]
 for lag in LAGS[:-1]:
  names += [f'past{lag}/'+base['state_names'][i].replace('previous_request/', 'previous_decision_request/') for i in PAST_INDICES]
  names += [f'past{lag}/age_s',f'past{lag}/integrated_vertical_gyro_rad']
  names += [f'past{lag}/integrated_request_{axis}'for axis in ('forward','lateral','yaw')]
 assert len(names)==127
 return dict(**{k:v for k,v in base.items()if k not in('version','state_names','observation')},version=3,state_names=names,
   observation='Four causal RGB captures, current proprioception and previous 12.5 Hz decision request (zero before the first decision), paired past sensors/decisions and measured/request integration; no object pose or teacher phase.',
   visual_history=dict(lags_samples=list(LAGS),image_keys=list(IMAGE_KEYS),padding='Repeat first observed sample of the physical episode',sample_hz=12.5,
    previous_request_semantics='Previous complete 8D decision at the preceding 80 ms sample, not the 20 ms motor command; first decision sees zeros. Rejected or skill-suppressed requests remain requests, not measured execution.',
    integrals='Trapezoidal integration at 12.5Hz; vertical gyro uses measured gravity. Request integrals are NOT measured displacement or pose.',
    reset='Physical episode only; history advances even when serving an existing action chunk.'))


class VisualHistory:
 def __init__(self):
  self.rows=deque(maxlen=73);self.index=0
 def observe(self,*,state,image,image_timestamp,timestamp):
  state=finite(state,58)
  if not all(math.isfinite(v)and v>=0 for v in (timestamp,image_timestamp)) or not 0<=timestamp-image_timestamp<=.08+1e-8:raise ValueError('Fresh causal image required')
  if self.rows and not math.isclose(timestamp-self.rows[-1]['timestamp'],.08,abs_tol=1e-8):raise ValueError('Contiguous 12.5Hz observations required')
  # Work on local values first; invalid inputs never advance the memory.
  g=state[3:6];norm=math.sqrt(sum(v*v for v in g))
  if norm<.5 or norm>1.5:raise ValueError('Measured gravity direction required')
  rate=-sum(w*v for w,v in zip(state[:3],g))/norm
  integral=[0.]*4
  if self.rows:
   prev=self.rows[-1]
   integral=[prev['integral'][0]+.04*(prev['rate']+rate)]+[prev['integral'][i+1]+.04*(prev['state'][48+i]+state[48+i])for i in range(3)]
  row=dict(index=self.index,timestamp=timestamp,image_timestamp=image_timestamp,state=state,image=image,rate=rate,integral=integral)
  self.rows.append(row);self.index+=1
  selected=[self.rows[max(0,len(self.rows)-1-lag)]for lag in LAGS]
  output=list(state)
  for past in selected[:-1]:
   output += [past['state'][i]for i in PAST_INDICES]
   output += [timestamp-past['timestamp']]+[a-b for a,b in zip(integral,past['integral'])]
  assert len(output)==127
  return dict(state=output,images=[r['image']for r in selected],image_timestamps=[r['image_timestamp']for r in selected],sample_timestamps=[r['timestamp']for r in selected],sample_indices=[r['index']for r in selected])
