"""Explicit four-camera-time packet decoder for the private walking history trial."""
import base64,io,math
from walking_visual_history import IMAGE_KEYS,LAGS

def decode_observation(packet, contract, expected_state):
 import numpy as np
 import torch
 from PIL import Image
 if packet.get('control_contract') != contract or expected_state!=127:
  raise ValueError('Exact history contract and 127 state dimensions required')
 if contract.get('visual_history',{}).get('image_keys')!=list(IMAGE_KEYS) or contract['visual_history'].get('lags_samples')!=list(LAGS):
  raise ValueError('Four ordered causal views required')
 state=np.asarray(packet['state'],dtype=np.float32)
 if state.shape!=(127,) or not np.isfinite(state).all():raise ValueError('Invalid history state')
 images=packet.get('images_base64'); captures=packet.get('history_image_timestamps');samples=packet.get('history_sample_timestamps')
 if not all(isinstance(v,list)and len(v)==4 for v in (images,captures,samples)):raise ValueError('Exactly four images and their original timestamps required')
 now=packet['observation_timestamp']
 if not all(isinstance(v,(int,float))and math.isfinite(v)and v>=0 for v in [now,*captures,*samples]):raise ValueError('Finite nonnegative observation times required')
 if abs(samples[-1]-now)>1e-8 or any(a>b+1e-8 for a,b in zip(samples,samples[1:])):raise ValueError('Past to present order required')
 ages=[now-t for t in samples]
 if any(age< -1e-8 or age>lag*.08+1e-8 for age,lag in zip(ages,LAGS)):raise ValueError('History age outside the declared window')
 # Before enough frames exist all unavailable past views repeat the first sample.
 first=now-ages[0]
 for i,lag in enumerate(LAGS):
  expected=max(first,now-lag*.08)
  if abs(samples[i]-expected)>1e-8:raise ValueError('Wrong history sample spacing')
  if not -1e-8<=samples[i]-captures[i]<=.08+1e-8:raise ValueError('Stale or future image at historical sample')
 for index,age in zip((76,99,122),ages[:-1]):
  if not math.isclose(float(state[index]),age,rel_tol=1e-6,abs_tol=1e-6):raise ValueError('History state/image ages differ')
 batch={'observation.state':torch.from_numpy(state).unsqueeze(0),'task':[str(packet['task'])]}
 for key,data in zip(IMAGE_KEYS,images,strict=True):
  if not isinstance(data,str)or len(data)>2_000_000:raise ValueError('Invalid image payload')
  image=Image.open(io.BytesIO(base64.b64decode(data,validate=True)))
  if image.size!=(960,540):raise ValueError('Original sensor image size required')
  rgb=np.array(image.convert('RGB').resize((480,270)))
  batch[key]=torch.from_numpy(rgb).permute(2,0,1).float().div(255).unsqueeze(0)
 return batch
