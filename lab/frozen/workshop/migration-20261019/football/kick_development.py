"""Explicit development pulse of the downloaded official v8 kick actor.

No deployment approval is inferred from matching ONNX metadata. A calibrated
placement is an initialization experiment, not an autonomous approach success.
"""
from pathlib import Path
import hashlib,math

class KickPulse:
 def __init__(self,start_after_release_s,duration_s):
  if not all(math.isfinite(x)for x in (start_after_release_s,duration_s))or start_after_release_s<.5 or not .1<=duration_s<=1.:raise ValueError('Explicit .5s+ standing delay and .1–1s development pulse required')
  self.start=start_after_release_s;self.duration=duration_s
 def active(self,task_time_s):return task_time_s is not None and self.start<=task_time_s<self.start+self.duration

def validate_actor(path,joint_names):
 import onnx
 model=onnx.load(path);meta={m.key:m.value for m in model.metadata_props}
 if meta.get('joint_names','').split(',')!=list(joint_names):raise ValueError('Kick joint order differs from native walk')
 if float(meta.get('action_scale','nan'))!=1.:raise ValueError('Unverified kick action scaling')
 ins=model.graph.input;outs=model.graph.output
 if len(ins)!=1 or len(outs)!=1 or [d.dim_value for d in ins[0].type.tensor_type.shape.dim]!=[1,61]or[d.dim_value for d in outs[0].type.tensor_type.shape.dim]!=[1,14]:raise ValueError('Kick61obs/14action required')
 return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'metadata':meta,'scope':'official v8 model; new limited-action contact development only','known_prior_unadapted_limit_failures_retained':True,'deployment_qualified':False}
