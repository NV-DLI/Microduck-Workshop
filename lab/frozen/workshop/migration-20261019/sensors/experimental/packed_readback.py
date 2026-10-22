"""Lossless grouped readback for retained physics diagnostics.

This utility has no sensor model and no control interface. World-state fields
returned by a caller remain privileged scoring data; do not pass the mapping
to a planner. Each dtype/device group uses one CPU synchronization. It never
casts double/int64 to float32, changes the sampling clock or invents samples.
"""
from collections import defaultdict
import numpy as np
import torch


def as_tensor(value):
    candidate=getattr(value,'torch',value)
    if isinstance(candidate,torch.Tensor):return candidate
    # Native BAM telemetry exposes Warp arrays. Import only when encountered.
    import warp as wp
    if isinstance(candidate,wp.array):return wp.to_torch(candidate)
    raise TypeError('Expected a native Torch/ArrayView/Warp array')


class PackedReadback:
    def __init__(self):
        self.calls=0
        self.cpu_transfers=0
        self.last_metadata=None

    def read(self,fields):
        if not isinstance(fields,dict)or not fields:raise ValueError('Named fields required')
        groups=defaultdict(list)
        for name,value in fields.items():
            if not isinstance(name,str)or not name:raise ValueError('Nonempty field names required')
            tensor=as_tensor(value)
            if tensor.dtype==torch.bfloat16:raise TypeError('NumPy cannot represent bfloat16 without casting')
            groups[(tensor.dtype,tensor.device)].append((name,tensor))
        result={};layout=[]
        for (dtype,device),members in groups.items():
            packed=torch.cat([value.detach().reshape(-1)for _,value in members])
            copied=packed.cpu().numpy()
            if copied.dtype.kind in 'fc' and not np.isfinite(copied).all():
                raise ValueError('Nonfinite native physics readback')
            cursor=0
            for name,value in members:
                count=value.numel()
                result[name]=copied[cursor:cursor+count].reshape(tuple(value.shape))
                cursor+=count
            layout.append({'dtype':str(dtype),'device':str(device),'values':cursor,'fields':[name for name,_ in members]})
        self.calls+=1;self.cpu_transfers+=len(groups)
        self.last_metadata={'groups':layout,'cpu_transfers':len(groups),'dtype_preserved':True}
        return result

    def metadata(self):
        return {'calls':self.calls,'cpu_transfers':self.cpu_transfers,'last':self.last_metadata,
                'sampling_changed':False,'world_state_is_controller_input':False,
                'scope':'Transport only; a caller must retain its sensor/scorer information boundary'}
