"""Exact-operator FP32 batch interpreter for the two pinned feedforward ONNXs.

No retraining, output clipping, normalizer update, or model-file mutation.
The caller owns actor-specific dispatch and actual previous-action feedback.
"""
from pathlib import Path
import hashlib
import numpy as np
import onnx
from onnx import helper, numpy_helper
import torch

class BatchPolicy:
    def __init__(self, path, expected_sha256, device='cpu'):
        path=Path(path)
        self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha256 != expected_sha256:
            raise ValueError('Pinned policy SHA mismatch')
        model=onnx.load(str(path))
        if len(model.graph.input)!=1 or len(model.graph.output)!=1:
            raise ValueError('Only a single feedforward input/output is supported')
        self.input=model.graph.input[0].name;self.output=model.graph.output[0].name
        self.constants={v.name:torch.from_numpy(numpy_helper.to_array(v).copy()).to(device)
                        for v in model.graph.initializer}
        self.nodes=[]
        for node in model.graph.node:
            if node.op_type not in ('Sub','Div','Gemm','Elu') or len(node.output)!=1:
                raise ValueError('Unsupported graph operator')
            attributes={a.name:helper.get_attribute_value(a) for a in node.attribute}
            self.nodes.append((node.op_type,list(node.input),node.output[0],attributes))
        self.path=str(path)

    def __call__(self, observations):
        if observations.ndim!=2 or observations.shape[1]!=61 or observations.dtype!=torch.float32:
            raise ValueError('Policy requires FP32 [batch,61] observations')
        values=dict(self.constants);values[self.input]=observations
        for op,inputs,output,attrs in self.nodes:
            a=[values[name] for name in inputs]
            if op=='Sub':result=a[0]-a[1]
            elif op=='Div':result=a[0]/a[1]
            elif op=='Elu':result=torch.nn.functional.elu(a[0],alpha=attrs.get('alpha',1.))
            else:
                left=a[0].T if attrs.get('transA',0) else a[0]
                right=a[1].T if attrs.get('transB',0) else a[1]
                result=attrs.get('alpha',1.)*(left@right)+attrs.get('beta',1.)*a[2]
            values[output]=result
        return values[self.output]

    def parity(self, observations, tolerance=3e-6):
        import onnxruntime as ort
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        session=ort.InferenceSession(self.path,sess_options=opts,providers=['CPUExecutionProvider'])
        inputs=observations.detach().cpu().numpy().astype(np.float32)
        expected=np.concatenate([session.run(None,{self.input:row[None]})[0] for row in inputs])
        with torch.inference_mode():actual=self(observations).cpu().numpy()
        error=float(np.abs(expected-actual).max())
        if not np.isfinite(error) or error>tolerance:
            raise RuntimeError(f'Batch graph numerical parity failed: {error}')
        return {'count':len(inputs),'maximum_absolute_error':error,'tolerance':tolerance,
                'device':str(observations.device),'reference':'original ONNX CPU scalar FP32',
                'model_sha256':self.sha256,'passed':True}
