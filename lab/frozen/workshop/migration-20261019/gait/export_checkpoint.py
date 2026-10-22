"""Export a local RSL-RL checkpoint without launching physics, checking parity."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('checkpoint',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if args.output.exists():raise ValueError('Refuse to overwrite an exported checkpoint')
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    obs=TensorDict({'policy':torch.zeros(1,61)},batch_size=[1])
    model=MLPModel(obs,{'actor':['policy']},'actor',14,hidden_dims=[512,256,128],activation='elu',obs_normalization=True)
    state=ckpt['actor_state_dict'].copy();state.pop('distribution.std_param')
    model.load_state_dict(state,strict=True);model.eval()
    exported=model.as_onnx(verbose=False);exported.eval()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(exported,exported.get_dummy_inputs(),str(args.output),opset_version=18,dynamo=False,
                      input_names=exported.input_names,output_names=exported.output_names)
    import onnxruntime as ort
    opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(args.output),sess_options=opts,providers=['CPUExecutionProvider'])
    generator=torch.Generator().manual_seed(970001)
    x=torch.randn(32,61,generator=generator)*.3
    with torch.inference_mode():expected=model(TensorDict({'policy':x},batch_size=[32])).numpy()
    actual=np.concatenate([session.run(None,{session.get_inputs()[0].name:row[None].numpy()})[0] for row in x])
    error=float(np.abs(expected-actual).max())
    report={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'checkpoint_iteration':int(ckpt['iter']),'output':str(args.output.resolve()),
            'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
            'parity_cases':32,'max_absolute_error':error,'passed':bool(np.isfinite(error) and error<1e-5),
            'method':'Native RSL-RL MLPModel vs exported ONNX CPU; empirical normalizer retained, stochastic std excluded for inference'}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit(2)

if __name__=='__main__':main()
