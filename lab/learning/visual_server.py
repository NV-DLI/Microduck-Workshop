"""Serve the small visual baseline through the same walking skill interface."""
import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import time
import torch
try:
    from .visual_baseline import SpatialEncoder,VisualPolicy,IMAGE_KEYS,read,sha
    from .visual_packet import decode_observation
    from .walking import bound_request,TASK
except ImportError:
    from visual_baseline import SpatialEncoder,VisualPolicy,IMAGE_KEYS,read,sha
    from visual_packet import decode_observation
    from walking import bound_request,TASK

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--port',type=int,default=8765)
    p.add_argument('--log',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--action-execution-steps',type=int,default=4);a=p.parse_args()
    if not 1<=a.action_execution_steps<=16:raise ValueError('Execution prefix must fit training horizon')
    info=read(a.checkpoint/'completion.json');contract=read(a.checkpoint/'control-contract.json')
    if not info['complete']or info['architecture']!='frozen_resnet18_spatial_gru':raise ValueError('Completed visual baseline required')
    encoder_file=Path(info['train_cache'])/'encoder.pt'
    if sha(encoder_file)!=info['encoder_sha256']:raise ValueError('Encoder identity differs')
    torch.set_num_threads(4)
    encoder=SpatialEncoder(pretrained=False).to(a.device)
    encoder.load_state_dict(torch.load(encoder_file,map_location=a.device,weights_only=True));encoder.eval()
    checkpoint=torch.load(a.checkpoint/'policy.pt',map_location=a.device,weights_only=True)
    model=VisualPolicy().to(a.device);model.load_state_dict(checkpoint['model']);model.eval()
    sm,ss,am,ascale=[torch.tensor(checkpoint[k],device=a.device)for k in ['state_mean','state_std','action_mean','action_std']]
    horizon=dict(native_prediction_horizon=16,saved_execution_horizon=16,actual_queue_steps=a.action_execution_steps,actual_queue_capacity=a.action_execution_steps)
    identity=dict(ready=True,method='visual_resnet18_gru',state_dim=127,control_contract=contract,checkpoint=str(a.checkpoint.resolve()),checkpoint_sha256=sha(a.checkpoint/'policy.pt'),encoder_sha256=info['encoder_sha256'],native_horizon_contract=horizon)
    queue=deque();previous=-1;episode=None
    with a.log.open('x',buffering=1)as log:
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path!='/health':self.send_error(404);return
                self.send_response(200);self.end_headers();self.wfile.write(json.dumps(identity).encode())
            def do_POST(self):
                nonlocal previous,episode
                if self.path!='/act':self.send_error(404);return
                try:
                    length=int(self.headers.get('Content-Length',0))
                    if not 0<length<8_000_000:raise ValueError('Invalid payload length')
                    start_all=time.monotonic();packet=json.loads(self.rfile.read(length));seq=packet['sequence']
                    if packet.get('task')!=TASK:raise ValueError('This trained baseline is single-task only')
                    if packet.get('reset')is True:
                        if seq!=0:raise ValueError('Only physical episode reset is supported')
                        queue.clear();previous=-1;episode=packet['episode_id']
                    if seq!=previous+1 or packet['episode_id']!=episode:raise ValueError('Noncontiguous observation or unreset episode')
                    batch=decode_observation(packet,contract,127);prediction=not queue;started=time.monotonic()
                    if prediction:
                        with torch.inference_mode():
                            images=torch.cat([batch[k]for k in IMAGE_KEYS]).to(a.device)
                            features=encoder(images).unsqueeze(0)
                            states=(batch['observation.state'].to(a.device)-sm)/ss
                            actions=(model(features,states)[0]*ascale+am).cpu().tolist()
                        queue.extend(actions[:a.action_execution_steps])
                    raw=queue.popleft();action=bound_request(raw);previous=seq
                    result=dict(sequence=seq,episode_id=episode,reset=packet.get('reset')is True,method='visual_resnet18_gru',
                        action=action,raw_action=raw,fresh_prediction=prediction,native_horizon_contract=horizon,
                        latency_s=time.monotonic()-started,decode_and_inference_s=time.monotonic()-start_all)
                    log.write(json.dumps(result)+'\n');body=json.dumps(result).encode()
                    self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                except Exception as e:self.send_error(422,str(e))
        print('Visual baseline ready',flush=True)
        HTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
if __name__=='__main__':main()
