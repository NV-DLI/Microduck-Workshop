"""GR00T's walking skill requests, using only previously captured RGB and BODY50.

Simulation waits for the workstation response. The inference server must carry
the exact walking action contract; a roller checkpoint is rejected.
"""
import base64
import io
import json
from pathlib import Path
import urllib.request

from .walking import TASK, KickLatch, bound_request, contract, pack_state


class WalkingPolicy:
    def __init__(self, server, output, joint_names, seed):
        self.server = server.rstrip('/')
        self.contract = contract(joint_names)
        with urllib.request.urlopen(self.server+'/health',timeout=20) as response:
            self.identity=json.load(response)
        if not self.identity.get('ready') or self.identity.get('control_contract') != self.contract:
            raise ValueError('A trained walking checkpoint with the matching contract is required')
        self.output=Path(output)/'walking-learning'
        self.output.mkdir()
        self.log=(self.output/'samples.jsonl').open('x',buffering=1)
        (self.output/'identity.json').write_text(json.dumps(self.identity,indent=2))
        self.seed=seed
        self.image=None
        self.image_stamp=None
        self.previous=[0.]*8
        self.request=[0.]*8
        self.latch=KickLatch()
        self.sequence=0
        self.next_inference_tick=None
        self.last_tick=None

    def capture(self, rgb, timestamp):
        from PIL import Image
        stream=io.BytesIO()
        Image.fromarray(rgb).save(stream,format='PNG')
        self.image=base64.b64encode(stream.getvalue()).decode()
        self.image_stamp=timestamp

    def step(self, *, timestamp, packet, default_q, previous_raw, enabled):
        t=timestamp; tick=round(t*50)
        if self.last_tick is not None and tick != self.last_tick+1:
            raise ValueError('Walking bridge requires contiguous 50 Hz updates')
        self.last_tick=tick
        if not enabled or packet is None:
            self.previous=[0.]*8
            return self.previous, None
        if not packet.get('orientation_valid') or not 0 <= t-packet['read_timestamp_s'] <= .08:
            raise ValueError('Stale walking sensor packet')
        if self.image is None or not 0 <= t-self.image_stamp <= .08+1e-9:
            raise ValueError('Missing fresh walking RGB')
        body=(packet['gyro_body_rad_s']+packet['gravity_body']
              +[q-home for q,home in zip(packet['encoder_q_rad'],default_q)]
              +packet['encoder_qd_rad_s']+list(previous_raw)+[0.]*13)
        phase=self.latch.phase(t)
        state=pack_state(body,self.previous,skill_phase_s=phase)
        if self.next_inference_tick is None:
            self.next_inference_tick=((tick+3)//4)*4
        if tick==self.next_inference_tick:
            payload=dict(sequence=self.sequence,reset=self.sequence==0,episode_seed=self.seed,
                         episode_id=self.output.parent.name,control_contract=self.contract,
                         image_base64=self.image,state=state,task=TASK)
            request=urllib.request.Request(self.server+'/act',data=json.dumps(payload).encode(),
                                           headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=120) as response:
                prediction=json.load(response)
            self.request=bound_request(prediction['action'])
            self.log.write(json.dumps(dict(timestamp_s=t,image_capture_s=self.image_stamp,
                                          sequence=self.sequence,state=state,request=self.request,
                                          prediction=prediction))+'\n')
            self.sequence+=1;self.next_inference_tick+=4
        desired=self.latch.update(t,self.request,gyro=packet['gyro_body_rad_s'],
                                 gravity=packet['gravity_body'],qd=packet['encoder_qd_rad_s'],
                                 previous_head=self.previous[3:7])
        # The local motor skill requires neutral head commands. Returning to
        # neutral is a bounded command transition, never a joint/pose teleport.
        if self.latch.started is None:
            desired[3:7]=[old+max(-.015,min(.015,new-old))
                          for old,new in zip(self.previous[3:7],desired[3:7])]
        self.previous=desired
        return desired,self.latch.phase(t)
