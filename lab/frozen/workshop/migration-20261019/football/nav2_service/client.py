"""Synchronous single-client file bridge; no simulated time is advanced here."""
import time
from pathlib import Path
if __package__:
 from .contract import atomic,read,digest,validate_response,SCHEMA
else:
 from contract import atomic,read,digest,validate_response,SCHEMA
class BridgeClient:
 def __init__(self,exchange_dir,run_nonce):self.directory=Path(exchange_dir);self.run_nonce=run_nonce;self.last_sequence=-1;self.last_sim=-1.
 def query(self,case,provenance,sequence,sim_query_s,timeout_s=5.):
  if not 0<timeout_s<=20:raise ValueError('Bounded wall timeout required')
  if sequence<=self.last_sequence or sim_query_s<=self.last_sim:raise ValueError('Monotone new request required')
  ready=read(self.directory/'ready.json')
  if ready['run_nonce']!=self.run_nonce or not ready['accepting_requests']:raise ValueError('Wrong/unready service')
  request={'schema':SCHEMA,'run_nonce':self.run_nonce,'sequence':sequence,'sim_query_s':sim_query_s,'sent_monotonic_ns':time.monotonic_ns(),'case':case,'provenance':provenance}
  atomic(self.directory/'request.json',request);self.last_sequence=sequence;self.last_sim=sim_query_s;end=time.monotonic()+timeout_s
  while time.monotonic()<end:
   p=self.directory/'response.json'
   if p.exists():
    r=read(p)
    if r.get('sequence')==sequence and r.get('request_sha256')==digest(request):return validate_response(r,request,sim_query_s)
   time.sleep(.02)
  raise TimeoutError('No matching Nav2 response; old path must not authorize movement')
 def shutdown(self):atomic(self.directory/'shutdown.json',{'run_nonce':self.run_nonce})
