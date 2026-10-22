"""Exact-ID/label owner for one isolated CPU-only Docker service; no existing service writes."""
import hashlib,json,os,signal,subprocess,time,uuid
from pathlib import Path
if __package__:
 from .contract import atomic,read
else:
 from contract import atomic,read
IMAGE='sha256:c60e955f781c31c00b46b415c9bad7dc8e5faa922f0e3ea0dbb44f604685bfbd'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(source):
 source=Path(source);m=read(source/'source-manifest.json');actual={str(p.relative_to(source))for p in source.rglob('*')if p.is_file()and p!=source/'source-manifest.json'and '__pycache__'not in p.parts and p.suffix!='.pyc'}
 if actual!=set(m['files']):raise ValueError('Complete bridge source inventory differs')
 for rel,digest in m['files'].items():
  p=source/rel
  if p.is_symlink()or sha(p)!=digest:raise ValueError('Bridge source changed: '+rel)
 return sha(source/'source-manifest.json')
class ServiceOwner:
 def __init__(self,source_dir,exchange_dir,output_dir,run_nonce,max_wall_s=1200):
  self.source=Path(source_dir).resolve();self.exchange=Path(exchange_dir).resolve();self.output=Path(output_dir).resolve();self.nonce=run_nonce;self.max_wall_s=max_wall_s;self.token=uuid.uuid4().hex;self.name='microduck-beta-nav2-'+self.token;self.identifier=None;self.process=None;self.log=None;self.cleanup=None
  if not isinstance(run_nonce,str)or not 16<=len(run_nonce)<=128 or not 1<=max_wall_s<=1800:raise ValueError('Explicit nonce/bounded lifetime required')
  if self.output==self.source or self.source in self.output.parents or self.exchange==self.source or self.source in self.exchange.parents:raise ValueError('Output outside source required')
 def start(self,timeout_s=20):
  pin=verify(self.source);self.source_pin=pin;self.exchange.mkdir(parents=True,exist_ok=False);self.output.mkdir(parents=True,exist_ok=False);(self.output/'result').mkdir()
  image=read_image=json.loads(subprocess.check_output(['docker','image','inspect',IMAGE]))[0]
  if image['Id']!=IMAGE:raise RuntimeError('Local image ID differs')
  command=['docker','create','--name',self.name,'--label','course.beta_nav2_token='+self.token,'--runtime=runc','--network=none','--cpus=1','--memory=384m','--memory-swap=384m','--pids-limit=128','--shm-size=32m','--read-only','--cap-drop=ALL','--security-opt=no-new-privileges','--user',f'{os.getuid()}:{os.getgid()}','--tmpfs','/tmp:rw,noexec,nosuid,size=32m']
  env={'CUDA_VISIBLE_DEVICES':'','NVIDIA_VISIBLE_DEVICES':'void','PYTHONDONTWRITEBYTECODE':'1','ROS_DOMAIN_ID':'179','ROS_LOCALHOST_ONLY':'1','RMW_IMPLEMENTATION':'rmw_fastrtps_cpp','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','ROS_LOG_DIR':'/bridge-output/ros-logs'}
  for k,v in env.items():command+=['--env',k+'='+v]
  for src,dst,ro in[(self.source,'/bridge-src',True),(self.exchange,'/bridge-exchange',False),(self.output,'/bridge-output',False)]:command+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly'if ro else'')]
  # Arguments/nonce are passed as positional shell arguments, never interpolated shell code.
  command+=['--entrypoint','/bin/bash',IMAGE,'-c','source /opt/ros/jazzy/setup.bash && /usr/bin/python3 -B /bridge-src/image_preflight.py --output /bridge-output/image-preflight.json && exec /usr/bin/python3 -B /bridge-src/service.py --exchange /bridge-exchange --output /bridge-output/result --run-nonce "$1" --max-wall-s "$2"','bridge-entry',self.nonce,str(self.max_wall_s)]
  atomic(self.output/'plan.json',{'kind':'persistent_Navfn_CPU_owner_v1','source_sha256':pin,'image_id':IMAGE,'container_name':self.name,'ownership_token':self.token,'run_nonce':self.nonce,'max_wall_s':self.max_wall_s,'cpu_limit':1,'memory_MiB':384,'runtime':'runc','network':'none','GPU':False,'command':command})
  try:
   self.identifier=subprocess.check_output(command,text=True,timeout=20).strip();self.log=(self.output/'container.log').open('wb');self.process=subprocess.Popen(['docker','start','-a',self.identifier],stdout=self.log,stderr=subprocess.STDOUT);end=time.monotonic()+timeout_s
   while time.monotonic()<end:
    p=self.exchange/'ready.json'
    if p.exists():
     r=read(p)
     if r.get('run_nonce')==self.nonce and r.get('accepting_requests'):return r
    if self.process.poll()is not None:raise RuntimeError('Service exited before ready')
    time.sleep(.02)
   raise TimeoutError('Service readiness deadline')
  except BaseException:self.close();raise
 def close(self):
  if self.cleanup is not None:return self.cleanup
  result={'owned_container':self.name,'ownership_token':self.token,'container_id':self.identifier,'removed':False,'owned_container_removed':False,'owned_processes_zero':False,'service_status':None,'container_exit_code':None}
  try:
   if self.identifier is not None:
    state=json.loads(subprocess.check_output(['docker','inspect',self.identifier],timeout=10))[0]
    if state['Id']!=self.identifier or state['Config']['Labels'].get('course.beta_nav2_token')!=self.token:raise RuntimeError('Exact container ID/token differs; refusing cleanup')
    if state['State']['Running']:
     atomic(self.exchange/'shutdown.json',{'run_nonce':self.nonce})
     try:self.process.wait(timeout=15)
     except subprocess.TimeoutExpired:subprocess.run(['docker','stop','--time','10',self.identifier],check=True,timeout=20,stdout=subprocess.DEVNULL)
    if self.process is not None:self.process.wait(timeout=10)
    state=json.loads(subprocess.check_output(['docker','inspect',self.identifier],timeout=10))[0];atomic(self.output/'container-final-inspect.json',state);result['container_exit_code']=state['State']['ExitCode'];result['final_running']=state['State']['Running']
    subprocess.run(['docker','rm',self.identifier],check=True,timeout=10,stdout=subprocess.DEVNULL)
    post=subprocess.run(['docker','ps','-aq','--filter','label=course.beta_nav2_token='+self.token],capture_output=True,text=True,check=True,timeout=10)
    result.update(removed=True,owned_container_removed=True,owned_processes_zero=not post.stdout.strip(),postscan_token_matches=post.stdout.split())
   if hasattr(self,'source_pin')and verify(self.source)!=self.source_pin:raise RuntimeError('Bridge source changed during ownership')
   summary=self.output/'result/summary.json'
   if summary.exists():result['service_status']=read(summary)['status'];result['summary_sha256']=sha(summary)
  except BaseException as exc:result['error']=repr(exc)
  finally:
   if self.log is not None:self.log.close()
   self.cleanup=result
   if self.output.exists():atomic(self.output/'cleanup.json',result)
  return result
