#!/usr/bin/env python3
"""Execute long episode: Microduck simulation and control."""
import hashlib,json,sys,time
from pathlib import Path
M=Path(__file__).resolve().parents[1];F=M/'football';sys.path.insert(0,str(F))
import isaac_execute
import nav2_bridge
BaseSession=nav2_bridge.Nav2Session

class BoundedLifetimeSession(BaseSession):
 def __init__(self,output,max_wall_s=1800):
  super().__init__(output,max_wall_s=1800.)
  Path(output).mkdir(exist_ok=True)
  (Path(output)/'declared-lifetime-adapter.json').write_text(json.dumps({'requested_wall_seconds':max_wall_s,'effective_wall_seconds':1800.,'owner_cap_unchanged':True,'algorithm_or_physics_changed':False},indent=2)+'\n')

if __name__=='__main__':
 output=Path(sys.argv[sys.argv.index('--output')+1]);record=output.with_name(output.name+'-launcher.json')
 if record.exists():raise FileExistsError('Retain each launcher record')
 files=[Path(__file__),F/'isaac_execute.py',F/'nav2_bridge.py',F/'nav2_service/owner.py'];hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
 report={'schema':'microduck.bounded-lifetime-launcher.v1','qualification_claimed':False,'argv':sys.argv,'max_service_wall_seconds':1800.,'source_sha256':hashes,'status':'starting','started_unix_s':time.time()};record.write_text(json.dumps(report,indent=2)+'\n')
 nav2_bridge.Nav2Session=BoundedLifetimeSession
 try:
  isaac_execute.main();report['status']='completed'
 except BaseException as e:
  report.update(status='failed',error=repr(e));raise
 finally:
  report['source_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items());report['ended_unix_s']=time.time();record.write_text(json.dumps(report,indent=2)+'\n')
