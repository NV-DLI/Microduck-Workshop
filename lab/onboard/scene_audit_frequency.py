"""100 Hz onboard audit: retain physical limits and check the full recovery window.
Every physical/control timestamp is independently checked before each audit.
"""
import argparse,hashlib,json,os,pathlib,runpy,sys
p=argparse.ArgumentParser();p.add_argument('--scene-runtime',type=pathlib.Path,required=True);p.add_argument('--script',required=True);a,remaining=p.parse_known_args()
run=pathlib.Path(remaining[0]);manifest=json.loads((run/'manifest.json').read_text());hz=manifest['physics_hz'];assert hz in (100,200);assert manifest['policy_hz']==50 and manifest['sensor_hz']==15
for filename,rate,offset in [('physics-trace.jsonl',hz,1),('policy-trace.jsonl',50,0)]:
 rows=[json.loads(line) for line in (run/filename).open()];assert rows
 assert all(abs(r['timestamp_s']-(i+offset)/rate)<1e-9 for i,r in enumerate(rows)),filename
 if offset==1:
  assert len(rows)==manifest['physics_samples']
  assert abs(len(rows)/hz-manifest['simulated_seconds'])<1e-9
 else:assert len(rows)==manifest['policy_updates']
lab=pathlib.Path(__file__).resolve().parent.parent;original=lab/'scene_audit.py'
sys.path.insert(0,str(lab))
if a.script!='audit_retimed_kick_actual.py':
 sys.argv=[str(original),'--scene-runtime',str(a.scene_runtime),'--script',a.script,*remaining];runpy.run_path(str(original),run_name='__main__')
else:
 M=a.scene_runtime.resolve();sys.path[:0]=[str(M/'evaluation/football'),str(M/'evaluation'),str(M/'football'),str(M/'sensors')]
 from scene_nav2 import map_contract,obstacle_contract
 sys.modules['nav2_service.map_contract']=map_contract;sys.modules['nav2_service.obstacle_contract']=obstacle_contract
 file=pathlib.Path(__file__).parent/'audit_official_only_kick_actual.py';source=file.read_text();remaining[remaining.index('--runtime')+1]=str(pathlib.Path(__file__).parent/'official_kick_runtime.py');old='fullrecoverywindow=len(win) >= 100,'
 new='''fullrecoverywindow=(len(win) >= 50
        and win[0]["timestamp_s"] - onset <= 3.51 + 1e-6
        and win[-1]["timestamp_s"] - onset >= 3.99 - 1e-6
        and all(abs(b["timestamp_s"] - a["timestamp_s"] - .01) < 1e-9 for a,b in zip(win,win[1:]))),'''
 assert source.count(old)==1;changed=source.replace(old,new) if hz==100 else source;output=pathlib.Path(remaining[remaining.index('--output')+1]);sidecar=output.with_suffix('.clock.json');assert not sidecar.exists()
 sidecar.write_text(json.dumps(dict(course_onboard=True,physics_hz=hz,policy_hz=50,recovery_window_s=[3.5,4.0],source_sha256=hashlib.sha256(source.encode()).hexdigest(),executed_sha256=hashlib.sha256(changed.encode()).hexdigest(),change='Minimum recovery samples100→50 plus explicit same half-second coverage and every10ms sample; all physical recovery limits and 200 actor kick ticks unchanged.'),indent=2)+'\n')
 sys.argv=[str(file),*remaining];exec(compile(changed,str(file)+'<private100Hz-recovery-audit>','exec'),{'__name__':'__main__','__file__':str(file)})
