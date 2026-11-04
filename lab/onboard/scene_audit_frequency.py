"""Validate every 100 Hz physics and 50 Hz policy clock before an offline audit."""
import argparse
import json
from pathlib import Path
import runpy
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--scene-runtime',type=Path,required=True)
parser.add_argument('--script',required=True)
args,remaining=parser.parse_known_args()
run=Path(remaining[0])
manifest=json.loads((run/'manifest.json').read_text())
assert manifest['physics_hz']==100
assert manifest['policy_hz']==50 and manifest['sensor_hz']==15
for filename,rate,offset in [('physics-trace.jsonl',100,1),('policy-trace.jsonl',50,0)]:
    rows=[json.loads(line) for line in (run/filename).open()]
    assert rows
    assert all(abs(row['timestamp_s']-(i+offset)/rate)<1e-9 for i,row in enumerate(rows)),filename
    if offset:
        assert len(rows)==manifest['physics_samples']
        assert abs(len(rows)/rate-manifest['simulated_seconds'])<1e-9
    else:
        assert len(rows)==manifest['policy_updates']

onboard=Path(__file__).resolve().parent
lab=onboard.parent
sys.path[:0]=[str(onboard),str(lab)]
local={'audit_shot_actual.py','audit_sensor_records.py','audit_direct_navigation.py'}
script=args.script
if script=='audit_media_terminal_cadence.py' and not manifest.get('media_archive',{}).get('videos'):
    script='audit_sensor_records.py'
if script in local:
    path=onboard/script
    sys.argv=[str(path),*remaining]
else:
    path=lab/'scene_audit.py'
    sys.argv=[str(path),'--scene-runtime',str(args.scene_runtime),'--script',script,*remaining]
runpy.run_path(str(path),run_name='__main__')
