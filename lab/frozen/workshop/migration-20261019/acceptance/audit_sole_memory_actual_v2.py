"""Independently replay actual public memory using actual fresh visual anchors; no physics inputs."""
from pathlib import Path
import argparse,hashlib,json,sys,importlib.util
import numpy as np
M=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(M/'acceptance'),str(M/'football'),str(M/'sensors')]
from public_ball_read_adapter_v2 import SameReadMemoryAdapter
from leg_odometry import LegFK

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--controller',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',action='store_true');a=p.parse_args();assert not a.output.exists()
 leg=LegFK(M/'sensors/sources/robot_allcollisions.xml',require_pinned=False)
 if not a.baseline:
  sys.path.insert(0,str(a.controller));from contact_odometry import SoleContactLegOdometry
  leg=SoleContactLegOdometry(leg)
 controls=[json.loads(l)for l in(a.run/'control.jsonl').open()];captures=[]
 for r in controls:
  m=r.get('public_handoff',{}).get('memory',{})
  if m.get('memory_available') and abs(m.get('ball_capture_timestamp_s',-10)-r['timestamp_s'])<1e-8:
   captures.append((r['timestamp_s'],m))
 reads=[json.loads(l)for l in(a.run/'public-memory-readback.jsonl').open()]
 first=next(r['readback']['actual_packet']for r in reads if r['readback'].get('actual_packet'));adapter=SameReadMemoryAdapter(leg,first['joint_names']);index=0;errors=[];availability=0;digests=0
 def observe(t,m):
  adapter.memory.observe_complete_ball(timestamp_s=t,point_body_m=m['point_body_m'],estimated_range_uncertainty_m=m['original_range_uncertainty_m'],complete=True,geometry_accepted=True,capture_read_age_s=t-adapter.last_read_s)
 for row in reads:
  t=row['timestamp_s']
  while index<len(captures) and captures[index][0]<=t+1e-8 and adapter.last_read_s is not None and abs(captures[index][0]-captures[index][1]['capture_sensor_read_age_s']-adapter.last_read_s)<1e-7:
   observe(*captures[index]);index+=1
  rb=row['readback'];q=rb.get('actual_packet');new=rb.get('reason')!='no_new_delivered_read' and q is not None
  packet=None
  if new:
   packet=dict(source='body_powerboard_control_imu',axes='trunk FLU',seq=q['seq'],timestamp_s=q['sample_timestamp_s'],read_timestamp_s=q['read_timestamp_s'],delivered_timestamp_s=q['delivered_timestamp_s'],encoder_read_timestamp_s=q['encoder_timestamp_s'],orientation_valid=q['orientation_valid'],encoder_q_rad=q['q_rad'],encoder_qd_rad_s=q['qd_rad_s'],gyro=q['gyro_body_rad_s'],gravity=q['gravity_body'],gyro_clipped=rb.get('source_gyro_clipped'),accel_clipped=rb.get('source_accel_clipped'))
  got=adapter.consume(packet,now_s=t)
  while index<len(captures) and abs(captures[index][0]-t)<1e-8:
   observe(*captures[index]);index+=1
  expected=rb['memory_readback'];actual=adapter.memory.status(t)
  if new:digests+=int(got.get('input_sha256')!=rb.get('input_sha256'))
  availability+=int(actual['memory_available']!=expected['memory_available'])
  if actual['memory_available'] and expected['memory_available']:errors.append(float(np.max(np.abs(np.array(actual['point_level_m'])-expected['point_level_m']))))
 passed=bool(errors and max(errors)<1e-9 and availability==0 and digests==0)
 result=dict(schema='microduck.public-sole-memory-actual-audit.v2',passed=passed,run=str(a.run),estimator='original_baseline'if a.baseline else'public_sole_contact',comparisons=len(errors),max_abs_point_error_m=max(errors,default=None),availability_mismatches=availability,input_digest_mismatches=digests,fresh_visual_anchors=len(captures),scope='Public packets and actual fresh same-capture control memory anchors only; no physics, force, pose or scored outcomes read.',sources={str(x):sha(x)for x in [a.run/'control.jsonl',a.run/'public-memory-readback.jsonl',Path(__file__),a.controller/'release.json']})
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k]for k in ['passed','comparisons','max_abs_point_error_m','availability_mismatches','input_digest_mismatches','fresh_visual_anchors']}));return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
