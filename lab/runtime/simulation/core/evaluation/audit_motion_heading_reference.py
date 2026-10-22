"""Read-only independent re-derivation from actual public packets, no truth."""
from pathlib import Path
import argparse,json,math,hashlib
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
reads={round(r['timestamp_s'],6):r['readback'].get('actual_packet') for r in map(json.loads,(a.run/'public-memory-readback.jsonl').open())}
anchors={};count=0;maximum=0.;source_fixed=True;source_equal=True;last=None;h=None
for r in map(json.loads,(a.run/'policy-trace.jsonl').open()):
 f=r.get('public_neutral_fine_step',{})
 if f.get('state')not in ('TRACK','MACRO'):continue
 anchor=f['motion_heading_anchor'];key=(f['attempts'],f['motion_kind']);packet=reads[round(r['timestamp_s'],6)]
 if key not in anchors:
  anchors[key]=anchor;h=anchor['heading_rad'];last=(anchor['seq'],anchor['read_s'])
  source_equal &= abs(h-f['goal_heading_rad'])<1e-10
  source_equal &= anchor['source']=='public_camera_goal_at_movement_start' and anchor['public_goal_fix_s']==f['goal_fix_s'] and 0<=r['timestamp_s']-anchor['public_goal_fix_s']<=.5
 else:source_fixed &= anchor==anchors[key]
 seq,read=packet['seq'],packet['read_timestamp_s']
 assert seq>=last[0] and read>=last[1]
 if seq>last[0]:
  dt=read-last[1];assert 0<dt<=.08
  g=packet['gravity_body'];w=packet['gyro_body_rad_s'];norm=math.sqrt(sum(v*v for v in g));rate=-sum(x*y for x,y in zip(w,g))/norm
  h=math.atan2(math.sin(h-rate*dt),math.cos(h-rate*dt));last=(seq,read)
 else:assert read==last[1]
 maximum=max(maximum,abs(math.atan2(math.sin(h-f['motion_heading_prediction_rad']),math.cos(h-f['motion_heading_prediction_rad']))));count+=1
 assert f['motion_heading_prediction_is_measurement'] is False
checks=dict(nonempty=count>0,actual_public_heading_rederived=maximum<1e-9,initial_camera_anchor_matches=bool(source_equal),anchor_immutable_during_movement=bool(source_fixed),not_relabelled_as_measurement=True)
out=dict(schema='microduck.motion-heading-reference-audit.v2',passed=all(checks.values()),checks=checks,moving_policy_ticks=count,maximum_heading_error_rad=maximum,anchors=list(anchors.values()),scope='Actual same-read public gyro/gravity/clock and logged initial camera heading only; no truth input or physics-success claim.',source_sha256={str(q):hashlib.sha256(q.read_bytes()).hexdigest()for q in [Path(__file__),a.run/'policy-trace.jsonl',a.run/'public-memory-readback.jsonl']});a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k]for k in ['passed','checks','moving_policy_ticks','maximum_heading_error_rad']}));assert out['passed']
