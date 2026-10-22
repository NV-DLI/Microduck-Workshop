"""Offline measured fixed durations/commands; only heading is interpolated.
Predictions are estimates, never a replacement for final current public gates.
"""
import json,math
from pathlib import Path
import numpy as np
TABLE=json.loads(Path(__file__).with_name('closedloop-step-table-v4.json').read_text())['worlds']
CANDIDATES=[dict(forward_duration_s=d,velocity_command=[.3,v,0.])for d in (.6,.8)for v in (-.2,-.1,0.)]
def select(point,heading):
 p=np.asarray(point,float)
 if p.shape!=(3,)or not np.isfinite(p).all()or not math.isfinite(heading):return None
 if not(.13<=p[0]<=.24 and -.10<=p[1]<=.06 and abs(heading)<=.08):return None
 c,s=math.cos(heading),math.sin(heading);pg=np.array([[c,s],[-s,c]])@p[:2]
 lo,hi=(-.1,0.)if heading<=0 else(0.,.1);a=(heading-lo)/.1
 choices=[]
 for row in CANDIDATES:
  d=row['forward_duration_s'];vy=row['velocity_command'][1]
  endpoints=[[w for w in TABLE if w['duration_s']==d and w['vy']==vy and w['h0_rad']==h]for h in (lo,hi)]
  assert all(len(ws)==4 for ws in endpoints)
  pred=[];ids=[]
  for wl,wh in zip(*endpoints):
   assert wl['replicate']==wh['replicate'] and wl['survived_15s'] and wh['survived_15s'] and not wl['controller_aborted'] and not wh['controller_aborted']
   el,eh=wl['endpoint_stop_plus_2s'],wh['endpoint_stop_plus_2s']
   delta=(1-a)*np.asarray(el['shot_frame_dx_dy_m'])+a*np.asarray(eh['shot_frame_dx_dy_m'])
   hf=(1-a)*el['truth_goal_heading_rad']+a*eh['truth_goal_heading_rad'];c,s=math.cos(hf),math.sin(hf)
   b=np.array([[c,-s],[s,c]])@(pg-delta);pred.append([float(b[0]),float(b[1]),hf]);ids.append([wl['index'],wh['index']])
  if not all(.075<=x<=.105 and -.05<=y<=-.02 and abs(h)<=.08 for x,y,h in pred):continue
  cost=max(((x-.09)/.015)**2+((y+.035)/.015)**2+(h/.08)**2 for x,y,h in pred)
  choices.append((cost,{**row,'predicted_endpoints':pred,'calibration_world_indices':ids,'heading_interpolation_weights':[1-a,a],'cost':cost,'truth_used':False,'calibration_is_not_a_bound':True,'fixed_duration_and_lateral_command':True}))
 return min(choices,key=lambda r:r[0])[1]if choices else None

def entry(**a):
 from settled_entry import neutral_turn_entry
 d=a['final_diag']
 if d.get('phase') not in ('OBSERVE','TURN','ADVANCE') or d.get('measured_quiet') is not True:return False,'primitive_not_measured_quiet'
 b=dict(a);b['final_diag']={**d,'phase':'TURN','alignment_mode':'final_goal_heading'};b['velocity_body_xyw']=[0.,0.,.6]
 ok,why=neutral_turn_entry(**b)
 if not ok:return False,why
 chosen=select(a['memory']['point_level_m'],a['goal_heading'])
 return (chosen is not None,'calibrated_short_primitive' if chosen else 'no_compatible_calibrated_primitive')
