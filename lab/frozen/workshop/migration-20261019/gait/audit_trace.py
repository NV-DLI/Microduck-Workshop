"""Recompute gait metrics from 200 Hz raw physical evidence, independently of PPO."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
 m=json.loads((a.directory/'trace-manifest.json').read_text());raw=a.directory/'raw-trace.npz'
 if hashlib.sha256(raw.read_bytes()).hexdigest()!=m['raw_trace_sha256']:raise ValueError('Raw trace hash mismatch')
 with np.load(raw,allow_pickle=False)as z:
  state=z['state'];control=z['control'];limits=z['joint_limits'];servos=z['servo_ids'];passive=z['passive_ids']
  dt=float(z['physics_dt']);cdt=float(z['control_dt'])
 if dt!=.005 or cdt!=.02 or len(state)!=4*len(control):raise ValueError('Missing physical substeps')
 cols=m['state_columns'];q=state[:,:,slice(*cols['joint_pos'])];v=state[:,:,7:10];w=state[:,:,10:13]
 root=state[:,:,:3];quat=state[:,:,3:7]
 yaw=np.unwrap(np.arctan2(2*(quat[:,:,3]*quat[:,:,2]+quat[:,:,0]*quat[:,:,1]),1-2*(quat[:,:,1]**2+quat[:,:,2]**2)),axis=0)
 desired=np.repeat(control[:,:,:3],4,axis=0)
 hard_violation=np.maximum(limits[None,:,:,0]-q,q-limits[None,:,:,1]).clip(min=0).max(axis=(0,2))
 effective=q[:,:,servos]+q[:,:,passive];effective_limits=limits[:,servos]+limits[:,passive]
 output_violation=np.maximum(effective_limits[None,:,:,0]-effective,effective-effective_limits[None,:,:,1]).clip(min=0).max(axis=(0,2))
 finite=np.isfinite(state).all(axis=(0,2))&np.isfinite(control).all(axis=(0,2))
 survived=(state[:,:,cols['fell_latched']].max(axis=0)==0)
 rows=[]
 for i,group in enumerate(m['groups']):
  metrics={'survived':bool(survived[i]),'joint_limit_violation_rad':float(max(hard_violation[i],output_violation[i])),
   'nonfinite':not bool(finite[i]),'sim_seconds':len(state)*dt}
  if m['protocol_id']=='walk-fixed-v1':
   metrics.update(planar_velocity_rmse_m_s=float(np.sqrt(np.mean(np.sum((v[:,i,:2]-desired[:,i,:2])**2,axis=1)))),
    yaw_rate_rmse_rad_s=float(np.sqrt(np.mean((w[:,i,2]-desired[:,i,2])**2))))
   behavior=metrics['planar_velocity_rmse_m_s']<=.05 and metrics['yaw_rate_rmse_rad_s']<=.25
  else:
   drift=[];angles=[];settle=[];cycles=0
   for c in range(3):
    t0=5.+25*c;lo=int((t0+1)/dt);hi=int((t0+20)/dt)
    drift.append(float(np.linalg.norm(root[lo:hi,i,:2]-root[lo,i,:2],axis=1).max()))
    angles.append(float(np.abs(yaw[lo:hi,i]-yaw[lo,i]).max()))
    start=int(t0/dt);end=int((t0+20)/dt)
    quiet=(np.linalg.norm(v[start:end,i,:2],axis=1)<.02)&(np.abs(w[start:end,i,2])<.1)
    lengths=np.convolve(quiet.astype(int),np.ones(100,dtype=int),'valid');found=np.flatnonzero(lengths==100)
    settle.append(float(found[0]*dt)if len(found)else 20.)
    pre=slice(int((t0-4)/dt),int(t0/dt));post=slice(int((t0+21)/dt),int((t0+25)/dt))
    active_pre=float(v[pre,i,0].mean())>=.05 if group=='forward' else float(w[pre,i,2].mean())>=.2
    active_post=float(v[post,i,0].mean())>=.05 if group=='forward' else float(w[post,i,2].mean())>=.2
    cycles+=bool(active_pre and active_post)
   metrics.update(cycles_completed=cycles,post_settle_drift_m=max(drift),post_settle_yaw_rad=max(angles),settle_time_s=max(settle))
   behavior=cycles==3 and max(drift)<=.03 and max(angles)<=math.radians(5) and max(settle)<=2.
  success=metrics['survived']and not metrics['nonfinite']and metrics['joint_limit_violation_rad']<=.001 and behavior
  rows.append({'episode_id':f"{m['seed']}-{group}-{i:03}",'group':group,'seed':m['seed'],'status':'completed','success':bool(success),'metrics':metrics})
 summary={g:{'successes':sum(r['success']for r in rows if r['group']==g),'trials':sum(r['group']==g for r in rows)}for g in sorted(set(m['groups']))}
 report={'schema':'microduck.gait-trace-audit.v1','runtime_id':m['runtime_id'],'hardware_profile_id':m['hardware_profile_id'],
  'protocol_id':m['protocol_id'],'split':m['split'],'seed':m['seed'],'model_sha256':m['model_sha256'],
  'raw_trace_sha256':m['raw_trace_sha256'],'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'trace_samples':len(state),'group_summary':summary,'episodes':rows,'release_passed':False,
  'note':'Physical metrics only; complete three-seed provenance and frozen-plan validation required separately.'}
 (a.directory/'independent-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(summary))

if __name__=='__main__':main()
