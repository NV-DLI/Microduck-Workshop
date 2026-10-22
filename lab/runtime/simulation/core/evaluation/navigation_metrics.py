"""Navigation metrics: Microduck simulation and control."""
from pathlib import Path
import hashlib,json,math
from functools import lru_cache
import numpy as np
PROTOCOL_PATH=Path(__file__).with_name('walk-navigation.json')
PROTOCOL_SHA='1709b6cad007cd6677b3b859a53c8be30d3a36fe3d79cc1c076195a762ded836'

@lru_cache(maxsize=1)
def protocol():
 if hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()!=PROTOCOL_SHA:raise ValueError('Frozen protocol hash changed')
 return json.loads(PROTOCOL_PATH.read_text())

def schedule(family,case,t):
 p=protocol();commands=p['fixed_commands'if family=='fixed'else'stop_restart']['cases']
 if case not in commands:raise ValueError('Unknown frozen case')
 if t<3:return np.zeros(3)
 if family=='fixed':return np.asarray(commands[case],float)
 return np.asarray(commands[case]if(t-3)%25<5 else[0.,0.,0.],float)

def rolling(x,n=200):
 a=np.cumsum(x,axis=0,dtype=np.float64);a=np.concatenate([np.zeros_like(a[:1]),a]);return(a[n:]-a[:-n])/n

def violation(q,limits):return np.maximum(np.maximum(limits[...,0]-q,q-limits[...,1]),0)
def wilson(k,n):
 if not n:return 0.
 z=1.959963984540054;f=k/n;return(f+z*z/(2*n)-z*math.sqrt(f*(1-f)/n+z*z/(4*n*n)))/(1+z*z/n)

def evaluate(arrays,*,family,case,seconds=None):
 p=protocol();duration=(20 if family=='fixed'else 78)if seconds is None else seconds
 t=np.asarray(arrays['time_s']);state=np.asarray(arrays['state']);n=state.shape[1]
 if t.ndim!=1 or state.ndim!=3 or len(t)!=len(state):raise ValueError('Time/state array dimensions invalid')
 if len(t)!=round(duration*200)or not np.allclose(t,np.arange(1,len(t)+1)/200,atol=1e-7,rtol=0):raise ValueError('Every actual200Hz sample must remain, including failed episodes')
 if not np.isfinite(state).all():raise ValueError('Nonfinite raw physics cannot be silently dropped')
 ids=np.asarray(arrays['servo_ids'],int);pids=np.asarray(arrays['passive_ids'],int);bounds=np.asarray(arrays['physical_output_limits']);limits=np.asarray(arrays['joint_limits'])
 if ids.shape!=(14,)or pids.shape!=(14,)or set(ids)&set(pids)or len(set(ids)|set(pids))!=28:raise ValueError('Exactly14servo and14distinctpassive coordinates required')
 if bounds.shape!=(14,2)or limits.shape!=(n,28,2):raise ValueError('Limit dimensions must include every episode')
 q=state[:,:,13:41];eff=q[:,:,ids]+q[:,:,pids];sv=violation(q[:,:,ids],limits[:,ids]);ov=violation(eff,bounds);pv=violation(q[:,:,pids],limits[:,pids]);all28=violation(q,limits)
 quat=state[:,:,3:7];x,y,z,w=(quat[:,:,i]for i in range(4));yaw=np.unwrap(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)),axis=0);pos=state[:,:,:2];velocity=np.concatenate([state[:,:,7:9],state[:,:,12:13]],axis=-1)
 means=rolling(velocity);abs_yaw_mean=rolling(abs(velocity[:,:,2:3]))[:,:,0];mt=t[199:];zero_index=int(np.argmin(abs(t-3)))
 if not np.allclose(np.linalg.norm(quat,axis=-1),1,atol=.001):raise ValueError('Invalid physical quaternion')
 if np.any(np.diff(state[:,:,-1],axis=0)<0):raise ValueError('Fell flag must be sticky; hidden reset detected')
 command=np.asarray(p['fixed_commands'if family=='fixed'else'stop_restart']['cases'][case]);planar_threshold=max(.01,.2*np.linalg.norm(command[:2]));yaw_threshold=max(.06,.2*abs(command[2]));rows=[]
 expected=np.stack([schedule(family,case,float(tt)-1e-10)for tt in t])
 for e in range(n):
  failures=[];r={'episode_index':e,'fell':bool(state[:,:,-1][:,e].max()),'servo_coordinate_peak_violation_rad':float(sv[:,e].max()),'physical_output_peak_violation_rad':float(ov[:,e].max()),'legacy_all28_peak_violation_rad':float(all28[:,e].max()),'passive_fit_status':'unverified; no hardware gear reliability claim','passive_soft_limit_per_joint':[{'joint_index':j,'peak_rad':float(pv[:,e,j].max()),'p95_rad':float(np.percentile(pv[:,e,j],95)),'p99_rad':float(np.percentile(pv[:,e,j],99)),'over_range_duration_s':float((pv[:,e,j]>0).sum()/200),'over_range_fraction':float((pv[:,e,j]>0).mean())}for j in range(14)]}
  if r['fell']:failures.append('fall')
  if r['servo_coordinate_peak_violation_rad']>.001:failures.append('servo_boundary')
  if r['physical_output_peak_violation_rad']>.001:failures.append('physical_output_boundary')
  if duration<(20 if family=='fixed'else 78):failures.append('development_duration_shorter_than_protocol')
  def tracking(mask,target):
   if not mask.any():return {'planar_mean_rmse_m_s':None,'yaw_mean_rmse_rad_s':None,'passed':False}
   v=means[mask,e];pe=float(np.sqrt(np.mean(np.sum((v[:,:2]-target[:2])**2,axis=1))));ye=float(np.sqrt(np.mean((v[:,2]-target[2])**2)))
   return {'planar_mean_rmse_m_s':pe,'yaw_mean_rmse_rad_s':ye,'passed':bool(pe<=max(.01,.2*np.linalg.norm(target[:2]))and ye<=max(.06,.2*abs(target[2])))}
  if family=='fixed':
   r['tracking']=tracking((mt>=4-1e-9)&(mt<=20),command)
   if not r['tracking']['passed']:failures.append('mean_tracking')
   span=t>=3;delta=pos[span,e]-pos[zero_index,e];dyaw=yaw[span,e]-yaw[zero_index,e]
   if np.linalg.norm(command[:2])>0:
    h=yaw[zero_index,e];c,s=np.cos(h),np.sin(h);direction=np.array([[c,-s],[s,c]])@command[:2];direction/=np.linalg.norm(direction);cross=delta@np.array([-direction[1],direction[0]])
    r['path']={'cross_track_rmse_m':float(np.sqrt(np.mean(cross**2))),'cross_track_peak_m':float(abs(cross).max()),'heading_drift_peak_rad':float(abs(dyaw).max())}
    if r['path']['cross_track_rmse_m']>.04 or r['path']['cross_track_peak_m']>.08 or r['path']['heading_drift_peak_rad']>.1745329252:failures.append('straight_path')
   elif command[2]:
    r['turn']={'translation_peak_m':float(np.linalg.norm(delta,axis=1).max()),'unwrapped_final_yaw_error_rad':float(abs(dyaw[-1]-command[2]*17))}
    if r['turn']['translation_peak_m']>.05 or r['turn']['unwrapped_final_yaw_error_rad']>.1745329252:failures.append('turn_completion')
   else:
    r['hold']={'position_peak_m':float(np.linalg.norm(delta,axis=1).max()),'yaw_peak_rad':float(abs(dyaw).max())}
    if r['hold']['position_peak_m']>.03 or r['hold']['yaw_peak_rad']>.0872664626:failures.append('hold_drift')
  else:
   cycles=[]
   for cycle in range(3):
    start=3+25*cycle;stop=start+5;end=stop+20;idx=np.flatnonzero((mt>=stop-1e-9)&(mt<=stop+2+1e-9));gate=(np.linalg.norm(means[:,e,:2],axis=1)<=.02)&(abs_yaw_mean[:,e]<=.1);first=None
    # Require 100 consecutive physical samples (.5s) and attainment by+2s.
    for i in idx:
     begin=i-99
     if begin>=0 and mt[begin]>=stop-1e-9 and gate[begin:i+1].all():first=begin;break
    axis=int(np.argmax(abs(command)));restart_idx=np.flatnonzero((mt>=start-1e-9)&(mt<=start+2+1e-9)&(means[:,e,axis]*np.sign(command[axis])>=.5*abs(command[axis])))
    tr=tracking((mt>=start+2-1e-9)&(mt<=stop+1e-9),command)
    cr={'cycle':cycle,'restart_delay_s':None if not len(restart_idx)else float(mt[restart_idx[0]]-start),'settle_gate_started_after_s':None if first is None else float(mt[first]-stop),'settle_gate_completed_after_s':None if first is None else float(mt[first+99]-stop),'tracking_after_restart':tr}
    if first is not None:
     ref=int(np.argmin(abs(t-mt[first])));span=(t>=mt[first]-1e-9)&(t<=end+1e-9);cr.update(post_settle_position_peak_m=float(np.linalg.norm(pos[span,e]-pos[ref,e],axis=1).max()),post_settle_yaw_peak_rad=float(abs(yaw[span,e]-yaw[ref,e]).max()))
    cr['passed']=bool(first is not None and len(restart_idx)and tr['passed']and cr['post_settle_position_peak_m']<=.03 and cr['post_settle_yaw_peak_rad']<=.0872664626);cycles.append(cr)
   r['cycles']=cycles
   if not all(v['passed']for v in cycles):failures.append('stop_restart')
  mask=t>=4;err=velocity[mask,e]-expected[mask]
  r['unfiltered_planar_velocity_RMSE']=float(np.sqrt(np.mean(np.sum(err[:,:2]**2,axis=1))))if mask.any()else None;r['unfiltered_yaw_velocity_RMSE']=float(np.sqrt(np.mean(err[:,2]**2)))if mask.any()else None
  r['v1_threshold_diagnostic_only']=bool(mask.any()and r['unfiltered_planar_velocity_RMSE']<=.05 and r['unfiltered_yaw_velocity_RMSE']<=.25 and r['legacy_all28_peak_violation_rad']<=.001 and not r['fell'])
  r['original_v1_outcome']='not recomputed as formal v1; raw threshold diagnostics retained';r['failure_reasons']=failures;r['passed']=not failures;rows.append(r)
 return rows
