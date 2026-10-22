"""Run two predeclared deep-gaze supervisor cohorts, retaining every world."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
M=Path(__file__).resolve().parents[1];O=M.parents[1]/'outputs/migration-20261019'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=O/'gait-dual-v12-deep-gaze-prospective-v1.json'
contract=json.loads(plan.read_text())
base=O/'gait-dual-v12-deep-gaze-dev-v1';base.mkdir(exist_ok=False)
log={'schema':'microduck.dual-deep-gaze-execution.v1','plan_sha256':sha(plan),'source_sha256':sha(__file__),'runs':[],'qualification_claimed':False,'state':'running'}
def save():
 p=base/'execution.tmp';p.write_text(json.dumps(log,indent=2)+'\n');p.replace(base/'execution.json')
def run(cmd,name):
 with (base/(name+'.log')).open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1'))
 if r.returncode:raise RuntimeError(f'{name} failed{r.returncode}; preserve failed original')
for pitch in contract['gaze_pitches']:
 for p,h in contract['source_sha256'].items():
  if sha(p)!=h:raise ValueError('Frozen source changed')
 output=base/('pitch'+str(pitch));item={'pitch':pitch,'output':str(output)};log['runs'].append(item);save()
 cmd=[sys.executable,str(M/'gait/evaluate_navigation_gaze_handoff_v12.py'),'--model',str(O/'gait-body50-v9-4000/policy.onnx'),'--stand-model',str(O/'gait-body50-v9-4600/policy.onnx'),'--recipe-dir',str(O/'gait-body50-v9-from3800-1024x1200-seed970321-r2'),'--output',str(output),'--family','stop','--adapter','trained-head-reference-v8','--model-line','from-scratch','--development','--seed',str(contract['seed']),'--num-envs','2','--gaze-pitch',str(pitch)]
 item['command']=cmd;save();run(cmd,'pitch'+str(pitch)+'-collector')
 report=json.loads((output/'report.json').read_text());audit=output/'independent-physical-audit.json'
 run([sys.executable,str(M/'acceptance/audit_navigation_v5.py'),*report['part_reports'],'--output',str(audit),'--development'],'pitch'+str(pitch)+'-physical')
 dual=output/'independent-dual-audit.json';run([sys.executable,str(M/'gait/audit_dual_policy_v12.py'),str(output/'report.json'),'--physical-audit',str(audit),'--output',str(dual)],'pitch'+str(pitch)+'-dual')
 rows=[]
 for part in report['part_reports']:
  r=json.loads(Path(part).read_text())
  with np.load(Path(part).parent/'raw-trace.npz',allow_pickle=False) as d:
   ready=d['actor_inferred_mask'];observed=d['actor_observation'][:,:,51:55]
   target=np.array(r['gaze_handoff']['head_delta_target_rad'],dtype=np.float32);head=np.zeros(4,dtype=np.float32);expected=[]
   for step in range(len(observed)):
    if step>=150:head+=np.clip(target-head,-.006,.006)
    expected.append(head.copy())
   expected=np.broadcast_to(np.array(expected)[:,None,:],observed.shape)
   error=float(np.abs(observed[ready]-expected[ready]).max())
   if error>2e-6:raise ValueError('Actual head command differs from declared slew')
   for i,e in enumerate(r['episodes']):
    t=d['time_s'];mask=(t>=10)&(t<=28);span=(t<=28);pos=d['state'][mask,i,:2];q=d['state'][span,i,13:41];ids=d['servo_ids'];pids=d['passive_ids'];limits=d['joint_limits'][i,ids];outlimits=d['physical_output_limits'];eff=q[:,ids]+q[:,pids]
    sv=np.maximum(np.maximum(limits[:,0]-q[:,ids],q[:,ids]-limits[:,1]),0).max();ov=np.maximum(np.maximum(outlimits[:,0]-eff,eff-outlimits[:,1]),0).max()
    first_survives=not bool(d['state'][span,i,-1].any());first_ready=bool(ready[:1400,i].all());cycle=e['cycles'][0]
    rows.append({'case':r['case'],'world':i,'all_task_strict_passed':e['passed'],'all_task_fell':e['fell'],'first_cycle_survived':first_survives,'first_cycle_mechanics_ok':bool(max(sv,ov)<=.001),'first_cycle_ready':first_ready,'position_peak_from10_to28s_m':float(np.linalg.norm(pos-pos[0],axis=-1).max()),'first_cycle_original_metrics':cycle,'actual_head_command_max_error_rad':error})
 result={'schema':'microduck.dual-deep-gaze-handoff-audit.v1','technical_passed':True,'qualification_claimed':False,'pitch':pitch,'all_worlds':rows,'first_cycle_standing_condition_met':sum(r['first_cycle_survived'] and r['first_cycle_mechanics_ok'] and r['first_cycle_ready'] and r['first_cycle_original_metrics']['settle_gate_completed_after_s'] is not None and r['first_cycle_original_metrics'].get('post_settle_position_peak_m',1)<=.03 and r['first_cycle_original_metrics'].get('post_settle_yaw_peak_rad',1)<=.0872664626 for r in rows),'trials':len(rows),'scope':'Development only. Conditional first stop includes no-fall, mechanical and ready gates; whole78s strict results are retained. No ball/goal was present.','physical_audit_sha256':sha(audit),'dual_audit_sha256':sha(dual),'source_sha256':sha(__file__)}
 p=output/'independent-head-handoff-audit.json';p.write_text(json.dumps(result,indent=2)+'\n');item.update(complete=True,audit_sha256=sha(p),standing_first_cycle=result['first_cycle_standing_condition_met']);save();print(json.dumps(item),flush=True)
log['state']='development-recorded';save()
