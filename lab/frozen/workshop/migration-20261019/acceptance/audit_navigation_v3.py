#!/usr/bin/env python3
"""Strict independent navigation audit; missing attempts never become passes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from navigation_metrics_v2 import protocol,evaluate,wilson,PROTOCOL_SHA

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit_part(path,*,allow_development=False):
 p=protocol();r=json.loads(Path(path).read_text());base=Path(path).parent
 for key,expected in [('schema','microduck.navigation-v2-part.v1'),('runtime_id',p['runtime_id']),('hardware_profile_id',p['hardware_profile_id']),('protocol_sha256',PROTOCOL_SHA),('physics_hz',200),('policy_hz',50),('automatic_resets',False),('state_projection',False),('support_during_task',False),('status','recorded')]:
  if r.get(key)!=expected:raise ValueError('Unacceptable '+key)
 if not allow_development and r['development_only']:raise ValueError('Development cannot qualify')
 if not allow_development and(r['seed']not in p['group_rule']['seeds']or r['expected_episodes']!=32 or not r.get('qualification_bundle_sha256')):raise ValueError('Formal seed/batch/freeze identity required')
 bundle=r['model_bundle']
 if not allow_development:
  frozen_file=Path(r['qualification_bundle_file']);frozen=json.loads(frozen_file.read_text())
  if sha(frozen_file)!=r['qualification_bundle_sha256']or frozen.get('model_bundle')!=bundle or frozen.get('protocol_sha256')!=PROTOCOL_SHA:raise ValueError('Frozen candidate evidence does not match')
 if set(bundle)!=set(p['required_model_bundle'])or any(not isinstance(v,str)or len(v)!=64 or any(c not in '0123456789abcdef'for c in v)for v in bundle.values()):raise ValueError('Complete real model-bundle hashes required')
 if sha(r['model_file'])!=bundle['network_sha256']:raise ValueError('Model file hash mismatch')
 assets=r['robot_asset_sources']
 if sha(assets['base'])!=assets['base_sha256']or sha(assets['manifest'])!=assets['manifest_sha256']:raise ValueError('Base robot USD/compiled-CoM provenance changed')
 for asset in assets['variants']:
  if sha(asset['path'])!=asset['sha256']:raise ValueError('Compiled robot USD changed')
 for file,h in r['adapter_file_sources'].items():
  if sha(file)!=h:raise ValueError('Frozen action dependency changed')
 bound=r['physical_output_bound_source']
 if sha(bound['source_usd'])!=bound['source_usd_sha256']:raise ValueError('Official14joint limit source changed')
 for file,key in [('raw-trace.npz','trace_sha256'),('post-reset-parameters.npz','post_reset_parameter_sha256'),('collector-env.yaml','collector_env_sha256')]:
  if sha(base/file)!=r[key]:raise ValueError('Evidence hash mismatch: '+file)
 for file,h in r['collector_sources'].items():
  if sha(file)!=h:raise ValueError('Collector source evidence changed')
 with np.load(base/'raw-trace.npz',allow_pickle=False)as f:arrays={k:f[k]for k in f.files}
 if arrays['state'].shape[1]!=r['expected_episodes']or len(arrays['state'])!=r['physics_samples']:raise ValueError('Dropped episode/sample')
 if len(arrays['control'])!=r['control_samples']or len(arrays['state'])!=4*len(arrays['control']):raise ValueError('Control/physics chronology mismatch')
 required={k for k in arrays if k.startswith('bam_')and k.endswith('effective_vin')}
 if not required:raise ValueError('Missing actual BAM voltage/torque/delay trace')
 for k in required:
  prefix=k[:-len('effective_vin')]
  for field in ['effective_vin','motor_torque','_delayed_target']:
   v=arrays.get(prefix+field)
   if v is None or len(v)!=len(arrays['state'])or not np.isfinite(v).all():raise ValueError('Incomplete actual BAM trace')
 if not r.get('actual_solver',{}).get('joint_solref')or not r['actual_solver'].get('joint_solimp'):raise ValueError('Missing actual solver readback')
 if r['adapter'] not in ('trained-internal8','trained-head-reference-v8','community-external3'):raise ValueError('Unknown action-history contract')
 expected_history=arrays['control'][:,:,:14]if r['adapter'] in ('trained-internal8','trained-head-reference-v8')else arrays['control'][:,:,14:28]
 if not np.allclose(arrays['control'][:,:,56:70],expected_history,atol=1e-6,rtol=0):raise ValueError('Wrong last-action adapter semantics')
 from navigation_metrics_v2 import schedule
 expected_commands=np.stack([schedule(r['family'],r['case'],i/50)for i in range(len(arrays['control']))])[:,None]
 if not np.allclose(arrays['control'][:,:,70:73],expected_commands,atol=1e-7,rtol=0):raise ValueError('Recorded commands differ from frozen schedule')
 recomputed=evaluate(arrays,family=r['family'],case=r['case'],seconds=r['expected_seconds'])
 if len(recomputed)!=len(r.get('episodes',[]))or any(v['episode_index']!=i for i,v in enumerate(r['episodes'])):raise ValueError('Failed episode rows omitted/reordered')
 # Never trust an edited summary pass flag.
 if [x['passed']for x in recomputed]!=[x['passed']for x in r['episodes']]:raise ValueError('Reported episode outcome differs from raw evidence')
 return {**r,'episodes':recomputed,'report_sha256':sha(path),'report_path':str(Path(path).resolve()),'audit_success_count':sum(x['passed']for x in recomputed)}

def audit(paths,*,allow_development=False):
 p=protocol();parts=[audit_part(f,allow_development=allow_development)for f in paths]
 if not parts:raise ValueError('No episodes supplied')
 baseline=parts[0]
 for r in parts:
  if r['model_bundle']!=baseline['model_bundle']or r['model_line']!=baseline['model_line']:raise ValueError('Cannot pool model/adapter/runtime identities')
 key=lambda r:(r['family'],r['case'],r['seed'],r['condition'])
 if len(set(map(key,parts)))!=len(parts):raise ValueError('Repeated subgroup, including rerun selection')
 expected={(family,case,seed,condition)for family,section in [('fixed','fixed_commands'),('stop','stop_restart')]for case in p[section]['cases']for seed in p['group_rule']['seeds']for condition in p['group_rule']['conditions']}
 actual=set(map(key,parts));missing=sorted(expected-actual);extra=sorted(actual-expected);groups=[]
 for family,section in [('fixed','fixed_commands'),('stop','stop_restart')]:
  for case in p[section]['cases']:
   for seed in p['group_rule']['seeds']:
    group=[r for r in parts if(r['family'],r['case'],r['seed'])==(family,case,seed)];passed=len(group)==2 and sum(r['audit_success_count']for r in group)>=61 and all(r['expected_episodes']==32 and r['audit_success_count']>=31 for r in group)
    groups.append({'family':family,'case':case,'seed':seed,'parts':len(group),'successes':sum(r['audit_success_count']for r in group),'trials':sum(r['expected_episodes']for r in group),'passed':passed})
 successes=sum(r['audit_success_count']for r in parts);trials=sum(r['expected_episodes']for r in parts)
 return {'schema':'microduck.navigation-v2-audit.v2','protocol_sha256':PROTOCOL_SHA,'runtime_id':p['runtime_id'],'hardware_profile_id':p['hardware_profile_id'],'model_bundle':baseline['model_bundle'],'model_line':baseline['model_line'],'passed':not allow_development and not missing and not extra and all(r['passed']for r in groups),'development_only':allow_development,'missing_subgroups':missing,'unexpected_subgroups':extra,'successes':successes,'trials':trials,'wilson95_lower':wilson(successes,trials),'groups':groups,'parts':[{'path':r['report_path'],'sha256':r['report_sha256'],'successes':r['audit_success_count'],'trials':r['expected_episodes']}for r in parts],'hardware_reliability_claimed':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('reports',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--development',action='store_true');a=p.parse_args()
 if a.output.exists():raise FileExistsError('Preserve audit history')
 result=audit(a.reports,allow_development=a.development);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'trials':result['trials'],'successes':result['successes'],'missing_groups':len(result['missing_subgroups'])}))
if __name__=='__main__':main()
