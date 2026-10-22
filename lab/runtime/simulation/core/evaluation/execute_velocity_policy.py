#!/usr/bin/env python3
"""Execute velocity policy: Microduck simulation and control."""
import hashlib,importlib,json,sys,time
from pathlib import Path
A=Path(__file__).resolve().parent;M=A.parent;sys.path.insert(0,str(M/'football'))
import isaac_execute
import nav2_bridge
from execute_long_episode import BoundedLifetimeSession
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old_build=isaac_execute.build_cfg;old_run=isaac_execute.run
contract={}

def build(a):
 if a.trained_recipe_dir is None:raise ValueError('An explicitly frozen v8 training recipe is required')
 if a.kick_model is not None:raise ValueError('v8 head mapping has not validated the community kick; no silent reuse')
 cfg,fixture=old_build(a)
 import torch
 initial=torch.load(a.trained_recipe_dir/'initial.pt',map_location='cpu',weights_only=False);metadata=json.loads(a.model.with_suffix('.json').read_text());updates=metadata['checkpoint_iteration']-int(initial['iter'])+1
 if updates<300:raise ValueError('Final head mapping requires completed300update transition')
 cls=importlib.import_module('head_reference_curriculum').FinalHeadResidualAction;cfg.actions.joint_pos.class_type=cls
 files=[Path(__file__),A/'execute_long_episode.py',a.trained_recipe_dir/'head_reference_curriculum.py',a.trained_recipe_dir/'head_reference.py',a.trained_recipe_dir/'initial.pt']
 contract.update(schema='microduck.actual-beta-v8-adapter.v1',qualification_claimed=False,adapter='trained-head-reference-v8',head_formula='reference +0.15*tanh(raw)',blend=1.,previous_action='raw network output',duplicate_external_adapter=False,completed_curriculum_updates=updates,source_sha256={str(p):sha(p)for p in files})
 return cfg,fixture

def run(a,cfg,fixture):
 try:old_run(a,cfg,fixture)
 finally:
  if a.output.is_dir():
   changed=[p for p,h in contract['source_sha256'].items()if sha(p)!=h];contract['sources_changed']=changed
   for i,p in enumerate(contract['source_sha256']):
    file=Path(p);target=a.output/'source-snapshot/v8-adapter'/f'{i:02}-{file.name}';target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(file.read_bytes())
   (a.output/'v8-adapter-contract.json').write_text(json.dumps(contract,indent=2)+'\n')
   path=a.output/'manifest.json'
   if path.exists():
    data=json.loads(path.read_text());data['trained_adapter'].update(kind='frozen FinalHeadResidualAction,reference+.15tanh(raw),internal8margin.035',head_reference_blend=1.,head_residual_rad=.15,adapter_contract_sha256=sha(a.output/'v8-adapter-contract.json'));data['source_changed_during_run']+=changed;path.write_text(json.dumps(data,indent=2)+'\n')
if __name__=='__main__':
 isaac_execute.build_cfg=build;isaac_execute.run=run;nav2_bridge.Nav2Session=BoundedLifetimeSession;isaac_execute.main()
