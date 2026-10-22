"""Fixed community action class and command mapping; never used for student exports."""
from pathlib import Path
import hashlib,importlib,json,sys
from control import motion as original_motion,KEYS
HERE=Path(__file__).resolve().parent
RECIPE=HERE/'community_recipe_v1'
def identity():
 data=json.loads((RECIPE/'manifest.json').read_text())
 for item in data['sourceFiles']:
  if Path(item['path']).name!=item['path'] or hashlib.sha256((RECIPE/item['path']).read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Frozen community recipe changed')
 return {'adapter':'raw-history-internal8','recipeManifestSha256':hashlib.sha256((RECIPE/'manifest.json').read_bytes()).hexdigest(),'maxJointTargetRateRadS':8.,'commandMapping':{'forwardBackward':.3,'yaw':.6,'lateral':0.},'liveDirectionQualification':'pending'}
def configure_community(cfg):
 identity()
 for name in ('bounded_actions','precision_recipe','precision_recipe_v3'):
  previous=sys.modules.get(name)
  if previous is not None and Path(previous.__file__).resolve()!=RECIPE/(name+'.py'):raise ValueError('Community recipe import would mix with another policy')
 sys.path.insert(0,str(RECIPE))
 try:module=importlib.import_module('precision_recipe_v3')
 finally:sys.path.remove(str(RECIPE))
 cfg.actions.joint_pos.class_type=module.BoundedTrainingAction
 return cfg

def motion(packet,now=None):
 # Reuse the fixed expiry/pause/protocol guard without inheriting lateral commands.
 guarded=original_motion(packet,now)
 if guarded==[0.,0.,0.]:return guarded
 held=set(packet['keys'])
 forward=int(bool(held&{'KeyW','ArrowUp'}))-int(bool(held&{'KeyS','ArrowDown'}))
 yaw=int(bool(held&{'KeyA','KeyQ','ArrowLeft'}))-int(bool(held&{'KeyD','KeyE','ArrowRight'}))
 return [.3*forward,0.,.6*yaw]
