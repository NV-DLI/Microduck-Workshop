"""Four causal views and motor skills with bounded kick head yaw."""
from pathlib import Path
import hashlib, importlib.util, json, sys
D=Path(__file__).resolve().parent
R=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(D),str(D.parent/'rollout'),str(R/'lab')]
import learning.walking_runtime as original
from walking_history_runtime import WalkingPolicy
import football_scene

def main():
 if '--walking-policy-server' not in sys.argv:
  raise ValueError('Provide a trained walking checkpoint server')
 output=Path(sys.argv[sys.argv.index('--output')+1])
 P=D.parent/'rollout'
 files=[Path(__file__),P/'walking_history_runtime.py',P/'walking_visual_history.py',P/'pending_kick_candidate.py',P/'kick_head_yaw_candidate.py',D/'walking_student_executor.py',D/'joint_brake.py',D/'grounded-pose.json']
 hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}
 with output.with_name(output.name+'-history-interface.json').open('x')as f:
  json.dump(dict(source_sha256=hashes,method='v3 127D/previous 12.5Hz decision/four causal RGB views; original pending quiet gate; bounded standing yaw during K10; no object or teacher observations; course standing61 joint-brake initialization'),f,indent=2)
 runtime=Path(sys.argv[sys.argv.index('--runtime')+1])
 sys.path[:0]=[str(runtime/'evaluation/football'),str(runtime/'evaluation'),str(runtime/'football'),str(runtime/'sensors')]
 spec=importlib.util.spec_from_file_location('kick_runtime',P/'kick_head_yaw_candidate.py')
 module=importlib.util.module_from_spec(spec);sys.modules['kick_runtime']=module;spec.loader.exec_module(module)
 original.WalkingPolicy=WalkingPolicy
 original_load=football_scene.load
 def load(name,path):
  return original_load(name,D/'walking_student_executor.py' if name=='isaac_execute' else path)
 football_scene.load=load
 football_scene.main()
 if any(hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
  raise RuntimeError('Frozen history/skill sources changed during physics')
if __name__=='__main__':main()
