"""Export four recent walking camera views with their paired sensor history."""
from pathlib import Path
import argparse,json,hashlib,sys
D=Path(__file__).resolve().parent
sys.path[:0]=[str(D),str(D.parent)]
from learning.walking_data import samples,score,digest
from learning.walking import ACTION_NAMES,TASK
from walking_visual_history import VisualHistory,specification,IMAGE_KEYS

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 old=json.loads((a.dataset/'provenance.json').read_text());split=json.loads((a.dataset/'split.json').read_text())['split']
 if split not in ('train','validation'):raise ValueError('No evaluation/test data admitted')
 if any(r['split']!=split for r in old):raise ValueError('Mixed splits')
 prepared=[]
 for r in old:
  source=Path(r['source']);actual=score(source)
  if actual!=r['score'] or actual['seed']!=r['seed'] or not actual['passed']:raise ValueError('Source score/seed differs')
  if digest(source/'policy-trace.jsonl')!=r['samples_sha256'] or digest(source/'sensor-inputs.jsonl')!=r['sensors_sha256']:raise ValueError('Source records changed')
  base,rows=samples(source,align_kick_onset=r.get('sampling_phase_policy')=='actual_kick_onset')
  if len(rows)!=r['frames']:raise ValueError('Source frame count differs')
  names=[x.removeprefix('q_relative/')for x in base['state_names'][6:20]];schema=specification(names)
  if prepared and schema!=prepared[0][3]:raise ValueError('Motor contract differs')
  prepared.append((r,source,rows,schema))
 if not prepared:raise ValueError('Empty source')
 import numpy as np
 from PIL import Image
 from lerobot.datasets.lerobot_dataset import LeRobotDataset
 schema=prepared[0][3]
 features={'observation.state':dict(dtype='float32',shape=(127,),names=schema['state_names']),'action':dict(dtype='float32',shape=(8,),names=ACTION_NAMES)}
 features.update({k:dict(dtype='image',shape=(270,480,3),names=['height','width','channels'])for k in IMAGE_KEYS})
 dataset=LeRobotDataset.create(repo_id='local/'+a.output.name,root=a.output,fps=12.5,robot_type='microduck_walking_visual_history',features=features,use_videos=False)
 provenance=[];alignment=a.output/'history-alignment.jsonl'
 with alignment.open('x')as log:
  for episode,(record,source,rows,_) in enumerate(prepared):
   memory=VisualHistory()
   for frame,row in enumerate(rows):
    transformed=memory.observe(state=row['state'],image=row['image'],image_timestamp=row['image_capture_s'],timestamp=row['timestamp_s'])
    item={'observation.state':np.asarray(transformed['state'],np.float32),'action':np.asarray(row['action'],np.float32),'task':TASK}
    item.update({k:np.asarray(Image.open(source/path).convert('RGB').resize((480,270)))for k,path in zip(IMAGE_KEYS,transformed['images'],strict=True)})
    dataset.add_frame(item)
    log.write(json.dumps(dict(episode_index=episode,frame_index=frame,seed=record['seed'],timestamp_s=row['timestamp_s'],images=transformed['images'],image_timestamps=transformed['image_timestamps'],sample_indices=transformed['sample_indices'],sample_timestamps=transformed['sample_timestamps']))+'\n')
   dataset.save_episode();provenance.append(dict(record,history_source=str(a.dataset.resolve()),history_transform_sha256=digest(D/'walking_visual_history.py')))
   print(json.dumps(dict(episode=episode,seed=record['seed'],frames=len(rows))),flush=True)
 dataset.finalize()
 for name,value in [('control-contract',schema),('provenance',provenance),('split',dict(split=split)),('preparation',dict(status='Dataset exported; model training is a separate step',source=str(a.dataset.resolve()),source_provenance_sha256=digest(a.dataset/'provenance.json'),transform_sha256=digest(D/'walking_visual_history.py'),frames=sum(r['frames']for r in provenance),episodes=len(provenance)))]:
  (a.output/(name+'.json')).write_text(json.dumps(value,indent=2))
if __name__=='__main__':main()
