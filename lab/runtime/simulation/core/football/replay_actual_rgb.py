#!/usr/bin/env python3
"""CPU sensor-only replay of real Beta RGB/encoders/IMU/ToF plus actual Nav2.

Does not read the privileged physics trace. Commands are recorded but not
executed: this validates integration, never a closed-loop football success rate.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'sensors'))
import cv2
from beta_sensor_profile import BetaCameraProfile
from controller import BetaNavigator,ControllerConfig
from detector_client import DetectorSession
from nav2_bridge import Nav2Session

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True);p.add_argument('--deployment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--hfov',type=float,default=70.);p.add_argument('--maximum-frames',type=int,default=0);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Retain every replay attempt')
 m=json.loads((a.episode/'manifest.json').read_text())
 if m['runtime_id']!='newton-bam-20261019'or m['hardware_profile_id']!='beta-gc2093-l8cx-v1':raise ValueError('Actual new Beta episode required')
 sensor_file=a.episode/'sensor-inputs.jsonl'
 if sha(sensor_file)!=m['trace_sha256']['sensor-inputs.jsonl']:raise ValueError('Recorded sensor trace changed')
 inputs=[json.loads(s)for s in sensor_file.read_text().splitlines()];a.output.mkdir(parents=True);detector=None;nav=None;report={'schema':'microduck.beta-real-sensor-replay.v1','runtime_id':m['runtime_id'],'hardware_profile_id':m['hardware_profile_id'],'episode_manifest_sha256':sha(a.episode/'manifest.json'),'sensor_inputs_sha256':sha(sensor_file),'deployment_sha256':sha(a.deployment),'status':'running','qualification_claimed':False,'closed_loop':False,'commands_executed':False,'physics_trace_read':False,'actual_nav2':True,'rows':[],'skipped_invalid_imu':0}
 try:
  detector=DetectorSession(a.deployment,a.output/'detector');nav=Nav2Session(a.output/'nav2',max_wall_s=1800.);controller=BetaNavigator(nav,camera=BetaCameraProfile(width_px=960,height_px=540,assumed_horizontal_fov_deg=a.hfov),config=ControllerConfig(layout=m['fixture']['layout']))
  for row in inputs:
   if row['proprio']is None:report['skipped_invalid_imu']+=1;continue
   if row['timestamp_s']<m['startup_support']['task_started_at_s']:continue  # Match the live runner's explicit release gate.
   rgb=cv2.cvtColor(cv2.imread(str(a.episode/row['rgb'])),cv2.COLOR_BGR2RGB)
   if hashlib.sha256(rgb.tobytes()).hexdigest()!=row['rgb_pixel_sha256']:raise ValueError('Actual captured RGB hash changed')
   result=detector.detect(rgb);control=controller.step(rgb,row['proprio'],row['tof_frame'],now_s=row['timestamp_s'],detections=result['detections'])
   report['rows'].append({'timestamp_s':row['timestamp_s'],'detector':result,'control':control,'geometry':controller.last_geometry})
   if a.maximum_frames and len(report['rows'])>=a.maximum_frames:break
  states={};balls=goals=0
  for row in report['rows']:
   states[row['control']['state']]=states.get(row['control']['state'],0)+1;g=row['geometry'];balls+=bool(g.get('ball')and g['ball'].get('accepted'));goals+=bool(g.get('goal')and g['goal'].get('accepted'))
  report.update(status='completed',states=states,accepted_ball_geometry_frames=balls,accepted_goal_geometry_frames=goals)
 except BaseException as error:report.update(status='failed',error=repr(error));raise
 finally:
  if detector is not None:report['detector_cleanup']=detector.close()
  if nav is not None:report['nav2_cleanup']=nav.close()
  report['source_sha256']={str(p.relative_to(ROOT)):sha(p)for p in [*ROOT.glob('*.py'),*ROOT.glob('vision/*.py')]}
  (a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items()if k!='rows'},indent=2))
if __name__=='__main__':main()
