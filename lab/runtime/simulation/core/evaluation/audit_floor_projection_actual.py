"""Audit floor projection actual: Microduck simulation and control."""
from pathlib import Path
import argparse,json,sys,math,hashlib
M=Path(__file__).resolve().parents[1];sys.path[:0]=[str(M/'football'),str(M/'sensors')]
import numpy as np
from hardware_head_fk import CourseHeadFK
from ground_plane import fit_virtual_floor
from perception import level_rotation

def quatmatrix(q):
 x,y,z,w=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def audit(run):
 requests={};nav_count=0;coarse_count=0
 for path in sorted((run/'nav2/service/result').glob('request-*.json')):
  req=json.loads(path.read_text());requests[req['case']['query_s']]=(path,req);nav_count+=1
 for line in(run/'control.jsonl').open():
  control=json.loads(line)
  for key in ('coarse_track_path','coarse_active_path'):
   route=control.get(key) or {};req=route.get('map_request')
   if not req:continue
   t=req['case']['query_s'];coarse_count+=1
   if t in requests:
    prev=requests[t][1]
    assert req['case']['tof_samples']==prev['case']['tof_samples'],'Different same-capture projection samples'
    assert req['case']['odom']['pose_xyyaw']==prev['case']['odom']['pose_xyyaw'],'Different same-capture pose'
   requests[t]=(run/'control.jsonl',req)
 fk=CourseHeadFK();count=0;corrected=0;errors=[];maxerr=0.;records=[]
 for line in(run/'sensor-inputs.jsonl').open():
  obs=json.loads(line);t=obs['timestamp_s']
  if t not in requests:continue
  path,req=requests[t];sample=req['case']['tof_samples'][0];frame=obs['tof_frame'];p=obs['proprio'];floor=fit_virtual_floor(frame,p,fk)
  assert floor['accepted'],'Request must have actual accepted current floor'
  assert sample['capture_s']==sample['fk_capture_s']==frame['timestamp_s']==p['timestamp_s']==t
  R=level_rotation((-np.array(floor['normal_body'])).tolist());head=fk.forward_policy14(p['q_rad'],joint_names=p['joint_names'])['tof_flu'];pose=req['case']['odom']['pose_xyyaw'];c,s=math.cos(pose[2]),math.sin(pose[2]);T=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@R
  expected_sensor=T@np.array(head.translation_m)+np.array([pose[0],pose[1],-floor['offset_m']]);rotation=T@np.array(head.rotation)
  e=max(float(np.max(np.abs(expected_sensor-np.array(sample['sensor_translation_odom'])))),float(np.max(np.abs(rotation-quatmatrix(sample['sensor_quaternion_odom_xyzw'])))));maxerr=max(maxerr,e);assert e<1e-8,'Public sensor transform mismatch'
  cells=sum(frame['cells'],[]);assert len(cells)==64
  assert sample['ranges_m']==[x['range_m']for x in cells]
  assert sample['isaac_valid_mask']==[x['status']=='valid'and not x.get('contains_too_close',False)for x in cells]
  assert sample['status']==[None]*64 and sample['near_m']==.1 and sample['far_m']==4.
  model=frame['model'];profile=frame['profile'];assert model['samples_per_zone_axis']==4 and model['source_width_px']==model['source_height_px']==32 and model['source_horizontal_fov_deg']==model['source_vertical_fov_deg']==50. and profile['horizontal_fov_deg']==profile['vertical_fov_deg']==45.
  changed=[]
  for zone,cell in enumerate(cells):
   expected=np.array(cell['ray_sensor_flu']);distance=cell['range_m'];valid=sample['isaac_valid_mask'][zone] and not cell.get('no_return_samples')and not cell.get('invalid_samples')and distance is not None
   if valid:
    # Reconstruct discrete camera pixel rays independently, not the controller's cache.
    parameterss=[];focal=16/math.tan(math.radians(25));center=15.5
    for rowoffset in (.125,.375,.625,.875):
     for coloffset in (.125,.375,.625,.875):
      az=math.radians(45*(.5-(cell['col']+coloffset)/8));el=math.radians(45*(.5-(cell['row']+rowoffset)/8))
      pixelx=math.floor(center-focal*math.tan(az)+.5);pixely=math.floor(center-focal*math.tan(el)/math.cos(az)+.5)
      if not(0<=pixelx<32 and 0<=pixely<32):continue
      ray=np.array([1.,(center-pixelx)/focal,(center-pixely)/focal]);ray/=np.linalg.norm(ray);parameterss.append(ray)
    depths=[float((rotation@ray)[2])for ray in parameterss];i=int(np.argmin(depths));residual=expected_sensor[2]+distance*depths[i]
    if depths[i]<0 and abs(residual)<=.005:expected=parameterss[i];changed.append(zone)
   e=float(np.max(np.abs(expected-np.array(sample['unit_rays_sensor'][zone]))));maxerr=max(maxerr,e)
   if e>1e-10:errors.append({'capture_s':t,'zone':zone,'error':e})
  corrected+=len(changed);count+=1;records.append({'capture_s':t,'corrected_zones':changed,'request_sha256':hashlib.sha256(json.dumps(req,sort_keys=True,separators=(',',':')).encode()).hexdigest()})
 assert count==len(requests) and count>0,'All actual requests must have matching public observations'
 return {'schema':'microduck.actual-floor-projection-independent-audit.v2','run':str(run),'passed':not errors,'actual_requests':count,'nav2_requests':nav_count,'coarse_path_maps':coarse_count,'corrected_zone_observations':corrected,'maximum_component_error':maxerr,'errors':errors,'records':records,'truth_consumed':False,'sensor_distances_and_masks_preserved':True}
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('run',type=Path);a.add_argument('--output',type=Path,required=True);v=a.parse_args();result=audit(v.run)
 with v.output.open('x')as f:json.dump(result,f,indent=2)
 print(json.dumps({k:x for k,x in result.items()if k!='records'}));raise SystemExit(0 if result['passed']else 1)
