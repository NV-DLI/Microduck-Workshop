"""Public8x8-only sphere forward residual; no physics/depth truth imports."""
from pathlib import Path
import sys,json,hashlib,math
import numpy as np
M=Path(__file__).resolve().parents[2];O=M.parents[1]/'outputs/migration-20261019';D=O/'football-v26-image-head-servo40s'
sys.path[:0]=[str(M/'football'),str(M/'sensors')]
from hardware_head_fk import CourseHeadFK
from perception import level_rotation
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=O/'football-v26-tof-independent-audit-v1.json';data=json.loads(base.read_text());s={round(r['timestamp_s'],6):r for r in map(json.loads,(D/'sensor-inputs.jsonl').open())};head=CourseHeadFK();rows=[]
for f in data['frames']:
 if not f['ball_geometry']or not f['ball_geometry']['accepted']:continue
 p=s[f['timestamp_s']]['proprio'];fk=head.forward_policy14(p['q_rad'],joint_names=p['joint_names'])['tof_flu'];R=level_rotation(-np.asarray(f['floor_observation']['normal_body']));ball=np.asarray(f['ball_geometry']['geometry']['point_reference_m']);c=np.asarray(fk.rotation).T@(ball-np.asarray(fk.translation_m));distance=np.linalg.norm(c)
 for h in f['hazards']:
  row,col=h['zone'];azlo,azhi=map(math.radians,(45*(.5-(col+1)/8),45*(.5-col/8)));ello,elhi=map(math.radians,(45*(.5-(row+1)/8),45*(.5-row/8)))
  az=float(np.clip(math.atan2(c[1],c[0]),azlo,azhi));horizontal=c[0]*math.cos(az)+c[1]*math.sin(az);el=float(np.clip(math.atan2(c[2],horizontal),ello,elhi));ray=np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)]);proj=float(c@ray);disc=.035**2-(distance**2-proj**2)
  minimum=None if disc<0 else proj-math.sqrt(max(0,disc));cell=s[f['timestamp_s']]['tof_frame']['cells'][row][col]
  rows.append({'timestamp_s':f['timestamp_s'],'zone':[row,col],'continuous_cell_sphere_minimum_range_m':minimum,'observed_range_m':cell['range_m'],'observed_minus_continuous_minimum_m':None if minimum is None else cell['range_m']-minimum,'sphere_intersects_cell':minimum is not None,'sample_range_spread_m':cell['sample_range_spread_m'],'mixed_edge':cell['mixed_edge'],'valid_samples':cell['valid_samples'],'vision_uncertainty':f['ball_geometry']['geometry'].get('estimated_range_uncertainty_m')})
v=[r['observed_minus_continuous_minimum_m']for r in rows if r['sphere_intersects_cell']];result={'schema':'microduck.public-zone-sphere-fit-offline.v1','source_sha256':sha(__file__),'inputs':{str(p):sha(p)for p in (base,D/'sensor-inputs.jsonl')},'reads_truth':False,'reads_raw32x32_depth':False,'inherited_audit_fields_consumed':['timestamp','observed ball geometry','ToF floor estimate','zone IDs'],'cell_count':len(rows),'sphere_intersects_cell_count':len(v),'range_residual_quantiles_m':dict(zip(('min','median','p95','max'),map(float,np.quantile(v,[0,.5,.95,1])))),'negative_residual_count':sum(x<0 for x in v),'rows':rows,'interpretation':['This only tests geometric range compatibility. A foreground object with indistinguishable range cannot be excluded from one zone.','Continuous cell minimum differs from finite emitter/raster nearest sampling. No new association tolerance is chosen from truth.','Original100mm reliability filter and original guard are unchanged.']};out=O/'football-v26-public-zone-sphere-fit-v1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n');print({k:result[k]for k in ('cell_count','sphere_intersects_cell_count','range_residual_quantiles_m','negative_residual_count')})
