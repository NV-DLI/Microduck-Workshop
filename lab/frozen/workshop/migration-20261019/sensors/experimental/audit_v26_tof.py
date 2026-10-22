"""Immutable recorded-data audit: public guard replay, then separate truth-only scoring."""
from pathlib import Path
import sys,json,hashlib,math,collections
import numpy as np
from PIL import Image
M=Path(__file__).resolve().parents[2];O=M.parents[1]/'outputs/migration-20261019';D=O/'football-v26-image-head-servo40s'
sys.path[:0]=[str(M/'football'),str(M/'sensors')]
from perception import BetaPerception,level_rotation
from beta_sensor_profile import BetaCameraProfile
from hardware_head_fk import quaternion_wxyz
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda n:[json.loads(l)for l in(D/n).read_text().splitlines()]
checked={}
for sub in ('football','sensors'):
 for p in (M/sub).glob('*.py'):
  q=D/'source-snapshot'/sub/p.name
  if q.exists():
   checked[str(p)]={'sha256':sha(p),'snapshot_sha256':sha(q)}
# The imported replay dependencies must still match the retained implementation.
for name in ('football/perception.py','football/ground_plane.py','football/hardware_head_fk.py','football/leg_odometry.py','sensors/beta_sensor_profile.py','sensors/beta_perception_geometry.py'):
 assert sha(M/name)==sha(D/'source-snapshot'/name),name
manifest=json.loads((D/'manifest.json').read_text());c=manifest['sensors']['camera'];cam=BetaCameraProfile(width_px=c['width_px'],height_px=c['height_px'],assumed_horizontal_fov_deg=c['assumed_horizontal_fov_deg']);vision=BetaPerception(cam)
sensors={round(r['timestamp_s'],6):r for r in read('sensor-inputs.jsonl')};dets={round(r['timestamp_s'],6):r for r in read('detections.jsonl')};physics={round(r['timestamp_s'],6):r for r in read('physics-trace.jsonl')}
controls=[r for r in read('control.jsonl')if 22<=r['timestamp_s']<=26 and r['reason']=='actual_ToF_near_obstacle_or_too_close']
frames=[];hazards=[]
for control in controls:
 t=round(control['timestamp_s'],6);s=sensors[t];p=s['proprio'];rgb=np.asarray(Image.open(D/s['rgb']).convert('RGB'));geo=vision.estimate(rgb,p,dets[t]['detections'],now_s=t,tof_frame=s['tof_frame'])
 assert geo['floor_observation']['accepted'];R=level_rotation(-np.asarray(geo['ground']['normal']));height=geo['height']['conditional_root_height_m'];ball=R@np.asarray(geo['ball']['geometry']['point_reference_m'])if geo['ball']and geo['ball']['accepted']else None
 fk=vision.head.forward_policy14(p['q_rad'],joint_names=p['joint_names'])['tof_flu'];Rf=np.asarray(fk.rotation);origin=np.asarray(fk.translation_m)
 # Truth starts here, isolated from public guard and geometry computations above.
 truth=physics[t];q=truth['root_quaternion_xyzw'];RB=np.asarray(quaternion_wxyz([q[3],*q[:3]]));root=np.asarray(truth['root_position_world_m']);ballworld=np.asarray(truth['ball_position_world_m']);depth=np.load(D/s['tof_range']);raw_origin_world=root+RB@origin
 framehazards=[];ball_exempted=[];close=[]
 for cell in sum(s['tof_frame']['cells'],[]):
  if cell['contains_too_close']:close.append([cell['row'],cell['col']]);continue
  if cell['status']!='valid' or cell['range_m']is None:continue
  body=origin+Rf@(np.asarray(cell['ray_sensor_flu'])*cell['range_m']);v=R@body;dist=None if ball is None else float(np.linalg.norm(v-ball))
  if dist is not None and dist<.047:ball_exempted.append([cell['row'],cell['col']]);continue
  if not(.04<v[2]+height<.45 and 0<v[0]<.105 and abs(v[1])<.09):continue
  # Recover which virtual subray supplied the nearest range, using only saved raw raster for scoring.
  subrays=[];fx=32/(2*math.tan(math.radians(25)));cx=cy=15.5
  for sr in range(4):
   for sc in range(4):
    az=math.radians(45*(.5-(cell['col']+(sc+.5)/4)/8));el=math.radians(45*(.5-(cell['row']+(sr+.5)/4)/8));f=math.cos(el)*math.cos(az);l=math.cos(el)*math.sin(az);u=math.sin(el);col=math.floor(cx-fx*l/f+.5);row=math.floor(cy-fx*u/f+.5);distance=float(depth[row,col])
    if .1<=distance<=4.:subrays.append((distance,row,col))
  selected=min(subrays);points={}
  for principal in (15.5,16.):
   ray=np.array([1.,(principal-selected[2])/fx,(principal-selected[1])/fx]);ray/=np.linalg.norm(ray);world=raw_origin_world+RB@Rf@(ray*selected[0]);points[str(principal)]={'point_world_m':world.tolist(),'height_m':float(world[2]),'sphere_surface_error_m':float(abs(np.linalg.norm(world-ballworld)-.035))}
  world=root+RB@body
  h={'timestamp_s':t,'zone':[cell['row'],cell['col']],'range_m':cell['range_m'],'mixed_edge':cell['mixed_edge'],'sample_spread_m':cell['sample_range_spread_m'],'public_level_point_m':v.tolist(),'public_height_above_floor_m':float(v[2]+height),'public_ball_center_m':None if ball is None else ball.tolist(),'public_distance_to_ball_center_m':dist,'public_ball_accepted':bool(geo['ball']and geo['ball']['accepted']),'public_ball_reason':None if geo['ball']is None else geo['ball']['reason'],'scoring_only':{'zone_center_point_world_m':world.tolist(),'zone_center_height_m':float(world[2]),'zone_center_sphere_surface_error_m':float(abs(np.linalg.norm(world-ballworld)-.035)),'selected_raw_range_m':selected[0],'selected_raster_row_col':list(selected[1:]),'nearest_range_matches_quantized_zone':abs(round(selected[0],3)-cell['range_m'])<1e-9,'selected_pixel_points_by_principal':points}}
  framehazards.append(h);hazards.append(h)
 frames.append({'timestamp_s':t,'hazard_count':len(framehazards),'too_close_zones':close,'ball_exempted_zones':ball_exempted,'ball_geometry':geo['ball'],'floor_observation':geo['floor_observation'],'hazards':framehazards})
summary={'stopped_frames':len(frames),'frames_too_close':sum(bool(f['too_close_zones'])for f in frames),'guard_replayed_frames':sum(bool(f['hazard_count']or f['too_close_zones'])for f in frames),'hazard_endpoint_count':len(hazards),'hazard_zones':dict(collections.Counter(str(h['zone'])for h in hazards)),'fresh_accepted_ball_frames':sum(bool(f['ball_geometry']and f['ball_geometry']['accepted'])for f in frames)}
for key in ('15.5','16.0'):
 h=[v['scoring_only']['selected_pixel_points_by_principal'][key]for v in hazards];summary['selected_raw_pixel_scoring_principal_'+key]={'sphere_surface_error_median_m':float(np.median([x['sphere_surface_error_m']for x in h])),'sphere_surface_error_p95_m':float(np.quantile([x['sphere_surface_error_m']for x in h],.95)),'within_5mm_ball_surface':sum(x['sphere_surface_error_m']<.005 for x in h),'within_5mm_ground':sum(abs(x['height_m'])<.005 for x in h)}
result={'schema':'microduck.v26-tof-independent-offline-audit.v1','source_sha256':sha(__file__),'inputs':{str(D/n):sha(D/n)for n in ('sensor-inputs.jsonl','control.jsonl','detections.jsonl','physics-trace.jsonl','manifest.json')},'replay_dependency_hashes':checked,'summary':summary,'scope':['Public guard uses only same-frame observed RGB boxes, encoder FK and64zone ToF, reproducing frozen geometry.','Truth world root and ball plus retained32x32 raster are used only after guard classification for independent scoring.','No source/controller/profile changed;100mm near filter retained.','Ray aggregation returns nearest subray range paired with zone centre; raw pixel location is diagnostic-only and unavailable on real8x8 hardware.'],'frames':frames}
out=O/'football-v26-tof-independent-audit-v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))
