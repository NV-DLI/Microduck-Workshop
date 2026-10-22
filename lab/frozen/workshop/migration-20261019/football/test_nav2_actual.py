"""Actual CPU Navfn service with explicitly synthetic sensor-packet fixtures.

This validates the new transport and empty/obstacle paths, not RGB detection,
robot motion, hardware performance or football success.
"""
import json,time,math,hashlib
from pathlib import Path
from nav2_bridge import Nav2Session,make_request
from beta_tof_geometry import tof_zone_rays
P=Path(__file__).resolve().parent
out=P/'reports'/('cpu-nav2-'+time.strftime('%Y%m%d-%H%M%S'));out.mkdir(parents=True)
rows=[];started=time.monotonic()
try:
 with Nav2Session(out,max_wall_s=180) as session:
  for i,layout in enumerate(['empty','one-obstacle']):
   t=float(i+1)
   # All 64 unknown rays stay unknown; known-map free cells come only from the
   # explicitly surveyed prior. No synthetic all-round scan is supplied.
   tof={'capture_s':t,'fk_capture_s':t,'provenance':'simulation_approximation','status':[None]*64,'ranges_m':[None]*64,'isaac_valid_mask':[False]*64,'near_m':.1,'far_m':4.,'unit_rays_sensor':tof_zone_rays(),'sensor_translation_odom':[-.66,0,.2],'sensor_quaternion_odom_xyzw':[0.,0.,0.,1.]}
   case,provenance=make_request(time_s=t,pose_map_xyyaw=[-.66,0.,0.],velocity_body_xyw=[0.,0.,0.],goal_map_xy=[.15,0],tof_sample=tof,layout=layout,image_sha256=hashlib.sha256(b'SYNTHETIC-CPU-FIXTURE').hexdigest(),localization_capture_s=t)
   provenance['fixture_scope']='Synthetic schema probe, not actual camera/ToF data'
   response=session.query(case,provenance);rows.append({'layout':layout,'response':response})
   if response['status']!='complete' or not response['path_xy'] or not response['envelope']['accepted']:raise AssertionError('Actual Navfn request failed: '+json.dumps(response))
  cleanup=session.close()
 report={'status':'passed','actual_cpu_navfn':True,'inputs':'synthetic CPU fixtures, not physical sensors','robot_or_gpu_started':False,'football_qualified':False,'requests':rows,'cleanup':cleanup,'wall_seconds':time.monotonic()-started}
except BaseException as e:
 report={'status':'failed','error':repr(e),'requests':rows,'robot_or_gpu_started':False};raise
finally:
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(out/'report.json')
