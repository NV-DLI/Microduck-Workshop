"""Owned CPU Navfn service and Beta sensor snapshot adapter."""
from pathlib import Path
import math,uuid,json,hashlib
from nav2_service.owner import ServiceOwner
from nav2_service.client import BridgeClient
from nav2_service.obstacle_contract import prior
from nav2_service.contract import atomic

def quaternion_xyzw(R):
 # Stable matrix-to-quaternion, branch on largest diagonal for 180-degree cases.
 tr=sum(R[i][i] for i in range(3))
 if tr>0:
  s=2*math.sqrt(tr+1);q=[(R[2][1]-R[1][2])/s,(R[0][2]-R[2][0])/s,(R[1][0]-R[0][1])/s,s/4]
 else:
  i=max(range(3),key=lambda k:R[k][k]);j=(i+1)%3;k=(i+2)%3;s=2*math.sqrt(max(0,1+R[i][i]-R[j][j]-R[k][k]));q=[0.,0.,0.,0.];q[i]=s/4;q[j]=(R[j][i]+R[i][j])/s;q[k]=(R[k][i]+R[i][k])/s;q[3]=(R[k][j]-R[j][k])/s
 n=math.hypot(*q);return[v/n for v in q]

def make_request(*,time_s,pose_map_xyyaw,velocity_body_xyw,goal_map_xy,tof_sample,layout,image_sha256,localization_capture_s):
 x,y,yaw=pose_map_xyyaw;dx,dy=goal_map_xy[0]-x,goal_map_xy[1]-y;c,s=math.cos(yaw),math.sin(yaw)
 # This adapter takes an already sensor-derived map pose. No command odometry.
 case={'name':'beta-live','query_s':time_s,'odom':{'capture_s':time_s,'pose_xyyaw':pose_map_xyyaw,'twist_body_xyw':velocity_body_xyw},'goal_body_xy':[c*dx+s*dy,-s*dx+c*dy],'tof_samples':[tof_sample],
 'recorded_evidence':{'raw_sensor_odom_xyyaw':pose_map_xyyaw,'map_to_odom_xyyaw':[0.,0.,0.],'RGB_localization_capture_end_s':localization_capture_s,'RGB_localization_available_s':localization_capture_s,'truth_read':False}}
 provenance={'kind':'isaac_recorded_tof_approximation_with_RGB_field_localization','hardware_profile_id':'beta-gc2093-l8cx-v1','runtime_id':'newton-bam-20261019','odometry_origin':'sensor_only','layout':layout,'static_prior':prior(layout),'source_file_sha256':{'rgb_pixels':image_sha256,'tof':hashlib.sha256(json.dumps(tof_sample,sort_keys=True).encode()).hexdigest()},'scope':'Known surveyed field with actual observed RGB localization and current8x8ToF; not unknown-obstacle discovery'}
 return case,provenance

class Nav2Session:
 def __init__(self,output,max_wall_s=1800):
  output=Path(output);nonce=uuid.uuid4().hex;self.sequence=0;self.last_time=-1.
  self.owner=ServiceOwner(Path(__file__).resolve().parent/'nav2_service',output/'exchange',output/'service',nonce,max_wall_s)
  self.ready=self.owner.start();self.client=BridgeClient(output/'exchange',nonce)
 def query(self,case,provenance):
  response=self.client.query(case,provenance,self.sequence,case['query_s']);self.sequence+=1;self.last_time=case['query_s'];return response
 def close(self):return self.owner.close()
 def __enter__(self):return self
 def __exit__(self,*args):self.close()
