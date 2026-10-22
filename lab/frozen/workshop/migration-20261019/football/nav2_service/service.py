"""One persistent official Navfn process, sequential bounded sensor transactions.

ROS uses real wall time for an immutable request snapshot. Original simulation
acquisition/query clocks remain separate in requests/responses and are never
converted to epoch stamps. No /cmd_vel, simulator, scene, actor or GPU access.
"""
import argparse,math,os,resource,signal,subprocess,threading,time,traceback,uuid
from pathlib import Path
from contract import atomic,read,digest,validate_request,map_compatible,RESPONSE
from map_contract import check_path,centerline_is_known_free,WIDTH,HEIGHT,RESOLUTION

def stamp_s(t):return t.sec+t.nanosec*1e-9
def pss(pid):
 try:
  return next(int(s.split()[1])*1024 for s in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines()if s.startswith('Pss:'))
 except (OSError,StopIteration):return None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--exchange',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--run-nonce',required=True);ap.add_argument('--max-wall-s',type=float,default=1200);a=ap.parse_args()
 if not 1<=a.max_wall_s<=1800:raise ValueError('Bounded service lifetime required')
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=''or os.environ.get('NVIDIA_VISIBLE_DEVICES')!='void':raise RuntimeError('GPU-hidden service required')
 import rclpy
 from rclpy.node import Node
 from rclpy.action import ActionClient
 from rclpy.executors import MultiThreadedExecutor
 from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy
 from nav_msgs.msg import OccupancyGrid,Odometry
 from nav2_msgs.msg import Costmap
 from geometry_msgs.msg import PoseStamped,TransformStamped
 from nav2_msgs.action import ComputePathToPose
 from lifecycle_msgs.srv import ChangeState
 from tf2_ros import TransformBroadcaster
 rclpy.init();node=Node('microduck_persistent_navfn_bridge');executor=MultiThreadedExecutor(num_threads=2);executor.add_node(node);spin=threading.Thread(target=executor.spin,daemon=True);spin.start()
 qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
 maps=node.create_publisher(OccupancyGrid,'/stock_observed_map',qos);odom_pub=node.create_publisher(Odometry,'/probe/sensor_odom_snapshot',qos);tf=TransformBroadcaster(node)
 latest=[None];node.create_subscription(Costmap,'/global_costmap/costmap_raw',lambda x:latest.__setitem__(0,(time.monotonic_ns(),x)),qos)
 client=ActionClient(node,ComputePathToPose,'/compute_path_to_pose');transition=node.create_client(ChangeState,'/planner_server/change_state')
 active_bundle=[None]
 def pose(x,y,yaw):
  p=PoseStamped();p.header.frame_id='probe_map';p.header.stamp=node.get_clock().now().to_msg();p.pose.position.x=float(x);p.pose.position.y=float(y);p.pose.orientation.z=math.sin(yaw/2);p.pose.orientation.w=math.cos(yaw/2);return p
 def publish_tf():
  bundle=active_bundle[0]
  if bundle is None:return
  now=node.get_clock().now().to_msg();case=bundle;e=case['recorded_evidence'];messages=[]
  for parent,child,xyyaw in [('probe_map','probe_odom',e['map_to_odom_xyyaw']),('probe_odom','probe_base',e['raw_sensor_odom_xyyaw'])]:
   x=TransformStamped();x.header.frame_id=parent;x.child_frame_id=child;x.header.stamp=now;x.transform.translation.x=float(xyyaw[0]);x.transform.translation.y=float(xyyaw[1]);x.transform.rotation.z=math.sin(xyyaw[2]/2);x.transform.rotation.w=math.cos(xyyaw[2]/2);messages.append(x)
  tf.sendTransform(messages)
  o=Odometry();o.header.frame_id='probe_odom';o.child_frame_id='probe_base';o.header.stamp=now;o.pose.pose=pose(*e['raw_sensor_odom_xyyaw']).pose;v=case['odom']['twist_body_xyw'];o.twist.twist.linear.x=float(v[0]);o.twist.twist.linear.y=float(v[1]);o.twist.twist.angular.z=float(v[2]);odom_pub.publish(o)
 node.create_timer(.02,publish_tf)
 def future(f,timeout):
  end=time.monotonic()+timeout
  while not f.done():
   if time.monotonic()>=end:raise TimeoutError('ROS future wall deadline')
   if child.poll()is not None:raise RuntimeError('Persistent planner exited')
   time.sleep(.005)
  return f.result()
 def barrier(snap,published_ns,published_wall_s):
  end=time.monotonic()+3.;reason='no_new_costmap_publication'
  while time.monotonic()<end:
   row=latest[0]
   if row is not None and row[0]>published_ns:
    received,m=row;md=m.metadata
    correct=(m.header.frame_id=='probe_map'and stamp_s(m.header.stamp)>=published_wall_s-1e-6 and md.size_x==WIDTH and md.size_y==HEIGHT and abs(md.resolution-RESOLUTION)<1e-8 and abs(md.origin.position.x-snap['origin'][0])<1e-7 and abs(md.origin.position.y-snap['origin'][1])<1e-7)
    if correct and map_compatible(m.data,snap):
     return {'matched':True,'cells_checked':len(m.data),'input_occupancy_sha256':digest(snap['data']),'raw_costmap_sha256':digest(list(m.data)),'costmap_header_wall_s':stamp_s(m.header.stamp),'published_map_wall_s':published_wall_s,'costmap_received_monotonic_ns':received,'expected_origin':snap['origin'],'verified':'new post-publication master-costmap full occupancy classes and origin/shape; known-free inflation may be 0..253, unknown 253/255; original envelope independently rechecked','map_update_time_field_used_as_ack':False}
    reason='master_content_or_frame_not_current_request'
   time.sleep(.01)
  raise TimeoutError('Map barrier: '+reason)
 started=time.monotonic();instance=uuid.uuid4().hex;log=(a.output/'planner.log').open('wb');child=None;report={'status':'starting','run_nonce':a.run_nonce,'service_instance':instance,'clock_scheme':'ROS real wall snapshot transaction; original simulator timestamps retained separately','GPU':False,'robot_hardware_measured':False,'requests':[],'planner_launch_count':0};lastseq=-1;lastsim=-1.;seen=None;activated=False;peak=0;resource_rows=[];sampler_stop=threading.Event()
 def sample():
  nonlocal peak
  while not sampler_stop.wait(.1):
   values=[pss(os.getpid()),pss(child.pid)if child else None];total=sum(x for x in values if x is not None);peak=max(peak,total)
   resource_rows.append({'wall_s':time.monotonic()-started,'bridge_PSS_bytes':values[0],'planner_PSS_bytes':values[1],'sum_PSS_bytes':total})
 sampler=threading.Thread(target=sample,daemon=True)
 def stop_signal(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
 for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop_signal)
 try:
  child=subprocess.Popen(['/opt/ros/jazzy/lib/nav2_planner/planner_server','--ros-args','--params-file',str(Path(__file__).with_name('nav2.yaml'))],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);report['planner_launch_count']=1;report['planner_pid']=child.pid;sampler.start()
  if not transition.wait_for_service(timeout_sec=15):raise TimeoutError('Planner lifecycle service unavailable')
  atomic(a.exchange/'ready.json',{'run_nonce':a.run_nonce,'service_instance':instance,'planner_pid':child.pid,'accepting_requests':True,'lifecycle_activation_requires_first_actual_snapshot':True,'clock_scheme':report['clock_scheme']})
  report['status']='serving';atomic(a.output/'summary.json',report)
  while time.monotonic()-started<a.max_wall_s:
   if child.poll()is not None:raise RuntimeError('Planner process exited during service')
   stop=a.exchange/'shutdown.json'
   if stop.exists()and read(stop).get('run_nonce')==a.run_nonce:report['status']='complete';break
   reqpath=a.exchange/'request.json'
   if not reqpath.exists():time.sleep(.02);continue
   if reqpath.stat().st_size>1_048_576:raise ValueError('Bounded request file exceeded1MiB')
   request=read(reqpath);request_hash=digest(request)
   if request_hash==seen:time.sleep(.02);continue
   seen=request_hash;began=time.monotonic();wall_started=time.time();r={'schema':RESPONSE,'run_nonce':a.run_nonce,'service_instance':instance,'sequence':request.get('sequence'),'request_sha256':request_hash,'source_sim_query_s':request.get('sim_query_s'),'planner_pid':child.pid,'status':'failed','path_xy':[],'envelope':{'accepted':False,'reason':'transaction_failed'}}
   try:
    snap=validate_request(request,a.run_nonce,lastseq,lastsim);lastseq=request['sequence'];lastsim=request['sim_query_s'];active_bundle[0]=request['case'];publish_tf()
    seq=f'{lastseq:08d}';atomic(a.output/f'request-{seq}.json',request)
    msg=OccupancyGrid();msg.header.frame_id='probe_map';msg.header.stamp=node.get_clock().now().to_msg();msg.info.map_load_time=msg.header.stamp;msg.info.width=WIDTH;msg.info.height=HEIGHT;msg.info.resolution=RESOLUTION;msg.info.origin.position.x=snap['origin'][0];msg.info.origin.position.y=snap['origin'][1];msg.info.origin.orientation.w=1.;msg.data=snap['data'];pub_ns=time.monotonic_ns();maps.publish(msg)
    if not activated:
     for tid in(1,3):
      q=ChangeState.Request();q.transition.id=tid
      if not future(transition.call_async(q),10).success:raise RuntimeError('Planner lifecycle transition failed')
     if not client.wait_for_server(timeout_sec=5):raise TimeoutError('Navfn action unavailable')
     activated=True
    r['map_barrier']=barrier(snap,pub_ns,stamp_s(msg.header.stamp));r['map_sha256']=digest({'origin':snap['origin'],'data':snap['data']});r['snapshot_counts']=snap['counts'];r['source_ToF_capture_s']=snap['source_ToF_capture_s'];r['map_pose']=snap['pose'];r['goal_map_xy']=snap['goal']
    action_start=node.get_clock().now().nanoseconds*1e-9;q=ComputePathToPose.Goal();q.start=pose(*snap['pose']);q.goal=pose(*snap['goal'],snap['pose'][2]);q.use_start=True;q.planner_id='GridBased';handle=future(client.send_goal_async(q),3)
    if not handle.accepted:raise RuntimeError('Navfn action rejected')
    result=future(handle.get_result_async(),3);path=result.result.path;points=[[p.pose.position.x,p.pose.position.y]for p in path.poses];err=int(result.result.error_code);status=int(result.status)
    permitted={result.result.NO_VALID_PATH,result.result.GOAL_OCCUPIED,result.result.START_OCCUPIED};operational=(err==0 and status==4 and bool(points))or(err in permitted and status==6 and not points)
    if points and(path.header.frame_id!='probe_map'or stamp_s(path.header.stamp)<action_start-1e-5):raise RuntimeError('Old/wrong-frame Navfn path')
    if not operational or not centerline_is_known_free(points,snap):raise RuntimeError('Navfn operational failure or stale/occupied/unknown centerline')
    # Recheck current full map after the action too; no next request enters until response.
    after=latest[0]
    if after is None or not map_compatible(after[1].data,snap):raise RuntimeError('Map changed during Navfn action')
    r.update(status='complete',planner_outcome_valid=operational,action_status=status,error_code=err,path_xy=points,path_header_wall_s=stamp_s(path.header.stamp),action_started_wall_s=action_start,envelope=check_path(points,snap),native_planning_time_s=stamp_s(result.result.planning_time),source_odom_capture_s=snap['source_odom_capture_s'])
   except Exception as exc:r.update(error=repr(exc),traceback=traceback.format_exc(),planner_outcome_valid=False)
   r.update(wall_elapsed_s=time.monotonic()-began,wall_started_epoch_s=wall_started,wall_finished_epoch_s=time.time(),sampled_peak_sum_PSS_bytes=peak,clock_scheme=report['clock_scheme'])
   atomic(a.exchange/'response.json',r);atomic(a.output/f"response-{len(report['requests']):04d}.json",r);report['requests'].append(r);atomic(a.output/'summary.json',report)
  else:report['status']='expired';report['reason']='bounded_service_wall_lifetime'
 except BaseException as exc:report.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
 finally:
  atomic(a.exchange/'ready.json',{'run_nonce':a.run_nonce,'service_instance':instance,'accepting_requests':False})
  if child is not None:
   if child.poll()is None:
    os.killpg(child.pid,signal.SIGINT)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
   report['planner_returncode']=child.returncode;report['owned_planner_reaped']=child.poll()is not None
  sampler_stop.set()
  if sampler.is_alive():sampler.join(timeout=1)
  report.update(wall_s=time.monotonic()-started,sampled_peak_sum_PSS_bytes=peak,service_cpu_s=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,planner_cpu_s=resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime)
  atomic(a.output/'resource-samples.json',resource_rows);atomic(a.output/'summary.json',report);log.close();executor.shutdown(timeout_sec=2);node.destroy_node();rclpy.shutdown();spin.join(timeout=2)
 return 0 if report['status']=='complete'and report.get('owned_planner_reaped')else 1
if __name__=='__main__':raise SystemExit(main())
