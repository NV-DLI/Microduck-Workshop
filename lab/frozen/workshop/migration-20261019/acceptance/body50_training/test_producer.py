import json,math,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy.spatial.transform import Rotation
M=Path(__file__).resolve().parents[2];sys.path.insert(0,str(M/'sensors'));sys.path.insert(0,str(M/'football'))
from isaac_beta_body_imu import IsaacBodyImu
from beta_body_imu import BetaBodyImuProfile
from body_imu_input import BodyImuInput
from physics_producer import Body50PhysicsProducer
from vector_body_imu import VectorBodyProfile
from test_equivalence import Noise

class Tests(unittest.TestCase):
 def test_private_physics_and_encoder_same_read(self):
  n=3;robot=SimpleNamespace(body_names=['trunk_base'],data=SimpleNamespace());refs=[IsaacBodyImu(robot,env_index=i,profile=BetaBodyImuProfile(orientation_correction_tau_s=2.))for i in range(n)];inputs=[BodyImuInput()for _ in range(n)];noise=[Noise(500+i)for i in range(n)]
  for r,ng in zip(refs,noise):r.sensor.chip.rng=ng
  batch=Body50PhysicsProducer(n,profile=VectorBodyProfile(orientation_correction_tau_s=2.));ticks=np.zeros(n,int);count=0;peak=0.
  for step in range(900):
   if step==510:refs[1].reset();inputs[1]=BodyImuInput();batch.reset(torch.tensor([1]));ticks[1]=0
   ticks+=1;t=ticks*.005;phase=np.maximum(t-1.1,0);position=np.stack([.03*np.sin(phase*3),.02*np.sin(phase*2),.125+.002*np.sin(phase*4)],-1);quat=Rotation.from_euler('xyz',np.stack([.05*np.sin(phase*4),.07*np.sin(phase*3),phase*.1],-1)).as_quat();omega=np.stack([.2*np.cos(phase*4),.21*np.cos(phase*3),phase*0+.1],-1)*(phase>0)[:,None]
   robot.data.body_pos_w=position[:,None];robot.data.body_quat_w=quat[:,None];robot.data.body_ang_vel_w=omega[:,None]
   q=np.sin(t[:,None]*np.arange(1,15)[None,:])*.1;qd=np.cos(t[:,None]*np.arange(1,15)[None,:])*.1;noises=np.zeros((n,6));expected=[]
   for i,r in enumerate(refs):
    inputs[i].record_encoders(float(t[i]),q[i].tolist(),qd[i].tolist());noise[i].draws=[];packets=r.update(float(t[i]));inputs[i].ingest(packets,now_s=float(t[i]));expected.append(packets)
    if noise[i].draws:noises[i]=noise[i].draws
   got=batch.update(trunk_position_world_m=position,trunk_quaternion_xyzw=quat,trunk_angular_velocity_world_rad_s=omega,effective_encoder_q_rad=q,effective_encoder_qd_rad_s=qd,noise_standard_normal=noises)
   for i,packets in enumerate(expected):
    self.assertEqual(bool(got['delivered_mask'][i]),bool(packets),(step,i))
    p=inputs[i].observe(float(t[i]));self.assertEqual(bool(got['orientation_valid'][i]),p is not None,(step,i,'ready'))
    if p is None:continue
    count+=1
    for key in('gyro_body_rad_s','gravity_body','encoder_q_rad','encoder_qd_rad_s','encoder_read_timestamp_s'):
     diff=float(np.max(abs(got[key][i].numpy()-p[key])));peak=max(peak,diff);self.assertLess(diff,1e-10,(step,i,key))
   base=torch.ones((n,61),dtype=torch.float32);out,ready=batch.build_actor_observation(base,native_home14=torch.zeros((n,14)),previous_raw_action14=torch.ones((n,14))*2)
   self.assertTrue(torch.equal(out[~ready],torch.zeros_like(out[~ready])));self.assertTrue(torch.equal(out[ready,34:48],torch.ones_like(out[ready,34:48])*2))
  self.assertGreater(count,1000);print(json.dumps({'producer_control_samples':count,'max_measurement_difference':peak,'independent_reset':True,'invalid_rows_no_truth':True}))
if __name__=='__main__':unittest.main()
