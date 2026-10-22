"""Same-noise full CPU packet checks; no GPU or truth-initialized sensor."""
import hashlib,json,math,random,struct,sys,unittest
from pathlib import Path
import numpy as np
import torch
M=Path(__file__).resolve().parents[2];sys.path.insert(0,str(M/'sensors'))
from beta_body_imu import BetaBodyImu,BetaBodyImuProfile
from vector_body_imu import VectorBodyImu,VectorBodyProfile
from scipy.spatial.transform import Rotation

class Noise:
 def __init__(self,seed):self.rng=random.Random(seed);self.draws=[]
 def gauss(self,mu,sigma):
  z=self.rng.gauss(0,1);self.draws.append(z);return mu+sigma*z

class TestEquivalence(unittest.TestCase):
 def test_dynamic_stationary_gap_clipping_and_independent_reset(self):
  n=4;model=VectorBodyImu(n,profile=VectorBodyProfile(orientation_correction_tau_s=2.))
  refs=[BetaBodyImu(profile=BetaBodyImuProfile(orientation_correction_tau_s=2.),seed=50+i)for i in range(n)];noise=[Noise(50+i)for i in range(n)]
  for r,ng in zip(refs,noise):r.chip.rng=ng
  elapsed=np.zeros(n);checks=0;valids=[0]*n;peak=0.
  for tick in range(1600):
   if tick in(631,1101):
    index=1 if tick==631 else 2;refs[index].reset();model.reset(torch.tensor([index]));elapsed[index]=0.
   elapsed+=.005
   # Skip a real chunk for all sensors to force lost-chronology invalidation.
   if 750<=tick<763:continue
   q=[];omega=[];acc=[];expected=[];noises=np.zeros((n,6))
   for i in range(n):
    t=elapsed[i];phase=max(0.,t-1.1)
    angles=[.11*math.sin(phase*7+i)*min(1,phase),.08*math.cos(phase*4)*min(1,phase),.2*phase]
    quat=Rotation.from_euler('xyz',angles).as_quat();w=[.5*math.sin(phase*6)*min(1,phase),.3*math.cos(phase*3)*min(1,phase),.2*min(1,phase)];a=[.6*math.sin(phase*5)*min(1,phase),.4*math.cos(phase*5)*min(1,phase),.1*math.sin(phase*4)*min(1,phase)]
    if i==3 and 330<tick<350:a=[0,0,-9.80665] # free fall cannot invent orientation
    if i==0 and tick==510:w=[14.,0,0] # gyro saturation invalidates chronology
    noise[i].draws=[];packets=refs[i].poll(float(t),q_world_from_trunk_xyzw=quat,angular_velocity_world_rad_s=w,linear_acceleration_world_m_s2=a)
    if noise[i].draws:noises[i]=noise[i].draws
    expected.append(packets);q.append(quat);omega.append(w);acc.append(a)
   got=model.poll(elapsed,q_world_from_trunk_xyzw=np.array(q),angular_velocity_world_rad_s=np.array(omega),linear_acceleration_world_m_s2=np.array(acc),noise_standard_normal=noises)
   for i,packets in enumerate(expected):
    self.assertEqual(bool(got['delivered_mask'][i]),bool(packets),(tick,i,'delivery'))
    if not packets:continue
    self.assertEqual(len(packets),1);e=packets[0];checks+=1;valids[i]+=e['orientation_valid']
    for key in('orientation_valid','seq','live_quaternion_reads'):
     self.assertEqual(got[key][i].item(),e[key],(tick,i,key,got[key][i],e[key]))
    for key in('timestamp_s','read_timestamp_s'):
     self.assertAlmostEqual(got[key][i].item(),e[key],places=10,msg=str((tick,i,key)))
    self.assertEqual(got['sample_seq'][i].item(),e['body_sensor_sample_seq'])
    raw=struct.pack('<3h3H',*got['gyro_counts'][i].tolist(),*[x&65535 for x in got['quaternion_xyz_half_bits'][i].tolist()])
    self.assertEqual(raw.hex(),e['raw_block_hex'],(tick,i,'wire bytes'))
    np.testing.assert_allclose(got['gyro_body_rad_s'][i],e['gyro'],atol=1e-11,rtol=0)
    if e['orientation_valid']:
     for key,ek in [('gravity_body','gravity'),('quat_wxyz','quat')]:
      error=float(np.max(abs(got[key][i].numpy()-e[ek])));peak=max(peak,error);self.assertLess(error,1e-10,(tick,i,key))
  self.assertGreater(checks,1200);self.assertTrue(all(v>10 for v in valids))
  print(json.dumps({'case':'dynamic-reset-gap-saturation','delivered_packets':checks,'valid_packets':valids,'wire_bytes_exact':True,'max_decoded_difference':peak}))
 def test_freefall_no_ready(self):
  sensor=VectorBodyImu(2)
  for tick in range(250):
   got=sensor.poll(tick*.005,q_world_from_trunk_xyzw=[[0,0,0,1]]*2,angular_velocity_world_rad_s=[[0,0,0]]*2,linear_acceleration_world_m_s2=[[0,0,-9.80665]]*2)
   self.assertFalse(got['orientation_valid'].any())
 def test_explicit_unsupported_delay(self):
  with self.assertRaises(ValueError):VectorBodyImu(1,profile=VectorBodyProfile(pipeline_delay_s=.02))

if __name__=='__main__':unittest.main()
