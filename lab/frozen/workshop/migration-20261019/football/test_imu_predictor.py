import unittest
import numpy as np
from imu_predictor import DeliveredBodyImu,rotation_vector_apply
class FakeHead:
 def body_imu(self,s,n,q,qd,*,capture_s):return {'gyro_body_rad_s':s['gyro'],'gravity_body':s['gravity'],'encoder_used':float(q[0]),'capture_s':capture_s}
class PredictorTests(unittest.TestCase):
 def create(self):
  x=DeliveredBodyImu(FakeHead(),['j'+str(i)for i in range(14)]);x.record_encoders(.02,np.ones(14),np.zeros(14));x.record_encoders(.04,np.ones(14)*2,np.zeros(14));return x
 def sample(self):return {'seq':1,'timestamp_s':.02,'delivered_timestamp_s':.04,'orientation_valid':True,'quat':[1.,0.,0.,0.],'gyro_clipped':[False]*3,'gyro':[0.,1.,0.],'gravity':[0.,0.,-1.]}
 def test_does_not_use_unreceived_measurement(self):self.assertIsNone(self.create().predict(.04))
 def test_acquisition_encoder_and_rotation(self):
  x=self.create();x.ingest([self.sample()],now_s=.04);r=x.predict(.06);self.assertEqual(r['encoder_used'],1.);np.testing.assert_allclose(r['gravity_body'],[np.sin(.04),0.,-np.cos(.04)],atol=1e-9);self.assertFalse(r['new_measurement'])
 def test_future_delivery_refused(self):
  with self.assertRaises(ValueError):self.create().ingest([self.sample()],now_s=.03)
 def test_missing_acquisition_encoder_refused(self):
  s=self.sample();s['timestamp_s']=.03
  with self.assertRaises(ValueError):self.create().ingest([s],now_s=.04)
 def test_stale_sample_holds(self):
  x=self.create();x.ingest([self.sample()],now_s=.04);self.assertIsNone(x.predict(.105))
 def test_invalid_sample_clears_prediction(self):
  x=self.create();x.ingest([self.sample()],now_s=.04);s=self.sample();s.update(seq=2,timestamp_s=.04,orientation_valid=False,quat=None);x.ingest([s],now_s=.04);self.assertIsNone(x.predict(.04))
if __name__=='__main__':unittest.main()
