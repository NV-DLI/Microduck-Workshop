import unittest
import copy
import numpy as np
from head_feedback_core import HeadSetpointFeedback,HOME,BOUND


class FeedbackTests(unittest.TestCase):
    def call(self,h,t,seq,read,valid=True,nominal=None,q=None,enabled=True):
        return h.update(now_s=t,nominal=np.zeros((1,4))if nominal is None else nominal,
            measured_q=HOME[None]+.3 if q is None else q,seq=[seq],read_s=[read],valid=[valid],enabled=[enabled])
    def test_new_read_integrates_measured_interval_only(self):
        h=HeadSetpointFeedback(1);self.call(h,1.,1,1.);self.call(h,1.02,2,1.02)
        np.testing.assert_allclose(h.bias,-.35*.3*.02)
    def test_duplicate_packet_never_integrates(self):
        h=HeadSetpointFeedback(1);self.call(h,1.,1,1.);self.call(h,1.02,2,1.02);before=h.bias.copy()
        self.call(h,1.04,2,1.02);np.testing.assert_array_equal(h.bias,before)
    def test_invalid_freezes_and_reacquired_first_packet_no_integration(self):
        h=HeadSetpointFeedback(1);self.call(h,1.,1,1.);self.call(h,1.02,2,1.02);b=h.bias.copy();s=h.sent.copy()
        self.call(h,1.04,2,1.02,valid=False);np.testing.assert_array_equal(h.bias,b);np.testing.assert_array_equal(h.sent,s)
        self.call(h,1.06,5,1.06);np.testing.assert_array_equal(h.bias,b)
    def test_domain_antiwindup_and_slew(self):
        h=HeadSetpointFeedback(1);nom=-BOUND[None].copy();prev=h.sent.copy()
        for i in range(400):
            sent,d=self.call(h,1+i*.02,i,1+i*.02,nominal=nom,q=HOME[None]+2.)
            self.assertLessEqual(abs(sent-prev).max(),.006000001);self.assertTrue(np.all(abs(sent)<=BOUND+1e-9));self.assertTrue(np.all(abs(h.bias)<=.3500001));prev=sent
        np.testing.assert_array_equal(h.bias,np.zeros((1,4)))
    def test_baseline_has_no_feedback(self):
        h=HeadSetpointFeedback(1)
        for i in range(10):self.call(h,1+i*.02,i,1+i*.02,enabled=False)
        np.testing.assert_array_equal(h.bias,np.zeros((1,4)))
    def test_bad_clock_or_future_packet_no_mutation(self):
        for t,read in ((float('nan'),1.),(1.,1.1)):
            h=HeadSetpointFeedback(1)
            with self.assertRaises(ValueError):self.call(h,t,1,read)
            self.assertIsNone(h.last_now);self.assertFalse(h.bias.any())
    def test_deadband(self):
        h=HeadSetpointFeedback(1)
        for i in range(10):self.call(h,1+i*.02,i,1+i*.02,q=HOME[None]+.004)
        self.assertFalse(h.bias.any())
    def test_duplicate_read_cannot_be_retimestamped(self):
        h=HeadSetpointFeedback(1);self.call(h,1.,1,1.)
        with self.assertRaises(ValueError):self.call(h,1.02,1,1.02)
        self.assertEqual(h.last_now,1.)


if __name__=='__main__':unittest.main()
