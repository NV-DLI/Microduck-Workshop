"""Causal input boundaries and one-shot skill execution."""
import math
import unittest

from learning.walking import KickLatch, contract, pack_state, validate_demonstrated_request


class WalkingContractTests(unittest.TestCase):
    def test_current_motor_requests_are_not_model_inputs(self):
        a=list(range(61)); b=a[:48]+[100.]*13
        self.assertEqual(pack_state(a,[0.]*8),pack_state(b,[0.]*8))
        self.assertEqual(len(pack_state(a,[0.]*8)),58)
        self.assertEqual(len(contract([str(i) for i in range(14)])['state_names']),58)

    def test_bad_teacher_actions_are_rejected_not_clipped(self):
        for v in (math.nan,math.inf,2.):
            with self.assertRaises(ValueError):
                validate_demonstrated_request([v]+[0.]*7)
        self.assertEqual(validate_demonstrated_request([0.,0.,.8]+[0.]*5)[2],.8)

    def test_one_kick_after_measured_quiet_then_stand(self):
        latch=KickLatch()
        packet=dict(gyro=[0.]*3,gravity=[0.,0.,-1.],qd=[0.]*14,previous_head=[0.]*4)
        for i in range(25):
            self.assertEqual(latch.update(i*.02,[0.]*7+[1.],**packet),[0.]*8)
        self.assertEqual(latch.update(.5,[0.]*7+[1.],**packet)[7],1.)
        self.assertEqual(latch.update(1.,[.3]+[0.]*7,**packet),[0.]*7+[1.])
        self.assertEqual(latch.update(4.5,[0.]*7+[1.],**packet),[0.]*8)
        self.assertEqual(latch.update(4.52,[0.]*7+[1.],**packet),[0.]*8)
        self.assertTrue(latch.completed)
        with self.assertRaises(ValueError):latch.update(math.nan,[0.]*8,**packet)

    def test_nearly_neutral_head_cannot_be_snapped_into_kick(self):
        latch=KickLatch()
        for i in range(40):
            out=latch.update(i*.02,[0.]*7+[1.],gyro=[0.]*3,
                             gravity=[0.,0.,-1.],qd=[0.]*14,previous_head=[.019,0.,0.,0.])
            self.assertEqual(out,[0.]*8)
        self.assertIsNone(latch.started)


if __name__=='__main__':unittest.main()
