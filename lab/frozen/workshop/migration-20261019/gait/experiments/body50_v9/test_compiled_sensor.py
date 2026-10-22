"""Dynamo fullgraph capture and same-noise CPU producer parity."""
import sys
from pathlib import Path
import unittest
import torch

M=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(M/'acceptance/body50_training'))
from physics_producer import Body50PhysicsProducer
from compiled_sensor import install


class TestCompiledSensor(unittest.TestCase):
    def test_full_graph_matches_all_packets_and_partial_reset(self):
        eager=Body50PhysicsProducer(4,seed=8451)
        candidate=Body50PhysicsProducer(4,seed=8451)
        candidate.sensor.strict_validation=False
        adapter=install(candidate,backend='eager')
        generator=torch.Generator().manual_seed(8452)
        valid=0
        for tick in range(360):
            if tick==220:
                ids=torch.tensor([1]);eager.reset(ids);candidate.reset(ids)
            args={'trunk_position_world_m':torch.tensor([[0.,0.,.125]]*4),
                  'trunk_quaternion_xyzw':torch.tensor([[0.,0.,0.,1.]]*4),
                  'trunk_angular_velocity_world_rad_s':torch.zeros(4,3),
                  'effective_encoder_q_rad':torch.full((4,14),tick*.0001),
                  'effective_encoder_qd_rad_s':torch.full((4,14),.02),
                  'noise_standard_normal':torch.randn((4,6),generator=generator,dtype=torch.float64)}
            expected=eager.update(**args);actual=candidate.update(**args)
            for name,value in expected.items():
                self.assertTrue(torch.equal(value,actual[name]),(tick,name))
            valid+=int(actual['orientation_valid'].sum())
        self.assertGreater(valid,100)
        self.assertEqual(adapter.calls,360)

    def test_original_rng_draw_sequence_preserved(self):
        eager=Body50PhysicsProducer(2,seed=62);candidate=Body50PhysicsProducer(2,seed=62)
        candidate.sensor.strict_validation=False
        install(candidate,backend='eager')
        args={'trunk_position_world_m':torch.tensor([[0.,0.,.125]]*2),
              'trunk_quaternion_xyzw':torch.tensor([[0.,0.,0.,1.]]*2),
              'trunk_angular_velocity_world_rad_s':torch.zeros(2,3),
              'effective_encoder_q_rad':torch.zeros(2,14),
              'effective_encoder_qd_rad_s':torch.zeros(2,14)}
        for _ in range(20):
            expected=eager.update(**args);actual=candidate.update(**args)
            for name,value in expected.items():self.assertTrue(torch.equal(value,actual[name]),name)
        self.assertTrue(torch.equal(eager.sensor.generator.get_state(),candidate.sensor.generator.get_state()))


if __name__=='__main__':unittest.main()
