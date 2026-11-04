"""State and cohort contracts independent of Kit/GPU."""
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import tempfile
import json
import math

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('onboard_session_test', ROOT/'lab/onboard/session.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class SessionTest(unittest.TestCase):
    def test_cleanup_error_preserves_physical_result(self):
        obj = module.ResetSession.__new__(module.ResetSession)
        obj.index = 1
        obj.rows = [dict(index=1, physical_success=True)]
        states = []
        obj.write_status = states.append
        obj.record_technical_error(RuntimeError('cleanup failed'), 'finalized')
        self.assertEqual(len(obj.rows), 1)
        self.assertTrue(obj.rows[0]['physical_success'])
        self.assertTrue(obj.rows[0]['technical_error'])
        self.assertEqual(states, ['technical_error'])
        obj.index = 2
        obj.args = SimpleNamespace(seed=43, root_x=-.55, root_y=.12, output=Path('episode-002'))
        obj.record_technical_error(RuntimeError('build failed'), 'build')
        self.assertEqual(len(obj.rows), 2)
        self.assertEqual(obj.rows[-1]['status'], 'technical_error')

    def test_positions_bounded_and_reproducible(self):
        center = (-.55, .12)
        self.assertEqual(module.initial_position(983402, 1, center), center)
        points = [module.initial_position(983402, i, center) for i in range(2, 502)]
        self.assertGreater(len(set(points)), 490)
        for i, (x,y) in enumerate(points, 2):
            self.assertLessEqual(abs(x-center[0]), .05)
            self.assertLessEqual(abs(y-center[1]), .05)
            self.assertEqual((x,y), module.initial_position(983402, i, center))

    def test_yaw_range_nominal_and_seed_reproducibility(self):
        center = (-.55, .12)
        self.assertEqual(module.initial_pose(983402, 1, center), (*center, 0.0))
        poses = [module.initial_pose(983402, i, center) for i in range(2, 1002)]
        for i, (x, y, yaw) in enumerate(poses, 2):
            self.assertLessEqual(abs(x-center[0]), .05)
            self.assertLessEqual(abs(y-center[1]), .05)
            self.assertLessEqual(abs(yaw), math.pi/12)
            self.assertEqual((x,y,yaw), module.initial_pose(983402,i,center))
        self.assertGreater(max(q[2] for q in poses), math.radians(14.9))
        self.assertLess(min(q[2] for q in poses), math.radians(-14.9))

    def test_argument_replacement_removes_duplicate_spellings(self):
        argv = ['run', '--root-yaw-rad=.5', '--seconds', '60', '--root-yaw-rad', '-.2']
        module.set_argument(argv, '--root-yaw-rad', .1)
        self.assertEqual(argv, ['run', '--seconds', '60', '--root-yaw-rad', '0.1'])

    def test_episode_paths_and_limit(self):
        with tempfile.TemporaryDirectory() as folder:
            obj = module.ResetSession.__new__(module.ResetSession)
            obj.root = Path(folder)
            obj.args = SimpleNamespace(output=obj.root, seed=42, root_x=-.55, root_y=.12,
                                       session_episodes=2, ball_xy=[.02,0])
            obj.seed=42; obj.center=(-.55,.12);obj.index=0;obj.requested=False;obj.rows=[]
            obj.experiment=None
            obj.app=SimpleNamespace(is_running=lambda:True)
            obj.resources=[]
            obj.resource_snapshot=lambda phase:dict(phase=phase)
            old=sys.argv
            try:
                sys.argv=['test', '--root-x=-9', '--root-yaw-rad=.3', '--seconds', '60']
                self.assertTrue(obj.next_episode())
                self.assertEqual(obj.args.output.name,'episode-001')
                self.assertEqual(obj.args.root_yaw_rad,0.0)
                self.assertEqual((obj.args.root_x,obj.args.root_y),obj.center)
                self.assertEqual(float(sys.argv[sys.argv.index('--root-yaw-rad')+1]),0.0)
                self.assertFalse(obj.next_episode())
                obj.requested=True
                self.assertTrue(obj.next_episode())
                self.assertEqual(obj.args.seed,43)
                self.assertEqual(obj.args.ball_xy,[.02,0])
                self.assertEqual(sys.argv[sys.argv.index('--output')+1],str(obj.args.output))
                self.assertEqual(float(sys.argv[sys.argv.index('--root-yaw-rad')+1]),obj.args.root_yaw_rad)
                self.assertEqual(float(sys.argv[sys.argv.index('--root-x')+1]),obj.args.root_x)
                self.assertEqual(float(sys.argv[sys.argv.index('--root-y')+1]),obj.args.root_y)
                self.assertEqual(sys.argv[sys.argv.index('--seconds')+1],'60')
                saved=json.loads((obj.root/'session.json').read_text())
                self.assertEqual(saved['root_yaw_offset_rad'],obj.args.root_yaw_rad)
                self.assertEqual(saved['jitter_xy_m'],[.05,.05])
                obj.requested=True
                self.assertFalse(obj.next_episode())
            finally:sys.argv=old

if __name__=='__main__':unittest.main()
