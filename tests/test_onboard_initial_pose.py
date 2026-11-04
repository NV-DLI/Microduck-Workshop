"""Independent CPU checks of reset attitude, fixture agreement and wiring order."""
import ast
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'lab/onboard/initial_pose.py'
spec = importlib.util.spec_from_file_location('onboard_pose_test', PATH)
pose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pose)
REFERENCE = json.loads((ROOT/'lab/course_labs/startup/grounded-pose.json').read_text())


def rotation(q):
    """Independent matrix oracle, normalized to allow measured float rounding."""
    norm=math.sqrt(sum(x*x for x in q))
    x,y,z,w=[v/norm for v in q]
    return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
            [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
            [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]


class InitialPoseTest(unittest.TestCase):
    def test_nominal_preserves_recorded_quaternion_exactly(self):
        self.assertEqual(pose.rotate_world_z(REFERENCE['quat'],0),tuple(REFERENCE['quat']))

    def test_left_world_rotation_matches_independent_matrix(self):
        original=rotation(REFERENCE['quat'])
        for degrees in [-30,-19,0,13,30]:
            yaw=math.radians(degrees)
            actual=rotation(pose.rotate_world_z(REFERENCE['quat'],yaw))
            c,s=math.cos(yaw),math.sin(yaw)
            expected=[[c*original[0][j]-s*original[1][j] for j in range(3)],
                      [s*original[0][j]+c*original[1][j] for j in range(3)],original[2]]
            for row,wanted in zip(actual,expected):
                for value,target in zip(row,wanted):self.assertAlmostEqual(value,target,places=13)

    def test_roll_pitch_and_all_local_point_heights_preserved(self):
        old=rotation(REFERENCE['quat'])
        for yaw in [-math.pi/6,.21,math.pi/6]:
            new=rotation(pose.rotate_world_z(REFERENCE['quat'],yaw))
            self.assertAlmostEqual(math.atan2(new[2][1],new[2][2]),math.atan2(old[2][1],old[2][2]),places=13)
            self.assertAlmostEqual(math.asin(-new[2][0]),math.asin(-old[2][0]),places=13)
            for point in [(1,0,0),(0,1,0),(0,0,1),(.02,-.03,-.1)]:
                self.assertAlmostEqual(sum(a*b for a,b in zip(new[2],point)),sum(a*b for a,b in zip(old[2],point)),places=13)

    def test_yaw_is_offset_not_absolute_or_discarded(self):
        baseline=pose.yaw_xyzw(REFERENCE['quat'])
        self.assertNotEqual(baseline,0)
        for yaw in [-math.pi/6,0,math.pi/6]:
            self.assertAlmostEqual(pose.yaw_xyzw(pose.rotate_world_z(REFERENCE['quat'],yaw)),baseline+yaw,places=13)

    def test_scene_and_fixture_share_grounded_pose(self):
        cfg=SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(init_state=SimpleNamespace(pos=None,rot=None))))
        fixture={'root_pose_xyzw':[-.6,.2,.12,0,0,0,1]}
        actual=pose.configure_grounded_pose(cfg,fixture,REFERENCE,.3)
        self.assertEqual(actual[:3],[-.6,.2,REFERENCE['root'][2]])
        self.assertEqual(tuple(actual[:3]),cfg.scene.robot.init_state.pos)
        self.assertEqual(tuple(actual[3:]),cfg.scene.robot.init_state.rot)
        self.assertEqual(actual,fixture['root_pose_xyzw'])
        self.assertAlmostEqual(fixture['root_yaw_rad'],pose.yaw_xyzw(REFERENCE['quat'])+.3)

    def test_rejects_nonfinite_nonunit_or_out_of_range(self):
        for q,yaw in [([0,0,0,2],0),([0,0,0,float('nan')],0),([0,0,0,1],float('inf'))]:
            with self.assertRaises(ValueError):pose.rotate_world_z(q,yaw)
        with self.assertRaises(ValueError):pose.configure_grounded_pose(None,{},REFERENCE,math.pi/6+.01)

    def test_executor_initializes_before_creation_and_keeps_brake_sequence(self):
        source=(ROOT/'lab/onboard/executor.py').read_text()
        tree=ast.parse(source)
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        code=ast.get_source_segment(source,run)
        self.assertLess(code.index('configure_grounded_pose('),code.index('env = gym.make('))
        self.assertNotIn('root[0, 3:] =',code)
        self.assertNotIn('root[0, 2] =',code)
        self.assertLess(code.index('robot.write_root_pose_to_sim_index(root_pose=root)'),code.index('initialization_brake.lock()'))
        self.assertIn('support = StartupSupport(\n            fixture["root_pose_xyzw"]',code)
        self.assertIn('fixture["root_pose_xyzw"] = root[0].detach().cpu().tolist()',code)
        self.assertLess(code.index('last_action = model(policy_obs)'),code.index('initialization_brake.release(t)',code.index('last_action = model(policy_obs)')))
        self.assertLess(code.index('initialization_brake.release(t)',code.index('last_action = model(policy_obs)')),code.index('raw_action = last_action.clone()'))
        self.assertIn('pose_helper = ONBOARD_DIR / "initial_pose.py"',code)
        arguments=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='arguments')
        declarations=[n for n in ast.walk(arguments) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value=='--root-yaw-rad']
        self.assertEqual(len(declarations),1)
        self.assertEqual(next(n.value.value for n in declarations[0].keywords if n.arg=='default'),0.0)

    def test_navigation_uses_observation_not_initial_truth(self):
        code=(ROOT/'lab/onboard/executor.py').read_text()
        start=code.index('controller = BetaNavigator(')
        end=code.index('report["navigation_ground_mode"]',start)
        self.assertNotIn('root_yaw',code[start:end])
        self.assertNotIn('fixture',code[start:end])
        # The existing bridge rotates using the observed map pose provided by its caller.
        bridge=(ROOT/'lab/runtime/simulation/core/football/path_request.py').read_text()
        self.assertIn('x, y, yaw = pose_map_xyyaw',bridge)


if __name__=='__main__':unittest.main()
