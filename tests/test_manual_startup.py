"""CPU regression checks for a reset queued while manual startup is settling.

Uses the real reset and takeover blocks extracted from lab/manual.py and the
real JointBrake with a minimal tensor-backed robot. No Isaac or CUDA is used.
Run with a Python environment that includes torch:
    python -m unittest discover -s tests -p test_manual_startup.py
"""
import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def source_block(predicate):
    tree = ast.parse((ROOT / "lab/manual.py").read_text())
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.If) and predicate(ast.unparse(node.test))]
    if len(matches) != 1:
        raise AssertionError(f"Expected one source block, found {len(matches)}")
    module = ast.Module(body=[copy.deepcopy(matches[0])], type_ignores=[])
    return compile(ast.fix_missing_locations(module), "<manual source block>", "exec")


@unittest.skipIf(torch is None, "torch is required for CPU JointBrake checks")
class ManualStartupTest(unittest.TestCase):
    def setUp(self):
        path = ROOT / "lab/groot_walking/student/joint_brake.py"
        spec = importlib.util.spec_from_file_location("manual_startup_brake", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.events = []
        events = self.events

        class Brake(module.JointBrake):
            def release(self, timestamp):
                if self.active:
                    events.append("release")
                super().release(timestamp)

        self.Brake = Brake
        ns = SimpleNamespace

        class Robot:
            def __init__(self):
                self.data = ns(
                    joint_pos_limits=ns(torch=torch.tensor([[[-1., 1.], [-2., 2.]]])),
                    default_joint_pos=ns(torch=torch.zeros((1, 2))),
                    joint_pos=ns(torch=torch.zeros((1, 2))))
                self.model = ns(
                    joint_limit_lower=torch.tensor([-1., -2.]),
                    joint_limit_upper=torch.tensor([1., 2.]),
                    joint_armature=torch.ones(2), body_mass=torch.ones(1),
                    body_inertia=torch.ones(1))
                self._backend_manager = ns(get_model=lambda: self.model)
                self.root_view = ns(is_floating_base=True)

            def write_joint_position_limit_to_sim_index(self, limits, warn_limit_violation):
                self.data.joint_pos_limits.torch.copy_(limits)
                self.model.joint_limit_lower.copy_(limits[0, :, 0])
                self.model.joint_limit_upper.copy_(limits[0, :, 1])

        self.robot = Robot()
        self.original = self.robot.data.joint_pos_limits.torch.clone()
        self.reset_block = source_block(lambda test: test == "flags['reset']")
        # Identify the takeover check by its actual diagnostic; the other check
        # releases the old brake before a reset.
        tree = ast.parse((ROOT / "lab/manual.py").read_text())
        matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "start_brake is not None and start_brake.active"
                   and "COURSE_POLICY_TAKEOVER" in ast.unparse(node)]
        self.assertEqual(len(matches), 1)
        self.takeover_block = compile(ast.fix_missing_locations(ast.Module(
            body=[copy.deepcopy(matches[0])], type_ignores=[])), "<manual takeover>", "exec")

    def run_reset(self, prior_state):
        prior = None if prior_state == "absent" else self.Brake(self.robot)
        if prior is not None:
            prior.lock()
            if prior_state == "released":
                prior.release(0)
        self.events.clear()

        def reset():
            self.events.append("reset")
            # State reset must never start with temporary joint limits still set.
            self.assertTrue(torch.equal(self.robot.data.joint_pos_limits.torch, self.original))

        def settle(env, render):
            self.events.append("settle")
            self.assertTrue(render)
            brake = self.Brake(self.robot)
            brake.lock()
            return brake, {"settled": True}

        namespace = dict(
            flags=dict(reset=True, kick=True),
            start_brake=prior, step=17, env=SimpleNamespace(reset=reset),
            held={"W"}, kick_start=5, fallen=True, scored=True, resets=2,
            a=SimpleNamespace(grounded_start=True),
            settle=settle, events=[], time=SimpleNamespace(monotonic=lambda: 1),
            json=__import__("json"), print=lambda *args, **kwargs: None)
        exec(self.reset_block, namespace)
        self.assertEqual(self.events, (["release"] if prior_state == "active" else [])
                         + ["reset", "settle"])
        self.assertTrue(namespace["start_brake"].active)
        self.assertEqual(namespace["held"], set())
        self.assertIsNone(namespace["kick_start"])
        self.assertFalse(namespace["flags"]["reset"])
        self.assertFalse(namespace["flags"]["kick"])
        self.assertFalse(namespace["fallen"])
        self.assertFalse(namespace["scored"])
        self.assertEqual(namespace["resets"], 3)
        # Execute the real takeover block after a simulated policy output.
        exec(self.takeover_block, namespace)
        self.assertFalse(namespace["start_brake"].active)
        self.assertTrue(torch.equal(self.robot.data.joint_pos_limits.torch, self.original))
        self.assertEqual(self.events[-1], "release")

    def test_real_actor_output_precedes_release_and_action_processing(self):
        tree = ast.parse((ROOT / "lab/manual.py").read_text())
        loop = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.For) and ast.unparse(node.target) == "step")
        first = next(i for i, node in enumerate(loop.body)
                     if isinstance(node, ast.Assign)
                     and ast.unparse(node) == "action = walk(obs)")
        last = next(i for i, node in enumerate(loop.body)
                    if isinstance(node, ast.Expr)
                    and ast.unparse(node) == "env.action_manager.process_action(action)")
        block = compile(ast.fix_missing_locations(ast.Module(
            body=copy.deepcopy(loop.body[first:last + 1]), type_ignores=[])),
            "<manual actor-to-processing>", "exec")
        brake = self.Brake(self.robot)
        brake.lock()
        observation = torch.zeros((1, 2))

        def walk(obs):
            self.assertTrue(brake.active)
            self.assertIs(obs, observation)
            self.events.append("actor_output")
            return torch.ones((1, 2))

        def process_action(action):
            self.assertFalse(brake.active)
            self.assertTrue(torch.equal(action, torch.ones((1, 2))))
            self.assertTrue(torch.equal(self.robot.data.joint_pos_limits.torch, self.original))
            self.events.append("process_action")

        exec(block, dict(
            obs=observation, walk=walk, kick_start=None, step=0,
            action_rows=[], cmd=SimpleNamespace(vel_command_b=torch.zeros((1, 3))),
            start_brake=brake, env=SimpleNamespace(action_manager=SimpleNamespace(
                process_action=process_action)), json=__import__("json"),
            print=lambda *args, **kwargs: None))
        self.assertEqual(self.events, ["actor_output", "release", "process_action"])

    def test_reset_queued_before_first_policy_restores_old_brake_first(self):
        self.run_reset("active")

    def test_reset_after_normal_takeover(self):
        self.run_reset("released")

    def test_reset_without_prior_brake(self):
        self.run_reset("absent")


if __name__ == "__main__":
    unittest.main()
