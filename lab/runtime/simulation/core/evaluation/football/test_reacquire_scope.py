import ast
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from neutral_fine_step import NeutralFineStep
from test_neutral_fine_step import packet, mem

D = Path(__file__).parent
R = D.parents[3] / "configuration/core/football-v57-full-task180s"


def restore(cls, row):
    f = cls()
    d = row["public_neutral_fine_step"]
    for name, field in {
        "state": "state", "reason": "reason", "started": "started_s",
        "step_start": "step_start_s", "step_end": "step_end_s",
        "anchor_s": "anchor_capture_s", "quiet_start": "quiet_started_s",
        "attempts": "attempts", "last_finished": "last_finished_s",
        "capture_s": "capture_guard_s", "capture_safe": "capture_safe",
        "goal_s": "goal_fix_s", "goal_heading": "goal_heading_rad",
        "step_goal_heading": "step_goal_heading_rad",
    }.items():
        setattr(f, name, d[field])
    f.head = np.zeros(4)
    f.last_policy = row["timestamp_s"]
    return f


class ReacquireScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policies = [json.loads(x) for x in (R / "policy-trace.jsonl").read_text().splitlines()]
        cls.controls = [json.loads(x) for x in (R / "control.jsonl").read_text().splitlines()]
        path = D.parent / "fine_step/neutral_fine_step.py"
        spec = importlib.util.spec_from_file_location("frozen_v57_fine_for_test", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        cls.Old = old.NeutralFineStep

    def test_actual_original_failure_and_new_head_only_continuation(self):
        previous, last = self.policies[-2:]
        control = self.controls[-1]
        self.assertEqual(previous["public_neutral_fine_step"]["state"], "REACQUIRE")
        self.assertTrue(control["public_attention_head_permission"]["accepted"])
        for cls, expected in ((self.Old, "FAILED"), (NeutralFineStep, "REACQUIRE")):
            f = restore(cls, previous)
            f.capture_s = control["timestamp_s"]
            f.capture_safe = control["public_neutral_fine_step"]["capture_safe"]
            if cls is NeutralFineStep:
                f.capture_head_safe = True
            out = f.policy(t=last["timestamp_s"], packet=last["IMU_prediction"],
                           memory=last["public_neutral_fine_step"]["memory"],
                           actual_head=last["actual_head_command_rad"])
            self.assertEqual(out["state"], expected)
            self.assertEqual(out["velocity"], [0., 0., 0.])
            self.assertFalse(out["kick_authorized"])
            self.assertFalse(out["memory"]["memory_available"])
            self.assertEqual(out["anchor_capture_s"], 54.535000000000004)
            if cls is self.Old:
                self.assertEqual(out["reason"], "neutral_step_stale_or_hazard_capture")

    def test_phase_matrix_hazard_revokes_movement_but_holds_zero_settle(self):
        for state in ("RETURN", "STEP", "SETTLE"):
            f = NeutralFineStep()
            f.state = state; f.started = 1.; f.last_policy = 1.98
            f.capture_s = 2.; f.capture_safe = False; f.capture_head_safe = True
            f.anchor_s = 1.; f.head = np.zeros(4); f.step_start = 1.6
            out = f.policy(t=2., packet=packet(2.), memory=mem(2.), actual_head=[0.] * 4)
            self.assertEqual(out["state"], "SETTLE")
            self.assertEqual(out["reason"], None if state=="SETTLE" else "public_guard_stop_requires_fresh_reacquisition")
            self.assertEqual(out["velocity"], [0., 0., 0.])

    def test_new_step_still_requires_original_body_safe_capture(self):
        f = NeutralFineStep()
        out = f.capture(t=1., guard_safe=False, head_only_safe=True, complete=True,
                        trigger=True, actual_head=[0.] * 4, memory=mem(1.),
                        goal_heading=0., goal_fix_s=1.)
        self.assertEqual(out["state"], "FAILED")
        self.assertEqual(out["reason"], "neutral_step_initial_public_domain_rejected")
        self.assertEqual(out["velocity"], [0., 0., 0.])

    def reacquire(self):
        f = restore(NeutralFineStep, self.policies[-2])
        f.capture_s = 62.935; f.capture_safe = False; f.capture_head_safe = True
        return f

    def test_complete_but_hazard_cannot_end_reacquire_or_start_step(self):
        f = self.reacquire()
        out = f.capture(t=62.94, guard_safe=False, head_only_safe=True, complete=True,
                        trigger=True, actual_head=[0.] * 4, memory=mem(62.94),
                        goal_heading=0., goal_fix_s=62.94)
        self.assertEqual(out["state"], "REACQUIRE")
        self.assertEqual(out["attempts"], 1)
        self.assertEqual(out["velocity"], [0., 0., 0.])

    def test_stale_invalid_or_missing_head_permission_and_deadline_fail(self):
        for mode in ("stale", "packet", "head", "deadline"):
            f = self.reacquire()
            t = 62.94
            if mode == "stale": f.capture_s = 62.7
            if mode == "head": f.capture_head_safe = False
            if mode == "deadline":
                t = f.started + 24.; f.last_policy = t - .02; f.capture_s = t
            p = packet(t)
            if mode == "packet": p["orientation_valid"] = False
            out = f.policy(t=t, packet=p, memory={"memory_available": False},
                           actual_head=[0.] * 4)
            self.assertEqual(out["state"], "FAILED")
            self.assertEqual(out["velocity"], [0., 0., 0.])

    def test_actual_executor_permission_assignment_is_scope_bound(self):
        tree = ast.parse((D / "executor.py").read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                    and any(isinstance(x, ast.Name) and x.id == "fine_head_only" for x in n.targets))
        expression = compile(ast.Expression(node.value), "<actual-executor-permission>", "eval")
        env = {"fine_step": SimpleNamespace(state="REACQUIRE"), "t": 2.,
               "attention_permission": {"accepted": True, "capture_s": 2., "owner": "BALL_REACQUIRE"},
               "last_control": {"public_attention_owner": {"owner": "BALL_REACQUIRE"},
                                "velocity_body_xyw": [0., 0., 0.], "kick_request": False}}
        self.assertTrue(eval(expression, {"bool": bool}, env))
        for field, value in (("state", "STEP"), ("capture", 1.9), ("owner", "FAILED"),
                             ("body", [.2, 0., 0.]), ("kick", True)):
            altered = copy.deepcopy(env)
            if field == "state": altered["fine_step"].state = value
            if field == "capture": altered["attention_permission"]["capture_s"] = value
            if field == "owner": altered["last_control"]["public_attention_owner"]["owner"] = value
            if field == "body": altered["last_control"]["velocity_body_xyw"] = value
            if field == "kick": altered["last_control"]["kick_request"] = value
            self.assertFalse(eval(expression, {"bool": bool}, altered))


if __name__ == "__main__":
    unittest.main()
