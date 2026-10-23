import unittest, math
import numpy as np
from neutral_fine_step import NeutralFineStep


def packet(t):
    return {
        "seq": round(t * 50),
        "timestamp_s": t,
        "read_timestamp_s": t,
        "encoder_read_timestamp_s": t,
        "orientation_valid": True,
        "encoder_q_rad": [0.0] * 14,
        "encoder_qd_rad_s": [0.0] * 14,
        "gyro_body_rad_s": [0.0, 0.0, 0.0],
        "gravity_body": [0.0, 0.0, -1.0],
    }


def mem(t, anchor=1.0, p=(0.2, -0.035, 0.0)):
    return {
        "memory_available": t - anchor <= 4.5,
        "ball_capture_timestamp_s": anchor,
        "point_level_m": list(p),
    }


class Fine(unittest.TestCase):
    def start(self, head=(0.0, 0.0, 0.0, 0.0)):
        f = NeutralFineStep()
        f.capture(
            t=1.0,
            guard_safe=True,
            complete=True,
            trigger=True,
            actual_head=head,
            memory=mem(1.0),
            goal_heading=0.0,
            goal_fix_s=1.0,
        )
        return f

    def drive(self, f, begin, end, head=None, guard=True):
        for k in range(round(begin * 50), round(end * 50) + 1):
            t = k / 50
            if k % 3 == 0:
                f.capture(
                    t=t,
                    guard_safe=guard,
                    complete=False,
                    trigger=False,
                    actual_head=f.head,
                    memory=mem(t),
                    goal_heading=0.0,
                    goal_fix_s=t,
                )
            h = f.head if head is None else head
            out = f.policy(t=t, packet=packet(t), memory=mem(t), actual_head=h)
        return out

    def test_one_committed_step_and_new_capture_required(self):
        f = self.start()
        values = []
        for k in range(51, 160):
            t = k / 50
            if k % 3 == 0:
                f.capture(
                    t=t,
                    guard_safe=True,
                    complete=False,
                    trigger=False,
                    actual_head=f.head,
                    memory=mem(t),
                    goal_heading=0.0,
                    goal_fix_s=t,
                )
            d = f.policy(t=t, packet=packet(t), memory=mem(t), actual_head=f.head)
            values.append(d["velocity"][0])
        self.assertEqual(values.count(0.2), 30)
        self.assertEqual(f.state, "REACQUIRE")
        self.assertEqual(f.attempts, 1)
        f.capture(
            t=3.2,
            guard_safe=True,
            complete=False,
            trigger=True,
            actual_head=[0] * 4,
            memory=mem(3.2),
            goal_heading=0.0,
            goal_fix_s=3.2,
        )
        self.assertEqual(f.state, "REACQUIRE")
        f.capture(
            t=3.27,
            guard_safe=True,
            complete=True,
            trigger=False,
            actual_head=[0] * 4,
            memory=mem(3.27),
            goal_heading=0.0,
            goal_fix_s=3.27,
        )
        self.assertEqual(f.state, "IDLE")
        self.assertEqual(f.attempts, 1)

    def test_physical_return_uses_sent_neutral_then_quiet(self):
        f = self.start((-0.49, 0.49, 0, 0))
        out = self.drive(f, 1.02, 2.6)
        self.assertEqual(f.state, "RETURN")
        self.assertEqual(out["velocity"], [0, 0, 0])
        out = self.drive(f, 2.62, 3.2)
        self.assertEqual(f.state, "STEP")
        self.assertGreaterEqual(f.step_start, 3.14)

    def test_no_fake_actual_neutral(self):
        f = self.start()
        self.drive(f, 1.02, 2.0, head=[0.1, 0, 0, 0])
        self.assertEqual(f.state, "RETURN")

    def test_expiry_does_not_reset_or_reanchor(self):
        f = self.start()
        d = f.policy(
            t=1.02, packet=packet(1.02), memory=mem(1.02, 1.01), actual_head=[0] * 4
        )
        self.assertEqual(f.state, "FAILED")
        self.assertEqual(d["velocity"], [0, 0, 0])

    def test_stale_capture_immediately_zero(self):
        f = self.start()
        self.drive(f, 1.02, 1.6)
        self.assertEqual(f.state, "STEP")
        f.capture_safe = False
        d = f.policy(t=1.62, packet=packet(1.62), memory=mem(1.62), actual_head=[0] * 4)
        self.assertEqual(f.state, "SETTLE")
        self.assertEqual(d["velocity"], [0, 0, 0])

    def test_invalid_packet_during_step(self):
        f = self.start()
        self.drive(f, 1.02, 1.6)
        q = packet(1.62)
        q["orientation_valid"] = False
        d = f.policy(t=1.62, packet=q, memory=mem(1.62), actual_head=[0] * 4)
        self.assertEqual(f.state, "FAILED")
        self.assertEqual(d["velocity"], [0, 0, 0])

    def test_goal_must_really_be_fresh_at_step(self):
        f = self.start()
        f.goal_s = 0.0
        for k in range(51, 80):
            t = k / 50
            f.capture_s = t
            f.capture_safe = True
            f.policy(t=t, packet=packet(t), memory=mem(t), actual_head=[0] * 4)
        self.assertEqual(f.state, "FAILED")
        self.assertEqual(f.reason, "neutral_step_fresh_goal_required")

    def test_total_deadline_even_in_reacquire(self):
        f = self.start()
        f.state = "REACQUIRE"
        f.step_end = 3.0
        f.last_policy = 24.98
        f.capture_s = 24.98
        d = f.policy(t=25.0, packet=packet(25.0), memory=mem(25.0), actual_head=[0] * 4)
        self.assertEqual(f.reason, "neutral_step_total_deadline")
        self.assertEqual(d["velocity"], [0, 0, 0])

    def test_bad_time_no_state_mutation(self):
        f = self.start()
        before = f.__dict__.copy()
        with self.assertRaises(ValueError):
            f.capture(
                t=math.nan,
                guard_safe=True,
                complete=True,
                trigger=True,
                actual_head=[0] * 4,
                memory=mem(1),
                goal_heading=0,
                goal_fix_s=1,
            )
        self.assertEqual(before, f.__dict__)

    def test_capture_exception_revokes_in_executor(self):
        import ast
        from pathlib import Path

        tree = ast.parse(Path(__file__).with_name("executor.py").read_text())
        handlers = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ExceptHandler)
            and n.type is not None
            and "TimeoutError" in ast.unparse(n.type)
        ]
        self.assertEqual(len(handlers), 1)
        self.assertIn(
            "fine_step.fail('neutral_step_capture_error')", ast.unparse(handlers[0])
        )

    def test_new_trigger_while_active_cannot_restart_clock(self):
        f = self.start()
        f.capture(
            t=1.07,
            guard_safe=True,
            complete=True,
            trigger=True,
            actual_head=[0] * 4,
            memory=mem(1.07),
            goal_heading=0,
            goal_fix_s=1.07,
        )
        self.assertEqual(f.started, 1.0)
        self.assertEqual(f.anchor_s, 1.0)
        self.assertEqual(f.attempts, 1)

    def test_expired_memory_during_settle_zero_then_fresh_reacquire_only(self):
        f = self.start()
        f.state = "SETTLE"
        f.step_start = 4.0
        f.step_end = 4.6
        f.last_policy = 5.48
        f.quiet_start = None
        for k in range(275, 302):
            t = k / 50
            f.capture_s = t
            f.capture_safe = True
            d = f.policy(
                t=t,
                packet=packet(t),
                memory={
                    "memory_available": False,
                    "point_level_m": None,
                    "ball_capture_timestamp_s": 1.0,
                },
                actual_head=[0] * 4,
            )
            self.assertEqual(d["velocity"], [0, 0, 0])
            self.assertFalse(d["kick_authorized"])
        self.assertEqual(f.state, "REACQUIRE")
        self.assertEqual(f.anchor_s, 1.0)
        f.capture(
            t=6.04,
            guard_safe=True,
            complete=False,
            trigger=True,
            actual_head=[0] * 4,
            memory=mem(6.04),
            goal_heading=0,
            goal_fix_s=6.04,
        )
        self.assertEqual(f.state, "REACQUIRE")
        self.assertEqual(f.attempts, 1)
        f.capture(
            t=6.1,
            guard_safe=True,
            complete=True,
            trigger=False,
            actual_head=[0] * 4,
            memory=mem(6.1),
            goal_heading=0,
            goal_fix_s=6.1,
        )
        self.assertEqual(f.state, "IDLE")
        self.assertEqual(f.anchor_s, 1.0)

    def test_expired_memory_still_revokes_return_and_step(self):
        for state in ("RETURN", "STEP"):
            f = self.start()
            f.state = state
            f.last_policy = 5.5
            f.capture_s = 5.52
            f.step_start = 5.4
            d = f.policy(
                t=5.52, packet=packet(5.52), memory=mem(5.52), actual_head=[0] * 4
            )
            self.assertEqual(f.state, "FAILED")
            self.assertEqual(f.reason, "neutral_step_memory_expired_or_reanchored")
            self.assertEqual(d["velocity"], [0, 0, 0])

    def test_settle_keeps_packet_hazard_and_total_deadline(self):
        for reason in ("packet", "capture", "deadline"):
            f = self.start()
            f.state = "SETTLE"
            f.last_policy = 5.5
            f.capture_s = 5.52
            t = 25.0 if reason == "deadline" else 5.52
            if reason == "deadline":
                f.last_policy = 24.98
                f.capture_s = 25.0
            p = packet(t)
            if reason == "packet":
                p["orientation_valid"] = False
            if reason == "capture":
                f.capture_safe = False
            d = f.policy(t=t, packet=p, memory=mem(t), actual_head=[0] * 4)
            self.assertEqual(f.state, "SETTLE" if reason == "capture" else "FAILED")
            self.assertEqual(d["velocity"], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
