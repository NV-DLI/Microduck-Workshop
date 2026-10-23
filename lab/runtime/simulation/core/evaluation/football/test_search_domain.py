import ast, json, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from ball_reacquisition import BallAwareGaze, BallSearch, LOWER, UPPER
from ball_view_hint import BallViewHint
from native_head_domain import HeadCommandDomainAdapter
from perception import BetaPerception
from hardware_head_fk import HEAD_JOINTS

D = Path(__file__).resolve().parent
M = D.parents[1]
O = M.parents[1] / "configuration/core"


class SearchDomain(unittest.TestCase):
    def owner(self):
        return SimpleNamespace(
            attention_owner=SimpleNamespace(state="BALL_REACQUIRE"),
            ball_recovery_active=True,
            ball_recovery_context_valid=True,
            refresh_too_close=False,
            ball_search=BallSearch(),
            ball_view_hint=BallViewHint(),
            ball_recovery_pixel_input=None,
            ball_time=1.0,
            vision=BetaPerception(),
        )

    def test_only_lost_ball_engineering_yaw_expanded(self):
        np.testing.assert_array_equal(LOWER, [-1.6, 0.0])
        np.testing.assert_array_equal(UPPER, [1.6, 1.4])
        for yaw in (-1.6, -1.508, 1.6):
            g = HeadCommandDomainAdapter().solve([yaw, 1.175])
            self.assertTrue(g["accepted"])
            self.assertLessEqual(g["source_command_excess_rad"], 1e-9)

    def test_actual_partial_frame_can_step_past_old_bound(self):
        d = json.loads(
            (O / "football-v50-head-tracking-public-audit-v1.json").read_text()
        )
        x = next(x for x in d["rows"] if x["timestamp_s"] == 30.0)
        owner = self.owner()
        g = BallAwareGaze(owner)
        g.current = np.array(x["actual_command_gaze_rad"])
        g.timestamp = 29.935
        R = g.parameters_domain.rotation(
            np.asarray(x["actual_public_head_absolute_rad"])
        )
        owner.ball_recovery_pixel_input = (
            {
                "observed_visible_box_center_px": x["visible_center_pixel"],
                "capture_timestamp_s": 30.0,
                "cropped": True,
            },
            R,
        )
        owner.attention_head_permission = {
            "accepted": True,
            "capture_s": 30.0,
            "owner": "BALL_REACQUIRE",
        }
        old = g.current.copy()
        g.update(old, 30.0)
        self.assertLess(g.current[0], -1.2)
        self.assertGreaterEqual(g.current[0], -1.6)
        self.assertTrue(
            owner.ball_recovery_diagnostic["parameters_native_domain"]["accepted"]
        )
        self.assertFalse(owner.ball_recovery_diagnostic["base_motion_authorized"])

    def test_rejected_parameters_holds_and_does_not_extend_deadline(self):
        owner = self.owner()
        g = BallAwareGaze(owner)
        g.current = np.array([-0.4, 1.1])
        g.timestamp = 1.0
        owner.ball_search.begin(1.0, g.current)
        start = owner.ball_search.started
        prior = g.current.copy()
        owner.attention_head_permission = {
            "accepted": True,
            "capture_s": 2.0,
            "owner": "BALL_REACQUIRE",
        }
        g.parameters_domain = SimpleNamespace(
            last=np.zeros(4),
            solve=lambda target: {"accepted": False, "rejection": "test-domain"},
        )
        g.update([0.0, 0.0], 2.0)
        np.testing.assert_array_equal(prior, g.current)
        self.assertEqual(start, owner.ball_search.started)
        self.assertFalse(
            owner.ball_recovery_diagnostic["parameters_native_domain"]["accepted"]
        )

    def test_original_image_jacobian_arithmetic_exact(self):
        a = ast.parse((M / "evaluation/image_head_servo.py").read_text())
        b = ast.parse((D / "image_head_servo.py").read_text())
        f = lambda t: next(x for x in t.body if isinstance(x, ast.FunctionDef))
        self.assertEqual(ast.dump(f(a)), ast.dump(f(b)))


if __name__ == "__main__":
    unittest.main()
