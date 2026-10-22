"""Test executor contract: Microduck simulation and control."""
import ast
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SRC = (HERE/'executor.py').read_text()


class ExecutorContract(unittest.TestCase):
    def test_parses_and_keeps_original_single_entry_points(self):
        ast.parse(SRC)
        self.assertEqual(SRC.count('fine_step.capture('), 1); self.assertEqual(SRC.count('settled_entry('), 1)
        self.assertEqual(SRC.count('route_at_step_end('), 1)

    def test_explicit_roots_no_inferred_M(self):
        for name in ('executor.py', 'execute.py', 'execute_media.py'):
            text = (HERE/name).read_text()
            self.assertNotIn('parents[1]', text, name); self.assertNotIn("M.parents[1]/'outputs", text, name)
            self.assertIn('paths', text, name)
        import paths
        self.assertEqual(paths.INSTALL_DIR, paths.M/'evaluation/football')
        with self.assertRaises(RuntimeError):
            paths.require_installed(HERE.parent/'not_installed')

    def test_approval_required_and_production_only(self):
        self.assertIn("p.add_argument('--recovery-approval',type=Path,required=True", SRC)
        self.assertIn('verify_approval(a.recovery_approval,parameters_dir=ENTRY_DIR,allow_test_only=False)', SRC)
        execute = (HERE/'execute.py').read_text()
        self.assertIn('verify_approval(a.recovery_approval,parameters_dir=D,allow_test_only=False)', execute)
        media = (HERE/'execute_media.py').read_text()
        self.assertIn('require_installed(D)', media); self.assertIn("release.get('test_only') is not False", media)
        self.assertFalse((HERE/'release.json').exists(), 'no release may be frozen before the physical gates')

    def test_policy_tick_order(self):
        order = ["sent_head_before=list(actual_head)",
                 "handoff_state=handoff.policy(now_s=t,packet=last_body_imu,actual_sent_head=sent_head_before,task_s=task_clock)",
                 "fine_state=fine_step.policy(",
                 "if handoff_state['override_body_zero']:velocity=[0.,0.,0.]",
                 "if fine_state['override']:velocity=fine_state['velocity']",
                 "recovery_velocity=handoff.recovery_velocity(t)",
                 "if recovery_velocity is not None:velocity=recovery_velocity",
                 "recovery_head_permission=recovery_reacquire_head_allowed(",
                 "if not recovery_head_permission['allowed']:actual_head=sent_head_before",
                 "actual_head,head_feedback_trace=head_feedback.update(",
                 "if head_feedback_trace['body_zero_for_withdrawal']:velocity=[0.,0.,0.]",
                 "velocity,recovery_row=handoff.recovery_after_head(",
                 "commands.vel_command_b[:]=torch.tensor([velocity],device=env.device)",
                 "head_cmd.command[:]=torch.tensor([actual_head],device=env.device)"]
        positions = [SRC.index(x) for x in order]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(SRC.count('handoff.recovery_after_head('), 1)

    def test_recovery_owns_controller_body_and_head(self):
        self.assertEqual(SRC.count('controller.external_fine_step_active=fine_step.active or handoff.recovery_owns_body'), 2)
        self.assertEqual(SRC.count('controller.external_fine_step_head_owned=fine_step.hold_navigation_head or handoff.recovery_owns_head'), 2)
        self.assertNotIn('controller.external_fine_step_active=fine_step.active\n', SRC)

    def test_capture_hook_after_guards_and_fine_capture(self):
        guard = SRC.index("rear_inputs=(fine_R,g['height']['conditional_root_height_m'])")
        fine = SRC.index('fine_state=fine_step.capture(')
        hook = SRC.index('recovery_row=handoff.recovery_capture(now_s=t,task_s=task_t,guard_safe=fine_guard,complete=complete')
        flags = SRC.index('controller.external_fine_step_active=fine_step.active or handoff.recovery_owns_body', fine)
        self.assertLess(guard, fine); self.assertLess(fine, hook); self.assertLess(hook, flags)
        self.assertIn("controller.final_approach.invalidate(t,'recovery_release_requires_new_original_observation')", SRC)
        self.assertEqual(SRC.count('handoff.recovery_capture_unavailable('), 2)

    def test_exception_handler_still_single_and_revokes(self):
        tree = ast.parse(SRC)
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler) and n.type is not None and 'TimeoutError' in ast.unparse(n.type)]
        self.assertEqual(len(handlers), 1)
        body = ast.unparse(handlers[0])
        self.assertIn("fine_step.fail('neutral_step_capture_error')", body)
        self.assertIn('recovery_capture_unavailable', body)

    def test_terminal_unchanged_and_recovery_archived(self):
        self.assertIn("if handoff.state in ('COMPLETE','FAILED'):", SRC)
        self.assertIn("'recovery-trace'", SRC); self.assertIn("report['retreat_recovery']=handoff.recovery_report()", SRC)
        self.assertNotIn('scorer.reset', SRC); self.assertNotIn('contact.reset', SRC)
        self.assertEqual(SRC.count('PhysicalGoalScorerV2(MICRODUCK_JOINT_NAMES)'), 1)

    def test_final_approach_held_without_budget_consumption(self):
        # Recovery sets external_fine_step_active, so TaskNavigator passes authorized=False:
        # an otherwise-due turn must not consume the cumulative 24-turn budget.
        from final_approach import FinalApproach
        f = FinalApproach(); f.turns = 5; f.advances = 7
        inputs = dict(point=[.25, .06, 0.], gyro_body=[0., 0., 0.], gyro_level=0., goal_heading=0., uncertainty=.02)
        for i in range(10):
            out, d = f.update(t=i/15, authorized=False, **inputs)
            self.assertEqual(out, [0., 0., 0.])
        self.assertEqual((f.turns, f.advances), (5, 7))
        g = FinalApproach(); g.turns = 5
        for i in range(10):
            out, d = g.update(t=i/15, authorized=True, **inputs)
        self.assertGreater(g.turns, 5)   # the same inputs would have consumed budget if not held
        text = (HERE/'task_navigation.py').read_text()
        self.assertIn("not getattr(self,'external_fine_step_active',False))", text)


FINAL_HELPER_SHA256 = 'eb55fe2ba8c61d7407c8046e131ca78411bd128251d210584c4bcdb95c32faf2'
FINAL_HELPER = Path('@COURSE@/configuration/core/heading_calibration/parameters/heading_feedback.py')


class V121Contract(unittest.TestCase):
    def test_helper_byte_matches_finished_bias_v121_helper(self):
        import hashlib
        data = (HERE/'heading_feedback.py').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), FINAL_HELPER_SHA256)
        if FINAL_HELPER.is_file():
            self.assertEqual(data, FINAL_HELPER.read_bytes())

    def test_handoff_and_helper_bound(self):
        self.assertIn('handoff=HeadHandoffV121(', SRC)
        import recovery_config as RC
        self.assertIn('heading_feedback.py', RC.BOUND_FILES)
        self.assertIn('microduck.football-retreat-recovery-approval.v1', RC.REFUSED_SCHEMAS)
        self.assertEqual(RC.STAGE_A_SPEC['seed'], 981131); self.assertEqual(RC.STAGE_A_SPEC['duration_s'], 1.6)

    def test_recovery_cannot_use_old_constant_motion(self):
        import retreat_recovery as rr
        src = (HERE/'retreat_recovery.py').read_text()
        self.assertNotIn('STOP_ALLOWANCE', src); self.assertNotIn('retreat_distance_reached', src)
        self.assertEqual((rr.STAGE_A_VX, rr.STAGE_A_DURATION_S, rr.STAGE_A_OFFSETS_RAD), (-.35, 1.6, (.10, .20)))
        self.assertEqual((rr.PLAN_DISTANCE_M, rr.MAX_BACKWARD_M, rr.MIN_BACKWARD_M, rr.ONSET_DEADLINE_S), (.28, .24, .10, 1.0))
        for bad in (dict(retreat_vx=-.40), dict(duration_s=1.8, move_steps=90), dict(heading_offset_rad=0.), dict(heading_offset_rad=.15)):
            p = rr.RecoveryParams(**{**rr.RecoveryParams.provisional().as_dict(), **bad})
            with self.assertRaises(ValueError):
                rr.RetreatRecovery(p)
        with self.assertRaises(ValueError):
            rr.RetreatRecovery(rr.RecoveryParams(**{**rr.RecoveryParams.provisional().as_dict(), 'helper_sha256': '0'*64}))

    def test_auditor_does_not_import_control_logic(self):
        tree = ast.parse((HERE/'audit_retreat_recovery.py').read_text())
        imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        loaded = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.Call) and 'spec_from_file_location' in ast.unparse(n.func)]
        for name in ('heading_feedback', 'retreat_recovery', 'recovery_handoff', 'recovery_fixture'):
            self.assertNotIn(name, imported)
            self.assertFalse(any(name in x for x in loaded), loaded)


if __name__ == '__main__':
    unittest.main()
