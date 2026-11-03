"""Failures must remain in the denominator; cleanup faults must remain visible."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('session_evaluation', ROOT/'lab/evaluate_onboard_session.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class EvaluationTest(unittest.TestCase):
    def test_lifecycle_errors_keep_completed_rounds_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'session.json').write_text(json.dumps(dict(state='closed', episodes=[
                dict(index=i, technical_error=True, technical_error_detail='cleanup failed',
                     technical_error_phase='finalized') for i in range(1,5)])))
            for i, success in ((1, True), (2, False)):
                p = root/f'episode-{i:03d}'; p.mkdir()
                (p/'manifest.json').write_text(json.dumps(dict(status='development_episode_finished',
                    terminal_reason='finished', fixture=dict(seed=i))))
                (p/'result.json').write_text(json.dumps(dict(course_evaluation=dict(
                    passed=success, audit_exit_codes=[0]))))
            p = root/'episode-004'; p.mkdir()
            (p/'manifest.json').write_text(json.dumps(dict(status='development_episode_finished',
                terminal_reason='user_reset')))
            with patch.object(m.subprocess, 'call', side_effect=AssertionError('must reuse audits')), patch('builtins.print'):
                self.assertEqual(m.evaluate(root, root), 1)
                first = (root/'summary.json').read_bytes()
                self.assertEqual(m.evaluate(root, root), 1)
                self.assertEqual(first, (root/'summary.json').read_bytes())
            report = json.loads(first)
            self.assertEqual((report['completed'], report['successes'], report['failures']), (2,1,1))
            self.assertEqual(report['technical_errors'], 4)
            self.assertEqual(report['interrupted'], 0)
            self.assertTrue(report['episodes'][0]['success'])

    def test_mixed_outcomes_and_repeated_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'session.json').write_text(json.dumps(dict(state='closed',episodes=[dict(index=i) for i in range(1,6)])))
            for i,success in ((1,True),(2,False)):
                p=root/f'episode-{i:03d}';p.mkdir()
                (p/'manifest.json').write_text(json.dumps(dict(status='development_episode_finished',terminal_reason='development_kick_window_complete',fixture=dict(seed=i))))
                (p/'result.json').write_text(json.dumps(dict(course_evaluation=dict(passed=success,audit_exit_codes=[0 if success else 1]))))
            for i,failed in ((3,False),(4,True)):
                p=root/f'episode-{i:03d}';p.mkdir()
                (p/'manifest.json').write_text(json.dumps(dict(status='failed' if failed else 'development_episode_finished',terminal_reason='user_reset',media_error='archive drain failed' if failed else None)))
            with patch.object(m.subprocess,'call',side_effect=AssertionError('must reuse completed audits')),patch('builtins.print'):
                self.assertEqual(m.evaluate(root,root),1)
                first=(root/'summary.json').read_bytes()
                self.assertEqual(m.evaluate(root,root),1)
                self.assertEqual(first,(root/'summary.json').read_bytes())
            result=json.loads(first)
            self.assertEqual((result['completed'],result['successes'],result['failures']),(2,1,1))
            self.assertEqual(result['success_rate'],.5)
            self.assertEqual(result['technical_errors'],2)
            self.assertEqual(result['interrupted'],1)
            self.assertEqual(result['audit_errors'],1)

    def test_corrupt_round_does_not_hide_later_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'session.json').write_text(json.dumps(dict(state='closed',episodes=[dict(index=i) for i in range(1,4)])))
            for i in range(1,4):
                p=root/f'episode-{i:03d}';p.mkdir()
                (p/'manifest.json').write_text('{broken' if i==1 else json.dumps(dict(status='development_episode_finished',terminal_reason='finished',fixture=dict(seed=i))))
                (p/'result.json').write_text('{broken' if i==2 else json.dumps(dict(course_evaluation=dict(passed=True,audit_exit_codes=[0]))))
            with patch('builtins.print'):
                self.assertEqual(m.evaluate(root,root),1)
            report=json.loads((root/'summary.json').read_text())
            self.assertEqual(report['completed'],2)
            self.assertEqual(report['successes'],1)
            self.assertEqual(report['technical_errors'],1)
            self.assertEqual(report['audit_errors'],1)
            self.assertEqual(report['episodes'][-1]['success'],True)

    def test_running_session_is_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'session.json').write_text(json.dumps(dict(state='waiting_for_r',episodes=[])))
            with self.assertRaisesRegex(RuntimeError,'Close'):
                m.evaluate(root,root)
            self.assertFalse((root/'summary.json').exists())

if __name__=='__main__':unittest.main()
