"""Regression from a real recorded trajectory; no new physics runs required."""
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_gaze_trace import audit as previous_audit
from audit_gaze_trace_v2 import audit

COHORT = Path('@COURSE@/outputs/migration-20261019/gait-v9-4600-navigation-gaze-dev-v1/gaze')


class PhysicalMetricNumerics(unittest.TestCase):
    def test_actual_reduction_failure_and_corrected_full_audit(self):
        with self.assertRaisesRegex(ValueError, 'camera_angular_rms'):
            previous_audit(COHORT/'report.json')
        result = audit(COHORT/'report.json')
        self.assertTrue(result['evidence_integrity_passed'])
        self.assertEqual(sum(g['trials'] for g in result['groups']), 20)
        self.assertEqual(sum(g['survivors'] for g in result['groups']), 19)

    def test_actual_angular_energy_against_accurate_scalar_sum(self):
        with np.load(COHORT/'raw-trace.npz') as data:
            omega = data['state'][1600:, 4, 37:40].astype(np.float64)
        expected = math.sqrt(math.fsum(float(x)*float(x) for x in omega.flat)/len(omega))
        recomputed = float(np.sqrt(np.mean(np.linalg.norm(omega, axis=1)**2)))
        self.assertAlmostEqual(expected, recomputed, places=13)

    def test_changed_metric_is_still_rejected(self):
        report = json.loads((COHORT/'report.json').read_text())
        report['episodes'][4]['camera_angular_rms_rad_s_after8s'] += .01
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            os.link(COHORT/'raw-trace.npz', temporary/'raw-trace.npz')
            (temporary/'report.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, 'camera_angular_rms'):
                audit(temporary/'report.json')


if __name__ == '__main__':
    unittest.main()
