"""CPU source-selection tests; fixtures are not student capture evidence."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import train_student_capture as entry


class OwnCaptureSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.mock = patch.object(entry, 'STUDENT', self.root)
        self.mock.start()

    def tearDown(self):
        self.mock.stop()
        self.temp.cleanup()

    def fixture(self, name='example', finished=1):
        folder = self.root / 'jobs' / name
        dataset = folder / 'captured-dataset'
        dataset.mkdir(parents=True)
        (dataset / 'manifest.json').write_text(json.dumps({'development_only': False, 'images': [{}]*384}))
        owner = {'action': 'capture-sensors', 'state': 'completed', 'exitCode': 0, 'processExitCode': 0,
                 'runtimeId': entry.RUNTIME_ID, 'profileId': entry.PROFILE_ID, 'finishedAt': finished}
        report = {'schema': 'microduck.student-new-dataset.v1', 'status': 'passed',
                  'runtime_id': entry.RUNTIME_ID, 'hardware_profile_id': entry.PROFILE_ID,
                  'student_capture_performed': True, 'sources_unchanged': True, 'dataset': str(dataset),
                  'dataset_manifest_sha256': entry.sha(dataset/'manifest.json')}
        (folder / 'supervisor.json').write_text(json.dumps(owner))
        (folder / 'course-result.json').write_text(json.dumps(report))
        return folder

    def edit(self, path, **changes):
        data = json.loads(path.read_text())
        data.update(changes)
        path.write_text(json.dumps(data))

    def test_missing_capture_has_no_teacher_fallback(self):
        with self.assertRaises(ValueError): entry.completed_capture()

    def test_latest_completed_capture_selected(self):
        self.fixture('earlier',1)
        self.fixture('latest',2)
        self.assertEqual(entry.completed_capture()['job_id'],'latest')
        self.assertEqual(entry.completed_capture('earlier')['job_id'],'earlier')

    def test_running_or_wrong_action_rejected(self):
        folder = self.fixture()
        self.edit(folder/'supervisor.json', state='running')
        with self.assertRaises(ValueError): entry.completed_capture()
        self.edit(folder/'supervisor.json', state='completed', action='detector-train')
        with self.assertRaises(ValueError): entry.completed_capture()

    def test_failed_or_incomplete_capture_rejected(self):
        folder = self.fixture()
        self.edit(folder/'supervisor.json', processExitCode=1)
        with self.assertRaises(ValueError): entry.completed_capture()
        self.edit(folder/'supervisor.json', processExitCode=0)
        self.edit(folder/'course-result.json', student_capture_performed=False)
        with self.assertRaises(ValueError): entry.completed_capture()

    def test_changed_or_other_release_data_rejected(self):
        folder = self.fixture()
        self.edit(folder/'captured-dataset/manifest.json', modified=True)
        with self.assertRaises(ValueError): entry.completed_capture()
        self.edit(folder/'course-result.json', dataset_manifest_sha256=entry.sha(folder/'captured-dataset/manifest.json'))
        self.edit(folder/'supervisor.json', runtimeId='historical-release')
        with self.assertRaises(ValueError): entry.completed_capture()

    def test_report_cannot_redirect_to_other_dataset(self):
        folder = self.fixture()
        other = self.fixture('other')
        self.edit(folder/'course-result.json', dataset=str(other/'captured-dataset'))
        with self.assertRaises(ValueError): entry.completed_capture('example')

    def test_path_traversal_and_job_symlink_rejected(self):
        folder = self.fixture()
        (folder.parent/'linked').symlink_to(folder, target_is_directory=True)
        for name in ('../example','linked'):
            with self.assertRaises(ValueError): entry.completed_capture(name)


if __name__ == '__main__': unittest.main()
