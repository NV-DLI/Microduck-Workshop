import copy
import math
import unittest
import numpy as np
from geometry import refine_grounded_complete_ball


class GroundedSphereContract(unittest.TestCase):
    def inputs(self):
        origin = np.array([.01, -.02, .17])
        target = np.array([.24, .03, .035])
        # Deliberately wrong box extent; its centre ray stays correct.
        point = origin + .9 * (target - origin)
        original = dict(kind='beta_YOLO_box_known_radius_v1', accepted=True,
            observed_boundary_required=True, missing_boundary_reconstructed=False,
            supported_boundary_sectors=24, required_boundary_sectors=21,
            capture_timestamp_s=2., reference_frame='body_capture_2', radius_assumption_m=.035,
            point_reference_m=point.tolist(), estimated_range_uncertainty_m=.02)
        pose = dict(translation_m=origin.tolist(), timestamp_s=2., reference_frame='body_capture_2')
        ground = dict(normal=[0., 0., 1.], offset_m=0., reference_frame='body_capture_2')
        return original, pose, ground, {'accepted': True}, target

    def test_exact_plane_geometry_and_no_mutation_or_new_authority(self):
        original, pose, ground, floor, target = self.inputs()
        before = copy.deepcopy(original)
        actual = refine_grounded_complete_ball(original, pose, ground, floor, now_s=2.)
        np.testing.assert_allclose(actual['point_reference_m'], target, atol=1e-14)
        self.assertEqual(original, before)
        self.assertFalse(actual['kick_authorized'])
        self.assertFalse(actual['ground_contact_verified'])
        self.assertEqual(actual['estimated_range_uncertainty_m'], .02)

    def test_rotated_translated_reference_frame(self):
        original, pose, ground, floor, target = self.inputs()
        c, s = math.cos(.4), math.sin(.4)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        translation = np.array([.07, -.03, .11])
        for obj, name in ((original, 'point_reference_m'), (pose, 'translation_m')):
            obj[name] = (R @ obj[name] + translation).tolist()
        normal = R @ np.array(ground['normal'])
        ground.update(normal=normal.tolist(), offset_m=float(normal @ translation))
        actual = refine_grounded_complete_ball(original, pose, ground, floor, now_s=2.)
        np.testing.assert_allclose(actual['point_reference_m'], R @ target + translation, atol=1e-14)

    def test_cropped_missing_floor_mixed_time_and_bad_geometry_rejected(self):
        for change in ('rejected', 'partial', 'missingfloor', 'time', 'frame', 'radius', 'normal', 'nan'):
            original, pose, ground, floor, _ = self.inputs()
            if change == 'rejected': original['accepted'] = False
            if change == 'partial': original['supported_boundary_sectors'] = 20
            if change == 'missingfloor': floor['accepted'] = False
            if change == 'time': pose['timestamp_s'] = 1.9
            if change == 'frame': ground['reference_frame'] = 'other'
            if change == 'radius': original['radius_assumption_m'] = .04
            if change == 'normal': ground['normal'] = [0, 0, 2]
            if change == 'nan': original['point_reference_m'][0] = float('nan')
            self.assertFalse(refine_grounded_complete_ball(original, pose, ground, floor, now_s=2.)['accepted'])

    def test_stale_or_future_now_rejected(self):
        original, pose, ground, floor, _ = self.inputs()
        for now in (1.99, 2.21, float('nan'), float('inf')):
            self.assertFalse(refine_grounded_complete_ball(original, pose, ground, floor, now_s=now)['accepted'])


if __name__ == '__main__':
    unittest.main()
