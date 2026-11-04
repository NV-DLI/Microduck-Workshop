"""Direct geometry keeps sensor freshness, obstacles and map bounds as vetoes."""
from pathlib import Path
import copy,json,math,sys,tempfile,unittest
R=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(R/'lab'),str(R/'lab/onboard'),str(R/'lab/runtime/simulation/core/football')]
from direct_path import DirectPath
from path_request import make_request

class DirectPathTest(unittest.TestCase):
    def inputs(self):
        sample=dict(capture_s=1.,fk_capture_s=1.,provenance='simulation_approximation',
            ranges_m=[None]*64,status=[None]*64,unit_rays_sensor=[[1.,0.,0.]]*64,
            sensor_translation_odom=[-.55,.12,.15],sensor_quaternion_odom_xyzw=[0,0,0,1],
            isaac_valid_mask=[False]*64,near_m=.1,far_m=4.)
        return make_request(time_s=1.,pose_map_xyyaw=[-.55,.12,0.],velocity_body_xyw=[0,0,0],
            goal_map_xy=[-.25,.12],tof_sample=sample,layout='empty',image_sha256='0'*64,
            localization_capture_s=1.)

    def query(self,case,provenance):
        with tempfile.TemporaryDirectory() as tmp:
            planner=DirectPath(Path(tmp)/'path')
            try:return planner.query(case,provenance)
            finally:planner.close()

    def test_shortest_line_reaches_target_and_has_no_service(self):
        case,prov=self.inputs();r=self.query(case,prov)
        self.assertTrue(r['envelope']['accepted'])
        self.assertFalse(r['external_service_started'])
        self.assertEqual(r['path_xy'][0],[-.55,.12]);self.assertEqual(r['path_xy'][-1],[-.25,.12])
        self.assertAlmostEqual(r['path_length_m'],.30)
        for a,b in zip(r['path_xy'],r['path_xy'][1:]):
            self.assertLessEqual(math.dist(a,b),.02+1e-12);self.assertAlmostEqual(b[1],.12)

    def test_current_tof_obstacle_is_not_erased_by_empty_field_prior(self):
        case,prov=self.inputs();sample=case['tof_samples'][0]
        sample['ranges_m'][0]=.2;sample['isaac_valid_mask'][0]=True
        r=self.query(case,prov)
        self.assertFalse(r['envelope']['accepted']);self.assertEqual(r['path_xy'],[])

    def test_stale_odom_tof_and_mismatched_exposure_are_rejected(self):
        for key in ['odom','tof','fk']:
            case,prov=self.inputs()
            if key=='odom':case['odom']['capture_s']=.89
            if key=='tof':case['tof_samples'][0]['capture_s']=.79
            if key=='fk':case['tof_samples'][0]['fk_capture_s']=.95
            with self.subTest(key),self.assertRaises(ValueError):self.query(case,prov)

    def test_unknown_space_and_wrong_layout_do_not_receive_paths(self):
        case,prov=self.inputs();case['goal_body_xy']=[2.,0.]
        r=self.query(case,prov);self.assertFalse(r['envelope']['accepted']);self.assertEqual(r['path_xy'],[])
        case,prov=self.inputs();prov['layout']='one-obstacle'
        with self.assertRaises(ValueError):self.query(case,prov)

    def test_hidden_or_command_integrated_odometry_is_rejected(self):
        for origin in ['simulation_truth','command_integrated']:
            case,prov=self.inputs();prov['odometry_origin']=origin
            with self.subTest(origin),self.assertRaises(ValueError):self.query(case,prov)

if __name__=='__main__':unittest.main()
