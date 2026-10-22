import copy
import json
from pathlib import Path
import unittest
from geometry import refine_grounded_complete_ball, parent


def inputs():
    pose={'translation_m':[.01,-.02,.17],'timestamp_s':2.,'reference_frame':'body_capture_2.000000000'}
    original={'kind':'beta_YOLO_box_known_radius_v1','accepted':True,'observed_boundary_required':True,'missing_boundary_reconstructed':False,'supported_boundary_sectors':24,'required_boundary_sectors':21,'capture_timestamp_s':2.,'reference_frame':pose['reference_frame'],'radius_assumption_m':.035,'point_reference_m':[.217,.025,.0485],'estimated_range_uncertainty_m':.02}
    ground={'normal':[0.,0.,1.],'offset_m':0.,'reference_frame':pose['reference_frame']}
    floor={'accepted':True,'truth_used':False,'normal_body':[0.,0.,1.],'offset_m':0.}
    perception={'schema':'microduck.beta-geometry.v1','truth_used':False,'timestamp_s':2.,'ball':{'accepted':True,'geometry':original},'pose':pose,'ground':ground,'floor_observation':floor}
    tof={'timestamp_s':2.,'frame_dropped':False,'cells':[[{'status':'valid'}for _ in range(8)]for _ in range(8)]}
    return perception,tof


class GeometryTests(unittest.TestCase):
    def call(self,p,tof,now=2.):return refine_grounded_complete_ball(p,tof_frame=tof,now_s=now)
    def test_valid_entire_result_equal_v1_no_mutation_or_authority(self):
        p,t=inputs();before=copy.deepcopy((p,t))
        old=parent.refine_grounded_complete_ball(p['ball']['geometry'],p['pose'],p['ground'],p['floor_observation'],now_s=2.)
        new=self.call(p,t)
        self.assertEqual(new,old);self.assertEqual((p,t),before)
        self.assertEqual(new['estimated_range_uncertainty_m'],.02)
        self.assertFalse(new['kick_authorized']);self.assertFalse(new['uncertainty_calibrated'])

    def test_all_capture_and_now_timestamps_require_finite_nonnegative(self):
        for field in('perception','original','pose','tof','now'):
            for invalid in(float('nan'),float('inf'),-float('inf'),-1.,True):
                with self.subTest(field=field,value=invalid):
                    p,t=inputs();now=2.
                    if field=='perception':p['timestamp_s']=invalid
                    elif field=='original':p['ball']['geometry']['capture_timestamp_s']=invalid
                    elif field=='pose':p['pose']['timestamp_s']=invalid
                    elif field=='tof':t['timestamp_s']=invalid
                    else:now=invalid
                    self.assertFalse(self.call(p,t,now)['accepted'])

    def test_matching_negative_times_cannot_pass(self):
        p,t=inputs();p['timestamp_s']=p['pose']['timestamp_s']=p['ball']['geometry']['capture_timestamp_s']=t['timestamp_s']=-2.
        self.assertFalse(self.call(p,t,-2.)['accepted'])

    def test_future_stale_and_mixed_capture_rejected(self):
        for now in(1.999,2.201):
            p,t=inputs();self.assertFalse(self.call(p,t,now)['accepted'])
        for field in('perception','original','pose','tof'):
            p,t=inputs()
            if field=='perception':p['timestamp_s']=1.99
            elif field=='original':p['ball']['geometry']['capture_timestamp_s']=1.99
            elif field=='pose':p['pose']['timestamp_s']=1.99
            else:t['timestamp_s']=1.99
            self.assertFalse(self.call(p,t)['accepted'])

    def test_all_references_must_match(self):
        for field in('pose','ground','original'):
            p,t=inputs();target=p['ball']['geometry']if field=='original'else p[field]
            target['reference_frame']='different_capture'
            self.assertFalse(self.call(p,t)['accepted'])

    def test_plane_must_be_same_accepted_floor(self):
        for mode in('offset','normal','nofloor','truth','nan','missing'):
            p,t=inputs();f=p['floor_observation']
            if mode=='offset':f['offset_m']=.001
            elif mode=='normal':f['normal_body']=[0.,.01,1.]
            elif mode=='nofloor':f['accepted']=False
            elif mode=='truth':f['truth_used']=True
            elif mode=='nan':f['normal_body'][0]=float('nan')
            else:del p['floor_observation']
            self.assertFalse(self.call(p,t)['accepted'])

    def test_missing_or_dropped_actual_tof_rejected(self):
        for mode in('drop','shape','none'):
            p,t=inputs()
            if mode=='drop':t['frame_dropped']=True
            elif mode=='shape':t['cells']=t['cells'][:7]
            else:t=None
            self.assertFalse(self.call(p,t)['accepted'])

    def test_parent_crop_and_unaccepted_ball_never_reopened(self):
        for mode in('parent','boundary','reconstructed','sectors'):
            p,t=inputs();g=p['ball']['geometry']
            if mode=='parent':p['ball']['accepted']=False
            elif mode=='boundary':g['observed_boundary_required']=False
            elif mode=='reconstructed':g['missing_boundary_reconstructed']=True
            else:g['supported_boundary_sectors']=20
            self.assertFalse(self.call(p,t)['accepted'])

    def test_uncertainty_preserved_or_rejected_never_reduced(self):
        for value in(0.,.02,.1):
            p,t=inputs();p['ball']['geometry']['estimated_range_uncertainty_m']=value
            self.assertEqual(self.call(p,t)['estimated_range_uncertainty_m'],value)
        for value in(-.01,float('nan'),float('inf')):
            p,t=inputs();p['ball']['geometry']['estimated_range_uncertainty_m']=value
            self.assertFalse(self.call(p,t)['accepted'])


if __name__=='__main__':unittest.main(verbosity=2)
