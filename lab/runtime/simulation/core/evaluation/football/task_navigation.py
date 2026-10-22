"""Task navigation: Microduck simulation and control."""
import copy,importlib.util,math
from pathlib import Path
import numpy as np
from goal_reacquisition_native import GoalReacquisitionNavigator
from final_approach import FinalApproach
from perception import level_rotation
from visual_localization import rotate,wrap
D=Path(__file__).resolve().parent;S=D.parents[1]/'sensors'
spec=importlib.util.spec_from_file_location('football_grounded_v2',S/'processing/ground_estimator/geometry.py')
geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)


class TaskNavigator(GoalReacquisitionNavigator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        estimate=self.vision.estimate
        def grounded(*args,**kwargs):
            g=estimate(*args,**kwargs)
            parameters=geometry.refine_grounded_complete_ball(g,tof_frame=kwargs.get('tof_frame'),now_s=kwargs['now_s'])
            g=copy.deepcopy(g)
            g['grounded_refinement']={k:v for k,v in parameters.items()if k not in ('point_reference_m',)}
            if g.get('ball') and g['ball'].get('accepted'):
                g['ball']['bbox_geometry_diagnostic']=g['ball']['geometry']
                if parameters['accepted']:g['ball']['geometry']=parameters
                else:g['ball'].update(accepted=False,reason='grounded_parameters_rejected')
            return g
        self.vision.estimate=grounded

    def reset(self):
        super().reset();self.final_approach=FinalApproach();self.final_entry=False
        from coarse_track import CoarseObserver
        self.coarse_observer=CoarseObserver()

    def _tof_sample(self,frame,o,R,height,observed_ball_level_xyz=None):
        sample,hazard=super()._tof_sample(frame,o,R,height,observed_ball_level_xyz)
        from zone_floor_projection import project_floor_zones
        g=self.last_geometry
        accepted=bool(g and g.get('timestamp_s')==o['timestamp_s'] and g.get('floor_observation',{}).get('accepted'))
        sample,diagnostic=project_floor_zones(sample,frame,floor_accepted=accepted)
        self.last_floor_projection=diagnostic
        # Immediate hazard/TooClose guards are inherited unchanged. Only the
        # navigation snapshot receives the supported floor-ray correction.
        return sample,hazard

    def step(self,rgb,proprio,tof_frame,**kwargs):
        out=super().step(rgb,proprio,tof_frame,**kwargs)
        out['public_floor_zone_projection']=getattr(self,'last_floor_projection',None)
        g=self.last_geometry;t=proprio['timestamp_s'];ball=g.get('ball')if g else None
        fresh=bool(g and g['timestamp_s']==t and ball and ball.get('accepted') and g['floor_observation']['accepted'])
        if not fresh:
            self.coarse_observer.clear()
            if self.final_entry:self.final_approach.invalidate(t,'fresh_complete_grounded_ball_required')
            return out
        R=level_rotation((-np.asarray(g['ground']['normal'])).tolist());point=R@np.asarray(ball['geometry']['point_reference_m'])
        uncertainty=float(ball['geometry']['estimated_range_uncertainty_m'])
        coarse_pose_valid=self.pose is not None and np.isfinite(self.pose).all() and 0<=t-self.goal_time<=8.
        coarse_heading=math.nan
        if coarse_pose_valid:
            rel=np.asarray(rotate(-self.pose[2],self.goal_map-self.pose[:2]))-point[:2];coarse_heading=math.atan2(rel[1],rel[0])
        coarse_geometry=(coarse_pose_valid and .38<point[0]<=.85 and np.linalg.norm(point[:2])<=.9
                         and abs(point[1])<=.25 and abs(coarse_heading)<=.35
                         and abs(-math.sin(coarse_heading)*point[0]+math.cos(coarse_heading)*point[1]+.028)<=.18)
        if (self.config.layout=='empty' and coarse_geometry
                and not getattr(self,'external_fine_step_active',False)):
            from coarse_track import path_authority
            valid_pose=self.pose is not None and np.isfinite(self.pose).all() and 0<=t-self.goal_time<=8.
            admitted=bool(out.get('motion_authorization',{}).get('allowed') and valid_pose and not self.refresh_too_close and out['state']!='REFRESH_GOAL')
            path=path_authority(self,tof_frame,proprio,R,g['height']['conditional_root_height_m'],point) if admitted else {'allowed':False,'capture_s':t,'layout':'empty','reason':'upstream_navigation_hold'}
            heading=math.nan
            if valid_pose:
                relative_goal=np.asarray(rotate(-self.pose[2],self.goal_map-self.pose[:2]));shot=relative_goal-point[:2];heading=math.atan2(shot[1],shot[0])
            diag=self.coarse_observer.update(t,point,proprio['gyro_body_rad_s'],heading,admitted and path['allowed'])
            out.update(velocity_body_xyw=[0.,0.,0.],final_approach=diag,coarse_track_path=path,kick_request=False)
            return out
        self.coarse_observer.clear()
        if np.linalg.norm(point[:2])>.4 and not self.final_entry:return out
        self.final_entry=True
        allowed=bool(out.get('motion_authorization',{}).get('allowed'))
        allowed|=out.get('reason') in ('zero_command_observe_real_settling','visual_and_IMU_settled_parameters')
        valid_pose=self.pose is not None and np.isfinite(self.pose).all() and 0<=t-self.goal_time<=8.
        allowed=bool(allowed and valid_pose and not self.refresh_too_close and out['state']!='REFRESH_GOAL' and not getattr(self,'external_fine_step_active',False))
        heading=math.nan
        if valid_pose:
            relative_goal=np.asarray(rotate(-self.pose[2],self.goal_map-self.pose[:2]))
            shot=relative_goal-point[:2];heading=math.atan2(shot[1],shot[0])
            try:_,hazard=self._tof_sample(tof_frame,proprio,R,g['height']['conditional_root_height_m'],point)
            except (ValueError,KeyError):hazard=True
            allowed=allowed and not hazard
        actual,diag=self.final_approach.update(t=t,point=point,gyro_body=proprio['gyro_body_rad_s'],
            gyro_level=float((R@np.asarray(proprio['gyro_body_rad_s']))[2]),goal_heading=heading,
            authorized=allowed,uncertainty=uncertainty)
        out.update(velocity_body_xyw=actual,final_approach=diag,kick_request=False)
        if allowed:out['state']='FINAL_ALIGN'
        out['final_motion_authorization']={'allowed':allowed,'fresh_complete_grounded':fresh,'goal_memory_age_s':t-self.goal_time,
            'grounded_source':'same result+same ToF capture guarded v2','truth_used':False}
        return out
