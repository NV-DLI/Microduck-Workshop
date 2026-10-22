import unittest
from unittest.mock import patch
import numpy as np
from command_navigation_native import NativeCommandNavigator, NativePhaseCommands
from guarded_navigation_feedback import UPSTREAM_HOLDS
from execute_observed_ball_turn import ObservedTurnNavigator


class ActualChain(unittest.TestCase):
    def nav(self, state):
        nav=NativeCommandNavigator(None); nav.state=state
        nav.gaze.current[:]=[.2,.4]; nav.gaze.last_time=.9
        nav.last_geometry={'timestamp_s':1.,'floor_observation':{'accepted':True},
                           'ground':{'normal':[0,0,1]},
                           'ball':{'accepted':True,'geometry':{'point_reference_m':[.6,.02,-.1],
                                   'estimated_range_uncertainty_m':.01}}}
        nav.ball_attention={'gaze_body_yaw_pitch_rad':[.2,.4],'capture_timestamp_s':1.}
        nav.yaw_feedback.integral=.12
        nav.native_commands.phase='ADVANCE'; nav.native_commands.started=.7
        nav.native_commands.last_t=.93
        return nav

    def apply(self, nav, reason, command=(.06,0,.2)):
        def decision(instance,*args,**kwargs):
            return instance._output(1.,command,reason)
        with patch.object(ObservedTurnNavigator,'step',decision):
            return nav.step(None,{'timestamp_s':1.,'gyro_body_rad_s':[0,0,-.5]},None)

    def test_every_guard_preempts_active_phase_integral_and_attention(self):
        reasons=UPSTREAM_HOLDS|{'Nav2_failed:timeout','Nav2_path_not_admitted','detector_unavailable'}
        for reason in reasons:
            for state in ('SEARCH_BALL','REFRESH_GOAL','VISUAL_ALIGN','IMU_WARMUP','RELEASE','RESET','TIMEOUT','SETTLE'):
                with self.subTest(reason=reason,state=state):
                    r=self.apply(self.nav(state),reason)
                    self.assertEqual(r['velocity_body_xyw'],[0.,0.,0.]);self.assertFalse(r['kick_request'])

    def test_too_close_and_missing_ball_or_floor_stop_actual_chain(self):
        for fault in ['too_close','ball','floor']:
            n=self.nav('NAV2_APPROACH')
            if fault=='too_close':n.refresh_too_close=True
            elif fault=='ball':n.last_geometry['ball']['accepted']=False
            else:n.last_geometry['floor_observation']['accepted']=False
            self.assertEqual(self.apply(n,'actual_Navfn_path')['velocity_body_xyw'],[0.,0.,0.])

    def test_old_yaw_integral_cannot_starve_aligned_forward_phase(self):
        n=self.nav('NAV2_APPROACH')
        self.assertEqual(self.apply(n,'actual_Navfn_path')['velocity_body_xyw'],[.3,0,0])
        self.assertEqual(n.yaw_feedback.integral,0.)

    def test_zero_request_disarms_even_current_forward(self):
        n=self.nav('NAV2_APPROACH')
        self.assertEqual(self.apply(n,'actual_Navfn_path',(0,0,0))['velocity_body_xyw'],[0.,0.,0.])

    def call(self,a,t,**kw):
        args=dict(timestamp_s=t,authorized=True,requested=[.06,0,.1],point_level=[.6,0,0],
                  range_uncertainty_m=.01,gyro_level_z=0.)
        args.update(kw)
        args.setdefault('gyro_body_rad_s',[0.,0.,args['gyro_level_z']])
        return a.update(**args)

    def ready(self,a,point=[.6,0,0]):
        for t in np.arange(7)/15:
            out=self.call(a,float(t),point_level=point)
        return out

    def test_observation_evidence_required_before_motion(self):
        a=NativePhaseCommands()
        for t in np.arange(6)/15:self.assertEqual(self.call(a,float(t))[0],[0.,0.,0.])
        self.assertEqual(self.call(a,.4)[0],[.3,0,0])

    def test_forward_duration_not_extended_by_refresh(self):
        a=NativePhaseCommands();self.ready(a)
        for t in [.47,.54,.61,.68,.75,.82,.89,.96]:self.assertEqual(self.call(a,t)[0],[.3,0,0])
        self.assertEqual(self.call(a,1.03)[0],[0.,0.,0.])

    def test_turn_finishes_on_observed_sign_crossing(self):
        a=NativePhaseCommands();self.assertEqual(self.ready(a,[.6,.2,0])[0],[0,0,.6])
        self.assertEqual(self.call(a,.47,point_level=[.6,-.01,0])[0],[0,0,0])

    def test_near_uncertainty_envelope_remains_held(self):
        a=NativePhaseCommands();self.ready(a)
        self.assertEqual(self.call(a,.47,point_level=[.35,0,0],range_uncertainty_m=.02)[0],[0,0,0])

    def test_nonquiet_stale_nonfinite_and_guard_inputs_hold(self):
        a=NativePhaseCommands()
        for t in np.arange(25)/15:
            self.assertEqual(self.call(a,float(t),gyro_level_z=.5)[0],[0,0,0])
        for kw in [{'timestamp_s':float('nan')},{'point_level':[float('nan'),0,0]},
                   {'range_uncertainty_m':float('inf')},{'authorized':False}]:
            a=NativePhaseCommands();self.ready(a)
            self.assertEqual(self.call(a,**({'t':.47}|kw))[0],[0,0,0])

    def test_near_goal_heading_owns_body_not_ball_bearing(self):
        a=NativePhaseCommands()
        for t in np.arange(7)/15:
            out=self.call(a,float(t),point_level=[.30,.08,0],near_goal_heading=0.,
                          near_position_error=[.20,.06])
        self.assertEqual(out[0],[.2,0.,0.])
        self.assertEqual(out[1]['body_heading_owner'],'public_preparation_waypoint')

    def test_near_goal_turn_and_crossing_are_public_error_controlled(self):
        a=NativePhaseCommands()
        for t in np.arange(7)/15:
            out=self.call(a,float(t),point_level=[.3,0,0],near_goal_heading=-.2,
                          near_position_error=[.2,0])
        self.assertEqual(out[0],[0.,0.,-.8])
        self.assertEqual(self.call(a,.47,point_level=[.3,0,0],near_goal_heading=.01,
                                  near_position_error=[.2,0])[0],[0,0,0])

    def test_lateral_phase_cannot_continue_after_goal_context_missing(self):
        a=NativePhaseCommands()
        for t in np.arange(7)/15:
            self.call(a,float(t),point_level=[.3,.08,0],near_goal_heading=0.,near_position_error=[.2,.06])
        self.assertEqual(self.call(a,.47,point_level=[.3,.08,0])[0],[0,0,0])

    def test_new_near_stop_envelope_preempts_active_forward(self):
        a=NativePhaseCommands();self.ready(a)
        self.assertEqual(self.call(a,.47,point_level=[.14,0,0],range_uncertainty_m=.02,
                                  near_goal_heading=0.,near_position_error=[.1,0])[0],[0,0,0])

    def test_new_sixteen_cm_development_envelope_only_changes_distance_gate(self):
        a=NativePhaseCommands();self.ready(a)
        self.assertEqual(self.call(a,.47,point_level=[.19,0,0],range_uncertainty_m=.02,
                                  near_goal_heading=0.,near_position_error=[.08,0])[0],[.3,0,0])
        self.assertEqual(self.call(a,.54,point_level=[.18,0,0],range_uncertainty_m=.02,
                                  near_goal_heading=0.,near_position_error=[.05,0])[0],[0,0,0])

    def test_signed_oscillation_cannot_pass_quiet_by_cancellation(self):
        a=NativePhaseCommands()
        for i,t in enumerate(np.arange(7)/15):
            out=self.call(a,float(t),gyro_level_z=.6 if i%2==0 else -.6)
        self.assertLess(abs(out[1]['gyro_window_mean_rad_s']),.12)
        self.assertGreater(out[1]['gyro_window_3D_RMS_rad_s'],.59)
        self.assertEqual(out[0],[0,0,0])

    def test_roll_pitch_motion_cannot_hide_behind_zero_yaw(self):
        a=NativePhaseCommands()
        for t in np.arange(7)/15:out=self.call(a,float(t),gyro_body_rad_s=[.3,0,0])
        self.assertEqual(out[0],[0,0,0])
        self.assertFalse(out[1]['measured_quiet'])

    def test_ball_roundtrip_cannot_pass_by_identical_endpoints(self):
        a=NativePhaseCommands()
        for i,t in enumerate(np.arange(7)/15):
            out=self.call(a,float(t),point_level=[.6+(.025 if i==3 else 0),0,0])
        self.assertEqual(out[1]['ball_relative_change_m'],0)
        self.assertGreater(out[1]['ball_window_XY_diameter_m'],.024)
        self.assertEqual(out[0],[0,0,0])

    def test_missing_nonfinite_gyro_vector_is_not_quiet(self):
        for v in [None,[0,0],[float('nan'),0,0]]:
            a=NativePhaseCommands();self.ready(a)
            self.assertEqual(self.call(a,.47,gyro_body_rad_s=v)[0],[0,0,0])

    def near_nav(self):
        n=self.nav('NAV2_APPROACH')
        n.pose=np.array([-.345,0.,0.]);n.ball=np.array([0.,0.]);n.goal_map=np.array([.93,0.])
        n.goal_time=1.;n.ball_time=1.;n.ball_attention=None
        n.last_geometry['ball']['geometry']['point_reference_m']=[.345,0.,-.1]
        n.last_geometry['ball']['geometry']['estimated_range_uncertainty_m']=.02
        n.last_geometry['height']={'conditional_root_height_m':.1}
        n._tof_sample=lambda *args:(None,False)
        return n

    def test_actual_navfn_shared_near_context_crosses_old_boundary_gap(self):
        n=self.near_nav();out=self.apply(n,'actual_Navfn_path')
        self.assertIn('heading_error_rad',out['near_ball_phase'])
        self.assertEqual(out['native_command_adapter']['body_heading_owner'],'public_preparation_waypoint')
        self.assertEqual(out['velocity_body_xyw'],[.3,0,0])

    def test_goal_refresh_shared_context_reaches_actual_chain(self):
        n=self.near_nav();out=self.apply(n,'visual_localization_refresh')
        self.assertIn('relative_position_error_m',out['near_ball_phase'])
        self.assertTrue(out['motion_authorization']['allowed'])

    def test_near_hard_guards_still_override_phase_and_context(self):
        for why in UPSTREAM_HOLDS | {'Nav2_failed:timeout','detector_unavailable'}:
            with self.subTest(why=why):
                n=self.near_nav();out=self.apply(n,why)
                self.assertEqual(out['velocity_body_xyw'],[0,0,0])
                self.assertFalse(out['motion_authorization']['allowed'])

    def test_near_explicit_zero_path_request_cannot_become_motion(self):
        n=self.near_nav();out=self.apply(n,'actual_Navfn_path',(0,0,0))
        self.assertEqual(out['velocity_body_xyw'],[0,0,0])

    def test_near_expired_goal_crop_floor_and_tof_cannot_move(self):
        for fault in ['expired','crop','floor','hazard','too_close']:
            with self.subTest(fault=fault):
                n=self.near_nav()
                if fault=='expired':n.goal_time=-8.
                elif fault=='crop':n.last_geometry['ball']['accepted']=False
                elif fault=='floor':n.last_geometry['floor_observation']['accepted']=False
                elif fault=='hazard':n._tof_sample=lambda *args:(None,True)
                else:n.refresh_too_close=True
                out=self.apply(n,'actual_Navfn_path')
                self.assertEqual(out['velocity_body_xyw'],[0,0,0])
                self.assertFalse(out['kick_request'])

    def test_near_waypoint_forward_cap_and_heading_guard(self):
        a=NativePhaseCommands()
        for t in np.arange(7)/15:
            out=self.call(a,float(t),point_level=[.3,.08,0],near_goal_heading=0.,near_position_error=[.2,.06])
        self.assertEqual(out[0],[.2,0.,0.])
        self.assertEqual(self.call(a,.47,point_level=[.3,.08,0],near_goal_heading=.16,near_position_error=[.2,.06])[0],[0,0,0])
        a=NativePhaseCommands()
        for t in np.arange(12)/15:
            out=self.call(a,float(t),point_level=[.3,.08,0],near_goal_heading=0.,near_position_error=[.2,.06])
        self.assertEqual(out[0],[.2,0.,0.])
        self.assertEqual(self.call(a,.8,point_level=[.3,.08,0],near_goal_heading=0.,near_position_error=[.2,.06])[0],[0,0,0])

    def test_forward_amplitude_is_latched_across_near_boundary(self):
        a=NativePhaseCommands()
        for t in np.arange(7)/15:
            out=self.call(a,float(t),point_level=[.3,0,0],near_goal_heading=0.,near_position_error=[.2,0])
        self.assertEqual(out[0],[.2,0.,0.])
        self.assertEqual(self.call(a,.47,point_level=[.4,0,0])[0],[.2,0.,0.])
        self.assertEqual(self.call(a,.8,point_level=[.4,0,0])[0],[0.,0.,0.])
        a=NativePhaseCommands();self.ready(a)
        self.assertEqual(self.call(a,.47,point_level=[.3,0,0],near_goal_heading=0.,near_position_error=[.2,0])[0],[.3,0.,0.])
        self.assertEqual(self.call(a,.54,point_level=[.3,0,0],near_goal_heading=0.,near_position_error=[.2,0],authorized=False)[0],[0.,0.,0.])

    def test_actual_chain_heading_is_toward_public_preparation_point(self):
        n=self.near_nav();n.native_commands.phase='OBSERVE';out=self.apply(n,'actual_Navfn_path')
        near=out['near_ball_phase']
        self.assertAlmostEqual(near['heading_error_rad'],0.)
        self.assertAlmostEqual(near['translation_heading_error_rad'],float(np.arctan2(.035,.255)))
        self.assertFalse(near['final_goal_heading_alignment_armed'])

    def test_near_turn_cap_is_latched_and_guards_still_preempt(self):
        a=NativePhaseCommands()
        for t in np.arange(7)/15:
            self.call(a,float(t),point_level=[.3,0,0],near_goal_heading=.3,near_position_error=[.2,.06])
        for t in np.arange(7,42)/15:
            out=self.call(a,float(t),point_level=[.3,0,0],near_goal_heading=.2,near_position_error=[.2,.04])
            self.assertEqual(out[0],[0,0,.8])
        self.assertEqual(self.call(a,2.81,point_level=[.3,0,0],near_goal_heading=.2,near_position_error=[.2,.04])[0],[0,0,0])
        a=NativePhaseCommands();self.ready(a,[.6,.2,0])
        for t in np.arange(7,24)/15:self.call(a,float(t),point_level=[.6,.2,0])
        self.assertEqual(self.call(a,1.61,point_level=[.6,.2,0])[0],[0,0,0])

if __name__=='__main__':unittest.main()
