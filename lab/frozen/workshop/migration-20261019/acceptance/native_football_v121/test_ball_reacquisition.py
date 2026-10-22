import unittest,json,math
from pathlib import Path
import numpy as np
from PIL import Image
from ball_reacquisition import BallReacquisitionNavigator,BallSearch
from controller import ControllerConfig
from visual_localization import rotate
R=Path('@COURSE@/outputs/migration-20261019/football-v47-full-task180s')
class Recovery(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.c=[json.loads(x)for x in (R/'control.jsonl').read_text().splitlines()];cls.s={x['timestamp_s']:x for x in map(json.loads,(R/'sensor-inputs.jsonl').read_text().splitlines())};cls.d={x['timestamp_s']:x for x in map(json.loads,(R/'detections.jsonl').read_text().splitlines())};cls.metric=max((x for x in cls.c if x.get('public_metric_ball')),key=lambda x:x['timestamp_s']);m=cls.metric;cls.ball_map=np.array(m['estimated_pose_map_xyyaw'][:2])+rotate(m['estimated_pose_map_xyyaw'][2],m['public_metric_ball']['point_level_m'][:2])
 def nav(self,t):
  row=min(self.c,key=lambda x:abs(x['timestamp_s']-t));t=row['timestamp_s'];n=BallReacquisitionNavigator(None,config=ControllerConfig(ground_mode='tof-virtual-floor'));n.pose=np.array(row['estimated_pose_map_xyyaw']);n.ball=self.ball_map.copy();n.ball_time=self.metric['timestamp_s'];n.first_time=2.34;n.last_time=t-.065;n.goal_time=t-row.get('near_ball_phase',{}).get('goal_fix_age_s',1.);n.last_observed_ball_gaze=[-.4,1.103413924839973];n.gaze.current[:]=n.last_observed_ball_gaze;n.gaze.timestamp=t-.065;n.final_entry=True;return n,t
 def actual(self,n,t):
  s=self.s[t];return n.step(np.asarray(Image.open(R/s['rgb'])),s['proprio'],s['tof_frame'],now_s=t,detections=self.d[t]['detections'])
 def test_four_actual_blank_inputs_override_indefinite_cache(self):
  for t in (60.,100.,140.,180.):
   n,t=self.nav(t);old=n.ball_time;z=self.actual(n,t)
   self.assertEqual(z['state'],'REACQUIRE_BALL');self.assertEqual(z['velocity_body_xyw'],[0.,0.,0.]);self.assertFalse(z['kick_request']);self.assertFalse(z['motion_authorization']['allowed']);self.assertEqual(n.ball_time,old);self.assertEqual(n.ball_search.started,t);self.assertGreater(len(n.ball_search.targets),0);json.dumps(z,allow_nan=False)
 def test_goal_refresh_does_not_restart_search_or_ball_age(self):
  n,t=self.nav(60.);self.actual(n,t);start=n.ball_search.started;old=n.ball_time;prior=n.gaze.current.copy();path=[]
  for k in range(1,46):
   now=t+k/15;n.goal_time=now;n.state='REFRESH_GOAL';n.attention_head_permission['capture_s']=now;n.gaze.update([0.,0.],now);path.append(n.gaze.current.copy())
  self.assertEqual(n.ball_search.started,start);self.assertEqual(n.ball_time,old);self.assertGreater(float(np.linalg.norm(np.array(path)-prior,axis=1).max()),.1)
 def test_actual_partial_pixels_only_retarget_head(self):
  n,t=self.nav(43.135);z=self.actual(n,t);self.assertEqual(z['state'],'REACQUIRE_BALL');self.assertEqual(z['velocity_body_xyw'],[0.,0.,0.]);self.assertFalse(z['kick_request']);self.assertIn('fresh visible ball pixels',z['ball_reacquisition']['mode'])
 def test_no_new_capture_holds_head_while_deadline_continues(self):
  n,t=self.nav(60.);self.actual(n,t);n.refresh_too_close=True;prior=n.gaze.current.copy();start=n.ball_search.started
  for k in range(1,61):n.gaze.update([0.,0.],t+k/15)
  self.assertTrue(np.array_equal(n.gaze.current,prior));self.assertEqual(n.ball_search.started,start)
 def test_invalid_public_pose_context_holds_head(self):
  n,t=self.nav(60.);self.actual(n,t);n.ball_recovery_context_valid=False;prior=n.gaze.current.copy()
  for k in range(1,61):n.gaze.update([0.,0.],t+k/15)
  self.assertTrue(np.array_equal(n.gaze.current,prior))
 def test_timeout_is_finite_and_no_budget_reset(self):
  s=BallSearch();s.begin(1.,[-.4,1.1]);start=s.started;s.begin(10.,[0.,0.]);s.target(25.,[-.4,1.1]);self.assertEqual(s.started,start);self.assertEqual(s.failed,'ball_reacquisition_timeout')
 def test_bad_seed_is_rejected_before_state_change(self):
  for bad in ([float('nan'),0.],[0.]):
   s=BallSearch()
   with self.assertRaises(ValueError):s.begin(1.,bad)
   self.assertIsNone(s.started)
 def test_fresh_complete_metric_clears_search(self):
  t=self.metric['timestamp_s'];n,t=self.nav(t);n.ball_search.begin(t-1.,[-.4,1.1]);self.actual(n,t);self.assertIsNone(n.ball_search.started);self.assertFalse(n.ball_recovery_active)
 def test_candidate_joint_source_and_physical_acceptance_recorded(self):
  s=BallSearch();s.begin(1.,[-.4,1.103]);self.assertEqual(len(s.domain_checks),9);self.assertTrue(all(x['accepted']for x in s.domain_checks));self.assertTrue(all(np.max(abs(np.array(x['head_offsets']))-np.array(x['source_command_bounds_rad']))<1e-9 for x in s.domain_checks))
if __name__=='__main__':unittest.main()
