"""Real final capture replay and bounded ownership negative cases; CPU only."""
import copy,importlib.util,json,math
from pathlib import Path
import unittest
import numpy as np
from PIL import Image
import perception
from visibility_gate import install
from attention_owner import AttentionOwner
from ball_reacquisition import BallReacquisitionNavigator
from neutral_fine_step import NeutralFineStep
from controller import ControllerConfig
from visual_localization import rotate

D=Path(__file__).parent
spec=importlib.util.spec_from_file_location('original_v62_attention_owner',D.parent/'native_football_v62/attention_owner.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class FineAttentionCommit(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.root=Path('@COURSE@/outputs/migration-20261019/football-v62-full-task180s')
  read=lambda name:[json.loads(s)for s in(cls.root/name).read_text().splitlines()]
  cls.controls=read('control.jsonl');cls.policies=read('policy-trace.jsonl');cls.sensor=read('sensor-inputs.jsonl')[-1];cls.detection=read('detections.jsonl')[-1]
 def setUp(self):self.original=install()
 def tearDown(self):perception.estimate_box_sphere=self.original
 def nav(self,baseline=False):
  row=self.controls[-1];prior=self.controls[-2];t=row['timestamp_s'];n=BallReacquisitionNavigator(None,config=ControllerConfig(ground_mode='tof-virtual-floor'))
  n.pose=np.array(row['estimated_pose_map_xyyaw']);n.final_entry=True;n.first_time=2.34;n.last_time=prior['timestamp_s'];n.goal_time=t-row['near_ball_phase']['goal_fix_age_s']
  valid=max((x for x in self.controls[:-1]if x.get('public_metric_ball')),key=lambda x:x['timestamp_s']);n.ball_time=valid['timestamp_s'];n.ball=n.pose[:2]+rotate(n.pose[2],valid['public_metric_ball']['point_level_m'][:2])
  if baseline:
   class OriginalCompatible(old.AttentionOwner):
    def update(self,**kwargs):kwargs.pop('defer_complete_release',None);return super().update(**kwargs)
   n.attention_owner=OriginalCompatible()
  a=n.attention_owner;ad=prior['public_attention_owner'];a.state=ad['owner'];a.started=ad['episode_started_s'];a.phase_started=ad['phase_started_s'];a.goal_completed=ad['new_goal_fix_s'];a.last_t=prior['timestamp_s']
  n.gaze.current=np.array(prior['gaze_body_yaw_pitch_rad']);n.gaze.timestamp=prior['timestamp_s'];n.ball_search.begin(a.started,n.gaze.current)
  n.external_fine_step_active=True;n.external_fine_step_head_owned=False
  return n
 def frame(self,n,proprio=None):
  s=self.sensor;return n.step(np.asarray(Image.open(self.root/s['rgb'])),s['proprio']if proprio is None else proprio,s['tof_frame'],now_s=s['timestamp_s'],detections=self.detection['detections'])
 def fine(self):
  r=self.policies[-2];s=r['public_neutral_fine_step'];f=NeutralFineStep()
  for attr,key in [('state','state'),('started','started_s'),('step_start','step_start_s'),('step_end','step_end_s'),('quiet_start','quiet_started_s'),('anchor_s','anchor_capture_s'),('goal_s','goal_fix_s'),('goal_heading','goal_heading_rad')]:setattr(f,attr,s[key])
  f.last_policy=r['timestamp_s'];f.last_sent=np.array(r['actual_head_command_rad']);f.head=f.last_sent.copy();return f
 def test_actual33_2_old_owner_releases_and_fine_fails_new_head_only_survives(self):
  for baseline in [True,False]:
   n=self.nav(baseline);out=self.frame(n);t=self.sensor['timestamp_s'];g=n.last_geometry;self.assertTrue(g['ball']['accepted']);self.assertTrue(g['floor_observation']['accepted']);self.assertTrue(n.refresh_too_close)
   self.assertEqual(out['velocity_body_xyw'],[0.,0.,0.]);self.assertFalse(out['kick_request']);permission=out.get('public_attention_head_permission')or{}
   f=self.fine();memory=self.policies[-1]['public_handoff']['memory'];before=f.started
   f.capture(t=t,guard_safe=False,complete=True,trigger=False,actual_head=self.policies[-2]['actual_head_command_rad'],memory=memory,goal_heading=f.goal_heading,goal_fix_s=f.goal_s,head_only_safe=permission.get('accepted')is True)
   after=f.policy(t=t,packet=self.policies[-1]['IMU_prediction'],memory=memory,actual_head=self.policies[-2]['actual_head_command_rad'])
   self.assertEqual(after['velocity'],[0.,0.,0.]);self.assertFalse(after['kick_authorized']);self.assertEqual(f.started,before)
   self.assertEqual(n.attention_owner.started,21.935)
   if baseline:self.assertEqual(out['public_attention_owner']['owner'],'NONE');self.assertEqual(after['reason'],'neutral_step_invalid_head_attention_capture')
   else:self.assertEqual(out['public_attention_owner']['owner'],'BALL_REACQUIRE');self.assertEqual(after['state'],'REACQUIRE');self.assertTrue(permission['accepted'])
 def test_actual_invalid_proprio_does_not_grant_head(self):
  n=self.nav();q=copy.deepcopy(self.sensor['proprio']);q['timestamp_s']=-1.
  before=n.gaze.current.copy()
  with self.assertRaisesRegex(ValueError,'Bad timestamp'):self.frame(n,q)
  np.testing.assert_array_equal(n.gaze.current,before);self.assertIsNone(n.attention_head_permission)
 def owner(self):
  a=AttentionOwner();a.update(t=10.,near=True,complete_capture_s=None,goal_fix_s=9.,request_goal=False);return a
 def test_complete_hazard_never_resets_attention_or_fine_deadline(self):
  a=self.owner()
  for t in [11.,20.,33.9]:a.update(t=t,near=True,complete_capture_s=t,goal_fix_s=t,request_goal=False,defer_complete_release=True);self.assertEqual(a.started,10.);self.assertEqual(a.phase_started,10.);self.assertEqual(a.state,'BALL_REACQUIRE')
  a.update(t=34.,near=True,complete_capture_s=34.,goal_fix_s=34.,request_goal=False,defer_complete_release=True);self.assertEqual(a.failed,'attention_reacquisition_total_timeout')
 def test_original_safe_complete_commit_then_next_new_capture_releases(self):
  a=self.owner();a.update(t=11.,near=True,complete_capture_s=11.,goal_fix_s=11.,request_goal=False,defer_complete_release=True);self.assertEqual(a.state,'BALL_REACQUIRE')
  f=self.fine();f.capture(t=33.2,guard_safe=True,complete=True,trigger=False,actual_head=[0.]*4,memory={},goal_heading=0.,goal_fix_s=33.2,head_only_safe=True);self.assertEqual(f.state,'IDLE')
  a.update(t=11.1,near=True,complete_capture_s=11.1,goal_fix_s=11.,request_goal=False,defer_complete_release=False);self.assertEqual(a.state,'NONE');self.assertEqual(a.started,10.)
 def test_partial_cannot_finish_and_future_complete_rejected(self):
  a=self.owner();a.update(t=11.,near=True,complete_capture_s=None,goal_fix_s=11.,request_goal=False,defer_complete_release=True);self.assertEqual(a.state,'BALL_REACQUIRE')
  before=copy.deepcopy(a.__dict__)
  with self.assertRaises(ValueError):a.update(t=11.1,near=True,complete_capture_s=11.2,goal_fix_s=11.,request_goal=False,defer_complete_release=True)
  self.assertEqual(a.__dict__,before)
 def test_expired_goal_still_refreshes_under_same_total_deadline(self):
  a=self.owner();a.update(t=20.,near=True,complete_capture_s=20.,goal_fix_s=9.,request_goal=True,defer_complete_release=True);self.assertEqual(a.state,'GOAL_REFRESH');self.assertEqual(a.started,10.)
 def test_nonfine_original_complete_behavior_unchanged(self):
  a=self.owner();a.update(t=11.,near=True,complete_capture_s=11.,goal_fix_s=11.,request_goal=False);self.assertEqual(a.state,'NONE')

if __name__=='__main__':unittest.main()
