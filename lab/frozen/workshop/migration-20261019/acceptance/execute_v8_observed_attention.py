#!/usr/bin/env python3
"""Development navigation with partial-image attention, strict metric geometry."""
import execute_v8_active_view as active
from observed_ball_attention import observed_attention


class AttentionNavigator(active.ActiveViewNavigator):
    def reset(self):
        super().reset();self.ball_attention=None

    def step(self,rgb,proprio,tof_frame,*,now_s,detections=None):
        fk=self.vision.head.forward_policy14(proprio['q_rad'],joint_names=proprio['joint_names'])['camera_raw_optical']
        self.ball_attention=observed_attention(detections,camera=self.vision.camera,
            rotation_body_from_raw=fk.rotation,capture_s=proprio['timestamp_s'],now_s=now_s)
        return super().step(rgb,proprio,tof_frame,now_s=now_s,detections=detections)

    def _output(self,t,command=(0.,0.,0.),reason='',gaze=None):
        use=self.state in ('SEARCH_BALL','GROUND_UNAVAILABLE') and self.ball_attention is not None
        if use:gaze=self.ball_attention['gaze_body_yaw_pitch_rad']
        out=super()._output(t,command,reason,gaze)
        if use:out['observed_ball_attention']=self.ball_attention
        return out


def build(a):
    cfg=active.build(a)
    for file in (__file__,active.base.entry.Path(__file__).with_name('observed_ball_attention.py')):
        active.base.entry.contract['source_sha256'][str(active.base.entry.Path(file).resolve())]=active.base.entry.sha(file)
    return cfg


if __name__=='__main__':
    entry=active.base.entry
    active.base.controller.BetaNavigator=AttentionNavigator
    entry.isaac_execute.build_cfg=build;entry.isaac_execute.run=entry.run
    entry.nav2_bridge.Nav2Session=entry.BoundedLifetimeSession;entry.isaac_execute.main()
