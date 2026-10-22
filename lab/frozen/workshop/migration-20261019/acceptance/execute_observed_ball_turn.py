#!/usr/bin/env python3
"""Separate development visual servo: turn the body toward visible ball pixels.

No partial box is used for distance, translation, kicking or map updates. It
only provides a fresh angular target when that target exceeds head yaw range.
"""
import math
import execute_v8_observed_attention as attention


class ObservedTurnNavigator(attention.AttentionNavigator):
    def _output(self,t,command=(0.,0.,0.),reason='',gaze=None):
        view=self.ball_attention
        valid=self.state=='SEARCH_BALL' and view is not None and 0<=t-view['capture_timestamp_s']<=.2
        evidence=None
        if valid:
            yaw=float(view['gaze_body_yaw_pitch_rad'][0])
            turn=max(-.2,min(.2,.5*yaw))if abs(yaw)>.35 else 0.
            if self.refresh_too_close:turn=0.
            command=(0.,0.,turn)
            evidence={'measured_visible_ray_yaw_rad':yaw,'body_turn_command_rad_s':turn,
                'head_domain_rad':[-.4,.4],'body_turn_deadband_rad':.35,
                'translation_authorized':False,'kick_authorized':False,
                'TooClose_guard':self.refresh_too_close,'source':'fresh RGB visible pixels plus same-time encoder FK'}
            reason='observed_ball_direction_body_turn'if turn else reason
        out=super()._output(t,command,reason,gaze)
        if evidence:out['observed_ball_body_turn']=evidence
        return out


def build(a):
    cfg=attention.build(a);entry=attention.active.base.entry
    entry.contract['source_sha256'][str(entry.Path(__file__).resolve())]=entry.sha(__file__)
    return cfg


if __name__=='__main__':
    entry=attention.active.base.entry
    attention.active.base.controller.BetaNavigator=ObservedTurnNavigator
    entry.isaac_execute.build_cfg=build;entry.isaac_execute.run=entry.run
    entry.nav2_bridge.Nav2Session=entry.BoundedLifetimeSession
    entry.isaac_execute.main()
