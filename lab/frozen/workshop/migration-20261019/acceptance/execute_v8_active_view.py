#!/usr/bin/env python3
"""New development controller: actively reacquire a mapped goal after turning.

Only the existing sensor-derived pose and declared map goal are used. The head
stays inside its trained yaw/pitch domain; the body turns to bring an out-of-view
goal back inside it. This fixes the old refresh state that always looked straight
ahead even after heading drift. No kicking is enabled by this wrapper.
"""
import math
import numpy as np
import execute_v8_navigation as base
from visual_localization import wrap


class ActiveViewNavigator(base.BoundedGazeNavigator):
    def reset(self):
        super().reset()
        self.refresh_too_close=False

    def step(self,rgb,proprio,tof_frame,**kwargs):
        self.refresh_too_close=any(c.get('contains_too_close',False)
            for row in tof_frame['cells'] for c in row)
        return super().step(rgb,proprio,tof_frame,**kwargs)

    def _output(self,t,command=(0.,0.,0.),reason='',gaze=None):
        observation=None
        if self.state=='REFRESH_GOAL' and self.pose is not None:
            delta=self.goal_map-self.pose[:2]
            bearing=wrap(math.atan2(delta[1],delta[0])-self.pose[2])
            gaze=[float(np.clip(bearing,-.4,.4)),0.]
            yaw=float(np.clip(.5*bearing,-.2,.2)) if abs(bearing)>.25 else 0.
            if self.refresh_too_close:yaw=0.
            command=(0.,0.,yaw)
            reason='sensor_pose_goal_reacquisition' if not self.refresh_too_close else 'goal_reacquisition_TooClose_body_turn_blocked'
            observation={'source':'existing visual/encoder/IMU map pose and declared goal map',
                         'goal_bearing_body_rad':bearing,'head_yaw_limit_rad':.4,
                         'body_turn_deadband_rad':.25,'body_yaw_limit_rad_s':.2,
                         'TooClose_guard':self.refresh_too_close,'privileged_state_read':False}
        out=super()._output(t,command,reason,gaze)
        if observation:out['active_goal_reacquisition']=observation
        return out


def build(a):
    result=base.navigation_build(a)
    base.entry.contract['source_sha256'][str(base.entry.Path(__file__).resolve())]=base.entry.sha(__file__)
    return result


if __name__=='__main__':
    base.controller.BetaNavigator=ActiveViewNavigator
    base.entry.isaac_execute.build_cfg=build
    base.entry.isaac_execute.run=base.entry.run
    base.entry.nav2_bridge.Nav2Session=base.entry.BoundedLifetimeSession
    base.entry.isaac_execute.main()
