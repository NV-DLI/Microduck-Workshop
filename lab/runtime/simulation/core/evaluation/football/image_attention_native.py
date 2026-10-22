#!/usr/bin/env python3
"""Image attention native: Microduck simulation and control."""
from pathlib import Path
import numpy as np
from near_ball_phase_native import NearBallNavigator
from bounded_image_head_servo import attention_correction


class ImageHeadNavigator(NearBallNavigator):
    def reset(self):
        super().reset(); self.head_servo_time=None

    def _output(self,t,command=(0.,0.,0.),reason='',gaze=None):
        original=self.ball_attention; corrected=None
        near=self.public_near_context(t) and 0<=t-self.goal_time<=8.
        eligible=near and reason in {'visual_localization_refresh','fresh_ball_required',
                                    'fresh_RGB_position_error','zero_command_observe_real_settling','actual_Navfn_path'}
        if (eligible and original is not None and self.current_sensor_pair is not None
                and np.all(self.gaze.current >= [-.4,0.]) and np.all(self.gaze.current <= [.4,1.4])):
            o,_=self.current_sensor_pair
            if abs(o['timestamp_s']-t)<1e-8:
                fk=self.vision.head.forward_policy14(o['q_rad'],joint_names=o['joint_names'])['camera_raw_optical']
                dt=0. if self.head_servo_time is None else t-self.head_servo_time
                corrected=attention_correction(original,command=self.gaze.current,
                    rotation_body_from_raw=fk.rotation,camera=self.vision.camera,now_s=t,dt_s=dt)
        self.head_servo_time=t
        if corrected is not None:
            self.ball_attention={**original,**corrected}
        try:
            out=super()._output(t,command,reason,gaze)
        finally:
            self.ball_attention=original
        if corrected is not None:out['image_head_servo']=corrected
        return out

