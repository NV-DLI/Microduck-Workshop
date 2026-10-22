"""Small public-IMU yaw-rate servo; no body pose or simulator truth input."""
import math

class MeasuredYawFeedback:
    def __init__(self):self.reset()
    def reset(self):
        self.last_time=None;self.filtered=0.;self.integral=0.;self.previous=0.
    def update(self,desired,measured,*,timestamp,valid=True):
        if not all(math.isfinite(v)for v in(desired,measured,timestamp)):
            raise ValueError('Finite desired/measured angular velocity and sensor timestamp required')
        dt=0. if self.last_time is None else timestamp-self.last_time
        if dt<0:raise ValueError('Measurements cannot move backward in time')
        if not valid or dt>.2:
            self.reset();self.last_time=timestamp
            return 0.,{'active':False,'reason':'invalid_ground_or_sensor_gap','integral_rad_s':0.}
        if self.last_time is None:self.filtered=measured
        self.last_time=timestamp
        alpha=1-math.exp(-dt/.25)
        self.filtered+=alpha*(measured-self.filtered)
        error=desired-self.filtered
        self.integral=max(-.12,min(.12,self.integral+.25*error*dt))
        requested=desired+.5*error+self.integral
        bounded=max(-.3,min(.3,requested))
        # Command slew is separate from physical BAM targets, which remain at
        # their trained50Hz/internal8 mapping. No physical state is overwritten.
        if dt>0:bounded=max(self.previous-.5*dt,min(self.previous+.5*dt,bounded))
        self.previous=bounded
        return bounded,{'active':True,'desired_yaw_rate_rad_s':desired,'measured_yaw_rate_rad_s':measured,
            'filtered_yaw_rate_rad_s':self.filtered,'integral_rad_s':self.integral,
            'output_yaw_command_rad_s':bounded,'p_gain':.5,'i_gain':.25,'filter_tau_s':.25,
            'absolute_command_limit_rad_s':.3,'command_slew_rad_s2':.5,
            'scope':'Public delivered BODY50 gyro; engineering feedback experiment, not trained gait qualification'}
