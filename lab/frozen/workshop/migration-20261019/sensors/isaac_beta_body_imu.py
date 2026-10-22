"""Isaac articulation-to-virtual-body-IMU boundary, without importing Kit."""
import numpy as np

from beta_body_imu import BetaBodyImu, BetaBodyImuProfile
from beta_head_imu import rotate


def _array(value):
    value = getattr(value, 'torch', value)
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


class IsaacBodyImu:
    def __init__(self, robot, *, seed=38112, profile=None, env_index=0, body_quaternion_order='xyzw'):
        self.robot = robot
        self.env_index = env_index
        self.body_id = list(robot.body_names).index('trunk_base')
        if body_quaternion_order not in ('xyzw', 'wxyz'):
            raise ValueError('Declare the articulation quaternion order')
        self.body_quaternion_order = body_quaternion_order
        self.profile = profile or BetaBodyImuProfile()
        self.sensor = BetaBodyImu(profile=self.profile, seed=seed)
        self.reset()

    @property
    def latest(self):
        return self.sensor.latest

    def reset(self):
        self.sensor.reset()
        self._time = self._position = self._velocity = None

    def update(self, timestamp_s):
        data = self.robot.data
        quaternion = _array(data.body_quat_w)[self.env_index, self.body_id]
        if self.body_quaternion_order == 'wxyz':
            quaternion = quaternion[[1, 2, 3, 0]]
        origin = _array(data.body_pos_w)[self.env_index, self.body_id]
        position = origin + rotate(quaternion, self.profile.sensor_position_trunk_m)
        delivered = []
        if self._time is not None:
            dt = timestamp_s - self._time
            if dt <= 0:
                raise ValueError('Body IMU update requires advancing simulation time')
            velocity = (position - self._position) / dt
            if self._velocity is not None:
                acceleration = (velocity - self._velocity) / dt
                angular = _array(data.body_ang_vel_w)[self.env_index, self.body_id]
                delivered = self.sensor.poll(timestamp_s, q_world_from_trunk_xyzw=quaternion,
                    angular_velocity_world_rad_s=angular, linear_acceleration_world_m_s2=acceleration)
            self._velocity = velocity
        self._position = position
        self._time = timestamp_s
        return delivered

    def metadata(self):
        return {**self.sensor.metadata(), 'articulation_body': 'trunk_base',
                'articulation_quaternion_order': self.body_quaternion_order,
                'virtual_sensor_acceleration': 'two finite differences of the configured rigidly-mounted sensor point at200Hz',
                'warmup': 'first two physics updates establish derivatives without invented acceleration',
                'world_kinematics_exposed_in_packet': False}
