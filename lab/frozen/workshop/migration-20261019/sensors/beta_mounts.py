"""Versioned CAD mount choices. No implicit link-name or position fallback."""
from dataclasses import asdict, dataclass
import math

@dataclass(frozen=True)
class SensorMount:
    name: str
    body_name: str
    camera_position_body_m: tuple[float,float,float]
    tof_position_body_m: tuple[float,float,float]
    imu_position_body_m: tuple[float,float,float]
    source: str
    site_quaternion_xyzw: tuple[float,float,float,float] = (0.,math.sqrt(.5),0.,math.sqrt(.5))
    camera_site_to_ros_xyzw: tuple[float,float,float,float] = (-.5,.5,-.5,.5)

    def to_dict(self):
        return {**asdict(self),'camera_mount_rotation_cw_deg':0,
                'imu_position_scope':'matching CAD head_imu site; Beta chip position has not been measured on this physical unit',
                'imu_orientation_scope':'Beta measured sensor-to-head mapping from robotd; head FLU follows the forward optical axis',
                'source_is_asset_reference_not_unit_calibration':True}

MOUNTS={
 'rl-jaw-soft-20261012':SensorMount('rl-jaw-soft-20261012','jaw_soft',
    (.0155,-.0000913778,-.0733),(.0135,.0224086,-.0733),(.0152323,.000111069,-.05106),
    'microduck_rl@b02fdea4902c2b93b7d34a83a6f05da4ececb7d3 robot_allcollisions.xml'),
 'runtime-bottom-head-shell-20261017':SensorMount('runtime-bottom-head-shell-20261017','bottom_head_shell',
    (.01175,0.,-.0735),(.0143,.0225,-.0735),(.0114823,.000202447,-.05126),
    'microduck@53230463927095ea890f231d29089bb0a538c06f kinematics/assets/alpha/robot_walk.xml'),
}

def get_mount(name):
    try:return MOUNTS[name]
    except KeyError:raise ValueError('Choose an explicit verified CAD mount: '+', '.join(MOUNTS)) from None
