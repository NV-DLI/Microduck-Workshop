"""Frozen actual384-image scene builder; independent of navigation experiments."""
import hashlib,json,math,sys
from pathlib import Path
M=Path(__file__).resolve().parent.parent
ROOT=M/"football"
for folder in (ROOT,M/"sensors",M/"gait"):
    if str(folder)not in sys.path:sys.path.insert(0,str(folder))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def configure_visual_assets(env):
    """Refine only the visible USD sphere; Newton sphere collision stays analytic."""
    from pxr import Sdf,Usd,UsdGeom
    ball=env.sim.stage.GetPrimAtPath('/World/envs/env_0/Ball');paths=[]
    for prim in Usd.PrimRange(ball):
        if prim.IsA(UsdGeom.Sphere):
            prim.CreateAttribute('refinementEnableOverride',Sdf.ValueTypeNames.Bool,custom=True).Set(True)
            prim.CreateAttribute('refinementLevel',Sdf.ValueTypeNames.Int,custom=True).Set(2)
            paths.append(str(prim.GetPath()))
    if not paths:raise RuntimeError('Expected the actual USD sphere for visual refinement')
    env.course_visual_geometry={'sphere_refinement_level':2,'sphere_prims':paths,'collision_changed':False,'sphere_radius_m':.035,'sphere_mass_kg':.045,'pitch_visual_top_z_m':.0002,'pitch_has_collision':False}

def sensor_prestartup(env,env_ids,*,width,height,hfov,seed):
    """Called by the environment before the first PHYSICS_READY event."""
    from isaac_beta_sensor_rig import IsaacBetaSensors
    from beta_sensor_profile import BetaCameraProfile
    from isaaclab.sensors.camera import Camera,CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg
    import isaaclab.sim as sim_utils
    configure_visual_assets(env)
    env.beta_rig=IsaacBetaSensors(env.sim,env.scene['robot'],mount_profile='rl-jaw-soft-20261012',camera_profile=BetaCameraProfile(width_px=width,height_px=height,assumed_horizontal_fov_deg=hfov),seed=seed)
    focal=width/(2*math.tan(math.radians(60)/2));K=[focal,0,width/2,0,focal,height/2,0,0,1]
    env.course_overhead=Camera(CameraCfg(prim_path='/World/envs/env_0/BetaSensors/Overhead',width=width,height=height,update_period=0.,data_types=['rgb'],renderer_cfg=IsaacRtxRendererCfg(),spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(K,width,height,clipping_range=(.03,20.),focal_length=20.)))

def build_cfg(a):
    import numpy as np
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg,AssetBaseCfg
    from isaaclab.managers import EventTermCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
    from isaaclab_tasks.utils import parse_env_cfg
    from nav2_service.obstacle_contract import OBSTACLE
    cfg=parse_env_cfg('IsaacContrib-Velocity-Flat-MicroDuck',device=a.device,num_envs=1,overrides=['presets=backlash'])
    if getattr(a,'trained_recipe_dir',None) is not None:
        import importlib
        a.trained_recipe_dir=a.trained_recipe_dir.resolve()
        if any(name in sys.modules for name in ('bounded_actions','precision_recipe','precision_recipe_v3')):raise RuntimeError('Frozen trained action import contaminated')
        sys.path.insert(0,str(a.trained_recipe_dir))
        cfg.actions.joint_pos.class_type=importlib.import_module('precision_recipe_v3').BoundedTrainingAction
        metadata=json.loads(a.model.with_suffix('.json').read_text())
        if metadata['sha256']!=sha(a.model)or Path(metadata['checkpoint']).resolve().parent!=a.trained_recipe_dir or sha(metadata['checkpoint'])!=metadata['checkpoint_sha256']:raise ValueError('Export does not bind this frozen training run')
    cfg.seed=a.seed;cfg.episode_length_s=a.seconds+5.;cfg.curriculum=None
    cfg.sim.dt=.005;cfg.decimation=4;cfg.sim.render_interval=4
    # Extra ball/goal contacts need more capacity than walking alone.
    cfg.sim.physics.solver_cfg.nconmax=64;cfg.sim.physics.solver_cfg.njmax=256
    cfg.scene.robot.spawn.usd_path=str(a.robot_usd.resolve())
    cfg.commands.base_velocity.debug_vis=False;cfg.commands.base_velocity.resampling_time_range=(1e8,1e8)
    cfg.commands.head_pose.resampling_time_range=(1e8,1e8);cfg.commands.head_pose.ranges=tuple((0.,0.)for _ in range(4))
    cfg.commands.body_pose.resampling_time_range=(1e8,1e8);cfg.commands.body_pose.ranges=tuple((0.,0.)for _ in range(6))
    cfg.events.reset_base.params['pose_range']={}
    if not a.randomized:
        for name in ('foot_friction','encoder_bias','mass_inertia','randomize_com','randomize_head_com','randomize_joint_friction','randomize_armature','push_robot'):setattr(cfg.events,name,None)
        cfg.scene.robot.actuators['servos'].vin_range=None;cfg.scene.robot.actuators['servos'].vin_drop_gain_range=None
        cfg.scene.robot.actuators['servos'].min_delay=4;cfg.scene.robot.actuators['servos'].max_delay=4
        cfg.observations.policy.enable_corruption=False
        cfg.observations.policy.imu.params.update(max_angle_deg=0,min_lag=0,max_lag=0)
    cfg.events.beta_sensors=EventTermCfg(func=sensor_prestartup,mode='prestartup',params={'width':a.width,'height':a.height,'hfov':a.assumed_hfov,'seed':a.seed})
    rng=np.random.default_rng(a.seed)
    root_y=float(rng.uniform(-.08,.08))if a.root_y is None else a.root_y
    ball_xy=[float(rng.uniform(-.08,.12)),float(rng.uniform(-.08,.08))]if a.ball_xy is None else a.ball_xy
    root_x=getattr(a,'root_x',-.66)
    cfg.scene.robot.init_state.pos=(root_x,root_y,.125)
    cfg.scene.ball=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/Ball',spawn=sim_utils.SphereCfg(radius=.035,rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),mass_props=sim_utils.MassPropertiesCfg(mass=.045),collision_props=sim_utils.CollisionPropertiesCfg(),activate_contact_sensors=True,visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.9,.91,.88),roughness=.75)),init_state=RigidObjectCfg.InitialStateCfg(pos=(*ball_xy,.035)))
    # Opening is .24m: post centres +/- .1325, each .025m thick.
    for name,size,pos in [('goal_left',(.025,.025,.1925),(.93,.1325,.09625)),('goal_right',(.025,.025,.1925),(.93,-.1325,.09625)),('goal_bar',(.025,.29,.025),(.93,0.,.1925))]:
        setattr(cfg.scene,name,AssetBaseCfg(prim_path='{ENV_REGEX_NS}/'+name,spawn=sim_utils.CuboidCfg(size=size,collision_props=sim_utils.CollisionPropertiesCfg(),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.9,.92,.88),roughness=.7)),init_state=AssetBaseCfg.InitialStateCfg(pos=pos)))
    cfg.scene.pitch=AssetBaseCfg(prim_path='{ENV_REGEX_NS}/PitchVisual',spawn=sim_utils.CuboidCfg(size=(2.,1.,.003),visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.12,.32,.16),roughness=.85)),init_state=AssetBaseCfg.InitialStateCfg(pos=(0.,0.,-.0013)))
    cfg.scene.sky_light.spawn.intensity=1800.
    # Sensor body is the ball; partner order is left then right. These are
    # forces ON the ball from the individual feet, not net ground contacts.
    feet=['{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/ankle_'+side for side in ('left','right')]
    cfg.scene.football_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Ball',filter_prim_paths_expr=feet,history_length=1,update_period=.005)
    if a.layout=='one-obstacle':
        pos=(*OBSTACLE['center_xy_m'],OBSTACLE['height_m']/2);size=(*OBSTACLE['size_xy_m'],OBSTACLE['height_m'])
        cfg.scene.obstacle=RigidObjectCfg(prim_path='{ENV_REGEX_NS}/KnownObstacle',spawn=sim_utils.CuboidCfg(size=size,rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True,kinematic_enabled=True),mass_props=sim_utils.MassPropertiesCfg(mass=1.),collision_props=sim_utils.CollisionPropertiesCfg(),activate_contact_sensors=True,visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.15,.2,.65),roughness=.8)),init_state=RigidObjectCfg.InitialStateCfg(pos=pos))
        cfg.scene.obstacle_contact=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/KnownObstacle',filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/Geometry/.*'],history_length=1,update_period=.005)
    return cfg,{'root_pose_xyzw':[root_x,root_y,.125,0.,0.,0.,1.],'ball_pose_xyzw':[*ball_xy,.035,0.,0.,0.,1.],'goal_plane_x_m':.93,'goal_opening_width_m':.24,'goal_opening_height_m':.18,'ball_radius_m':.035,'ball_mass_kg':.045,'seed':a.seed,'layout':a.layout,'scope':'Initialization and independent evaluation only; never controller input'}
