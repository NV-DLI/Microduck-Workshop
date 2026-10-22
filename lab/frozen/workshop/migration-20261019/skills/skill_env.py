"""Physical skill objectives in Isaac Lab, with explicit metrics and fresh PPO.

Community ONNX receives exactly the published 61D contract. Native actors get
additional simulator state and an explicit phase clock (75D); they are separate
policies, trained by PPO without demonstrations or imported weights.
"""
from __future__ import annotations
import math
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz, quat_mul, quat_conjugate
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from copy import deepcopy
from isaaclab_assets.robots.microduck import (
    MICRODUCK_CFG, MICRODUCK_ALLCOLLISIONS_CFG,
    MICRODUCK_BACKLASH_CFG, MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG,
)
from isaaclab_tasks.contrib.microduck.flat_env_cfg import MicroDuckVelocityFlatEnvCfg
from contract import HOME_POSE, POLICY_JOINTS
from backlash_contract import resolve_joint_channels, encoder_channels
from skill_randomization import SkillEventsCfg, SkillDomainRandomizationCfg, events_for_skill


def physics_config():
    return deepcopy(MicroDuckVelocityFlatEnvCfg().sim.physics)


def T(x):
    return getattr(x, "torch", x)


@configclass
class SkillSceneCfg(InteractiveSceneCfg):
    robot = None
    ball = None
    foot_0 = None
    foot_1 = None
    whole_robot_contacts = None
    record_camera = None
    ground = None
    light = None


def configure_scene(cfg):
    """Declare all assets before InteractiveScene builds the shared clone plan."""
    scene=cfg.scene
    if cfg.robot_variant == 'backlash28':
        base=MICRODUCK_ALLCOLLISIONS_BACKLASH_CFG if cfg.skill=='backflip' else MICRODUCK_BACKLASH_CFG
    elif cfg.robot_variant == 'servo14':
        base=MICRODUCK_ALLCOLLISIONS_CFG if cfg.skill=='backflip' else MICRODUCK_CFG
    else:raise ValueError(f'Unknown robot variant: {cfg.robot_variant}')
    robot_cfg=deepcopy(base);robot_cfg.prim_path='{ENV_REGEX_NS}/Robot'
    # Official USD is rooted at zero, without the old asset's +0.12 m offset.
    robot_cfg.init_state.pos=(0.,0.,.368 if cfg.skill=='basketball' else cfg.root_height)
    if cfg.robot_usd:robot_cfg.spawn.usd_path=cfg.robot_usd
    if cfg.actuator!='bam':raise ValueError('This migration requires official Newton-native BAM')
    if not cfg.domain_randomization:
        robot_cfg.actuators['servos'].vin_range=(7.4,7.4)
        robot_cfg.actuators['servos'].vin_drop_gain_range=(0.,0.)
        robot_cfg.actuators['servos'].min_delay=3
        robot_cfg.actuators['servos'].max_delay=3
    scene.robot=robot_cfg
    scene.ball=None
    if cfg.skill=='basketball':
        scene.ball=RigidObjectCfg(
            prim_path='{ENV_REGEX_NS}/Ball',
            spawn=sim_utils.SphereCfg(radius=.12,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.),
                mass_props=sim_utils.MassPropertiesCfg(mass=.62),
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.001,rest_offset=0.),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2,dynamic_friction=1.2,restitution=0.),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.85,.24,.035),roughness=.8)),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0,0,.121)))
    scene.ground=AssetBaseCfg(prim_path='/World/ground',collision_group=-1,
        spawn=sim_utils.GroundPlaneCfg(color=(.18,.22,.28),physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.,dynamic_friction=1.,restitution=0.)))
    for i,body in enumerate(('ankle_left','ankle_right')):
        setattr(scene,f'foot_{i}',ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/Robot/Geometry/trunk_base/.*/'+body,
            update_period=.005,history_length=1,track_air_time=True))
    scene.whole_robot_contacts=ContactSensorCfg(
        prim_path='{ENV_REGEX_NS}/Robot/Geometry/.*',update_period=.005,history_length=1
    ) if cfg.skill=='backflip' else None
    scene.record_camera=None
    if cfg.record_video:
        from isaaclab.sensors.camera import Camera,CameraCfg
        from isaaclab_physx.renderers import IsaacRtxRendererCfg
        class OverviewCamera(Camera):
            def reset(self,env_ids=None):
                # One global camera is shared by all parallel environments.
                return super().reset(None)
        scene.record_camera=CameraCfg(class_type=OverviewCamera,
            prim_path='/World/SkillCamera',update_period=0.,height=720,width=1280,
            data_types=['rgb'],renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.,horizontal_aperture=20.955,clipping_range=(.01,100.)))
    scene.light=AssetBaseCfg(prim_path='/World/light',spawn=sim_utils.DomeLightCfg(intensity=1600.,color=(.9,.93,1.)))


@configclass
class SkillEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 8.0
    action_space = 14
    observation_space = 75
    state_space = 0
    sim = SimulationCfg(dt=0.005, render_interval=4, physics=physics_config(), use_newton_actuators=True)
    scene = SkillSceneCfg(num_envs=512, env_spacing=2.0)
    skill: str = "dance"
    evaluation: bool = False
    record_video: bool = False
    jump_reset: str = "mixed"
    disable_sleep: bool = False
    basketball_recipe: str = "legacy"
    root_height: float = .125
    actuator: str = "bam"
    reset_noise: float = 1.0
    jump_objective: str = "discovery"
    arabesque_objective: str = "legacy"
    backflip_objective: str = "legacy"
    backflip_reset: str = "mixed"
    backflip_recovery_curriculum: bool = False
    backflip_strict_finish: bool = False
    prime_reset_delay: bool = True
    domain_randomization: bool = True
    events: SkillEventsCfg | None = SkillEventsCfg()
    robot_usd: str | None = None
    robot_variant: str = 'backlash28'


class SkillEnv(DirectRLEnv):
    cfg: SkillEnvCfg

    def _setup_scene(self):
        # Modern InteractiveScene constructs and clones declared entities before
        # this hook. Newton's clone plan handles independent physics worlds.
        self.robot=self.scene['robot']
        self.ball=self.scene['ball'] if self.cfg.skill=='basketball' else None
        self.feet_sensors=[self.scene[f'foot_{i}'] for i in range(2)]
        self.body_contacts=self.scene['whole_robot_contacts'] if self.cfg.skill=='backflip' else None

    def __init__(self, cfg, **kwargs):
        configure_scene(cfg)
        cfg.events=events_for_skill(cfg.skill,cfg.robot_variant) if cfg.domain_randomization else None
        super().__init__(cfg, **kwargs)
        if cfg.skill=='backflip':
            # Rotation/contact chronology is evaluated in _apply_action at
            # 200 Hz. A single internal four-step graph would skip its ticks.
            self.sim.backend_manager.set_decimation(1)
            self._physics_handles_decimation=False
        self.jids,self.passive_jids=resolve_joint_channels(self.robot.joint_names,cfg.robot_variant)
        if not hasattr(self,'encoder_bias'):self.encoder_bias=torch.zeros(self.num_envs,14,device=self.device)
        self.dr_evidence=SkillDomainRandomizationCfg.to_dict(enabled=cfg.domain_randomization,variant=cfg.robot_variant)
        self.dr_evidence['asset_usd']=self.robot.cfg.spawn.usd_path
        self.dr_evidence['sleep_configuration']='Newton contacts; MJWarp sleeping is unavailable in this selected pipeline'
        self.fids, _ = self.robot.find_bodies(["ankle_left", "ankle_right"], preserve_order=True)
        # Cache device indices: Python-list CUDA indexing copies a host tensor
        # and synchronizes on every sensor read, dominating a single GUI world.
        self.jids_t=torch.tensor(self.jids,device=self.device,dtype=torch.long)
        self.passive_jids_t=torch.tensor(self.passive_jids,device=self.device,dtype=torch.long)
        self.fids_t=torch.tensor(self.fids,device=self.device,dtype=torch.long)
        self.home = torch.tensor(HOME_POSE, device=self.device)
        self.actions = torch.zeros(self.num_envs,14,device=self.device)
        self.prev_actions = self.actions.clone()
        self.command = torch.zeros(self.num_envs,13,device=self.device)
        self.max_rise = torch.zeros(self.num_envs,device=self.device)
        self.air_ticks = self.max_rise.clone()
        self.max_air = self.max_rise.clone()
        self.took_off = self.max_rise.bool()
        self.landed = self.max_rise.bool()
        self.stable_ticks = self.max_rise.clone()
        self.peak_vz = self.max_rise.clone()
        self.start_xy = torch.zeros(self.num_envs,2,device=self.device)
        self.metrics = {}
        self.rollout_metrics = {}
        self.reset_count = 0
        self.jump_enabled = True
        for name in ("support_ticks", "takeoff_com", "ground_clearance", "visible_jump", "com_rise", "sole_clearance", "qualified_air_ticks"):
            setattr(self,name,torch.zeros(self.num_envs,device=self.device))
        self.qualified_launch=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self.qualified_landed=self.qualified_launch.clone()
        left=[[-.030161,-.017313,-.034446],[-.025834,-.017479,-.035201],[-.012899,-.017724,-.035217],[-.006433,-.017971,-.035231],[0,-.018026,-.035245],[.009570,-.017631,-.035216],[.016052,-.017224,-.034977]]
        right=[[-.016052,-.017224,-.034977],[-.009570,-.017631,-.035216],[0,-.018026,-.035245],[.006433,-.017971,-.035231],[.012899,-.017724,-.035217],[.025834,-.017479,-.035201],[.030161,-.017313,-.034446]]
        self.sole_probes=torch.tensor([left,right],device=self.device)
        if self.cfg.skill == "backflip":
            from backflip import BackflipTracker
            self.flip = BackflipTracker(self.num_envs, self.device)
            self.flip_last_quat = T(self.robot.data.root_quat_w).clone()
            self.flip_paid = torch.zeros(self.num_envs, device=self.device)
            self.flip_first_body=torch.full((self.num_envs,),-1,device=self.device,dtype=torch.long)
            self.flip_first_feet=torch.zeros(self.num_envs,device=self.device)
            self.flip_first_tilt=torch.zeros(self.num_envs,device=self.device)
            self.flip_contact_paid=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            self.flip_hold_paid=torch.zeros(self.num_envs,device=self.device)
            self.flip_training_landed=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            self.flip_training_hold=torch.zeros(self.num_envs,device=self.device)
            self.flip_training_max_hold=torch.zeros(self.num_envs,device=self.device)
            self.flip_step_started=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            self.flip_step_succeeded=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            self.flip_launch_tilt=torch.zeros(self.num_envs,device=self.device)
        self.pose_hold = torch.zeros(self.num_envs, device=self.device)
        self.pose_max_hold = self.pose_hold.clone()
        self.balance_horizon=torch.where(torch.arange(self.num_envs,device=self.device)%4==0,self.max_episode_length,min(self.max_episode_length,150))


    def _pre_physics_step(self, actions):
        if hasattr(self,'flip'):
            self.flip_step_started.zero_();self.flip_step_succeeded.zero_()
        if self.ball is not None and self.cfg.basketball_recipe in ("robust","reset_curriculum") and not self.cfg.evaluation:
            ids=torch.nonzero((self.episode_length_buf>0)&(self.episode_length_buf%100==0)).flatten()
            if len(ids):
                velocity=T(self.ball.data.root_vel_w)[ids].clone()
                velocity[:,:2]+=(torch.rand(len(ids),2,device=self.device)-.5)*.08
                self.ball.write_root_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
        self.prev_actions.copy_(self.actions)
        self.actions.copy_(actions)
        # Joint hard limits are enforced by Newton. Do not crop exported model actions.
        self.robot.set_joint_position_target_index(target=self.home + self.actions - self.encoder_bias, joint_ids=self.jids_t)

    def _apply_action(self):
        # Inspect all-body contacts and orientation at the physical 200 Hz rate.
        # This prevents summing separated hops or a ground roll into a flip.
        if hasattr(self, 'flip'):
            pos, tilt, q, feet_contact, feet = self._measure()
            quat = T(self.robot.data.root_quat_w)
            delta = quat_mul(quat_conjugate(self.flip_last_quat), quat)
            delta = torch.where(delta[:, 3:4]<0, -delta, delta)
            pitch_delta = -2*torch.atan2(delta[:,1],delta[:,3])
            off_axis = 2*delta[:,[0,2]].norm(dim=-1)
            body_force=T(self.body_contacts.data.net_forces_w).norm(dim=-1)
            any_contact=body_force.amax(-1)>.08
            first_contact=self.flip.started&~self.flip.touched&any_contact
            self.flip_first_body=torch.where(first_contact,body_force.argmax(-1),self.flip_first_body)
            self.flip_first_feet=torch.where(first_contact,feet_contact.any(-1).float(),self.flip_first_feet)
            self.flip_first_tilt=torch.where(first_contact,tilt,self.flip_first_tilt)
            if self.cfg.backflip_recovery_curriculum:
                self.flip_training_landed|=first_contact&feet_contact.any(-1)&(tilt<math.radians(35))&(self.flip.frontier>=math.radians(320))
                quiet=self.flip_training_landed&feet_contact.all(-1)&(tilt<math.radians(20))&((pos[:,2]-self.scene.env_origins[:,2])>.095)
                quiet&=(T(self.robot.data.root_lin_vel_w).norm(dim=-1)<.2)&(T(self.robot.data.root_ang_vel_b).norm(dim=-1)<2)
                self.flip_training_hold=torch.where(quiet,self.flip_training_hold+self.physics_dt,torch.zeros_like(self.flip_training_hold))
                self.flip_training_max_hold=torch.maximum(self.flip_training_max_hold,self.flip_training_hold)
            self.flip.update(pitch_delta, off_axis,
                pos[:,2]-self.scene.env_origins[:,2], feet_contact,
                any_contact,
                tilt, T(self.robot.data.root_lin_vel_w).norm(dim=-1),
                T(self.robot.data.root_ang_vel_b).norm(dim=-1),
                T(self.robot.data.root_lin_vel_w)[:,2], self.physics_dt)
            # A 50 Hz reward must retain events from all four 200 Hz substeps.
            self.flip_step_started|=self.flip.just_started
            self.flip_step_succeeded|=self.flip.just_succeeded
            self.flip_launch_tilt=torch.where(self.flip.just_started,tilt,self.flip_launch_tilt)
            self.flip_last_quat.copy_(quat)
        # Newton-native DriveBam computes efforts at each physical substep.

    def community_obs(self):
        d = self.robot.data
        self.command.zero_()
        if self.cfg.skill == "jump":
            self.command[:,0] = ((self.episode_length_buf*self.step_dt < 0.75) & ~self.landed & self.jump_enabled).float()
        if self.cfg.skill == "backflip":
            phase=(self.episode_length_buf*self.step_dt/.30).pow(3)
            self.command[:,7]=phase/(1+phase)  # Original body-command slot 0.
            self.command[:,8]=0.0  # No external assistance in train or evaluation.
            if self.cfg.backflip_objective in ('launch_spin','compact_flight','compact_landing','landing_upright'):
                # Preserve the native jump actor's observation contract during
                # curriculum transfer. The physical task and gate remain a flip.
                self.command.zero_()
                self.command[:,0]=((self.episode_length_buf*self.step_dt<.75)&~self.flip.touched).float()
        gyro=T(d.root_ang_vel_b)
        gravity=T(d.projected_gravity_b)
        if self.cfg.domain_randomization:
            gyro=gyro+torch.empty_like(gyro).uniform_(-.03,.03)
            gravity=gravity+torch.empty_like(gravity).uniform_(-.01,.01)
        return torch.cat((gyro, gravity,
            encoder_channels(T(d.joint_pos),self.jids_t,self.passive_jids_t)+self.encoder_bias-self.home,
            encoder_channels(T(d.joint_vel),self.jids_t,self.passive_jids_t), self.actions, self.command),dim=-1)

    def _get_observations(self):
        obs = self.community_obs()
        phi = self.episode_length_buf.float()*self.step_dt*(2*math.pi/1.2)
        phase = torch.stack((phi.sin(),phi.cos()),dim=-1)
        extra = torch.zeros(self.num_envs,9,device=self.device)
        if self.ball is not None:
            extra = torch.cat((T(self.ball.data.root_pos_w)-T(self.robot.data.root_pos_w),
                T(self.ball.data.root_lin_vel_w),T(self.ball.data.root_ang_vel_w)),dim=-1)
        if self.ball is not None and self.cfg.basketball_recipe in ("robust","reset_curriculum"):
            # A balance policy has no rhythm target. Express all privileged ball
            # signals in the robot frame so randomized yaw remains observable.
            phase.zero_()
            quat=T(self.robot.data.root_quat_w)
            extra=torch.cat([quat_apply_inverse(quat,extra[:,i:i+3]) for i in (0,3,6)],dim=-1)
        if self.cfg.skill=='arabesque' and self.cfg.arabesque_objective in ('static','hold'):
            # A hold has no rhythmic phase. Expose displacement from its own
            # start and the raised foot geometry so the policy can stop drift.
            phase.zero_();d=self.robot.data;quat=T(d.root_quat_w)
            feet=T(d.body_pos_w).index_select(1,self.fids_t)
            fq=T(d.body_quat_w).index_select(1,self.fids_t)
            probes=quat_apply(fq[:,:,None,:].expand(-1,-1,7,-1).reshape(-1,4),self.sole_probes[None].expand(self.num_envs,-1,-1,-1).reshape(-1,3)).reshape(self.num_envs,2,7,3)+feet[:,:,None,:]
            height=probes[:,1,:,2].amin(-1)-self.scene.env_origins[:,2]
            contact=torch.stack([T(s.data.net_forces_w).norm(dim=-1).sum(-1)>.08 for s in self.feet_sensors],dim=-1)
            drift=torch.cat((T(d.root_pos_w)[:,:2]-self.start_xy,torch.zeros(self.num_envs,1,device=self.device)),dim=-1)
            extra=torch.cat((quat_apply_inverse(quat,drift),quat_apply_inverse(quat,feet[:,1]-feet[:,0]),height[:,None],contact.float()),dim=-1)
        if self.cfg.skill=='backflip' and self.cfg.backflip_objective in ('compact_flight','compact_landing','landing_upright'):
            d=self.robot.data;f=self.flip
            masses=T(d.body_mass);points=T(d.body_com_pos_w)
            com=(points*masses[:,:,None]).sum(1)/masses.sum(1)[:,None]
            com_velocity=(T(d.body_com_lin_vel_w)*masses[:,:,None]).sum(1)/masses.sum(1)[:,None]
            spread=((points[:,:,0]-com[:,None,0]).square()+(points[:,:,2]-com[:,None,2]).square()).mul(masses).sum(1)
            feet=T(d.body_pos_w).index_select(1,self.fids_t)
            extra=torch.stack(((T(d.root_pos_w)[:,2]-self.scene.env_origins[:,2])*5,com_velocity[:,2],f.angle/(2*math.pi),f.flying.float(),f.touched.float(),f.feet_first.float(),f.airtime*2,(feet[:,:,2].amin(1)-self.scene.env_origins[:,2])*5,spread/.006),dim=-1)
        return {"policy":torch.cat((obs,phase,extra,T(self.robot.data.root_lin_vel_b)),dim=-1)}

    def _measure(self):
        d = self.robot.data
        pos = T(d.root_pos_w)
        g = T(d.projected_gravity_b)
        tilt = torch.acos((-g[:,2]).clamp(-1,1))
        q = T(d.joint_pos).index_select(1,self.jids_t)
        contact = torch.stack([T(s.data.net_forces_w).norm(dim=-1).sum(dim=-1)>0.08 for s in self.feet_sensors],dim=-1)
        feet = T(d.body_pos_w).index_select(1,self.fids_t)
        return pos, tilt, q, contact, feet

    def _get_dones(self):
        self._last_measure=self._measure()
        pos,tilt,q,contact,feet = self._last_measure
        finite = torch.isfinite(T(self.robot.data.root_state_w)).all(-1) & torch.isfinite(q).all(-1)
        relative_height = pos[:,2]-self.scene.env_origins[:,2]
        if self.ball is not None:
            bp = T(self.ball.data.root_pos_w)
            bad = (tilt>math.radians(55)) | (pos[:,2]-bp[:,2]<0.17) | ((pos[:,:2]-bp[:,:2]).norm(dim=-1)>0.16)
        elif self.cfg.skill == "backflip":
            bad = (relative_height<.025) | ((self.flip.touched)&~self.flip.feet_first)
            if self.cfg.backflip_recovery_curriculum:
                bad=(relative_height<.025)|(self.flip.touched&~self.flip.feet_first&~self.flip_training_landed)
        else:
            bad = (tilt>1.0) | (relative_height<0.055)
        self.metrics = {"tilt":tilt, "root_height":relative_height, "finite":finite.float()}
        self.rollout_metrics = {"failed":bad|~finite}
        horizon=self.balance_horizon if self.ball is not None and self.cfg.basketball_recipe=='reset_curriculum' and not self.cfg.evaluation else self.max_episode_length
        return bad|~finite, self.episode_length_buf >= horizon-1

    def _get_rewards(self):
        # DirectRLEnv calls dones immediately before rewards, with no intervening
        # physics or resets. Reuse these exact sensor values.
        pos,tilt,q,contact,feet = self._last_measure
        d = self.robot.data
        v,w = T(d.root_lin_vel_w),T(d.root_ang_vel_b)
        up = torch.exp(-tilt.square()/0.09)
        pose = torch.exp(-(q-self.home).square().mean(-1)/0.2)
        rate = (self.actions-self.prev_actions).square().sum(-1)
        z = pos[:,2]-self.scene.env_origins[:,2]
        common = -0.02*w.square().sum(-1)-0.03*rate-0.00001*T(d.applied_torque).index_select(1,self.jids_t).square().sum(-1)
        if self.cfg.skill == "basketball":
            bp = T(self.ball.data.root_pos_w)
            offset = (pos[:,:2]-bp[:,:2]).square().sum(-1)
            if self.cfg.basketball_recipe in ("robust","reset_curriculum"):
                # Use the public foot SITE, not the ankle BODY origin.
                offset_local=torch.tensor([0,-.0238146,-.0140852],device=self.device).expand(self.num_envs*2,3)
                feet=feet+quat_apply(T(d.body_quat_w).index_select(1,self.fids_t).reshape(-1,4),offset_local).reshape(self.num_envs,2,3)
                common=common-.07*rate
            gap = ((feet-bp[:,None,:]).norm(dim=-1)-0.132).square().mean(-1)
            centered = torch.exp(-offset/0.04**2)
            height = torch.exp(-(pos[:,2]-bp[:,2]-0.245).square()/0.03**2)
            reward = 2*centered + up + torch.exp(-gap/0.02**2) + height + 0.3*pose + common -0.05*T(self.ball.data.root_lin_vel_w).square().sum(-1)
            self.metrics.update(center_error=offset.sqrt(), upright=up)
        elif self.cfg.skill == "arabesque":
            fq=T(d.body_quat_w).index_select(1,self.fids_t)
            probes=quat_apply(fq[:,:,None,:].expand(-1,-1,7,-1).reshape(-1,4), self.sole_probes[None].expand(self.num_envs,-1,-1,-1).reshape(-1,3)).reshape(self.num_envs,2,7,3)+feet[:,:,None,:]
            heights=probes[:,:,:,2].amin(-1)-self.scene.env_origins[:,None,2]
            relative=quat_apply_inverse(T(d.root_quat_w),feet[:,1]-feet[:,0])
            single=contact[:,0]&~contact[:,1]
            lift=torch.exp(-(heights[:,1]-.055).square()/.025**2)
            behind=torch.exp(-(relative[:,0]+.045).square()/.05**2)
            drift=(pos[:,:2]-self.start_xy).norm(dim=-1)
            held=single&(heights[:,1]>.035)&(heights[:,1]<.080)&(relative[:,0]<-.01)&(tilt<.35)&(drift<.15)
            self.pose_hold=torch.where(held,self.pose_hold+self.step_dt,torch.zeros_like(self.pose_hold))
            self.pose_max_hold=torch.maximum(self.pose_max_hold,self.pose_hold)
            reward=10*lift*up+4*single.float()*up+3*behind*lift+3*up+torch.exp(-(z-.115).square()/.025**2)+common-2*v[:,:2].square().sum(-1)-2*drift
            if self.cfg.arabesque_objective in ('static','hold'):
                masses=T(d.body_mass)
                com=(T(d.body_com_pos_w)*masses[:,:,None]).sum(1)/masses.sum(1)[:,None]
                support_error=(com[:,:2]-probes[:,0,:,:2].mean(1)).square().sum(-1)
                pose_quality=torch.exp(-(heights[:,1]-.055).square()/.015**2)*behind*up
                foot_speed=T(d.body_lin_vel_w).index_select(1,self.fids_t)[:,1].square().sum(-1)
                reward=12*pose_quality+5*single.float()*pose_quality+8*held.float()+3*torch.exp(-support_error/.035**2)*up+common
                reward-=12*drift.square()+4*v[:,:2].square().sum(-1)+.6*foot_speed
                if self.cfg.arabesque_objective=='hold':
                    # A wide discovery objective keeps lift gradients visible.
                    # Double-foot standing receives no positive support annuity.
                    pose_quality=lift*behind*up
                    reward=14*pose_quality+6*single.float()*pose_quality+10*held.float()+common
                    reward-=12*drift.square()+4*v[:,:2].square().sum(-1)+.4*foot_speed
                    reward-=2*support_error/.035**2*single.float()
                self.metrics.update(pose_quality=pose_quality,support_center_error=support_error.sqrt(),raised_foot_speed=foot_speed.sqrt())
            self.metrics.update(right_sole_height=heights[:,1],right_rear_offset=-relative[:,0],single_support=single.float(),pose_hold_seconds=self.pose_max_hold,drift=drift)
        elif self.cfg.skill == "backflip":
            t=self.episode_length_buf*self.step_dt
            f=self.flip
            frontier=(f.frontier-self.flip_paid).clamp(min=0)
            self.flip_paid.copy_(f.frontier)
            preload=torch.exp(-(z-.080).square()/.025**2)*(t<.25)
            launch=torch.exp(-(v[:,2]-1.2).square()/.6**2)*((t>=.2)&(t<.6))*~f.touched
            back_rate=(-w[:,1]).clamp(0,15)/15
            tuck=torch.exp(-(q[:,[3,12]].abs()-1.2).square().mean(-1)/.4**2)
            settling=(2*up+pose+contact.float().sum(-1))*f.touched*f.feet_first
            dense=4*preload+14*launch+4*back_rate*launch+2*tuck*f.flying+4*settling-.03*rate-.01*w[:,[0,2]].square().sum(-1)
            reward=self.step_dt*dense+12*frontier/(2*math.pi)+self.flip_step_started.float()+30*self.flip_step_succeeded.float()-3*self.reset_terminated.float()
            if self.cfg.backflip_objective in ('launch_spin','compact_flight','compact_landing','landing_upright'):
                push=((t>.15)&(t<.8))&~f.touched
                upward=v[:,2].clamp(0,1.5)
                pitch=(-w[:,1]).clamp(0,30)/20
                dense=4*preload+10*upward*push+6*pitch*f.flying+3*pitch*upward*push+2*tuck*f.flying
                dense+=4*settling-.01*w[:,[0,2]].square().sum(-1)-.01*rate
                reward=self.step_dt*dense+20*frontier/(2*math.pi)+4*self.flip_step_started.float()+40*self.flip_step_succeeded.float()-2*self.reset_terminated.float()
                self.metrics.update(upward_speed=v[:,2],backward_pitch_rate=-w[:,1])
                if self.cfg.backflip_objective in ('compact_flight','compact_landing','landing_upright'):
                    masses=T(d.body_mass);points=T(d.body_com_pos_w)
                    com=(points*masses[:,:,None]).sum(1)/masses.sum(1)[:,None]
                    # Point-mass spread is a shaping proxy, not a claim to
                    # measure the full articulated rotational inertia.
                    spread=((points[:,:,0]-com[:,None,0]).square()+(points[:,:,2]-com[:,None,2]).square()).mul(masses).sum(1)
                    compact=torch.exp(-spread/.003)*f.flying*(f.frontier<5.0)
                    com_velocity=(T(d.body_com_lin_vel_w)*masses[:,:,None]).sum(1)/masses.sum(1)[:,None]
                    peak=com_velocity[:,2].clamp(0,1.8)
                    gain=(peak-self.peak_vz).clamp(min=0)*push*~f.started
                    self.peak_vz=torch.maximum(self.peak_vz,peak)
                    near_finish=(f.frontier>5.0)&f.flying
                    braking=(up+pose)*near_finish
                    reward+=6*gain/1.2+self.step_dt*(12*compact+6*braking-.03*w[:,1].square()*near_finish)
                    self.metrics.update(pitch_mass_spread=spread,compact_score=compact,peak_com_upward_speed=self.peak_vz)
                    if self.cfg.backflip_objective in ('compact_landing','landing_upright'):
                        # Reach the complete revolution before penalizing its
                        # angular speed. Progress beyond one turn earns no extra
                        # rotation credit; the unchanged tracker gates landing.
                        full=(f.frontier>=2*math.pi)&((f.frontier-frontier)<2*math.pi)
                        turn_progress=f.frontier.clamp(max=2*math.pi)-(f.frontier-frontier).clamp(max=2*math.pi)
                        late=(f.frontier>6.15)&f.flying
                        dense=4*preload+12*com_velocity[:,2].clamp(0,1.8)*push
                        dense+=4*pitch*f.flying*(f.frontier<6.0)+12*compact+6*settling
                        dense+=8*(up+pose)*late-.002*w[:,1].square()*late
                        dense-=.01*w[:,[0,2]].square().sum(-1)+.01*rate
                        reward=self.step_dt*dense+50*turn_progress/(2*math.pi)+6*gain/1.2
                        reward+=4*self.flip_step_started.float()+30*full.float()+100*self.flip_step_succeeded.float()-5*self.reset_terminated.float()
                        if self.cfg.backflip_objective=='landing_upright':
                            # Broad first-contact feedback gives a gradient
                            # toward the original 20-degree acceptance gate.
                            # Pay it once, even if a landed episode continues.
                            new_contact=f.touched&~self.flip_contact_paid
                            self.flip_contact_paid|=f.touched
                            contact_quality=self.flip_first_feet*torch.exp(-self.flip_first_tilt.square()/.65**2)
                            if self.cfg.backflip_strict_finish:
                                contact_quality*=f.frontier.clamp(0,2*math.pi).div(2*math.pi).pow(4)
                            reward+=new_contact*(50*contact_quality+30*f.feet_first.float())
                            reward+=24*gain/1.2
                            reward+=self.step_dt*12*torch.exp(-tilt.square()/.65**2)*late
                            landed=f.touched&f.feet_first
                            hold_frontier=f.max_stable.clamp(max=.5)
                            if self.cfg.backflip_recovery_curriculum:
                                landed|=self.flip_training_landed
                                hold_frontier=torch.maximum(hold_frontier,self.flip_training_max_hold.clamp(max=.5))
                                # Teach launch alignment while preserving the
                                # community's 0.135 m start of rotation tracking.
                                launch_scale=.12 if self.cfg.backflip_strict_finish else .5
                                reward+=30*self.flip_step_started.float()*torch.exp(-self.flip_launch_tilt.square()/launch_scale**2)
                            hold_gain=(hold_frontier-self.flip_hold_paid).clamp(min=0)
                            self.flip_hold_paid.copy_(hold_frontier)
                            # Replace contact-count annuity with upright height
                            # and stillness. Partial stable holds earn progress
                            # toward the same half-second physical gate.
                            # Keep height gradients visible from deep crouches.
                            # The old narrow height Gaussian was almost zero at
                            # 5 cm and left a profitable crouching solution.
                            stand_height=torch.exp(-(z-.12).square()/.05**2)
                            settle_dense=10*up+3*pose+30*stand_height*up
                            settle_dense-=10*(.095-z).clamp(min=0)/.04
                            settle_dense-=8*v.square().sum(-1)+.1*w.square().sum(-1)
                            recovery_scale=torch.where(f.frontier>=2*math.pi,1.,.15) if self.cfg.backflip_strict_finish else 1.
                            reward+=self.step_dt*(settle_dense*landed*recovery_scale-6*settling)+40*hold_gain/.5*recovery_scale
                            if self.cfg.backflip_strict_finish:
                                reward-=self.step_dt*2*w[:,[0,2]].square().sum(-1)*f.flying
                            if self.cfg.backflip_recovery_curriculum:
                                self.metrics.update(curriculum_landed=self.flip_training_landed.float(),curriculum_stable_hold=self.flip_training_max_hold)
            self.metrics.update(flip_takeoff=f.started.float(),flip_airtime=f.airtime,airborne_rotation_deg=f.frontier*180/math.pi,feet_first=f.feet_first.float(),stable_landing_seconds=f.max_stable,backflip_success=f.success.float(),assist_force_n=torch.zeros_like(z))
            self.metrics.update(first_contact_body_index=self.flip_first_body.float(),first_contact_includes_feet=self.flip_first_feet,first_contact_tilt_degrees=self.flip_first_tilt*180/math.pi,takeoff_tilt_degrees=self.flip_launch_tilt*180/math.pi)
            self.rollout_metrics.update({k:v.clone() for k,v in self.metrics.items()})
            return reward
        elif self.cfg.skill == "dance":
            phase = (self.episode_length_buf*self.step_dt/1.2)%1.0
            tr = (phase<0.5).float()*0.55
            tl = (phase>=0.5).float()*0.55
            err = (q[:,0]-tl).square()+(q[:,9]-tr).square()
            track = torch.exp(-err/0.25**2)
            reward = 8*track + 2*torch.exp(-v[:,:2].square().sum(-1)/0.08**2) +1.5*contact.float().sum(-1)+2*up+1.5*pose+torch.exp(-(z-.115).square()/.04**2)+common
            self.metrics.update(yaw_rms=(err/2).sqrt(), tracking=track, upright=up, drift=(pos[:,:2]-self.start_xy).norm(dim=-1))
        else:
            t = self.episode_length_buf*self.step_dt
            # Independent community-style measurement: min(whole-body CoM rise
            # since takeoff, clearance of every sole underside probe). Keep the
            # original root-above-standing metric for comparison, not as a proxy.
            masses=T(d.body_mass)
            com=(T(d.body_com_pos_w)*masses[:,:,None]).sum(1)/masses.sum(1)[:,None]
            com_vel=(T(d.body_com_lin_vel_w)*masses[:,:,None]).sum(1)/masses.sum(1)[:,None]
            foot_quat=T(d.body_quat_w).index_select(1,self.fids_t)
            probes=quat_apply(foot_quat[:,:,None,:].expand(-1,-1,7,-1).reshape(-1,4), self.sole_probes[None].expand(self.num_envs,-1,-1,-1).reshape(-1,3)).reshape(self.num_envs,2,7,3)+feet[:,:,None,:]
            clearance=(probes[:,:,:,2].amin(dim=(1,2))-self.scene.env_origins[:,2]).clamp(min=0)
            pre=~self.qualified_launch
            self.support_ticks=torch.where(pre&contact.all(-1),self.support_ticks+1,self.support_ticks)
            self.ground_clearance=torch.where(pre&contact.any(-1),clearance,self.ground_clearance)
            new_launch=pre&(self.support_ticks>=2)&~contact.any(-1)&(com_vel[:,2]>.05)&(tilt<.5)
            self.takeoff_com=torch.where(new_launch,com[:,2],self.takeoff_com)
            self.qualified_launch|=new_launch
            flight=self.qualified_launch&~self.qualified_landed&~contact.any(-1)
            self.qualified_air_ticks+=flight.float()
            effective=(clearance-self.ground_clearance).clamp(min=0)
            com_rise=(com[:,2]-self.takeoff_com).clamp(min=0)
            previous_visible=self.visible_jump
            self.visible_jump=torch.maximum(self.visible_jump,torch.minimum(com_rise,effective)*flight)
            self.com_rise=torch.maximum(self.com_rise,com_rise*flight)
            self.sole_clearance=torch.maximum(self.sole_clearance,effective*flight)
            self.qualified_landed|=self.qualified_launch&contact.any(-1)
            self.metrics.update(visible_jump=self.visible_jump,com_rise=self.com_rise,sole_clearance=self.sole_clearance,qualified_airtime=self.qualified_air_ticks*self.step_dt)

            # Foot-body origin is above the sole. Contact loss must persist 80 ms;
            # rising body height without contact loss receives no flight reward.
            airborne = ~contact.any(-1) & (t>0.06)
            self.air_ticks = torch.where(airborne,self.air_ticks+1,torch.zeros_like(self.air_ticks))
            self.max_air = torch.maximum(self.max_air,self.air_ticks*self.step_dt)
            takeoff = (self.air_ticks>=4) & ~self.took_off
            self.took_off |= takeoff
            touchdown = self.took_off & contact.any(-1) & ~self.landed
            self.landed |= touchdown
            rise = (z-.125).clamp(min=0)*airborne.float()
            frontier = (rise-self.max_rise).clamp(min=0)
            self.max_rise = torch.maximum(self.max_rise,rise)
            launch = (v[:,2].clamp(min=0)-self.peak_vz).clamp(min=0)*(t<.75)*~self.landed
            self.peak_vz = torch.maximum(self.peak_vz,v[:,2].clamp(min=0))
            drift = (pos[:,:2]-self.start_xy).square().sum(-1)
            quiet = (tilt<.2)&(v.norm(dim=-1)<.12)&contact.all(-1)&(z>.095)&self.landed
            stable_bonus = (self.stable_ticks==49)&quiet
            self.stable_ticks = torch.where(quiet,self.stable_ticks+1,torch.zeros_like(self.stable_ticks))
            # Sparse progress bonuses are not multiplied by dt; dense costs are.
            reward = 2*launch/.6 + takeoff.float() + 6*frontier/.012 +3*touchdown.float()*up+6*stable_bonus.float()
            if self.cfg.jump_objective == "legacy":
                dense = up+pose
            else:
                # Reward physical objectives only. No joint targets or actions
                # are supplied by this phase schedule. Standing through the
                # launch window must not collect a large survival annuity.
                crouch_score = torch.exp(-(z-.085).square()/.025**2)*(t<.2)
                launch_score = torch.exp(-(v[:,2]-.55).square()/.3**2)*((t>=.2)&(t<.5))
                settle_score = (up+pose)*self.landed*(t>.5)
                dense = 4*crouch_score+12*launch_score+2*settle_score-0.5*tilt.square()
                self.metrics.update(crouch_score=crouch_score,launch_score=launch_score)
            if self.cfg.jump_objective == "physical":
                # Optimize the independently measured flight, not root extension.
                # A tiny bounce followed by standing cannot collect landing credit.
                valid=(self.visible_jump>=.012)&(self.qualified_air_ticks*self.step_dt>=.08)
                crouch_score=torch.exp(-(z-.075).square()/.025**2)*(t<.3)
                launch_score=torch.exp(-(com_vel[:,2]-.85).square()/.35**2)*((t>=.2)&(t<.65))*~self.qualified_landed
                dense=4*crouch_score+16*launch_score+3*(up+pose)*self.qualified_landed*valid-tilt.square()
                progress=(self.visible_jump-previous_visible).clamp(min=0)
                reward=14*progress/.025+6*new_launch.float()*com_vel[:,2].clamp(0,1.)
                reward+=4*flight.float()*self.step_dt/.12+10*stable_bonus.float()*valid
                reward+=2*touchdown.float()*up*valid
                self.metrics.update(crouch_score=crouch_score,launch_score=launch_score)
            reward += self.step_dt*(dense+common-3*drift-0.2*v[:,:2].square().sum(-1))
            self.metrics.update(foot_support=contact.float().sum(-1), base_speed=v.norm(dim=-1), settle_seconds=self.stable_ticks*self.step_dt, jump_rise=self.max_rise, airtime=self.max_air, landed=self.landed.float(), stable=(self.stable_ticks>=50).float())
            self.rollout_metrics.update({k:v.clone() for k,v in self.metrics.items()})
            return reward-2*self.reset_terminated.float()
        self.rollout_metrics.update({k:v.clone() for k,v in self.metrics.items()})
        return self.step_dt*reward-2*self.reset_terminated.float()

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if hasattr(self,"metrics"):
            for k,v in self.metrics.items():
                self.extras.setdefault("log",{})["Metrics/"+k] = v[env_ids].mean()
            self.extras.setdefault("log",{})["Metrics/episode_seconds"] = (self.episode_length_buf[env_ids].float()*self.step_dt).mean()
        super()._reset_idx(env_ids)
        n = len(env_ids)
        pose = T(self.robot.data.default_root_pose)[env_ids].clone()
        pose[:,:3] = self.scene.env_origins[env_ids]
        pose[:,2] += .368 if self.ball is not None else self.cfg.root_height
        noise = self.cfg.reset_noise*(5.0 if self.ball is not None and self.cfg.basketball_recipe in ("robust","reset_curriculum") else 1.0)
        pose[:,:2] += (torch.rand(n,2,device=self.device)-.5)*.004*noise
        angles = (torch.rand(n,3,device=self.device)-.5)*.02*noise
        if self.ball is not None and self.cfg.basketball_recipe in ("robust","reset_curriculum"):
            angles[:,2]=(torch.rand(n,device=self.device)-.5)*2*math.pi
        pose[:,3:] = quat_from_euler_xyz(angles[:,0],angles[:,1],angles[:,2])
        q = T(self.robot.data.default_joint_pos)[env_ids].clone()
        q[:,self.jids_t] += (torch.rand(n,14,device=self.device)-.5)*.01*noise
        # Reset every physical joint. A passive hinge is not a second action:
        # initialize its play to zero, safely inside every randomized stop.
        if self.passive_jids:q[:,self.passive_jids_t]=0.
        standing_only=(self.cfg.skill=='jump' and self.cfg.jump_reset=='standing') or (self.cfg.skill=='backflip' and self.cfg.backflip_reset=='standing')
        if self.cfg.skill in ("jump", "backflip") and not self.cfg.evaluation and not standing_only:
            # Community jump recipe uses standing / foot-flat crouch resets for
            # discovery. This changes only initial physical states, never actions.
            crouch = torch.rand(n,device=self.device)<0.5
            pose[crouch,2] = self.scene.env_origins[env_ids][crouch,2]+.071
            for policy_i,value in {2:.4188,3:1.3776,4:.9588,11:-.4188,12:-1.3776,13:-.9588}.items():
                q[crouch,self.jids[policy_i]] = value
        self.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n,6,device=self.device),env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=q,env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q),env_ids=env_ids)
        # Reset actuator targets as well as measured state. DirectRLEnv writes
        # targets once during reset; stale/zero absolute targets would seed the
        # BAM delay buffer with a spurious command before the first policy tick.
        self.robot.set_joint_position_target_index(
            target=encoder_channels(q,self.jids_t,self.passive_jids_t),joint_ids=self.jids_t,env_ids=env_ids)
        # Native DriveBam owns/reset its lag ring and holds the current target
        # until history fills. Do not call the old Python actuator's prime API.
        if self.ball is not None:
            bp=pose.clone();bp[:,:3]=self.scene.env_origins[env_ids];bp[:,2]+=.121;bp[:,3:]=0;bp[:,6]=1
            self.ball.write_root_pose_to_sim_index(root_pose=bp,env_ids=env_ids)
            self.ball.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n,6,device=self.device),env_ids=env_ids)
        if hasattr(self,"actions"):
            self.actions[env_ids]=0;self.prev_actions[env_ids]=0
            for name in ("max_rise","air_ticks","max_air","took_off","landed","stable_ticks","peak_vz"):
                getattr(self,name)[env_ids]=0
            self.start_xy[env_ids]=pose[:,:2]
            self.pose_hold[env_ids]=0;self.pose_max_hold[env_ids]=0
            if hasattr(self, 'flip'):
                self.flip.reset(env_ids);self.flip_last_quat[env_ids]=pose[:,3:];self.flip_paid[env_ids]=0
                self.flip_first_body[env_ids]=-1;self.flip_first_feet[env_ids]=0
                self.flip_first_tilt[env_ids]=0;self.flip_contact_paid[env_ids]=False;self.flip_hold_paid[env_ids]=0
                self.flip_training_landed[env_ids]=False;self.flip_training_hold[env_ids]=0;self.flip_training_max_hold[env_ids]=0
                self.flip_step_started[env_ids]=False;self.flip_step_succeeded[env_ids]=False;self.flip_launch_tilt[env_ids]=0
            for name in ("support_ticks","takeoff_com","ground_clearance","visible_jump","com_rise","sole_clearance","qualified_air_ticks","qualified_launch","qualified_landed"):
                getattr(self,name)[env_ids]=0
