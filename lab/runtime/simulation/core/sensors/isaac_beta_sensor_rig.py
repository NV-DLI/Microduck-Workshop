"""Separate RTX approximation of the Beta GC2093 and SPI L8CX 8x8 ToF.

Attach to the actual Newton head-link pose, including all rotation. There is no
world-stabilized gimbal, object truth, semantic label or synthetic 360 scan API.
Optics are explicit experiment assumptions. No alpha calibration is inherited.
"""
from __future__ import annotations
import math
import numpy as np
import cv2
from pxr import Gf,UsdGeom
import usdrt
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera,CameraCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_rtx_hydra_engine_attached,ensure_isaac_rtx_render_update
from beta_sensor_profile import CAMERA_PROFILE, PROFILE_ID
from beta_tof_geometry import TOF_PROFILE, sample_tof_from_depth
from beta_mounts import get_mount
from beta_head_imu import BetaHeadImu
from capture_clock import CaptureClock


def array(value):
    value=getattr(value,'torch',value)
    return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)


def qmul(a,b):
    """Hamilton product, explicit XYZW at the Isaac/Newton boundary."""
    x,y,z,w=a;X,Y,Z,W=b
    return np.asarray([w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z])


def rotate(q,v):
    q=np.asarray(q,dtype=float);q=q/np.linalg.norm(q)
    return np.asarray(v)+2*np.cross(q[:3],np.cross(q[:3],v)+q[3]*np.asarray(v))


class IsaacBetaSensors:
    def __init__(self,sim,robot,*,mount_profile, camera_profile=CAMERA_PROFILE, seed=38111,noise_std_m=0.,dropout_probability=0.,body_quaternion_order="xyzw"):
        self.mount = get_mount(mount_profile)
        if body_quaternion_order not in ("xyzw","wxyz"): raise ValueError("Explicit quaternion convention required")
        self.body_quaternion_order=body_quaternion_order
        self.imu=BetaHeadImu(seed=seed)
        self._imu_last_position=None;self._imu_last_velocity=None;self._imu_last_time=None;self._imu_batch=[]
        self.sim,self.robot,self.seed=sim,robot,seed
        self.noise_std_m,self.dropout_probability=noise_std_m,dropout_probability
        self.sequence=0;self.head_id=None
        self._capture_clock=CaptureClock()
        sim.set_setting('/isaaclab/render/rtx_sensors',True);ensure_rtx_hydra_engine_attached()
        self.profile=camera_profile
        self.width,self.height=self.profile.width_px,self.profile.height_px
        self.k=np.asarray(self.profile.K,dtype=np.float64)
        self.d=np.asarray(self.profile.distortion_k1_k2_p1_p2_k3,dtype=np.float64)
        # RTX renders an undistorted overscan. An inverse Brown-Conrady map then
        # samples actual rendered pixels into the assumed Beta camera's distorted raw
        # image; overscan avoids inventing black borders from missing input FOV.
        uu,vv=np.meshgrid(np.arange(self.width),np.arange(self.height))
        points=np.stack((uu,vv),axis=-1).astype(np.float64).reshape(-1,1,2)
        undistorted=cv2.undistortPoints(points,self.k,self.d).reshape(self.height,self.width,2)
        # Isaac's USD adapter discards off-centre principal points and averages
        # fx/fy. Render a centred, square-pixel pinhole it can represent exactly;
        # all asymmetric K and lens distortion then reside in this inverse map.
        self.render_focal_px=float((self.k[0,0]+self.k[1,1])/2)
        self.render_width=2*int(math.ceil(float(np.abs(undistorted[:,:,0]).max())*self.render_focal_px+4))
        self.render_height=2*int(math.ceil(float(np.abs(undistorted[:,:,1]).max())*self.render_focal_px+4))
        if not 0<self.render_width<=4096 or not 0<self.render_height<=4096:
            raise ValueError('Assumed Beta lens requires an invalid overscan size')
        render_k=np.asarray([[self.render_focal_px,0,self.render_width/2],[0,self.render_focal_px,self.render_height/2],[0,0,1]])
        self.map_x=(undistorted[:,:,0]*self.render_focal_px+self.render_width/2).astype(np.float32)
        self.map_y=(undistorted[:,:,1]*self.render_focal_px+self.render_height/2).astype(np.float32)
        # RTX partitions clone geometry by env; cameras must belong to the same
        # environment or they see only the shared ground. This rig is single-env.
        rgb_path='/World/envs/env_0/BetaSensors/RawCamera'
        self.rgb_camera=Camera(CameraCfg(prim_path=rgb_path,update_period=0.,
            width=self.render_width,height=self.render_height,data_types=['rgb'],
            renderer_cfg=IsaacRtxRendererCfg(),spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                render_k.flatten().tolist(),self.render_width,self.render_height,clipping_range=(.03,20.),focal_length=15.)))
        # A distinct optical origin and 32x32 RTX depth product. A 50-degree
        # pinhole view covers every subray of the official 45x45 spherical FoV.
        self.tof_resolution=32;self.tof_render_fov=50.
        focal=self.tof_resolution/(2*math.tan(math.radians(self.tof_render_fov)/2))
        tof_k=[focal,0,16.,0,focal,16.,0,0,1]
        tof_path='/World/envs/env_0/BetaSensors/TofDepth'
        self.tof_camera=Camera(CameraCfg(prim_path=tof_path,update_period=0.,
            width=32,height=32,data_types=['distance_to_camera'],
            renderer_cfg=IsaacRtxRendererCfg(),spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                tof_k,32,32,clipping_range=(.03,4.5),focal_length=15.)))
        self.paths=(rgb_path,tof_path);self.matrices={};self.poses={}

    def reset(self,*,timestamp_s=0.0):
        self.rgb_camera.reset();self.tof_camera.reset()
        self._capture_clock.reset(timestamp_s)
        self.imu.reset();self._imu_last_position=None;self._imu_last_velocity=None;self._imu_last_time=None;self._imu_batch=[]
        self._enable_color_render()

    def _enable_color_render(self):
        # Isaac Sim 6 / Lab 3's depth-only camera initializer changes this
        # setting globally, blacking out an earlier RGB render product. Both
        # sensors coexist here, so restore color after all cameras initialize.
        self.sim.set_setting('/rtx/sdg/force/disableColorRender',False)

    def _publish(self,path,matrix):
        UsdGeom.Xformable(self.sim.stage.GetPrimAtPath(path)).MakeMatrixXform().Set(matrix)
        rt_stage=usdrt.Usd.Stage.Attach(sim_utils.get_current_stage_id())
        transform=usdrt.Rt.Xformable(rt_stage.GetPrimAtPath(path))
        transform.SetLocalXformFromUsd();transform.SetWorldXformFromUsd()
        rt_matrix=usdrt.Gf.Matrix4d(*[float(matrix[i][j]) for i in range(4) for j in range(4)])
        transform.CreateFabricHierarchyLocalMatrixAttr().Set(rt_matrix)
        transform.CreateFabricHierarchyWorldMatrixAttr().Set(rt_matrix)

    def _place(self,camera,path,position,q_ros):
        camera.set_world_poses(positions=np.asarray(position)[None,:],orientations=np.asarray(q_ros)[None,:],convention='ros')
        # USD/OpenGL looks down -Z with +Y up; preserve the entire head rotation,
        # unlike look-at with world-up which would silently stabilize head roll.
        q_gl=qmul(q_ros,[1.,0.,0.,0.]);q_gl/=np.linalg.norm(q_gl)
        matrix=Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Quatd(float(q_gl[3]),Gf.Vec3d(*q_gl[:3].tolist()))))
        matrix.SetTranslateOnly(Gf.Vec3d(*np.asarray(position).tolist()))
        self.matrices[path]=matrix;self._publish(path,matrix)
        self.poses[path]={'position_world_m':np.asarray(position).tolist(),'quaternion_ros_xyzw':np.asarray(q_ros).tolist()}

    def _head_pose(self):
        if self.head_id is None:
            ids,_=self.robot.find_bodies([self.mount.body_name],preserve_order=True)
            if len(ids)!=1:raise RuntimeError(f'Expected {self.mount.body_name} for explicit mount {self.mount.name}; inspect the asset and select its matching profile')
            self.head_id=ids[0]
        head=array(self.robot.data.body_pos_w)[0,self.head_id]
        head_q=array(self.robot.data.body_quat_w)[0,self.head_id].astype(float)
        if self.body_quaternion_order=='wxyz':head_q=head_q[[1,2,3,0]]
        head_q/=np.linalg.norm(head_q)
        return head,head_q

    def update_mount(self):
        head,head_q=self._head_pose()
        site_q=np.asarray(self.mount.site_quaternion_xyzw)
        upright_q=qmul(qmul(head_q,site_q),self.mount.camera_site_to_ros_xyzw)
        # Beta's native 0-degree mount: raw and upright use identical axes.
        camera_pos=head+rotate(head_q,self.mount.camera_position_body_m)
        tof_pos=head+rotate(head_q,self.mount.tof_position_body_m)
        self._place(self.rgb_camera,self.paths[0],camera_pos,upright_q)
        self._place(self.tof_camera,self.paths[1],tof_pos,upright_q)
        return dict(self.poses)

    def update_imu(self,*,timestamp_s):
        """Call each physics tick; 60 Hz samples are delivered in 15 Hz batches.

        Finite differences of the chosen CAD IMU site include head lever-arm
        acceleration. The first two ticks warm the derivative; they do not
        synthesize acceleration samples. Returned fields have no world pose.
        """
        self._imu_batch=[]
        head,head_q=self._head_pose()
        position=head+rotate(head_q,self.mount.imu_position_body_m)
        if self._imu_last_time is not None:
            dt=timestamp_s-self._imu_last_time
            if dt<=0:raise ValueError('IMU update requires advancing physics time')
            velocity=(position-self._imu_last_position)/dt
            if self._imu_last_velocity is not None:
                acceleration=(velocity-self._imu_last_velocity)/dt
                angular=array(self.robot.data.body_ang_vel_w)[0,self.head_id]
                q_flu=qmul(head_q,self.mount.site_quaternion_xyzw)
                self._imu_batch=self.imu.poll(timestamp_s,q_world_from_head_xyzw=q_flu,
                    angular_velocity_world_rad_s=angular,linear_acceleration_world_m_s2=acceleration)
            self._imu_last_velocity=velocity
        self._imu_last_position=position;self._imu_last_time=timestamp_s
        return list(self._imu_batch)

    def capture(self,*,timestamp_s,render_subframes=3):
        capture_elapsed_s=self._capture_clock.elapsed(timestamp_s)
        self._enable_color_render()
        self.update_mount()
        for _ in range(render_subframes):
            self.sim.render()
            for path,matrix in self.matrices.items():self._publish(path,matrix)
            ensure_isaac_rtx_render_update()
        # These products are sampled at 15 Hz, independently of the 200 Hz
        # physics/IMU loop. Their internal clocks must advance by real elapsed
        # simulation time, including the first capture since reset.
        self.rgb_camera.update(capture_elapsed_s,force_recompute=True)
        self.tof_camera.update(capture_elapsed_s,force_recompute=True)
        rendered=array(self.rgb_camera.data.output['rgb'])[0,:,:,:3]
        depth=array(self.tof_camera.data.output['distance_to_camera'])[0].squeeze()
        if rendered.shape!=(self.render_height,self.render_width,3) or depth.shape!=(32,32):
            raise RuntimeError(f'Invalid real RTX products: RGB {rendered.shape}; ToF depth {depth.shape}')
        raw=cv2.remap(rendered,self.map_x,self.map_y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        tof=sample_tof_from_depth(depth,profile=TOF_PROFILE,depth_kind='ray_range',
            source_horizontal_fov_deg=50.,source_vertical_fov_deg=50.,samples_per_zone=4,
            noise_std_m=self.noise_std_m,dropout_probability=self.dropout_probability,
            seed=self.seed+self.sequence,timestamp_s=timestamp_s)
        self.sequence+=1
        self._capture_clock.commit(timestamp_s)
        return {'rgb_raw':raw,'rgb_upright':raw.copy(),
                'tof_source_ray_range':depth.copy(),'tof':tof.to_dict(),
                'simulated_mount_poses':dict(self.poses),'timestamp_s':timestamp_s,'capture_elapsed_s':capture_elapsed_s,
                'head_imu':self.imu.latest,'hardware_profile_id':PROFILE_ID}

    def metadata(self):
        return {'kind':'beta_rtx_sensor_approximation','hardware_profile_id':PROFILE_ID,
            'camera':{**self.profile.to_dict(),'distortion_application':'inverse Brown-Conrady remap of real RTX overscan using explicit assumed D'},
            'camera_upright_display':self.profile.display_intrinsics(),'tof':TOF_PROFILE.to_dict(),
            'head_imu':self.imu.metadata(),'mount':self.mount.to_dict(),
            'body_quaternion_order':self.body_quaternion_order,'camera_api_quaternion_order':'xyzw',
            'rtx_scene_partition':'env_0 (single-environment sensor rig)',
            'camera_clock':'actual capture-to-capture simulation elapsed time; first capture from explicit reset timestamp (default zero)',
            'world_stabilized_gimbal':False,'independent_tof_optical_origin':True,
            'rgb_raw_resolution':[self.width,self.height],'rgb_upright_rotation':'identity; Beta native 0 degrees',
            'rtx_undistorted_overscan_resolution':[self.render_width,self.render_height],
            'rtx_pinhole_intrinsics':{'fx':self.render_focal_px,'fy':self.render_focal_px,'cx':self.render_width/2,'cy':self.render_height/2},
            'rgb_depth_coexistence':'global RTX disableColorRender false after depth camera initialization',
            'tof_rtx_depth_resolution':[32,32],'tof_render_fov_deg':[50.,50.],
            'tof_sampling':'16 spherical subrays per 8x8 zone from independent RTX range product',
            'near_clip_m':.03,'near_clip_reason':'avoid decorative opaque lens housing; objects within 3 cm are not observed',
            'noise_assumptions':{'range_noise_std_m':self.noise_std_m,'zone_dropout_probability':self.dropout_probability,'origin':'explicit course assumptions, not measured hardware noise'},
            'missing_models':['exposure and motion blur','rolling shutter','auto exposure','ToF photon/reflectance/ambient IR interference','device crosstalk','ST SFLP firmware'],
            'controller_boundary':'simulated_mount_poses is renderer/debug metadata, not admissible navigation localization',
            'semantic_truth_available':False,'autonomous_navigation_validated':False}
