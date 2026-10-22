"""Meaningful geometry/frame/cadence regressions; no GPU or robot success claim."""
from dataclasses import replace
import math,sys,unittest,xml.etree.ElementTree as ET
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from beta_sensor_profile import BetaCameraProfile,BetaHeadImuProfile
from beta_perception_geometry import raw_to_upright,upright_to_raw,raw_pixel_to_unit_ray,GeometryError
from beta_tof_geometry import TOF_PROFILE,tof_zone_rays,sample_tof_from_depth,tof_to_forward_scan
from beta_head_imu import BetaHeadImu,rotate,quaternion_between
from beta_mounts import MOUNTS

class CameraTests(unittest.TestCase):
    def test_beta_never_borrows_alpha_calibration(self):
        p=BetaCameraProfile();d=p.to_dict()
        self.assertFalse(d['calibrated']);self.assertIsNone(d['physical_K']);self.assertIsNone(d['physical_D'])
        self.assertEqual(p.mount_rotation_cw_deg,0);self.assertEqual(d['profile_id'],'beta-gc2093-l8cx-v1')
        self.assertEqual(p.display_intrinsics()['pixel_operation'],'identity')
    def test_identity_is_correct_for_landscape_corners(self):
        p=BetaCameraProfile()
        for uv in ((0.,0.),(1919.,1079.),(1919.,0.),(959.5,539.5)):
            self.assertEqual(raw_to_upright(uv,p),uv);self.assertEqual(upright_to_raw(uv,p),uv)
    def test_reject_alpha_rotation(self):
        with self.assertRaises(ValueError):BetaCameraProfile(mount_rotation_cw_deg=90)
    def test_scale_keeps_principal_centre_and_fov(self):
        p=BetaCameraProfile();k=p.intrinsics_at(960,540);q=BetaCameraProfile(width_px=960,height_px=540)
        self.assertEqual(k,q.K)
    def test_fov_edge_ray_and_optical_centre(self):
        for fov in (60.,70.,80.,90.):
            p=BetaCameraProfile(assumed_horizontal_fov_deg=fov)
            r=raw_pixel_to_unit_ray((p.cx,p.cy),p);self.assertEqual(r,(0.,0.,1.))
            r=raw_pixel_to_unit_ray((p.width_px-1,p.cy),p)
            expected=math.atan(((p.width_px-1)/2)/p.fx)
            self.assertAlmostEqual(math.atan2(r[0],r[2]),expected,places=12)
    def test_brown_inverse_roundtrip(self):
        for fov in (60.,70.,80.,90.):
            p=BetaCameraProfile(assumed_horizontal_fov_deg=fov,distortion_k1_k2_p1_p2_k3=(-.08,.01,.002,-.002,0.))
            for x,y in ((.2,.1),(-.3,.25),(.4,-.2)):
                r2=x*x+y*y;k1,k2,p1,p2,k3=p.distortion_k1_k2_p1_p2_k3
                radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
                uv=(p.fx*(x*radial+2*p1*x*y+p2*(r2+2*x*x))+p.cx,p.fy*(y*radial+p1*(r2+2*y*y)+2*p2*x*y)+p.cy)
                r=raw_pixel_to_unit_ray(uv,p);self.assertAlmostEqual(r[0]/r[2],x,places=10);self.assertAlmostEqual(r[1]/r[2],y,places=10)
    def test_bad_optics_rejected(self):
        for kwargs in ({'assumed_horizontal_fov_deg':float('nan')},{'width_px':720,'height_px':1280},{'fps':60.}):
            with self.assertRaises(ValueError):BetaCameraProfile(**kwargs)

class ToFTests(unittest.TestCase):
    def test_official_beta_cadence_sector_and_bus(self):
        p=TOF_PROFILE.to_dict();self.assertEqual((p['rows'],p['cols'],p['fps']),(8,8,15.));self.assertEqual(p['bus'],'SPI3')
        self.assertEqual(p['sensor_generations'],('VL53L8CX',));self.assertEqual(p['horizontal_fov_deg'],45.)
        for ray in tof_zone_rays():self.assertAlmostEqual(sum(x*x for x in ray),1.)
    def test_wall_range_not_axial_range(self):
        f=sample_tof_from_depth([[1.]*32 for _ in range(32)],depth_kind='camera_z')
        self.assertTrue(all(c.status=='valid' and c.range_m>=1. for row in f.cells for c in row))
        self.assertGreater(f.cells[0][0].range_m,f.cells[3][3].range_m)
    def test_no_return_is_never_free_space(self):
        f=sample_tof_from_depth([[float('inf')]*32 for _ in range(32)])
        scan=tof_to_forward_scan(f);self.assertEqual(scan['ranges_m'],[None]*8);self.assertFalse(scan['free_space_clearing_allowed'])
    def test_too_close_is_invalid(self):
        f=sample_tof_from_depth([[.02]*32 for _ in range(32)])
        self.assertTrue(all(c.status=='invalid' and c.contains_too_close for row in f.cells for c in row))

class ImuTests(unittest.TestCase):
    def profile(self):return BetaHeadImuProfile(gyro_bias_rad_s=(0.,0.,0.),gyro_noise_std_rad_s=0.,accel_noise_std_m_s2=0.)
    def settle(self,imu,q=(0.,0.,0.,1.),start=0.):
        for i in range(21):
            imu.poll(start+i/60,q_world_from_head_xyzw=q,angular_velocity_world_rad_s=(0.,0.,0.),linear_acceleration_world_m_s2=(0.,0.,0.))
        return imu.latest
    def test_beta_measured_mount_axes(self):
        q=(.5,-.5,-.5,.5) # official WXYZ .5,.5,-.5,-.5 -> XYZW
        self.assertEqual(rotate(q,(1.,2.,3.)),(-3.,-1.,2.))
    def test_world_yaw_is_not_leaked(self):
        for yaw in (0.,.8,-1.2):
            imu=BetaHeadImu(profile=self.profile());q=(0.,0.,math.sin(yaw/2),math.cos(yaw/2))
            out=self.settle(imu,q)
            self.assertTrue(out['orientation_valid'])
            self.assertEqual(out['quat'],[1.,0.,0.,0.]);self.assertAlmostEqual(out['accel'][2],9.80665,places=2)
            self.assertFalse(out['absolute_world_yaw'])
    def test_rate_60_with_fifo_15_and_original_timestamps(self):
        imu=BetaHeadImu(profile=self.profile());batches=[]
        for tick in range(201):
            out=imu.poll(tick/200,q_world_from_head_xyzw=(0.,0.,0.,1.),angular_velocity_world_rad_s=(0.,0.,0.),linear_acceleration_world_m_s2=(0.,0.,0.))
            if out:batches.append(out)
        self.assertEqual(imu.seq,61);self.assertEqual(len(batches),16);self.assertEqual(imu.missed,0)
        frames=[f for batch in batches for f in batch];self.assertEqual(len(frames),61)
        self.assertTrue(any(f['fifo_latency_s']>.04 for f in frames));self.assertTrue(all(0<=f['fifo_latency_s']<=.07 for f in frames))
    def test_late_poll_does_not_fabricate_readings(self):
        imu=BetaHeadImu(profile=self.profile())
        for t in (0.,1.):imu.poll(t,q_world_from_head_xyzw=(0.,0.,0.,1.),angular_velocity_world_rad_s=(0.,0.,0.),linear_acceleration_world_m_s2=(0.,0.,0.))
        self.assertEqual(imu.seq,2);self.assertEqual(imu.missed,59)
    def test_gyro_integrates_local_yaw(self):
        imu=BetaHeadImu(profile=self.profile())
        self.settle(imu)
        for i in range(1,61):imu.poll((20+i)/60,q_world_from_head_xyzw=(0.,0.,0.,1.),angular_velocity_world_rad_s=(0.,0.,.5),linear_acceleration_world_m_s2=(0.,0.,0.))
        w,x,y,z=imu.latest['quat'];self.assertAlmostEqual(2*math.atan2(z,w),.5,places=3)
    def test_orientation_tracks_tilt_without_magnetometer(self):
        imu=BetaHeadImu(profile=self.profile());angle=.3;q=(math.sin(angle/2),0.,0.,math.cos(angle/2))
        self.settle(imu,q)
        w,x,y,z=imu.latest['quat'];self.assertAlmostEqual(2*math.atan2(x,w),angle,places=3)
    def test_noisy_freefall_never_initializes_orientation(self):
        # Original bug: the direction of ~zero-g noise became a fake tilt.
        for seed in range(8):
            imu=BetaHeadImu(seed=seed)
            for i in range(40):
                imu.poll(i/60,q_world_from_head_xyzw=(.3,0.,0.,math.sqrt(.91)),
                    angular_velocity_world_rad_s=(0.,0.,0.),linear_acceleration_world_m_s2=(0.,0.,-9.80665))
                self.assertFalse(imu.latest['orientation_valid']);self.assertIsNone(imu.latest['quat'])
    def test_freefall_then_landing_impulse_then_stable_initializes(self):
        imu=BetaHeadImu(seed=17)
        for i in range(12):
            imu.poll(i/60,q_world_from_head_xyzw=(0.,0.,0.,1.),angular_velocity_world_rad_s=(0.,0.,0.),
                linear_acceleration_world_m_s2=(0.,0.,-9.80665 if i<10 else 9.80665))
        self.assertFalse(imu.latest['orientation_valid'])
        for i in range(11):
            imu.poll((12+i)/60,q_world_from_head_xyzw=(0.,0.,0.,1.),angular_velocity_world_rad_s=(0.,0.,0.),linear_acceleration_world_m_s2=(0.,0.,0.))
            self.assertFalse(imu.latest['orientation_valid'])
        out=self.settle(imu,start=23/60)
        self.assertTrue(out['orientation_valid']);self.assertGreaterEqual(out['orientation_initialized_at_s'],23/60)
        w,x,y,z=out['quat'];up=rotate((x,y,z,w),(0.,0.,1.))
        self.assertGreater(up[2],.9999)
    def test_noisy_tilt_is_estimated_from_accel_not_world_yaw(self):
        angle=.35;q=(math.sin(angle/2),0.,0.,math.cos(angle/2))
        for seed in range(8):
            out=self.settle(BetaHeadImu(seed=seed),q)
            self.assertTrue(out['orientation_valid'])
            w,x,y,z=out['quat'];self.assertAlmostEqual(2*math.atan2(x,w),angle,delta=.006)
    def test_high_gyro_and_unstable_near_one_g_do_not_initialize(self):
        imu=BetaHeadImu(profile=self.profile())
        for i in range(30):
            imu.poll(i/60,q_world_from_head_xyzw=(0.,0.,0.,1.),angular_velocity_world_rad_s=(0.,0.,.5),linear_acceleration_world_m_s2=(0.,0.,0.))
        self.assertFalse(imu.latest['orientation_valid'])
        for i in range(30,60):
            # Magnitude is exactly one g, but direction alternates too quickly.
            tilt=.12 if i%2 else -.12
            imu.poll(i/60,q_world_from_head_xyzw=(0.,0.,0.,1.),angular_velocity_world_rad_s=(0.,0.,0.),
                linear_acceleration_world_m_s2=(9.80665*math.sin(tilt),0.,9.80665*(math.cos(tilt)-1)))
        self.assertFalse(imu.latest['orientation_valid'])
        self.assertTrue(self.settle(imu,start=1.)['orientation_valid'])
    def test_dynamic_impulse_after_init_uses_gyro_without_tilt_snap(self):
        imu=BetaHeadImu(profile=self.profile());self.settle(imu)
        before=imu.latest['quat']
        for i in range(21,27):
            imu.poll(i/60,q_world_from_head_xyzw=(0.,0.,0.,1.),angular_velocity_world_rad_s=(0.,0.,0.),linear_acceleration_world_m_s2=(15.,0.,0.))
            self.assertTrue(imu.latest['orientation_valid']);self.assertEqual(imu.latest['quat'],before)
    def test_sample_gap_invalidates_until_another_stable_window(self):
        imu=BetaHeadImu(profile=self.profile());self.settle(imu)
        imu.poll(1.,q_world_from_head_xyzw=(0.,0.,0.,1.),angular_velocity_world_rad_s=(0.,0.,0.),linear_acceleration_world_m_s2=(0.,0.,0.))
        self.assertFalse(imu.latest['orientation_valid']);self.assertIsNone(imu.latest['quat'])
        self.assertTrue(self.settle(imu,start=1+1/60)['orientation_valid'])

class MountTests(unittest.TestCase):
    def test_mounted_positions_match_their_pinned_asset(self):
        sources=Path(__file__).resolve().parents[1]/'sources'
        for name,filename in [('rl-jaw-soft-20261012','robot_allcollisions.xml'),('runtime-bottom-head-shell-20261017','runtime_robot_walk.xml')]:
            m=MOUNTS[name];b=next(b for b in ET.parse(sources/filename).getroot().iter('body') if b.get('name')==m.body_name)
            for field,site_name in [('camera_position_body_m','head_camera'),('tof_position_body_m','tof'),('imu_position_body_m','head_imu')]:
                site=next(s for s in b.findall('site') if s.get('name')==site_name)
                self.assertEqual(getattr(m,field),tuple(float(x) for x in site.get('pos').split()))

if __name__=='__main__':unittest.main()
