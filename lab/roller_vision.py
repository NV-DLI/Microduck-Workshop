"""Experimental roller visual servo: RGB/ToF + encoders and trunk IMU only.

This controller does not reuse walking sole odometry or its step controller.
It is a separate, limited empty-field experiment, not a Nav2 deployment.
"""

import functools
import json
import math
from pathlib import Path
import numpy as np
from roller_imu import turn_command


def create_sensors(env, env_ids):
    from beta_sensor_profile import BetaCameraProfile
    from isaac_beta_sensor_rig import IsaacBetaSensors

    env.beta_rig = IsaacBetaSensors(
        env.sim,
        env.scene["robot"],
        mount_profile="rl-jaw-soft-20261012",
        camera_profile=BetaCameraProfile(
            width_px=960, height_px=540, assumed_horizontal_fov_deg=70
        ),
        seed=int(env.cfg.seed) + 53959,
    )


class RollerVision:
    def __init__(self, env, deployment, output):
        import perception
        from scene_goal_points import extract_scene_goal_bases
        from vision import known_goal_geometry
        from detector_client import DetectorSession
        from head_adapter import HeadAdapter

        self.env = env
        self.output = Path(output)
        manifest = json.loads(Path(deployment).read_text())
        for key in ("weights", "onnx"):
            if key in manifest and not Path(manifest[key]).is_absolute():
                manifest[key] = str((Path(deployment).parent / manifest[key]).resolve())
        resolved = self.output / "detector-deployment.json"
        resolved.write_text(json.dumps(manifest, indent=2))
        self.detector = DetectorSession(
            resolved, self.output / "detector", backend="torch", device="cuda:0"
        )
        original = perception.estimate_box_sphere

        @functools.wraps(original)
        def estimate(*args, **kwargs):
            kwargs["radius_m"] = 0.05
            return original(*args, **kwargs)

        perception.estimate_box_sphere = estimate
        perception.extract_beta_goal_bases = extract_scene_goal_bases
        known_goal_geometry.POST_SEPARATION_M = 0.394
        self.vision = perception.BetaPerception(camera=env.beta_rig.profile)
        self.head = HeadAdapter()
        self.command = [0.0, 0.0, 0.0]
        self.head_offsets = [0.0] * 4
        self.goal_heading = None
        self.goal_at = -1e9
        self.ball = None
        self.ball_at = -1e9
        self.speed = 0.0
        self.optical_speed = 0.0
        self.last_ball = None
        self.last_ball_at = None
        self.quiet_since = None
        self.kick_at = None
        self.state = "DETECT_GOAL"
        self.failure = None
        self.log = (self.output / "visual-control.jsonl").open("x")
        self.frames = self.output / "sensor-frames"
        self.frames.mkdir()
        self.count = 0
        self.last_capture = -1e9
        self.last_t = 0.0
        self.lost_braked_at = None
        self.brake_until = None
        self.settle_until = 0.0
        self.gaze = np.asarray([0.0, 0.0])
        self.gaze_object = None

    def step(self, t, observation, inertial_velocity_body_m_s=None):
        """Accept the explicit encoder/IMU schema; simulator object poses are absent."""
        from perception import validate_observation, level_rotation
        from PIL import Image

        validate_observation(observation)
        if self.goal_heading is not None:
            self.goal_heading -= observation["gyro_body_rad_s"][2] * (t - self.last_t)
        self.last_t = t
        if t - self.last_capture < 1 / 15:
            return self.command, self.head_offsets, False
        self.last_capture = t
        capture = self.env.beta_rig.capture(timestamp_s=t, render_subframes=4)
        rgb = capture["rgb_raw"]
        detected = self.detector.detect(rgb)
        geometry = self.vision.estimate(
            rgb, observation, detected["detections"], now_s=t, tof_frame=capture["tof"]
        )
        R = level_rotation(observation["gravity_body"])
        tof_valid = bool(
            geometry.get("floor_observation")
            and geometry["floor_observation"].get("accepted")
        )
        fresh_ball = (
            tof_valid and geometry.get("ball") and geometry["ball"].get("accepted")
        )
        if fresh_ball:
            self.ball = R @ np.asarray(
                geometry["ball"]["geometry"]["point_reference_m"]
            )
            self.ball_at = t
            if self.last_ball is not None and 0 < t - self.last_ball_at < 0.3:
                measured = (self.last_ball[0] - self.ball[0]) / (t - self.last_ball_at)
                self.optical_speed = 0.8 * self.optical_speed + 0.2 * float(
                    np.clip(measured, -1, 1)
                )
            self.last_ball, self.last_ball_at = self.ball.copy(), t
        self.speed = (
            self.optical_speed
            if inertial_velocity_body_m_s is None
            else float(inertial_velocity_body_m_s[0])
        )
        fresh_goal = (
            tof_valid and geometry.get("goal") and geometry["goal"].get("accepted")
        )
        if fresh_goal:
            goal = R @ np.asarray(
                geometry["goal"]["geometry"]["goal_center_reference_m"]
            )
            direction = goal[:2] - (
                self.ball[:2]
                if self.ball is not None and t - self.ball_at < 0.2
                else [0, 0]
            )
            self.goal_heading = math.atan2(direction[1], direction[0])
            self.goal_at = t
        command = [0.0, 0.0, 0.0]
        kick = False
        pitch = 0.35
        if self.kick_at is not None:
            self.state = "RECOVER"
            pitch = 0.2
        elif self.failure:
            self.state = "STOP"
        elif t < 2.0:
            self.state = "INITIAL_STAND"
        elif self.brake_until is not None and t < self.brake_until:
            self.state = "BRAKE"
            command[0] = -0.25
        elif t < self.settle_until:
            self.state = "SETTLE"
        elif t - self.goal_at > 6.0:
            self.state = "DETECT_GOAL"
            pitch = -0.18
        elif not fresh_ball:
            self.state = "DETECT_BALL"
            pitch = 0.7
            if self.speed > 0.05 and self.lost_braked_at != self.ball_at:
                self.brake_until = t + 0.12
                self.settle_until = t + 0.5
                self.lost_braked_at = self.ball_at
                self.state = "BRAKE"
                command[0] = -0.25
        else:
            x, y = self.ball[:2]
            pitch = min(1.15, math.atan2(0.15, max(0.07, x)))
            if self.brake_until is not None and t < self.brake_until:
                self.state = "BRAKE"
                command[0] = -0.25
            elif t < self.settle_until:
                self.state = "SETTLE"
            elif x < 0.09:
                self.failure = "ball_passed_safe_kick_window"
                self.state = "STOP"
            elif x < 0.40 and self.speed > 0.05:
                self.brake_until = t + 0.12
                self.settle_until = t + 0.5
                self.state = "BRAKE"
                command[0] = -0.25
            elif x < 0.40 and abs(self.goal_heading) >= 0.06:
                self.state = "ALIGN_GOAL"
                command[2] = turn_command(
                    self.goal_heading, observation["gyro_body_rad_s"][2]
                )
            elif (
                0.16 <= x <= 0.32
                and abs(y + 0.045) < 0.015
                and abs(self.goal_heading) < 0.06
            ):
                self.state = "AIM"
                quiet = (
                    abs(self.speed) < 0.035
                    and np.linalg.norm(observation["gyro_body_rad_s"]) < 0.25
                )
                self.quiet_since = (
                    (t if self.quiet_since is None else self.quiet_since)
                    if quiet
                    else None
                )
                if self.quiet_since is not None and t - self.quiet_since >= 0.24:
                    self.kick_at = t
                    kick = True
                    self.state = "KICK"
            elif x < 0.23 and self.speed > 0.08:
                self.brake_until = t + 0.12
                self.settle_until = t + 0.5
                self.state = "BRAKE"
                command[0] = -0.25
            else:
                self.state = "APPROACH"
                heading = math.atan2(y + 0.045, max(0.06, x - 0.11))
                command[2] = turn_command(heading, observation["gyro_body_rad_s"][2])
                command[0] = 0.6 if x > 0.30 else 0.3
                if abs(heading) > 0.4:
                    command[0] = 0.0
        if self.state != "AIM":
            self.quiet_since = None
        target_class = "goal" if self.state == "DETECT_GOAL" else "ball"
        if target_class != self.gaze_object:
            self.gaze_object = target_class
            if target_class == "goal":
                self.gaze = np.asarray([0.0, -0.15])
            elif self.ball is not None:
                self.gaze = np.asarray(
                    [
                        math.atan2(self.ball[1], self.ball[0]),
                        math.atan2(0.18, max(0.07, self.ball[0] - 0.04)),
                    ]
                )
            else:
                self.gaze = np.asarray([0.0, 0.35])
        boxes = [d for d in detected["detections"] if d["class_name"] == target_class]
        if boxes and self.state not in ("RECOVER", "STOP"):
            box = max(boxes, key=lambda d: d["confidence"])["bbox_xyxy_px"]
            u = (box[0] + box[2]) / 2
            v = (box[1] + box[3]) / 2
            # Image-based gaze feedback also handles a cropped object before
            # metric geometry is accepted; it never invents its hidden boundary.
            self.gaze += [
                -0.08 * math.atan2(u - 480, 685.5),
                0.08 * math.atan2(v - 270, 685.5),
            ]
            self.gaze = np.clip(self.gaze, [-0.45, -0.30], [0.45, 1.15])
        if not boxes and self.state not in ("RECOVER", "STOP"):
            if target_class == "ball" and self.ball is not None:
                self.gaze = np.asarray(
                    [
                        math.atan2(self.ball[1], self.ball[0]),
                        math.atan2(0.18, max(0.07, self.ball[0] - 0.04)),
                    ]
                )
            else:
                self.gaze[1] = (
                    (0.55 + 0.30 * math.sin(2 * t))
                    if target_class == "ball"
                    else (-0.15 + 0.15 * math.sin(2 * t))
                )
            self.gaze = np.clip(self.gaze, [-0.45, -0.30], [0.45, 1.15])
        solution = self.head.solve(self.gaze)
        if not solution["accepted"]:
            self.failure = "head_inverse_kinematics_rejected"
            command = [0.0, 0.0, 0.0]
            kick = False
        target = (
            solution["head_pose_native_offsets_rad"]
            if solution["accepted"]
            else self.head_offsets
        )
        self.head_offsets = (
            np.clip(np.asarray(target) - self.head_offsets, -0.045, 0.045)
            + self.head_offsets
        )
        self.head_offsets = self.head_offsets.tolist()
        self.command = command
        filename = f"{self.count:05d}.png"
        Image.fromarray(rgb).save(self.frames / filename)
        self.log.write(
            json.dumps(
                dict(
                    timestamp_s=t,
                    state=self.state,
                    command=command,
                    head_offsets=self.head_offsets,
                    head_ik=solution,
                    raw_tof=capture["tof"],
                    kick_request=kick,
                    goal_age_s=t - self.goal_at,
                    goal_heading_rad=self.goal_heading,
                    ball_body_level_m=None if self.ball is None else self.ball.tolist(),
                    ball_age_s=t - self.ball_at,
                    optical_forward_speed_m_s=self.optical_speed,
                    inertial_velocity_body_m_s=inertial_velocity_body_m_s,
                    estimated_forward_speed_m_s=self.speed,
                    detections=detected,
                    geometry=geometry,
                    observation=observation,
                    rgb_file="sensor-frames/" + filename,
                    failure=self.failure,
                    truth_used=False,
                )
            )
            + "\n"
        )
        self.log.flush()
        self.count += 1
        return command, self.head_offsets, kick

    def close(self):
        self.log.close()
        self.detector.close()
