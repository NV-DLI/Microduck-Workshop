"""A separate high-level action stream above the preserved roller actor.

Only the expert receives simulator object poses. Learner packets contain camera,
encoders and the declared ideal IMU. Physics truth is recorded separately for
scoring. One sample is taken before the action it labels, every four actor ticks.
"""

from pathlib import Path
import base64
import io
import json
import math
import time
import urllib.request
import numpy as np

TASK = "Roll forward, bump the football into the goal, and stop in roller mode."
ACTION_NAMES = [
    "forward_request_native",
    "yaw_request_native",
    "gaze_pitch_rad",
    "gaze_weight",
]


def tensor(x):
    return getattr(x, "torch", x)


def vector(x):
    return tensor(x)[0].detach().cpu().numpy()


class LearningController:
    def __init__(self, env, config, output, actor_sha256):
        from isaaclab_assets.robots.microduck import MICRODUCK_JOINT_NAMES
        from head_adapter import HeadAdapter

        self.env = env
        self.cfg = json.loads(Path(config).read_text())
        self.mode = self.cfg["mode"]
        self.performance_diagnostic = self.cfg.get("performance_diagnostic", False)
        self.recovery = None
        if self.cfg.get("automatic_release_brake"):
            if self.mode != "policy" or not self.cfg.get("low_level_stop"):
                raise ValueError(
                    "Automatic braking corrections require the learner and local stop controller"
                )
            from learning.recovery import ReleaseBrakeExpert

            self.recovery = ReleaseBrakeExpert()
        self.render_subframes = 4
        self.diagnostic_no_rgb = self.cfg.get("diagnostic_no_rgb", False)
        if self.diagnostic_no_rgb and self.mode != "expert":
            raise ValueError("Only expert controller diagnostics may omit camera data")
        if self.mode not in ("expert", "policy", "human"):
            raise ValueError("mode must be expert, policy or human")
        self.robot = env.scene["robot"]
        self.names = list(MICRODUCK_JOINT_NAMES)
        self.servo, _ = self.robot.find_joints(self.names, preserve_order=True)
        self.play, _ = self.robot.find_joints(
            ["passive_" + n + "_backlash" for n in self.names], preserve_order=True
        )
        self.head = HeadAdapter()
        self.contract = dict(
            version=2,
            action_names=ACTION_NAMES,
            longitudinal=(
                "forward_request_or_local_PI_stop"
                if self.cfg.get("low_level_stop")
                else "native_actor_request"
            ),
            stop_threshold=0.05 if self.cfg.get("low_level_stop") else None,
            stop_feedback_hz=50,
            stop_kp=float(self.cfg.get("stop_kp", 3.0)),
            stop_ki=float(self.cfg.get("stop_ki", 4.0)),
            gaze_rate_rad_per_actor_tick=0.015,
            sample_hz=12.5,
            actor_hz=50,
            locomotion="official_rollers",
        )
        self.history = None
        if self.cfg.get("state_context"):
            from learning.history import CausalHistory, MODE

            if self.cfg["state_context"] != MODE:
                raise ValueError("Unknown temporal state contract")
            self.history = CausalHistory()
        self.action = np.zeros(4)
        self.head_offsets = np.zeros(4)
        self.output = Path(output) / "learning"
        self.output.mkdir()
        (self.output / "images").mkdir()
        self.rows = (self.output / "samples.jsonl").open("x")
        self.timings = (self.output / "timing.jsonl").open("x")
        self.privileged = (self.output / "expert-truth.jsonl").open("x")
        self.phase = "OBSERVE"
        self.policy_identity = None
        if self.mode == "policy":
            with urllib.request.urlopen(
                self.cfg["server"] + "/health", timeout=10
            ) as response:
                self.policy_identity = json.load(response)
            if not self.policy_identity.get("ready") or not self.policy_identity.get(
                "checkpoint_sha256"
            ):
                raise ValueError(
                    "Inference server must report a frozen checkpoint identity"
                )
        self.touch_at = None
        self.release_brake_at = None
        self.brake_integral = 0.0
        self.last_gate = False
        self.reset_inference = True
        self.inference_sequence = 0
        from roller_imu import RollerBrake

        self.local_brake = RollerBrake()
        self.local_integral = 0.0
        self.segment = 0
        self.sequence = 0
        self.done = False
        self.start = time.monotonic()
        state_names = (
            ["q/" + n for n in self.names]
            + ["qd/" + n for n in self.names]
            + ["gyro/x", "gyro/y", "gyro/z", "gravity/x", "gravity/y", "gravity/z"]
            + ["imu_velocity/x", "imu_velocity/y", "imu_velocity/z"]
        )
        if self.history is not None:
            from learning.history import names, specification

            state_names = names(state_names, ACTION_NAMES)
            self.contract["temporal_state"] = specification()
        self.contract.update(
            state_names=state_names,
            actor_sha256=actor_sha256,
            camera_raw_shape=[540, 960, 3],
            learner_rgb_shape=[270, 480, 3],
            image_resize="PIL bicubic RGB",
            state_units=(
                "q rad; qd rad/s; gyro rad/s; gravity unit vector; ideal integrated velocity m/s; same units for each state lag; previous request: forward/yaw native requests, gaze pitch rad, gaze weight unitless"
                if self.history is not None
                else "q rad; qd rad/s; gyro rad/s; gravity unit vector; ideal integrated velocity m/s"
            ),
        )
        (self.output / "schema.json").write_text(
            json.dumps(
                dict(
                    version=2,
                    task=TASK,
                    state_names=state_names,
                    action_names=ACTION_NAMES,
                    sample_hz=12.5,
                    actor_hz=50,
                    physics_hz=200,
                    action_is_absolute=True,
                    camera="beta_rgb",
                    imu="ideal trunk specific-force integration",
                    policy_observation_excludes=[
                        "world_pose",
                        "ball_pose",
                        "goal_pose",
                        "time",
                        "expert_phase",
                    ],
                    configuration=self.cfg,
                    policy_identity=self.policy_identity,
                    trainable_recording=not (
                        self.diagnostic_no_rgb or self.performance_diagnostic
                    ),
                    control_contract=self.contract,
                ),
                indent=2,
            )
        )

    def expert(self, t, velocity, gravity, gyro):
        # Privileged simulator teacher: it is not the RGB-only baseline and not
        # human demonstration. All privileged values stay outside learner state.
        r = vector(self.robot.data.root_pos_w)
        b = vector(self.env.scene["ball"].data.root_pos_w)
        quat = vector(self.robot.data.root_quat_w)
        x, y, z, w = quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        self.privileged.write(
            json.dumps(dict(timestamp_s=t, root=r.tolist(), ball=b.tolist(), yaw=yaw))
            + "\n"
        )
        launch = float(self.cfg.get("launch_s", 5.0))
        push = float(self.cfg.get("push_s", 2.1))
        ramp = float(self.cfg.get("ramp_down_s", 0.0))
        speed = float(self.cfg.get("push_request", 0.6))
        if self.cfg.get("release_brake") and t >= launch:
            ball_velocity = vector(self.env.scene["ball"].data.root_lin_vel_w)
            # A privileged demonstration teacher may use simulated object
            # velocity; this is never added to the learner's observations.
            released = (
                b[0] > float(self.cfg.get("release_ball_x", 0.10))
                and ball_velocity[0] > float(self.cfg.get("release_ball_vx", 0.4))
                and gravity[2] < -0.95
                and np.linalg.norm(gyro) < 2.0
            )
            if released and self.release_brake_at is None:
                self.release_brake_at = t
            if self.release_brake_at is not None:
                push = min(push, self.release_brake_at - launch)
        vx = wz = 0.0
        gaze = float(self.cfg.get("observe_pitch_rad", 0.55))
        if t < launch - 1.5:
            self.phase = "OBSERVE"
            pitch = gaze * min(1.0, max(0.0, (t - 1.0) / 0.8))
        else:
            pitch = gaze * max(0.0, 1 - (t - (launch - 1.5)) / 0.5)
            if t < launch:
                self.phase = "SETTLE"
            elif t < launch + push:
                self.phase = "PUSH"
                vx = speed
            elif ramp > 0 and t < launch + push + ramp:
                self.phase = "DECELERATE"
                vx = speed * (1 - (t - launch - push) / ramp)
            else:
                self.phase = "BRAKE"
                v = float(velocity[0])
                if abs(v) < 0.12 and np.linalg.norm(gyro) < 0.5:
                    self.brake_integral = float(
                        np.clip(self.brake_integral + v * 0.08, -0.20, 0.20)
                    )
                vx = float(np.clip(-3 * v - self.brake_integral, -0.5, 0.35))
        weight = (
            min(1.0, max(0.0, (t - 1.0) / 0.8))
            if t < launch - 1.5
            else max(0.0, 1 - (t - (launch - 1.5)) / 0.5)
        )
        if self.cfg.get("low_level_stop") and self.phase == "BRAKE":
            vx = 0.0
        if gaze == 0.0:
            weight = 0.0
        return np.asarray([vx, wz, pitch, weight])

    def motor_command(self, velocity):
        vx = float(self.action[0])
        if self.cfg.get("low_level_stop") and vx <= 0.05:
            v = float(velocity[0])
            candidate = float(np.clip(self.local_integral + v * 0.02, -0.15, 0.15))
            requested = (
                -self.contract["stop_kp"] * v - self.contract["stop_ki"] * candidate
            )
            if -0.5 <= requested <= 0.35:
                self.local_integral = candidate
            vx = float(
                np.clip(
                    -self.contract["stop_kp"] * v
                    - self.contract["stop_ki"] * self.local_integral,
                    -0.5,
                    0.35,
                )
            )
        else:
            self.local_brake.reset()
            self.local_integral = 0.0
        return [vx, 0.0, float(self.action[1])]

    def apply_head(self, actor_action):
        solution = self.head.solve([0.0, float(self.action[2])])
        if not solution["accepted"]:
            raise RuntimeError("Gaze is outside head IK tolerance")
        target = np.asarray(solution["head_pose_native_offsets_rad"])
        self.head_offsets += np.clip(target - self.head_offsets, -0.015, 0.015)
        import torch

        actor_action[:, 5:9] = torch.lerp(
            actor_action[:, 5:9],
            torch.tensor(
                self.head_offsets, device=self.env.device, dtype=actor_action.dtype
            ),
            float(self.action[3]),
        )

    def step(self, step, inertial, held):
        from PIL import Image

        t = step * 0.02
        if step % 4:
            return self.action
        sample_started = time.monotonic()
        q = vector(self.robot.data.joint_pos)
        qd = vector(self.robot.data.joint_vel)
        gyro = vector(self.robot.data.root_ang_vel_b)
        gravity = vector(self.robot.data.projected_gravity_b)
        state = np.r_[
            q[self.servo] + q[self.play],
            qd[self.servo] + qd[self.play],
            gyro,
            gravity,
            inertial,
        ]
        if self.history is not None:
            state = np.asarray(
                self.history.observe(
                    state.tolist(), self.action.tolist(), self.sequence
                )
            )
        frame = f"images/{self.sequence:06d}.png"
        capture_started = time.monotonic()
        if not self.diagnostic_no_rgb:
            capture = self.env.beta_rig.capture(
                timestamp_s=t, render_subframes=self.render_subframes
            )
            rgb = capture["rgb_raw"]
            image = Image.fromarray(rgb)
            image.save(self.output / frame)
        capture_elapsed = time.monotonic() - capture_started
        gate = self.mode == "human" or "H" in held
        if gate != self.last_gate:
            self.segment += 1
            self.reset_inference = True
        self.last_gate = gate
        if gate:
            source = "human"
            pitch = float(
                np.clip(0.55 * ("I" in held) - 0.15 * ("K" in held), -0.15, 0.55)
            )
            solution = self.head.solve([0.0, pitch])
            if not solution["accepted"]:
                raise RuntimeError("Human gaze is outside head IK tolerance")
            action = np.asarray(
                [
                    0.6 * ("W" in held) - 0.5 * ("S" in held),
                    0.3 * (("Q" in held) - ("E" in held)),
                    pitch,
                    float("I" in held or "K" in held),
                ]
            )
        elif self.mode == "expert":
            source = "automatic_expert"
            action = self.expert(t, inertial, gravity, gyro)
        elif self.recovery is not None and self.recovery.active:
            source = "automatic_expert"
            action = np.zeros(4)
        else:
            source = "learner"
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            packet = dict(
                episode_id=self.output.parent.name,
                episode_seed=int(self.env.cfg.seed),
                control_contract=self.contract,
                sequence=self.inference_sequence,
                frame=self.sequence,
                reset=self.reset_inference,
                state=state.tolist(),
                task=TASK,
                image_base64=base64.b64encode(buffer.getvalue()).decode(),
            )
            request = urllib.request.Request(
                self.cfg["server"] + "/act",
                data=json.dumps(packet).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(
                request, timeout=float(self.cfg.get("timeout_s", 120))
            ) as response:
                result = json.load(response)
            if result["sequence"] != self.inference_sequence:
                raise ValueError("Stale inference response")
            action = np.asarray(result["action"], dtype=float)
            self.inference_sequence += 1
            self.reset_inference = False
        proposal = action.tolist() if source == "learner" else None
        if self.recovery is not None and not gate:
            truth = dict(
                robot_position=vector(self.robot.data.root_pos_w).tolist(),
                ball_position=vector(self.env.scene["ball"].data.root_pos_w).tolist(),
                ball_velocity=vector(
                    self.env.scene["ball"].data.root_lin_vel_w
                ).tolist(),
                forward_velocity=float(inertial[0]),
                gravity_z=float(gravity[2]),
                angular_speed=float(np.linalg.norm(gyro)),
                timestamp_s=t,
            )
            was_active = self.recovery.active
            if self.recovery.update(proposal=action, **truth):
                if not was_active:
                    self.segment += 1
                    self.reset_inference = True
                source = "automatic_expert"
                action = np.zeros(4)
                self.phase = "RECOVERY_BRAKE"
            self.privileged.write(
                json.dumps(
                    dict(
                        **truth,
                        recovery_active=self.recovery.active,
                        trigger=self.recovery.trigger,
                    )
                )
                + "\n"
            )
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Expected four finite high-level action values")
        if not (
            -0.5 <= action[0] <= 0.6
            and abs(action[1]) <= 0.3
            and -0.2 <= action[2] <= 1.0
            and 0 <= action[3] <= 1.0
        ):
            raise ValueError("Action outside controller contract")
        self.action = action
        self.rows.write(
            json.dumps(
                dict(
                    frame=self.sequence,
                    timestamp_s=t,
                    image=frame,
                    state=state.tolist(),
                    action=action.tolist(),
                    source=source,
                    segment=self.segment,
                    correction=gate
                    or (self.recovery is not None and self.recovery.active),
                    learner_proposal=proposal,
                    recovery_scope=(
                        "released_ball_braking_only"
                        if self.recovery is not None
                        else None
                    ),
                    task=TASK,
                    eligible_for_training=(
                        source != "learner"
                        and not (self.diagnostic_no_rgb or self.performance_diagnostic)
                    ),
                    expert_phase=self.phase if source == "automatic_expert" else None,
                )
            )
            + "\n"
        )
        self.rows.flush()
        self.timings.write(
            json.dumps(
                dict(
                    frame=self.sequence,
                    capture_s=capture_elapsed,
                    total_s=time.monotonic() - sample_started,
                )
            )
            + "\n"
        )
        self.timings.flush()
        self.sequence += 1
        return action

    def close(self):
        self.timings.close()
        self.rows.close()
        self.privileged.close()
        (self.output / "timing.json").write_text(
            json.dumps(
                dict(
                    samples=self.sequence, elapsed_wall_s=time.monotonic() - self.start
                )
            )
        )
