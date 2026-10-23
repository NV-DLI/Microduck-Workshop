"""Recovery fixture: Microduck simulation and control."""

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = Path("@COURSE@/simulation/core")
for p in (str(HERE), str(M / "evaluation"), str(M / "football"), str(M / "sensors")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
from fixture_leg import FixtureLeg  # noqa: E402
from recovery_handoff import (
    HeadHandoffV121,
    recovery_reacquire_head_allowed,
    complete_same_capture,
)  # noqa: E402
from retreat_recovery import RecoveryParams, rear_path_authority  # noqa: E402
from head_feedback import ObservationHeadFeedback  # noqa: E402
from final_approach import in_training_box  # noqa: E402

NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
V117_POSE = (-0.018582903114977494, 0.053301156041499737, 0.022864512722746654)
V117_POINT = (0.09028235245258838, -0.05894066203680877, -0.08633258107262796)


def packet(t, seq, v=(0.0, 0.0), valid=True, yaw_rate=0.0):
    qd = [0.0] * 14
    qd[0], qd[1] = float(v[0]), float(v[1])
    w = [0.0, 0.0, float(yaw_rate)]
    return {
        "source": "body_powerboard_control_imu",
        "axes": "trunk FLU",
        "seq": seq,
        "timestamp_s": t,
        "read_timestamp_s": t,
        "delivered_timestamp_s": t,
        "encoder_read_timestamp_s": t,
        "orientation_valid": valid,
        "encoder_q_rad": [0.0] * 14,
        "encoder_qd_rad_s": qd,
        "gyro": list(w),
        "gravity": [0.0, 0.0, -1.0],
        "gyro_body_rad_s": list(w),
        "gravity_body": [0.0, 0.0, -1.0],
        "gyro_clipped": [False] * 3,
        "accel_clipped": [False] * 3,
    }


def empty_tof(t, sensor_xy):
    rays = []
    for row in range(8):
        for col in range(8):
            a, e = math.radians(-19.7 + col * 5.625), math.radians(19.7 - row * 5.625)
            v = np.array(
                [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
            )
            rays.append((v / np.linalg.norm(v)).tolist())
    return {
        "capture_s": t,
        "fk_capture_s": t,
        "provenance": "simulation_approximation",
        "status": [None] * 64,
        "ranges_m": [None] * 64,
        "isaac_valid_mask": [False] * 64,
        "near_m": 0.1,
        "far_m": 4.0,
        "unit_rays_sensor": rays,
        "sensor_translation_odom": [float(sensor_xy[0]), float(sensor_xy[1]), 0.2],
        "sensor_quaternion_odom_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def capture_times(start, end):
    out = []
    n = math.ceil(start * 15)
    while True:
        tick = math.ceil(n * 200 / 15 - 1e-10)
        t = round(tick * 0.005, 6)
        n += 1
        if t > end:
            return out
        if t >= start:
            out.append(t)


def rot(yaw, v):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class Episode:
    """faults: set of names; see run()."""

    def __init__(
        self,
        faults=(),
        *,
        params=None,
        task_origin=2.34,
        layout="empty",
        end_s=30.0,
        odometry_bias=0.0,
        onset_s=0.25,
        speed=0.11,
        coast_m=0.0,
        ball_visible_after_s=0.0,
        cycles=0,
        drift_rad_s=-0.08,
        wz_gain=0.8
    ):
        self.faults = set(faults)
        self.params = params or RecoveryParams.provisional()
        self.layout = layout
        self.task_origin = task_origin
        self.end_s = end_s
        self.bias = odometry_bias
        self.onset_s = onset_s
        self.speed = speed
        self.coast_m = coast_m
        self.ball_visible_after_s = ball_visible_after_s
        self.cycles = cycles
        self.handoff = HeadHandoffV121(
            FixtureLeg(), NAMES, params=self.params, task_limit_s=180.0
        )
        self.feedback = ObservationHeadFeedback()
        self.pose = np.array(V117_POSE, float)
        self.ball_world = self.pose[:2] + rot(self.pose[2], V117_POINT[:2])
        self.seq = 0
        self.actual_head = [0.0] * 4
        self.velocity = [0.0, 0.0, 0.0]
        self.command_since = None
        self.coasting = 0.0
        self.files = {
            k: []
            for k in (
                "recovery-trace",
                "policy-trace",
                "public-memory-readback",
                "control",
                "physics-trace",
            )
        }
        self.last_control = {
            "velocity_body_xyw": [0.0, 0.0, 0.0],
            "kick_request": False,
            "state": "FINAL_ALIGN",
        }
        self.last_capture_s = -1.0
        self.counters = {"final_turns": 5, "final_advances": 7, "fine_attempts": 3}
        # Fine TRACK holds an actual neutral head before HEAD_RETURN; the controller's
        # BALL_REACQUIRE gaze is only used as nominal in recovery REACQUIRE.
        self.gaze_head = (
            [0.0, 0.25, 0.0, 0.0]
            if "entry_head_nonneutral" in self.faults
            else [0.0, 0.0, 0.0, 0.0]
        )
        self.reacquire_gaze = [0.0, 0.25, 0.0, 0.0]
        self.terminal = None
        self.gap_done = False
        self.physical_vx = 0.0
        self.physical_rate = 0.0
        self.last_packet = None
        self.drift = -0.9 if "yaw_drift_big" in self.faults else drift_rad_s
        self.wz_gain = wz_gain
        if "overshoot" in self.faults:
            self.speed = 0.2
        if "short_travel" in self.faults:
            self.speed = 0.05

    # -- public observations ----------------------------------------------------
    def body_point(self):
        d = rot(-self.pose[2], self.ball_world - self.pose[:2])
        return [float(d[0]), float(d[1]), -0.086]

    def geometry(self, t, *, complete):
        if not complete:
            return {
                "timestamp_s": t,
                "ball": {"accepted": False},
                "floor_observation": {"accepted": True},
            }
        return {
            "timestamp_s": t,
            "floor_observation": {"accepted": True},
            "grounded_refinement": {"accepted": True},
            "ball": {
                "accepted": True,
                "geometry": {
                    "point_reference_m": self.body_point(),
                    "estimated_range_uncertainty_m": 0.02,
                    "observed_boundary_required": True,
                    "accepted": True,
                },
            },
        }

    # -- one 50 Hz policy tick in executor order ----------------------------------
    def policy_tick(self, t):
        h = self.handoff
        v_public = self.public_velocity(t)
        skip = bool(
            "odometry_reset" in self.faults
            and not self.gap_done
            and h.recovery.state == "RETREAT"
            and h.recovery.retreat_start is not None
            and t - h.recovery.retreat_start >= 0.6
        )
        self.gap_done |= skip
        stale = bool(
            "helper_read_gap" in self.faults
            and h.recovery.state == "RETREAT"
            and h.recovery.motion_index == 20
            and self.last_packet is not None
            and not getattr(self, "stale_done", False)
        )
        if stale:
            # No new delivery this tick: the unchanged previous packet is re-observed (valid for
            # memory/guards); the next new read is .04 s after the last one -> helper read-gap fault.
            self.stale_done = True
            pk = dict(self.last_packet)
        else:
            self.seq += (
                2 if skip else 1
            )  # skip: one delivered public read lost (sequence gap)
            pk = packet(t, self.seq, v=v_public, yaw_rate=self.physical_rate)
        self.last_packet = pk
        task = t - self.task_origin
        sent_before = list(self.actual_head)
        hs = h.policy(now_s=t, packet=pk, actual_sent_head=sent_before, task_s=task)
        velocity = list(self.last_control["velocity_body_xyw"])
        if hs["override_body_zero"]:
            velocity = [0.0, 0.0, 0.0]
        rv = h.recovery_velocity(t)
        if rv is not None:
            velocity = rv
        gaze = (
            self.reacquire_gaze
            if (hs["state"] == "RECOVERY" and h.recovery.state == "REACQUIRE")
            else self.gaze_head
        )
        nominal = (
            hs["head_offsets"]
            if hs["override_body_zero"] and hs["head_offsets"] is not None
            else list(gaze)
        )
        perm = None
        if hs["state"] == "RECOVERY" and hs["head_offsets"] is None:
            perm = recovery_reacquire_head_allowed(
                now_s=t, capture_s=self.last_capture_s, control=self.last_control
            )
            if not perm["allowed"]:
                nominal = sent_before
        requested = list(nominal)
        self.actual_head, hf = self.feedback.update(
            now_s=t,
            nominal=nominal,
            packet=pk,
            velocity=velocity,
            state=self.last_control.get("state", ""),
            handoff_state=hs["state"],
            navigation_enabled=True,
            head_motion_allowed=True,
            fine_reacquire_head_only=False,
            macro_neutral_return=False,
        )
        if hf["body_zero_for_withdrawal"]:
            velocity = [0.0, 0.0, 0.0]
        velocity, row = h.recovery_after_head(
            now_s=t,
            task_s=task,
            requested_velocity=rv,
            velocity=velocity,
            requested_head=requested,
            sent_head=self.actual_head,
            head_feedback=hf,
            head_permission=perm,
            packet=pk,
        )
        if row is not None:
            hs = h.status(t)
            self.files["recovery-trace"].append(row)
        self.velocity = velocity
        self.files["public-memory-readback"].append(
            {"timestamp_s": t, "readback": h.latest_readback, "handoff": hs}
        )
        self.files["policy-trace"].append(
            {
                "timestamp_s": t,
                "policy": "kick_right" if hs["kick_runtime_authorized"] else "walk",
                "kick_requested": hs["kick_runtime_authorized"],
                "actual_velocity_command": velocity,
                "actual_head_command_rad": list(self.actual_head),
                "public_handoff": hs,
                "IMU_prediction": dict(pk),
            }
        )
        return hs

    def public_velocity(self, t):
        # Public odometry reports the velocity actually realised over the last interval.
        return (self.physical_vx * (1.0 + self.bias), 0.0)

    def actual_vx(self, t):
        if self.velocity[0] < 0:
            if self.command_since is None:
                self.command_since = t
            return -self.speed if t - self.command_since >= self.onset_s else 0.0
        self.command_since = None
        return 0.0

    def physics(self, t0):
        vx = 0.0 if "dead_zone" in self.faults else self.actual_vx(t0)
        rate = (
            self.wz_gain * self.velocity[2] + self.drift if any(self.velocity) else 0.0
        )
        self.physical_vx = vx
        self.physical_rate = rate
        for i in range(1, 5):
            t = round(t0 + i * 0.005, 6)
            self.pose[:2] += rot(self.pose[2], (vx * 0.005, 0.0))
            self.pose[2] += rate * 0.005
            q = [0.0, 0.0, math.sin(self.pose[2] / 2), math.cos(self.pose[2] / 2)]
            self.files["physics-trace"].append(
                {
                    "timestamp_s": t,
                    "task_time_s": t - self.task_origin,
                    "root_position_world_m": [*self.pose[:2].tolist(), 0.12],
                    "root_quaternion_xyzw": q,
                    "ball_position_world_m": [*self.ball_world.tolist(), 0.035],
                    "raw_paired_foot_ball_forces_n": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    "kick_contact_observed": False,
                    "qualified_goal_seen": False,
                    "fell": False,
                    "kick_onset_s": self.handoff.kick_start,
                    "scope": "CPU fixture kinematic toy; not physics",
                }
            )

    def capture(self, t):
        h = self.handoff
        self.last_capture_s = t
        rs = h.recovery.state
        visible = (
            h.state == "RECOVERY"
            and rs == "REACQUIRE"
            and "no_reacquire" not in self.faults
            and t >= (h.recovery.reacquire_start or 1e9) + self.ball_visible_after_s
        )
        g = self.geometry(t, complete=visible)
        complete = complete_same_capture(g, t)
        hs = h.capture(
            now_s=t,
            geometry=g,
            head_offsets=self.actual_head,
            parameters=False,
            goal_heading=-0.0187,
            goal_fix_s=t,
        )
        guard = True
        if "guard_unsafe_in_retreat" in self.faults and rs == "RETREAT":
            guard = False
        if "unsafe_reacquire" in self.faults and rs == "REACQUIRE":
            guard = False
        if h.state == "RECOVERY":
            if "stale_capture" in self.faults and rs == "RETREAT":
                return  # capture callback never arrives (dropped)
            path = {
                "allowed": False,
                "capture_s": t,
                "layout": self.layout,
                "reason": "fresh_public_guard_inputs_required",
                "truth_used": False,
            }
            if rs in ("AUTHORIZE", "RETREAT"):
                path = rear_path_authority(
                    now_s=t,
                    pose_map_xyyaw=self.pose.tolist(),
                    goal_fix_s=t,
                    distance_m=h.recovery.distance,
                    tof_sample=empty_tof(t, self.pose[:2]),
                    layout=self.layout,
                    image_sha256="0" * 64,
                    fixed_target_map_xy=h.recovery.fixed_target,
                )
            control = {
                "public_attention_owner": {"owner": "BALL_REACQUIRE"},
                "public_attention_head_permission": {
                    "accepted": True,
                    "owner": "BALL_REACQUIRE",
                    "capture_s": t,
                    "body_authorized": False,
                    "kick_authorized": False,
                },
                "velocity_body_xyw": [0.0, 0.0, 0.0],
                "kick_request": False,
                "state": "REACQUIRE_BALL",
            }
            row = h.recovery_capture(
                now_s=t,
                task_s=t - self.task_origin,
                guard_safe=guard,
                complete=complete,
                rear_path=path,
                geometry=g,
                control=control,
                controller_pose=self.pose.tolist(),
                goal_fix_s=t,
                approach_counters=dict(self.counters),
            )
            self.files["recovery-trace"].append(row)
            self.last_control = control
            hs = h.status(t)
        else:
            self.last_control = {
                "velocity_body_xyw": [0.0, 0.0, 0.0],
                "kick_request": False,
                "state": "FINAL_ALIGN",
            }
        self.files["control"].append(
            {
                "timestamp_s": t,
                "state": self.last_control.get("state"),
                "velocity_body_xyw": [0.0, 0.0, 0.0],
                "public_handoff": hs,
                "final_approach": {
                    "turn_attempts": self.counters["final_turns"],
                    "advance_attempts": self.counters["final_advances"],
                },
            }
        )

    # -- episode -------------------------------------------------------------------
    def run(self):
        h = self.handoff
        start = 12.98
        k0 = round(start / 0.02)
        # A complete same-capture anchor while ordinary approach owns control.
        for k in range(k0, k0 + 2):
            self.policy_tick(round(k * 0.02, 6))
        t_anchor = round((k0 + 1) * 0.02, 6)
        g = {
            "ball": {
                "accepted": True,
                "geometry": {
                    "point_reference_m": list(V117_POINT),
                    "estimated_range_uncertainty_m": 0.02,
                    "observed_boundary_required": True,
                    "accepted": True,
                },
            }
        }
        h.capture(
            now_s=t_anchor,
            geometry=g,
            head_offsets=[0.0] * 4,
            parameters=False,
            goal_heading=-0.0187,
            goal_fix_s=t_anchor,
        )
        assert not in_training_box(h.adapter.memory.status(t_anchor)["point_level_m"])
        k = k0 + 2
        enter_at = 14.8
        entered = False
        release_seen = None
        caps = capture_times(start, self.end_s)
        ci = 0
        while True:
            t = round(k * 0.02, 6)
            if t > self.end_s:
                self.terminal = "task_duration_limit"
                break
            while ci < len(caps) and caps[ci] <= t + 1e-9:
                if caps[ci] > round((k - 1) * 0.02, 6) + 1e-9:
                    self.capture(caps[ci])
                ci += 1
            if enter_at is not None and t >= enter_at - 1e-9 and h.state == "APPROACH":
                h.enter_from_fine(now_s=t, goal_heading=-0.0187, goal_fix_s=t)
                entered = True
                enter_at = None
            hs = self.policy_tick(t)
            if (
                "force_kick_in_recovery" in self.faults
                and hs["state"] == "RECOVERY"
                and h.recovery.state == "SETTLE"
            ):
                self.files["policy-trace"][-1]["kick_requested"] = True
            if h.state in ("FAILED", "COMPLETE"):
                self.terminal = (
                    "public_handoff_failed"
                    if h.state == "FAILED"
                    else "development_kick_window_complete"
                )
                break
            if entered and h.state == "APPROACH" and h.recovery.state == "RELEASED":
                release_seen = release_seen if release_seen is not None else t
                if t - release_seen > 0.3:
                    if self.cycles <= 0:
                        self.terminal = "fixture_stop_after_release"
                        break
                    # Toy re-approach (NOT physics): the original approach brings the robot back
                    # to the same near pose, consuming original budgets cumulatively; a new
                    # ordinary APPROACH anchor is taken, then the fine TRACK route re-enters HEAD_RETURN.
                    self.cycles -= 1
                    release_seen = None
                    entered = False
                    self.pose[:] = V117_POSE
                    self.counters["final_turns"] += 1
                    self.counters["fine_attempts"] += 1
                    h.capture(
                        now_s=t,
                        geometry={
                            "ball": {
                                "accepted": True,
                                "geometry": {
                                    "point_reference_m": self.body_point(),
                                    "estimated_range_uncertainty_m": 0.02,
                                    "observed_boundary_required": True,
                                    "accepted": True,
                                },
                            }
                        },
                        head_offsets=[0.0] * 4,
                        parameters=False,
                        goal_heading=-0.0187,
                        goal_fix_s=t,
                    )
                    enter_at = round(t + 0.2, 6)
            self.physics(t)
            k += 1
        return self

    def write(self, out_dir, *, approval_test_only=True):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=False)
        for name, values in self.files.items():
            with (out / (name + ".jsonl")).open("x") as f:
                for v in values:
                    f.write(json.dumps(v, allow_nan=False) + "\n")
        manifest = {
            "schema": "microduck.beta-football-development.v1",
            "terminal_reason": self.terminal,
            "retreat_recovery_approval": {
                "parameters": self.params.as_dict(),
                "test_only": approval_test_only,
                "manifest_path": "CPU_FIXTURE",
                "manifest_sha256": None,
            },
            "retreat_recovery_scope": {"task_limit_s": 180.0, "layout": self.layout},
            "retreat_recovery": self.handoff.recovery_report(),
            "scorer": {
                "contact": {"pre_kick_ball_contact_observed": False},
                "physical_episode_success": False,
            },
            "fixture_scope": "CPU kinematic toy; no physics, contact, gait or success evidence",
        }
        (out / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        )
        return out
