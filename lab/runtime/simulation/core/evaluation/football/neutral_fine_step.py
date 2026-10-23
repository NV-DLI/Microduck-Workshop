"""Neutral fine step: Microduck simulation and control."""

import math
import numpy as np


class NeutralFineStep:
    def __init__(self):
        self.state = "IDLE"
        self.reason = None
        self.started = None
        self.last_policy = None
        self.head = None
        self.return_target = np.zeros(4)
        self.quiet_start = None
        self.step_start = None
        self.step_end = None
        self.capture_s = None
        self.capture_safe = False
        self.capture_head_safe = False
        self.goal_s = None
        self.goal_heading = None
        self.macro_shot_error = None
        self.macro_lateral_command = 0.0
        self.macro_initial_y = None
        self.macro_lateral_error = 0.0
        self.macro_velocity = [0.0, 0.0, 0.0]
        self.macro_yaw_rate = 0.0
        self.primitive = None
        self.track_lateral_lp = None
        self.track_heading_lp = None
        self.track_velocity = [0.0, 0.0, 0.0]
        self.motion_kind = "STEP"
        self.turn_sign = 0.0
        self.turn_start = None
        self.turn_reason = None
        self.coarse_track = False
        self.turn_goal_only = False
        self.turn_initial_goal_heading = None
        self.track_brake_start = None
        self.track_brake_command = None
        self.final_stride = False
        self.maximum_step_s = 0.6
        self.head_slew = 0.3
        self.motion_heading_ref = None
        self.motion_heading_seq = None
        self.motion_heading_read_s = None
        self.motion_heading_anchor = None
        self.anchor_s = None
        self.attempts = 0
        self.last_finished = None
        self.step_goal_heading = None

    @property
    def active(self):
        return self.state != "IDLE"

    @property
    def hold_navigation_head(self):
        return self.state in (
            "RETURN",
            "STEP",
            "TURN",
            "TRACK",
            "MACRO",
            "SETTLE",
            "FAILED",
        )

    def fail(self, reason):
        self.state = "FAILED"
        self.reason = reason
        if hasattr(self, "last_sent"):
            self.head = self.last_sent.copy()

    @staticmethod
    def packet_valid(packet, t):
        try:
            return bool(
                packet is not None
                and packet.get("orientation_valid") is True
                and isinstance(packet["seq"], int)
                and not isinstance(packet["seq"], bool)
                and packet["seq"] >= 0
                and all(
                    math.isfinite(packet[k])
                    and packet[k] >= 0
                    and 0 <= t - packet[k] <= 0.08
                    for k in (
                        "timestamp_s",
                        "read_timestamp_s",
                        "encoder_read_timestamp_s",
                    )
                )
                and packet["encoder_read_timestamp_s"] == packet["read_timestamp_s"]
                and np.asarray(packet["gyro_body_rad_s"]).shape == (3,)
                and np.asarray(packet["gravity_body"]).shape == (3,)
                and np.asarray(packet["encoder_q_rad"]).shape == (14,)
                and np.asarray(packet["encoder_qd_rad_s"]).shape == (14,)
                and all(
                    np.isfinite(packet[k]).all()
                    for k in (
                        "encoder_q_rad",
                        "encoder_qd_rad_s",
                        "gyro_body_rad_s",
                        "gravity_body",
                    )
                )
            )
        except (ValueError, KeyError, TypeError):
            return False

    @staticmethod
    def waypoint_bearing(point, goal_heading):
        p = np.asarray(point, float)
        if (
            p.shape != (3,)
            or not np.isfinite(p).all()
            or goal_heading is None
            or not math.isfinite(goal_heading)
        ):
            return None
        c, s = math.cos(goal_heading), math.sin(goal_heading)
        target = np.array(
            [0.09, -0.035]
        )  # v71 body-forward right-foot strike point; goal gate separately checked
        error = p[:2] - target
        return float(math.atan2(error[1], error[0]))

    @staticmethod
    def angle_delta(current, reference):
        return math.atan2(math.sin(current - reference), math.cos(current - reference))

    @staticmethod
    def quiet(packet):
        gravity = np.asarray(packet["gravity_body"], float)
        gyro = np.asarray(packet["gyro_body_rad_s"], float)
        qd = np.asarray(packet["encoder_qd_rad_s"], float)
        tilt = math.atan2(float(np.linalg.norm(gravity[:2])), -float(gravity[2]))
        return bool(
            np.linalg.norm(gyro) <= 0.1
            and np.sqrt(np.mean(qd**2)) <= 0.25
            and tilt <= 0.35
        )

    def capture(
        self,
        *,
        t,
        guard_safe,
        complete,
        trigger,
        actual_head,
        memory,
        goal_heading,
        goal_fix_s,
        head_only_safe=False,
        motion_kind="STEP"
    ):
        if (
            not math.isfinite(t)
            or t < 0
            or self.capture_s is not None
            and t <= self.capture_s
        ):
            raise ValueError("Fresh monotone capture required")
        self.capture_s = t
        self.capture_safe = bool(guard_safe)
        self.capture_head_safe = bool(head_only_safe)
        if (
            math.isfinite(goal_heading)
            and math.isfinite(goal_fix_s)
            and 0 <= t - goal_fix_s <= 8
        ):
            self.goal_heading = float(goal_heading)
            self.goal_s = float(goal_fix_s)
        if self.state == "IDLE" and trigger:
            p = np.asarray(memory.get("point_level_m"), float)
            head = np.asarray(actual_head, float)
            coarse = motion_kind == "COARSE_TRACK"
            turn = motion_kind == "TURN"
            far_step = motion_kind in ("FAR_STEP", "TRACK", "COARSE_TRACK")
            track = motion_kind in ("TRACK", "MACRO", "COARSE_TRACK")
            valid = bool(
                complete
                and guard_safe
                and memory.get("memory_available")
                and memory.get("ball_capture_timestamp_s") == t
                and p.shape == (3,)
                and np.isfinite(p).all()
                and 0.13 <= p[0]
                and np.linalg.norm(p[:2])
                <= (
                    0.9
                    if coarse
                    else (
                        0.4
                        if turn or far_step
                        else 0.27 if motion_kind == "MACRO" else 0.22
                    )
                )
                and (
                    -0.25 <= p[1] <= 0.25
                    if coarse
                    else -0.3 <= p[1] <= 0.3 if turn else -0.10 <= p[1] <= 0.08
                )
                and self.goal_s is not None
                and 0 <= t - self.goal_s <= 8
                and (
                    turn
                    or track
                    or abs(self.waypoint_bearing(p, self.goal_heading)) <= 0.15
                )
                and head.shape == (4,)
                and np.isfinite(head).all()
            )
            if not valid:
                self.fail("neutral_step_initial_public_domain_rejected")
            else:
                self.return_target = np.zeros(4)
                if turn and abs(self.goal_heading) > 0.45:
                    self.return_target[2] = max(-1.1, min(1.1, self.goal_heading))
                self.coarse_track = False
                self.turn_goal_only = False
                self.turn_initial_goal_heading = None
                self.track_brake_start = None
                self.track_brake_command = None
                self.coarse_track = coarse
                self.motion_kind = "TRACK" if coarse else motion_kind
                self.track_lateral_lp = None
                self.track_heading_lp = None
                self.motion_heading_ref = None
                self.motion_heading_seq = None
                self.motion_heading_read_s = None
                self.motion_heading_anchor = None
                self.final_stride = bool(
                    not turn
                    and 0.14 <= p[0] < 0.195
                    and abs(p[1] + 0.035) <= 0.008
                    and max(abs(head)) <= 0.75
                )
                self.maximum_step_s = 1.2 if self.final_stride else 0.6
                self.head_slew = (
                    0.65
                    if motion_kind in ("MACRO", "TRACK", "COARSE_TRACK")
                    else 0.5 if self.final_stride or turn or far_step else 0.3
                )
                self.reason = None
                self.state = "RETURN"
                self.started = t
                self.last_policy = t
                self.head = head.copy()
                self.anchor_s = t
                self.quiet_start = None
                self.step_start = None
                self.step_end = None
                self.step_goal_heading = None
                self.attempts += 1
        elif (
            self.state == "REACQUIRE" and complete and guard_safe and t > self.step_end
        ):
            self.state = "IDLE"
            self.last_finished = t
            self.quiet_start = None
        return self.status(t, memory)

    def policy(self, *, t, packet, memory, actual_head):
        if (
            not math.isfinite(t)
            or t < 0
            or self.last_policy is not None
            and t < self.last_policy
        ):
            raise ValueError("Finite monotone policy clock required")
        self.last_sent = np.asarray(actual_head, float).copy()
        if self.last_sent.shape != (4,) or not np.isfinite(self.last_sent).all():
            raise ValueError("Actual sent four-head command required")
        if self.state in ("IDLE", "FAILED"):
            return self.status(t, memory)
        dt = t - self.last_policy
        if not 0 <= dt <= 0.025:
            self.fail("neutral_step_policy_clock_gap")
            return self.status(t, memory)
        self.last_policy = t
        if t - self.started >= 24.0 - 1e-9:
            self.fail("neutral_step_total_deadline")
            return self.status(t, memory)
        if not self.packet_valid(packet, t):
            self.fail("neutral_step_invalid_same_read")
            return self.status(t, memory)
        if self.capture_s is None or not 0 <= t - self.capture_s <= 0.2:
            self.fail("neutral_step_stale_or_hazard_capture")
            return self.status(t, memory)
        if self.state == "REACQUIRE":
            # This phase has no body/kick authority and cannot start another
            # STEP. Preserve the real hazard; allow only independently gated
            # head attention until a new complete AND body-safe capture arrives.
            if not (self.capture_safe or self.capture_head_safe):
                self.fail("neutral_step_invalid_head_attention_capture")
            return self.status(t, memory)
        if not self.capture_safe:
            if self.state == "SETTLE":
                # Body authority was already irrevocably zero at step end.
                # Wait for a new safe capture without moving the head or
                # carrying the quiet interval across missing geometry.
                self.head = self.last_sent.copy()
                self.quiet_start = None
                return self.status(t, memory)
            # A current rejected capture withdraws all motion immediately.
            # Keep the existing bounded zero-body settle/reobserve procedure;
            # neither the old anchor nor this rejection can restart motion.
            self.state = "SETTLE"
            self.step_end = t
            self.quiet_start = None
            self.head = self.last_sent.copy()
            self.track_velocity = [0.0, 0.0, 0.0]
            self.reason = "public_guard_stop_requires_fresh_reacquisition"
            return self.status(t, memory)
        # SETTLE has irrevocably zero body authority. Old ball memory cannot
        # authorize another step here; fresh current packet/capture and the
        # original quiet/attention clocks still govern head-only reacquisition.
        if self.state == "SETTLE":
            if self.quiet(packet):
                if self.quiet_start is None:
                    self.quiet_start = t
            else:
                self.quiet_start = None
            if self.quiet_start is not None and t - self.quiet_start >= 0.5 - 1e-9:
                self.state = "REACQUIRE"
            return self.status(t, memory)
        if (
            not memory.get("memory_available")
            or memory.get("ball_capture_timestamp_s") != self.anchor_s
            or not 0 <= t - self.anchor_s <= 4.5
        ):
            self.fail("neutral_step_memory_expired_or_reanchored")
            return self.status(t, memory)
        p = np.asarray(memory["point_level_m"], float)
        if (
            p.shape != (3,)
            or not np.isfinite(p).all()
            or not (
                0.075
                < p[0]
                <= (
                    0.9
                    if self.coarse_track
                    else (
                        0.4
                        if self.motion_kind in ("TURN", "FAR_STEP", "TRACK")
                        else 0.25
                    )
                )
                and (
                    -0.25 <= p[1] <= 0.25
                    if self.coarse_track
                    else (
                        -0.3 <= p[1] <= 0.3
                        if self.motion_kind == "TURN"
                        else (
                            -0.15 <= p[1] <= 0.15
                            if self.motion_kind == "TRACK"
                            else -0.10 <= p[1] <= 0.08
                        )
                    )
                )
            )
        ):
            self.fail("neutral_step_propagated_point_invalid")
            return self.status(t, memory)
        if self.state == "RETURN":
            self.head += np.clip(
                self.return_target - self.head,
                -self.head_slew * dt,
                self.head_slew * dt,
            )
            neutral = (
                np.max(abs(self.head - self.return_target)) <= 1e-9
                and np.max(abs(np.asarray(actual_head) - self.return_target)) <= 1e-9
            )
            if neutral and self.quiet(packet):
                if self.quiet_start is None:
                    self.quiet_start = t
            else:
                self.quiet_start = None
            if self.quiet_start is not None and t - self.quiet_start >= 0.5 - 1e-9:
                if self.goal_s is None or not 0 <= t - self.goal_s <= 0.5:
                    self.fail("neutral_step_fresh_goal_required")
                elif (
                    self.motion_kind not in ("TURN", "TRACK", "MACRO")
                    and abs(self.waypoint_bearing(p, self.goal_heading)) > 0.15
                ):
                    self.fail("neutral_step_waypoint_direction_required")
                elif p[0] < 0.13:
                    self.fail("neutral_step_start_range_too_close")
                elif self.motion_kind == "MACRO":
                    from short_primitive import select

                    self.primitive = select(p, self.goal_heading)
                    if self.primitive is None:
                        self.state = "SETTLE"
                        self.step_end = t
                        self.quiet_start = None
                        self.reason = "primitive_unavailable_after_headreturn"
                    else:
                        self.state = "MACRO"
                        self.step_start = t
                        self.step_goal_heading = self.goal_heading
                        self.quiet_start = None
                        self.macro_initial_y = float(p[1])
                        self.macro_lateral_error = 0.0
                        self.macro_shot_error = None
                        self.macro_lateral_command = 0.0
                        self.macro_velocity = self.macro_command(p, packet, dt)
                elif self.motion_kind == "TRACK":
                    if abs(self.goal_heading) > 0.35:
                        self.fail("neutral_track_initial_goal_angle")
                    elif abs(self.goal_heading) > 0.08 and not self.coarse_track:
                        # Neutral head and a real quiet window are already
                        # established. Align to the fresh goal before advancing;
                        # TURN must stop and reobserve the ball before TRACK.
                        self.motion_kind = "TURN"
                        self.turn_goal_only = True
                        self.state = "TURN"
                        self.turn_start = t
                        self.turn_sign = math.copysign(1.0, self.goal_heading)
                        self.turn_initial_goal_heading = float(self.goal_heading)
                        self.quiet_start = None
                        self.reason = "track_start_goal_pre_alignment"
                    else:
                        self.state = "TRACK"
                        self.step_start = t
                        self.step_goal_heading = self.goal_heading
                        self.quiet_start = None
                        self.track_velocity = self.track_filtered(
                            p, self.goal_heading, dt, packet
                        )
                elif self.motion_kind == "TURN":
                    self.state = "TURN"
                    self.turn_start = t
                    self.turn_sign = math.copysign(
                        1.0, self.turn_bearing(p, self.goal_heading)
                    )
                    self.quiet_start = None
                else:
                    self.state = "STEP"
                    self.step_start = t
                    self.step_goal_heading = self.goal_heading
                    self.quiet_start = None
        elif self.state == "MACRO":
            if (
                self.goal_s is None
                or not 0 <= t - self.goal_s <= 0.5
                or abs(self.goal_heading) > 0.4
            ):
                self.fail("primitive_public_goal_guard")
            elif t - self.step_start >= self.primitive["forward_duration_s"] - 1e-9:
                self.state = "SETTLE"
                self.step_end = t
                self.quiet_start = None
            else:
                self.macro_velocity = self.macro_command(p, packet, dt)
        elif self.state == "TRACK":
            if self.coarse_track and (
                p[0] <= 0.36 or t - self.step_start >= 2.0 - 1e-9
            ):
                self.state = "SETTLE"
                self.step_end = t
                self.quiet_start = None
                self.track_velocity = [0.0, 0.0, 0.0]
                self.reason = "coarse_stop_requires_fresh_ball"
                return self.status(t, memory)
            if self.goal_s is None or not 0 <= t - self.goal_s <= 0.5:
                # A missing goal observation withdraws motion. It cannot extend
                # the old observation clock or authorize a kick at this stop.
                self.state = "SETTLE"
                self.step_end = t
                self.quiet_start = None
                self.head = self.last_sent.copy()
                self.track_velocity = [0.0, 0.0, 0.0]
                self.reason = "public_goal_stop_requires_fresh_reacquisition"
                return self.status(t, memory)
            if abs(self.goal_heading) > 0.45:
                self.fail("neutral_track_fresh_goal_or_heading_required")
                return self.status(t, memory)
            if (
                p[0] <= 0.09
                or t - self.step_start >= 2.3 - 1e-9
                or self.track_brake_start is None
                and p[0] <= 0.115
            ):
                self.state = "SETTLE"
                self.step_end = t
                self.quiet_start = None
                self.track_velocity = [0.0, 0.0, 0.0]
            elif self.track_brake_start is not None:
                self.motion_heading(self.goal_heading, packet)
                fraction = max(0.0, 1.0 - (t - self.track_brake_start) / 0.16)
                if fraction <= 1e-9:
                    self.state = "SETTLE"
                    self.step_end = t
                    self.quiet_start = None
                    self.track_velocity = [0.0, 0.0, 0.0]
                else:
                    self.track_velocity = [
                        float(v * fraction) for v in self.track_brake_command
                    ]
            else:
                self.track_velocity = self.track_filtered(
                    p, self.goal_heading, dt, packet
                )
                if p[0] <= 0.125:
                    # Reduce only the actor's requested body velocity, never
                    # project actions or alter physical state. Freshness and
                    # memory gates above remain mandatory on every brake tick.
                    self.track_brake_start = t
                    self.track_brake_command = self.track_velocity.copy()
        elif self.state == "TURN":
            if self.goal_s is None or not 0 <= t - self.goal_s <= 0.5:
                self.fail("neutral_turn_fresh_goal_required")
                return self.status(t, memory)
            bearing = (
                self.goal_heading
                if self.turn_goal_only
                else self.turn_bearing(p, self.goal_heading)
            )
            gravity = np.asarray(packet["gravity_body"], float)
            up = -gravity / np.linalg.norm(gravity)
            yaw_rate = float(np.dot(np.asarray(packet["gyro_body_rad_s"], float), up))
            elapsed = t - self.turn_start
            predictive = (
                yaw_rate * self.turn_sign > 0.1
                and (bearing - 0.12 * yaw_rate) * self.turn_sign <= 0.015
            )
            if self.turn_goal_only:
                # An instantaneous sway-rate peak must not stop prealignment
                # before the fresh goal observation shows any net rotation.
                progress = (self.turn_initial_goal_heading - bearing) * self.turn_sign
                predictive = bool(
                    predictive
                    and elapsed >= 0.16 - 1e-9
                    and self.goal_s > self.turn_start
                    and progress >= 0.03
                )
            stop = elapsed >= 0.08 - 1e-9 and (
                bearing * self.turn_sign <= 0.015 or predictive
            )
            if stop or elapsed >= 0.8 - 1e-9:
                self.state = "SETTLE"
                self.step_end = t
                self.quiet_start = None
                self.turn_reason = "public_imu_brake" if stop else "bounded_turn_time"
        elif self.state == "STEP":
            if t - self.step_start >= self.maximum_step_s - 1e-9 or p[0] <= (
                0.112 if self.final_stride else 0.105
            ):
                self.state = "SETTLE"
                self.step_end = t
                self.quiet_start = None
            elif (
                self.goal_s is None
                or not 0 <= t - self.goal_s <= 1.1
                or self.step_goal_heading is None
                or abs(self.angle_delta(self.goal_heading, self.step_goal_heading))
                > 0.15
            ):
                self.fail("neutral_step_goal_or_heading_guard")
        return self.status(t, memory)

    def motion_heading(self, camera_heading, packet):
        # Anchor only once, at the actual start of this bounded movement.
        # Later camera captures retain their own freshness/guard authority but
        # do not inject measurement jitter into the moving yaw reference.
        seq = packet["seq"]
        read = packet["read_timestamp_s"]
        if self.motion_heading_ref is None:
            self.motion_heading_ref = float(camera_heading)
            self.motion_heading_anchor = {
                "heading_rad": float(camera_heading),
                "public_goal_fix_s": self.goal_s,
                "read_s": read,
                "seq": seq,
                "source": "public_camera_goal_at_movement_start",
            }
        elif seq > self.motion_heading_seq and read > self.motion_heading_read_s:
            delta = read - self.motion_heading_read_s
            if delta > 0.08:
                raise ValueError("Motion heading cannot integrate a stale read gap")
            gravity = np.asarray(packet["gravity_body"], float)
            up = -gravity / np.linalg.norm(gravity)
            rate = float(np.dot(np.asarray(packet["gyro_body_rad_s"], float), up))
            self.motion_heading_ref = self.angle_delta(
                self.motion_heading_ref - rate * delta, 0.0
            )
        elif (
            seq < self.motion_heading_seq
            or read < self.motion_heading_read_s
            or (seq == self.motion_heading_seq and read != self.motion_heading_read_s)
        ):
            raise ValueError(
                "Non-monotone or re-timestamped public motion heading read"
            )
        self.motion_heading_seq = seq
        self.motion_heading_read_s = read
        return self.motion_heading_ref

    def macro_command(self, p, packet, dt):
        gravity = np.asarray(packet["gravity_body"], float)
        up = -gravity / np.linalg.norm(gravity)
        rate = float(np.dot(np.asarray(packet["gyro_body_rad_s"], float), up))
        self.macro_yaw_rate += (1.0 - math.exp(-dt / 0.1)) * (
            rate - self.macro_yaw_rate
        )
        heading = self.motion_heading(self.goal_heading, packet)
        vx, vy, _ = self.primitive["velocity_command"]
        # Match one physically sampled duration and lateral command exactly.
        # Public goal heading remains closed-loop; public final gates remain.
        self.macro_lateral_command = float(vy)
        self.macro_shot_error = (
            -math.sin(heading) * float(p[0]) + math.cos(heading) * float(p[1]) + 0.035
        )
        # Keep heading equilibrium independent of lateral hysteresis.
        # This is algebraically the v82 successful vy-.2 law (.15-2.5*.06=0).
        wz = max(-0.5, min(0.5, 2.5 * heading - 0.25 * self.macro_yaw_rate))
        return [vx, vy, wz]

    def track_filtered(self, p, heading, dt, packet):
        heading = self.motion_heading(heading, packet)
        lateral = (
            -math.sin(heading) * p[0]
            + math.cos(heading) * p[1]
            + (0.028 if self.coarse_track else 0.035)
        )
        if self.track_lateral_lp is None:
            self.track_lateral_lp = lateral
            self.track_heading_lp = heading
        else:
            a = 1.0 - math.exp(-dt / 0.20)
            self.track_lateral_lp += a * (lateral - self.track_lateral_lp)
            b = 1.0 - math.exp(-dt / 0.12)
            self.track_heading_lp += b * (heading - self.track_heading_lp)
        # Counter the observed lateral and yaw oscillations using mean errors.
        vy = max(
            -0.2,
            min(
                0.2,
                0.65 * math.sin(self.track_heading_lp) + 10.0 * self.track_lateral_lp,
            ),
        )
        gravity = np.asarray(packet["gravity_body"], float)
        up = -gravity / np.linalg.norm(gravity)
        rate = float(np.dot(np.asarray(packet["gyro_body_rad_s"], float), up))
        self.macro_yaw_rate += (1.0 - math.exp(-dt / 0.1)) * (
            rate - self.macro_yaw_rate
        )
        wz = max(
            -0.5, min(0.5, 2.5 * self.track_heading_lp - 0.25 * self.macro_yaw_rate)
        )
        # Public near-ball range lowers forward command before the final stop.
        # .25 is an already physically sampled actor command, not assumed speed.
        return [0.3 if float(p[0]) > 0.17 else 0.25, vy, wz]

    @staticmethod
    def track_command(p, goal_heading):
        # Public shot-line error, separately controlled lateral and yaw axes.
        c, s = math.cos(goal_heading), math.sin(goal_heading)
        lateral = -s * p[0] + c * p[1] + 0.035
        vy = max(-0.2, min(0.2, 5.0 * lateral))
        # Rotate desired shot-frame translation into current gravity-level body axes.
        vx_body = max(0.25, min(0.3, 0.3 * c - vy * s))
        vy_body = max(-0.2, min(0.2, 0.3 * s + vy * c))
        return [vx_body, vy_body, max(-0.2, min(0.2, 2.0 * goal_heading))]

    @staticmethod
    def turn_bearing(p, goal_heading):
        p = np.asarray(p, float)
        r = float(np.linalg.norm(p[:2]))
        foot = math.atan2(p[1], p[0]) + math.asin(min(1.0, 0.035 / max(r, 0.035)))
        predicted = math.atan2(
            math.sin(goal_heading - foot), math.cos(goal_heading - foot)
        )
        return goal_heading - max(-0.06, min(0.06, predicted))

    def status(self, t, memory):
        return {
            "state": self.state,
            "reason": self.reason,
            "started_s": self.started,
            "step_start_s": self.step_start,
            "step_end_s": self.step_end,
            "return_head_target_native": self.return_target.tolist(),
            "anchor_capture_s": self.anchor_s,
            "head_offsets": (
                None
                if self.state in ("IDLE", "REACQUIRE") or self.head is None
                else self.head.tolist()
            ),
            "motion_heading_prediction_rad": self.motion_heading_ref,
            "motion_heading_anchor": self.motion_heading_anchor,
            "motion_heading_prediction_is_measurement": False,
            "macro_shot_error_m": self.macro_shot_error,
            "macro_lateral_command": self.macro_lateral_command,
            "macro_lateral_error_m": self.macro_lateral_error,
            "macro_initial_y_m": self.macro_initial_y,
            "primitive": self.primitive,
            "velocity": (
                self.macro_velocity
                if self.state == "MACRO"
                else (
                    [(0.25 if self.final_stride else 0.2), 0.0, 0.0]
                    if self.state == "STEP"
                    else (
                        self.track_velocity
                        if self.state == "TRACK"
                        else (
                            [
                                0.0,
                                0.0,
                                (0.8 if self.turn_goal_only else 0.6) * self.turn_sign,
                            ]
                            if self.state == "TURN"
                            else [0.0, 0.0, 0.0]
                        )
                    )
                )
            ),
            "motion_kind": self.motion_kind,
            "coarse_track": self.coarse_track,
            "track_brake_started_s": self.track_brake_start,
            "track_brake_initial_command": self.track_brake_command,
            "track_brake_duration_s": 0.16,
            "track_lateral_filtered_m": self.track_lateral_lp,
            "track_goal_heading_filtered_rad": self.track_heading_lp,
            "turn_start_s": self.turn_start,
            "turn_goal_only": self.turn_goal_only,
            "turn_initial_goal_heading_rad": self.turn_initial_goal_heading,
            "turn_stop_reason": self.turn_reason,
            "override": self.active,
            "head_owned": self.hold_navigation_head,
            "quiet_started_s": self.quiet_start,
            "attempts": self.attempts,
            "last_finished_s": self.last_finished,
            "capture_guard_s": self.capture_s,
            "capture_safe": self.capture_safe,
            "capture_head_only_safe": self.capture_head_safe,
            "goal_fix_s": self.goal_s,
            "goal_heading_rad": self.goal_heading,
            "waypoint_bearing_rad": self.waypoint_bearing(
                memory.get("point_level_m"), self.goal_heading
            ),
            "step_goal_heading_rad": self.step_goal_heading,
            "step_heading_drift_rad": (
                None
                if self.goal_heading is None or self.step_goal_heading is None
                else self.angle_delta(self.goal_heading, self.step_goal_heading)
            ),
            "direction_contract": "entry/RETURN preparation-waypoint bearing; STEP change in goal bearing from STEP onset; final kick absolute goal heading remains separate",
            "memory": memory,
            "maximum_memory_age_s": 4.5,
            "maximum_attention_s": 24.0,
            "maximum_step_s": (
                2.0 if self.motion_kind == "TRACK" else self.maximum_step_s
            ),
            "final_stride": self.final_stride,
            "head_slew_rad_s": self.head_slew,
            "kick_authorized": False,
            "qualification_claimed": False,
            "truth_used": False,
        }
