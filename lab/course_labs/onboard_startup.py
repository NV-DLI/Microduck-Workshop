"""Course start-up gate for the onboard football route (standing61 joint brake).

The robot starts at HOME on the measured grounded root of the same football pitch
(groot_walking/student/grounded-pose.json), every joint temporarily locked by the
joint-limit brake, root free and zero external wrench. This gate wraps the GR00T
route's NoSupportStartup (same stationary check) and adds what the onboard route
needs before its walking actor may produce its first real output:

- the locked robot has settled (NoSupportStartup readiness) and is still stationary:
  the same thresholds must hold continuously for the last 0.5 s at the release tick
  (the inner check latches once passed, so a late camera could otherwise release a
  robot that moved after settling);
- the head camera and ToF delivered a complete, fresh frame;
- the BODY50 control IMU and same-read encoders are finite.

Until the executor calls mark_released(t), the policy, navigation, scoring and the
task clock are all off and the executor holds HOME without evaluating any network.
The executor releases the brake after the first actor output is computed and
before it is applied. If the gate is not ready by the initialization timeout, the
run fails with the joints still locked and nothing is scored.

Pure Python/numpy: no Kit, torch or GPU, so the CPU tests exercise it directly.
"""

import math

import numpy as np

RGB_SHAPE = (540, 960, 3)
TOF_ROWS = TOF_COLS = 8
TOF_STATUSES = ("valid", "no_return", "invalid")
# Captures arrive every 1/15 s; the 50 Hz tick reads the latest one.
MAX_CAPTURE_AGE_S = 0.1
JOINTS = 14
# Stationary check of joint_brake.NoSupportStartup (same values), kept true until release.
MIN_UPRIGHT = 0.995
MAX_SPEED_M_S = 0.002
MAX_ANGULAR_SPEED_RAD_S = 0.03
STABLE_WINDOW_S = 0.5


def _finite(values, n):
    try:
        values = [float(v) for v in values]
    except (TypeError, ValueError):
        return False
    return len(values) == n and all(math.isfinite(v) for v in values)


def capture_status(timestamp_s, rgb, tof):
    """Check one delivered capture: the RGB array and ToF dict from IsaacBetaSensors.capture()."""
    reasons = []
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != RGB_SHAPE:
        reasons.append("rgb_not_960x540_uint8")
    elif int(rgb.max()) == int(rgb.min()):
        reasons.append("rgb_constant_image")
    valid = 0
    if not isinstance(tof, dict) or tof.get("frame_dropped") is not False:
        reasons.append("tof_frame_missing_or_dropped")
    else:
        if not isinstance(tof.get("timestamp_s"), (int, float)) or abs(
            float(tof["timestamp_s"]) - float(timestamp_s)
        ) > 1e-9:
            reasons.append("tof_timestamp_mismatch")
        cells = tof.get("cells")
        if not isinstance(cells, (list, tuple)) or len(cells) != TOF_ROWS or any(
            not isinstance(row, (list, tuple)) or len(row) != TOF_COLS for row in cells
        ):
            reasons.append("tof_not_8x8")
        else:
            for row in cells:
                for cell in row:
                    status = cell.get("status")
                    if status not in TOF_STATUSES:
                        reasons.append("tof_unknown_status")
                    elif status == "valid":
                        r = cell.get("range_m")
                        if not isinstance(r, (int, float)) or not math.isfinite(r) or r <= 0:
                            reasons.append("tof_valid_zone_without_finite_range")
                        else:
                            valid += 1
            if valid == 0 and "tof_valid_zone_without_finite_range" not in reasons:
                reasons.append("tof_no_valid_zone")
    return dict(
        timestamp_s=float(timestamp_s),
        ok=not reasons,
        reasons=sorted(set(reasons)),
        tof_valid_zones=valid,
    )


def imu_status(packet):
    """Check the BODY50 packet the walking actor would read (gyro, gravity, same-read encoders)."""
    if packet is None:
        return dict(ok=False, reasons=["body_imu_not_delivered"])
    reasons = []
    if not _finite(packet.get("gyro_body_rad_s", ()), 3):
        reasons.append("gyro_not_finite")
    gravity = packet.get("gravity_body", ())
    if not _finite(gravity, 3):
        reasons.append("gravity_not_finite")
    elif not 0.9 <= float(np.linalg.norm(np.asarray(gravity, float))) <= 1.1:
        reasons.append("gravity_not_unit")
    if not _finite(packet.get("encoder_q_rad", ()), JOINTS):
        reasons.append("encoder_q_not_finite")
    if not _finite(packet.get("encoder_qd_rad_s", ()), JOINTS):
        reasons.append("encoder_qd_not_finite")
    return dict(ok=not reasons, reasons=reasons)


def stationary(*, quaternion_xyzw, velocity_world, angular_velocity_world, imu_orientation_valid, **_):
    """NoSupportStartup's stationary test; nonfinite state is never stationary."""
    if not (
        _finite(quaternion_xyzw, 4) and _finite(velocity_world, 3) and _finite(angular_velocity_world, 3)
    ):
        return False
    qx, qy = float(quaternion_xyzw[0]), float(quaternion_xyzw[1])
    return bool(
        imu_orientation_valid
        and 1 - 2 * (qx * qx + qy * qy) > MIN_UPRIGHT
        and np.linalg.norm(np.asarray(velocity_world, float)) < MAX_SPEED_M_S
        and np.linalg.norm(np.asarray(angular_velocity_world, float)) < MAX_ANGULAR_SPEED_RAD_S
    )


class OnboardBrakeStartup:
    """Wrap NoSupportStartup; nothing is enabled until the brake is released."""

    def __init__(self, inner, *, timeout_s=None):
        self.inner = inner
        self.cfg = inner.cfg
        self.timeout_s = float(inner.cfg.calibration_timeout_s if timeout_s is None else timeout_s)
        self.release_at = None
        self.failure = None
        self.capture = None
        self.last_check = None
        self.max_wrench = 0.0
        # The inner check latches once settled; this window stays current until release.
        self.stable_since = None
        self.last_update_s = None
        self.stability_resets = 0

    @property
    def settled_at(self):
        """Time the locked robot passed the stationary check (not the task start)."""
        return self.inner.task_started_at

    @property
    def task_started_at(self):
        # The executor derives the task clock, scorer time and recovery origin from this.
        return self.release_at

    def stable_for_s(self, timestamp_s):
        """Length of the current stationary window at this tick (0 if not stationary now)."""
        if self.stable_since is None or self.last_update_s != float(timestamp_s):
            return 0.0
        return float(timestamp_s) - self.stable_since

    def update(self, timestamp_s, **state):
        row = self.inner.update(timestamp_s, **state)
        if self.release_at is not None:
            return dict(
                row,
                phase="TASK",
                task_started_at_s=self.release_at,
                task_time_s=float(timestamp_s) - self.release_at,
            )
        t = float(timestamp_s)
        if stationary(**state):
            self.stable_since = t if self.stable_since is None else self.stable_since
        else:
            if self.stable_since is not None:
                self.stability_resets += 1
            self.stable_since = None
        self.last_update_s = t
        failure = row["failure"] or self.failure
        if failure is None and t >= self.timeout_s:
            failure = self.failure = (
                "locked_grounded_initialization_not_stable"
                if self.settled_at is None
                else "onboard_sensors_not_ready_before_timeout"
                if self.stable_for_s(t) >= STABLE_WINDOW_S - 1e-9
                else "locked_robot_not_stationary_before_timeout"
            )
        return dict(
            row,
            phase=(
                "JOINT_BRAKE_INITIALIZATION"
                if self.settled_at is None
                else "JOINT_BRAKE_AWAITING_FIRST_WALK_OUTPUT"
            ),
            policy_enabled=False,
            navigation_enabled=False,
            score_enabled=False,
            task_started_at_s=None,
            task_time_s=None,
            failure=failure,
        )

    def note_capture(self, timestamp_s, rgb, tof):
        self.capture = capture_status(timestamp_s, rgb, tof)
        return self.capture

    def ready(self, timestamp_s, packet):
        """True when the first walking-actor output may be computed and the brake released."""
        reasons = []
        if self.release_at is not None:
            reasons.append("already_released")
        if self.failure is not None:
            reasons.append("initialization_failed")
        if self.settled_at is None:
            reasons.append("locked_robot_not_settled")
        elif self.stable_for_s(timestamp_s) < STABLE_WINDOW_S - 1e-9:
            reasons.append("locked_robot_not_stationary")
        if self.capture is None:
            reasons.append("no_capture_delivered")
        else:
            reasons += self.capture["reasons"]
            age = float(timestamp_s) - self.capture["timestamp_s"]
            if not 0 <= age <= MAX_CAPTURE_AGE_S:
                reasons.append("capture_not_fresh")
        reasons += imu_status(packet)["reasons"]
        self.last_check = dict(timestamp_s=float(timestamp_s), ready=not reasons, reasons=reasons)
        return not reasons

    def mark_released(self, timestamp_s):
        if self.release_at is not None:
            raise RuntimeError("Brake release recorded twice")
        if not self.last_check or not self.last_check["ready"] or (
            self.last_check["timestamp_s"] != float(timestamp_s)
        ):
            raise RuntimeError("Release requires a ready check at the same tick")
        self.release_at = float(timestamp_s)

    def check_wrench(self, peak):
        """The actual external wrench buffer must stay zero at all times, not only while scoring."""
        peak = float(peak)
        if not math.isfinite(peak) or peak > 1e-12:
            raise RuntimeError("Nonzero external wrench on the onboard brake start-up")
        self.max_wrench = max(self.max_wrench, peak)

    def metadata(self):
        return dict(
            self.inner.metadata(),
            kind="standing61 joint brake at HOME on measured grounded root; free root; no body support",
            settled_at_s=self.settled_at,
            task_started_at_s=self.release_at,
            task_clock_starts_at_release=True,
            failure=self.failure or self.inner.failure,
            last_ready_check=self.last_check,
            last_capture_check=self.capture,
            max_capture_age_s=MAX_CAPTURE_AGE_S,
            stationary_until_release=dict(
                min_upright=MIN_UPRIGHT,
                max_speed_m_s=MAX_SPEED_M_S,
                max_angular_speed_rad_s=MAX_ANGULAR_SPEED_RAD_S,
                window_s=STABLE_WINDOW_S,
                stable_since_s=self.stable_since,
                resets_before_release=self.stability_resets,
            ),
            actual_external_wrench_peak=self.max_wrench,
        )
