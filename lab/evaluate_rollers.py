"""Independently score the recorded gliding task; never called by its controller."""

from pathlib import Path
import argparse
import hashlib
import json
import math
from PIL import Image


def evaluate(run):
    import numpy as np

    report = json.loads((run / "report.json").read_text())
    physics = [json.loads(x) for x in (run / "physical-evaluation.jsonl").open()]
    actions = [json.loads(x) for x in (run / "action-trace.jsonl").open()]
    visual = (
        [json.loads(x) for x in (run / "visual-control.jsonl").open()]
        if report["autonomous"]
        else []
    )
    kick = report["task_result"]["kick_at_s"]
    window = report["task_result"]["contact_window_s"]
    contacts = [r for r in physics if r["ball_contact_force_n"] > 0.05]
    early = any(kick is None or r["timestamp_s"] < kick for r in contacts)
    impacts = [
        r
        for r in contacts
        if kick is not None and 0 <= r["timestamp_s"] - kick <= window
    ]
    first_impact = impacts[0]["timestamp_s"] if impacts else None
    scene = report["scene_profile"]
    radius = scene["ball_diameter_m"] / 2
    plane = scene["goal_plane_x_m"] + radius
    goal = None
    for a, b in zip(physics, physics[1:]):
        p, q = a["ball_position"], b["ball_position"]
        if p[0] < plane <= q[0]:
            f = (plane - p[0]) / (q[0] - p[0])
            y = p[1] + f * (q[1] - p[1])
            z = p[2] + f * (q[2] - p[2])
            if (
                abs(y) + radius <= scene["goal_opening_width_m"] / 2
                and z - radius >= -0.001
                and z + radius <= scene["goal_opening_height_m"]
            ):
                goal = a["timestamp_s"] + f * (b["timestamp_s"] - a["timestamp_s"])
                break
    fell = any(r["gravity_z"] > -0.6 or r["robot_position"][2] < 0.065 for r in physics)
    tail = []
    for r in reversed(physics):
        if (
            kick is None
            or r["timestamp_s"] < kick + window
            or r["linear_speed"] >= 0.02
            or r["angular_speed"] >= 0.3
            or r["gravity_z"] >= -0.94
        ):
            break
        tail.append(r)
    recovery = tail[0]["timestamp_s"] - tail[-1]["timestamp_s"] if tail else 0.0
    cadence = all(
        abs(r["timestamp_s"] - (i + 1) * 0.005) < 1e-6 for i, r in enumerate(physics)
    ) and len(physics) == 4 * len(actions)
    shapes = all(
        len(r["observation"]) == 61
        and len(r["executed_action"]) == 14
        and len(r["raw_action"]) == 14
        for r in actions
    )
    sensor_schema = all(
        set(r["observation"])
        == {
            "timestamp_s",
            "joint_names",
            "q_rad",
            "qd_rad_s",
            "gyro_body_rad_s",
            "gravity_body",
        }
        and r["truth_used"] is False
        and len(r["observation"]["q_rad"]) == 14
        for r in visual
    )
    requests = [r for r in visual if r["kick_request"]]
    handoff = bool(requests) and all(
        r["ball_age_s"] < 0.1 and r["goal_age_s"] <= 6 and r["head_ik"]["accepted"]
        for r in requests
    )
    image_integrity = all(
        hashlib.sha256(
            Image.open(run / r["rgb_file"]).convert("RGB").tobytes()
        ).hexdigest()
        == r["detections"]["image_sha256"]
        for r in visual
    )

    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, (float, int)) or math.isfinite(value)

    finite_records = all(finite(r) for r in physics + actions + visual)
    expected_model = "cf05651d2708a2f9364212e86b866c97a70ace8131c492500105e8f28bf99afd"
    mode = (
        report["locomotion"] == "rollers"
        and report["roller_shot"] == "glide"
        and report["physics_backend"] == "newton"
        and report["resets"] == 0
        and report["models"]["locomotion"] == expected_model
        and report["physics_hz"] == 200
        and report["policy_hz"] == 50
    )
    policy_replay = False
    policy_max_error = None
    if finite_records and shapes:
        import numpy as np
        import onnxruntime as ort

        actor_path = (
            Path(__file__).resolve().parents[1] / "models/official-browser/rollers.onnx"
        )
        if hashlib.sha256(actor_path.read_bytes()).hexdigest() != expected_model:
            raise ValueError("Official roller actor identity changed")
        session = ort.InferenceSession(
            str(actor_path), providers=["CPUExecutionProvider"]
        )
        expected = np.concatenate(
            [
                session.run(
                    None,
                    {
                        session.get_inputs()[0].name: np.asarray(
                            [r["observation"]], dtype=np.float32
                        )
                    },
                )[0]
                for r in actions
            ],
            axis=0,
        )
        policy_max_error = float(
            np.max(np.abs(expected - np.asarray([r["raw_action"] for r in actions])))
        )
        policy_replay = policy_max_error < 1e-4
    action_cadence = all(
        abs(r["timestamp_s"] - i * 0.02) < 1e-6 for i, r in enumerate(actions)
    )
    request_at = report["task_result"].get("visual_request_at_s", kick)
    request_link = (
        len(requests) == 1
        and kick is not None
        and request_at is not None
        and abs(requests[0]["timestamp_s"] - request_at) < 1e-6
        and 0 <= kick - request_at <= 4.001
    )
    visual_cadence = (
        bool(visual)
        and visual[0]["timestamp_s"] <= 0.021
        and all(
            0 < b["timestamp_s"] - a["timestamp_s"] <= 0.081
            for a, b in zip(visual, visual[1:])
        )
    )
    visual_link = visual_cadence and all(
        abs(r["observation"]["timestamp_s"] - r["timestamp_s"]) < 1e-6
        and 0 <= round(r["timestamp_s"] / 0.02) < len(actions)
        and abs(
            actions[round(r["timestamp_s"] / 0.02)]["timestamp_s"] - r["timestamp_s"]
        )
        < 1e-6
        for r in visual
    )
    tof_handoff = bool(requests) and all(
        r["geometry"].get("floor_observation", {}).get("accepted", False)
        for r in requests
    )
    preparation = True
    if (
        report["task_result"].get("visual_request_at_s") is not None
        and kick is not None
    ):
        # During head release the robot may correct heading, but cannot propel.
        preparation_rows = [r for r in actions if request_at <= r["timestamp_s"] < kick]
        quiet_rows = [r for r in actions if kick - 0.4 <= r["timestamp_s"] < kick]
        preparation = (
            (2.0 if report.get("imu_velocity_estimator") else 1.0) - 1e-6
            <= kick - request_at
            <= 4.0 + 1e-6
            and bool(preparation_rows)
            and all(
                abs(r["command"][1]) < 1e-8
                and (
                    r.get("brake_active", False)
                    if report.get("brake_controller")
                    else abs(
                        r["command"][0]
                        - (
                            float(
                                np.clip(
                                    -3 * r["inertial_velocity_body_m_s"][0], -0.5, 0.35
                                )
                            )
                            if abs(r.get("inertial_velocity_body_m_s", [0])[0]) > 0.008
                            else 0.0
                        )
                    )
                    < 1e-7
                )
                for r in preparation_rows
            )
            and len(quiet_rows) >= 19
            and all(
                sum(v * v for v in r["observation"][:3]) < 0.25**2 for r in quiet_rows
            )
        )
    handoff = handoff and request_link and visual_link and tof_handoff and preparation
    stopped_before_goal = bool(tail) and all(
        r["robot_position"][0] + 0.06 < scene["goal_plane_x_m"] for r in tail
    )

    import numpy as np
    from bisect import bisect_right

    visual_times = [v["timestamp_s"] for v in visual]
    executed_chain = bool(shapes and finite_records)
    if executed_chain:
        for a in actions:
            expected = np.asarray(a["raw_action"], dtype=float).copy()
            t = a["timestamp_s"]
            if visual and t >= 2.0:
                j = bisect_right(visual_times, t + 1e-9) - 1
                if j < 0:
                    executed_chain = False
                    break
                weight = (
                    1.0
                    if request_at is None or t < request_at
                    else max(0.0, 1.0 - (t - request_at) / 0.4)
                )
                expected[5:9] = (1 - weight) * expected[5:9] + weight * np.asarray(
                    visual[j]["head_offsets"]
                )
            if np.max(np.abs(expected - np.asarray(a["executed_action"]))) > 2e-6:
                executed_chain = False
                break
    executed_chain = (
        executed_chain
        and all(
            np.max(np.abs(np.asarray(a["observation"][48:51]) - a["command"])) < 1e-6
            for a in actions
        )
        and all(
            np.max(np.abs(np.asarray(b["observation"][34:48]) - a["executed_action"]))
            < 1e-6
            for a, b in zip(actions, actions[1:])
        )
    )
    handoff_heading = True
    if kick is not None and visual and request_at is not None:
        j = bisect_right(visual_times, kick + 1e-9) - 1
        heading = visual[j].get("goal_heading_rad") if j >= 0 else None
        if heading is None:
            handoff_heading = False
        else:
            heading -= sum(
                a["observation"][2] * 0.02
                for a in actions
                if visual_times[j] + 1e-8 < a["timestamp_s"] <= kick + 1e-8
            )
            handoff_heading = abs(heading) < 0.04001
    handoff = handoff and handoff_heading

    imu_valid = not report.get("imu_velocity_estimator")
    imu_replay_error = None
    imu_speed_error = None
    if report.get("imu_velocity_estimator"):
        import numpy as np

        packets = [json.loads(x) for x in (run / "imu.jsonl").read_text().splitlines()]
        keys = {
            "timestamp_s",
            "dt_s",
            "specific_force_body_m_s2",
            "gyro_body_rad_s",
            "gravity_body",
            "estimated_velocity_body_m_s",
        }
        packet_contract = len(packets) == len(physics) and all(
            set(p) == keys
            and finite(p)
            and abs(p["timestamp_s"] - (i + 1) * 0.005) < 1e-6
            and abs(p["dt_s"] - 0.005) < 1e-9
            and all(len(p[k]) == 3 for k in keys - {"timestamp_s", "dt_s"})
            for i, p in enumerate(packets)
        )
        velocity = np.zeros(3)
        replay = []
        if packet_contract:
            for p in packets:
                velocity = velocity + 0.005 * (
                    np.asarray(p["specific_force_body_m_s2"])
                    + 9.81 * np.asarray(p["gravity_body"])
                    - np.cross(p["gyro_body_rad_s"], velocity)
                )
                replay.append(velocity.copy())
            imu_replay_error = float(
                np.max(
                    np.abs(
                        np.asarray(replay)
                        - np.asarray(
                            [p["estimated_velocity_body_m_s"] for p in packets]
                        )
                    )
                )
            )
            imu_speed_error = float(
                np.max(
                    np.abs(
                        np.linalg.norm(replay, axis=1)
                        - np.asarray([p["linear_speed"] for p in physics])
                    )
                )
            )
            action_estimate_link = all(
                np.max(
                    np.abs(
                        np.asarray(a["inertial_velocity_body_m_s"])
                        - (np.zeros(3) if i == 0 else replay[4 * i - 1])
                    )
                )
                < 1e-7
                for i, a in enumerate(actions)
            )
            gyro_gravity_link = all(
                np.max(
                    np.abs(
                        np.asarray(a["observation"][:6])
                        - np.r_[
                            packets[4 * i - 1]["gyro_body_rad_s"],
                            packets[4 * i - 1]["gravity_body"],
                        ]
                    )
                )
                < 1e-4
                for i, a in enumerate(actions)
                if i > 0
            )
            visual_estimate_link = all(
                np.max(
                    np.abs(
                        np.asarray(v["inertial_velocity_body_m_s"])
                        - np.asarray(
                            actions[round(v["timestamp_s"] / 0.02)][
                                "inertial_velocity_body_m_s"
                            ]
                        )
                    )
                )
                < 1e-7
                for v in visual
            )
            imu_valid = bool(
                imu_replay_error < 1e-7
                and imu_speed_error < 0.02
                and action_estimate_link
                and gyro_gravity_link
                and visual_estimate_link
            )
    brake_chain = True
    if report.get("brake_controller"):
        brake_kind = report["brake_controller"].get("kind")
        brake_chain = brake_kind in ["PI", "PI_wait_P_stop"] and report[
            "brake_controller"
        ] == dict(
            kind=brake_kind,
            kp=3.0,
            ki=4.0,
            limits=[-0.5, 0.35],
            integral_limits=[-0.15, 0.15],
            dt_s=0.02,
        )
        integral = 0.0
        for a in actions:
            if a["brake_active"]:
                v = a["inertial_velocity_body_m_s"][0]
                if (
                    brake_kind == "PI_wait_P_stop"
                    and kick is not None
                    and a["timestamp_s"] >= kick
                ):
                    integral = 0.0
                    expected_command = (
                        max(-0.5, min(0.35, -3 * v)) if abs(v) > 0.008 else 0.0
                    )
                else:
                    candidate = max(-0.15, min(0.15, integral + 0.02 * v))
                    if -0.5 <= -3 * v - 4 * candidate <= 0.35:
                        integral = candidate
                    expected_command = max(-0.5, min(0.35, -3 * v - 4 * integral))
                brake_chain = (
                    brake_chain and abs(a["command"][0] - expected_command) < 1e-6
                )
            else:
                integral = 0.0
            brake_chain = brake_chain and abs(a["brake_integral_m"] - integral) < 1e-7
        if kick is not None:
            push = report["fixture"].get(
                "planned_push_seconds", report["fixture"]["shot_push_seconds"]
            )
            for a in actions:
                if a["timestamp_s"] < kick - 1e-8:
                    continue
                propelling = a["timestamp_s"] < kick + push - 1e-8
                brake_chain = brake_chain and (
                    not a["brake_active"] and abs(a["command"][0] - 0.6) < 1e-6
                    if propelling
                    else a["brake_active"]
                )
                brake_chain = (
                    brake_chain and abs(a["command"][1]) + abs(a["command"][2]) < 1e-7
                )
            if visual and len(requests) == 1:
                expected_push = report["fixture"]["shot_push_seconds"] + max(
                    0.0, min(0.5, (requests[0]["ball_body_level_m"][0] - 0.18) / 0.35)
                )
                brake_chain = brake_chain and abs(push - expected_push) < 1e-7

    stopped_before_goal = stopped_before_goal and all(
        r["robot_position"][0] + 0.06 < scene["goal_plane_x_m"]
        for r in physics
        if kick is not None and r["timestamp_s"] >= kick
    )

    cadence = cadence and action_cadence

    physical = (
        first_impact is not None
        and goal is not None
        and goal >= first_impact
        and not early
        and not fell
        and recovery >= 1.0
        and stopped_before_goal
    )
    result = dict(
        scope="Empty-field gliding shot; ideal trunk IMU, RGB/ToF, no Nav2; not hardware qualification",
        checks=dict(
            physics_action_cadence=cadence,
            actor_dimensions=shapes,
            explicit_sensor_schema=sensor_schema,
            image_integrity=image_integrity,
            visual_handoff=handoff,
            mode_and_model=mode,
            official_actor_replay=policy_replay,
            finite_records=finite_records,
            visual_time_chain=visual_link,
            tof_at_handoff=tof_handoff,
            head_release_preparation=preparation,
            stopped_before_goal=stopped_before_goal,
            imu_packet_chain=imu_valid,
            executed_action_chain=executed_chain,
            handoff_heading=handoff_heading,
            brake_command_chain=bool(brake_chain),
        ),
        real_wheel_ball_contact=first_impact is not None,
        pre_shot_contact=early,
        goal_time_s=goal,
        contact_time_s=first_impact,
        fallen=fell,
        continuous_final_recovery_s=recovery,
        local_physical_sequence_passed=bool(
            physical
            and mode
            and finite_records
            and cadence
            and shapes
            and policy_replay
            and imu_valid
            and executed_chain
            and brake_chain
        ),
        autonomous_task_passed=bool(
            report["autonomous"]
            and physical
            and mode
            and finite_records
            and cadence
            and shapes
            and policy_replay
            and sensor_schema
            and handoff
            and image_integrity
            and imu_valid
            and executed_chain
            and brake_chain
        ),
        official_actor_max_abs_error=policy_max_error,
        imu_replay_max_abs_error=imu_replay_error,
        imu_speed_magnitude_max_error_m_s=imu_speed_error,
        final_speed_threshold_m_s=0.02,
        all_files_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                run / "report.json",
                run / "physical-evaluation.jsonl",
                run / "action-trace.jsonl",
            ]
            + [
                p
                for p in [run / "imu.jsonl", run / "visual-control.jsonl"]
                if p.exists()
            ]
        },
    )
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = evaluate(a.run)
    with a.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
