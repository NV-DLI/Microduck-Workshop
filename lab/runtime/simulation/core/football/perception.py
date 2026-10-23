"""Perception: Microduck simulation and control."""

from dataclasses import asdict
from pathlib import Path
import hashlib, json, math
import numpy as np
import cv2
from beta_sensor_profile import BetaCameraProfile, PROFILE_ID
from beta_perception_geometry import CaptureCameraPose, PlaneEstimate
from hardware_head_fk import CourseHeadFK, MJCF_PATH
from leg_odometry import LegFK
from vision.contour import extract_white_ball_contour
from vision.grounded_direction import estimate_grounded_direction
from vision.goal_points import extract_goal_bases
from vision.goal_geometry import project_goal_parameterss
from vision.beta_box_geometry import estimate_box_sphere
from vision.beta_goal_points import extract_beta_goal_bases
from vision.known_goal_geometry import project_known_post_pair
from ground_plane import fit_virtual_floor

P = Path(__file__).resolve().parent
OBS_KEYS = {
    "timestamp_s",
    "joint_names",
    "q_rad",
    "qd_rad_s",
    "gyro_body_rad_s",
    "gravity_body",
}


def validate_observation(o):
    if set(o) != OBS_KEYS:
        raise ValueError(
            "Encoder/IMU schema only; hidden state or unknown fields forbidden"
        )
    if len(o["joint_names"]) != 14 or len(set(o["joint_names"])) != 14:
        raise ValueError("14 explicit effective output encoder channels required")
    for key, n in [
        ("q_rad", 14),
        ("qd_rad_s", 14),
        ("gyro_body_rad_s", 3),
        ("gravity_body", 3),
    ]:
        v = np.asarray(o[key], dtype=float)
        if v.shape != (n,) or not np.isfinite(v).all():
            raise ValueError("Invalid " + key)
    if (
        isinstance(o["timestamp_s"], bool)
        or not math.isfinite(o["timestamp_s"])
        or o["timestamp_s"] < 0
    ):
        raise ValueError("Bad timestamp")
    if not 0.5 < float(np.linalg.norm(o["gravity_body"])) < 1.5:
        raise ValueError("Projected gravity not unit")


def level_rotation(gravity):
    n = -np.asarray(gravity, float)
    norm = np.linalg.norm(n)
    if not np.isfinite(norm) or not 0.5 < norm < 1.5:
        raise ValueError("Projected gravity not unit")
    n /= norm
    x = np.array([1.0, 0.0, 0.0])
    x -= n * np.dot(n, x)
    if np.linalg.norm(x) < 0.3:
        raise ValueError("Body pitch too steep for level frame")
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    return np.stack([x, y, n])


class BetaPerception:
    def __init__(self, camera=None):
        self.camera = camera or BetaCameraProfile(width_px=960, height_px=540)
        self.head = CourseHeadFK()
        self.leg = LegFK(MJCF_PATH, require_pinned=False)
        self.vertices = json.loads(
            (P / "sources/sole-collision-geometry.json").read_text()
        )
        # Independent current USD/CAD zero-position agreement, not old geometry reuse.
        import xml.etree.ElementTree as ET

        names = [
            j.get("name")
            for j in ET.parse(MJCF_PATH).getroot().findall(".//joint")
            if j.get("name")
        ]
        names = list(dict.fromkeys(names))
        for name, geom in self.vertices.items():
            point = self.leg.foot(name, [0.0] * len(names), names)["position_body_m"]
            if math.dist(point, geom["USD_zero_joint_site_body_m"]) > 2e-6:
                raise ValueError("New CAD/USD foot-frame mismatch")

    def pose(self, o, floor=None):
        validate_observation(o)
        t = o["timestamp_s"]
        g = np.asarray(o["gravity_body"], float)
        n = -g / np.linalg.norm(g)
        if not 0.5 < np.linalg.norm(g) < 1.5:
            raise ValueError("Projected gravity not unit")
        lows = []
        for name, geometry in self.vertices.items():
            foot = self.leg.foot(name, o["q_rad"], o["joint_names"])
            R = np.asarray(foot["rotation"])
            site = np.asarray(self.leg.paths[name][1])
            ankle = np.asarray(foot["position_body_m"]) - R @ site
            verts = np.asarray(geometry["vertices_ankle_m"])
            lows.append(float((verts @ (R.T @ n) + n @ ankle).min()))
        height = -min(lows)
        fk = self.head.forward_policy14(o["q_rad"], joint_names=o["joint_names"])[
            "camera_raw_optical"
        ]
        frame = f"body_capture_{t:.9f}"
        pose = CaptureCameraPose(
            t,
            frame,
            fk.rotation,
            fk.translation_m,
            "October CAD + same-time effective output joint encoders; no world pose",
        )
        ground = PlaneEstimate(
            frame,
            tuple(n),
            -height,
            "IMU gravity + lowest current USD sole hull; conditional flat-ground support, no contact truth",
        )
        if floor is not None and floor["accepted"]:
            n = np.asarray(floor["normal_body"])
            height = -floor["offset_m"]
            ground = PlaneEstimate(
                frame,
                tuple(n),
                floor["offset_m"],
                "Actual64zone distances + declared virtual nearest-zone raster model; not hardware calibrated",
            )
        return (
            pose,
            ground,
            {
                "conditional_root_height_m": height,
                "ground_normal_body": n.tolist(),
                "flat_ground_support_assumption": not bool(floor and floor["accepted"]),
                "actual_ToF_plane_used": bool(floor and floor["accepted"]),
            },
        )

    def parameterss(self, rgb):
        """Image-only white course props; not a trained detector or YOLO result."""
        a = np.asarray(rgb)
        white = (
            (a.min(axis=2) >= 160) & ((a.max(axis=2).astype(int) - a.min(axis=2)) < 35)
        ).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
        out = []
        for i in range(1, count):
            x, y, w, h, area = map(int, stats[i])
            fill = area / (w * h)
            if (
                area < 35
                or min(w, h) < 7
                or x < 3
                or y < 3
                or x + w > a.shape[1] - 3
                or y + h > a.shape[0] - 3
            ):
                continue
            name = (
                "ball"
                if 0.65 < w / h < 1.5 and fill > 0.55
                else (
                    "goal"
                    if 0.8 < w / h < 6 and fill < 0.65 and min(w, h) > 18
                    else None
                )
            )
            if name:
                out.append(
                    {
                        "class_name": name,
                        "confidence": 0.5,
                        "bbox_xyxy_px": [x, y, x + w, y + h],
                    }
                )
        return out

    def estimate(self, rgb, o, detections=None, *, now_s, tof_frame=None):
        a = np.asarray(rgb)
        cam = self.camera
        if a.dtype != np.uint8 or a.shape != (cam.height_px, cam.width_px, 3):
            raise ValueError(
                "Actual original Beta landscape RGB required at declared camera size"
            )
        validate_observation(o)
        floor = (
            fit_virtual_floor(tof_frame, o, self.head)
            if tof_frame is not None
            else None
        )
        pose, ground, height = self.pose(o, floor)
        stamp = o["timestamp_s"]
        result = {
            "schema": "microduck.beta-geometry.v1",
            "hardware_profile_id": PROFILE_ID,
            "timestamp_s": stamp,
            "image_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "camera": cam.to_dict(),
            "pose": asdict(pose),
            "ground": asdict(ground),
            "height": height,
            "ball": None,
            "goal": None,
            "truth_used": False,
            "detector_source": (
                "image_only_white_course_prop_fallback"
                if detections is None
                else "caller_YOLO_original_pixel_boxes"
            ),
        }
        result["floor_observation"] = floor
        if floor is not None and not floor["accepted"]:
            return {**result, "reason": "actual_ToF_floor_unavailable"}
        if not 0 <= now_s - stamp <= 0.2:
            return {**result, "reason": "stale_or_future_RGB"}
        rows = self.parameterss(a) if detections is None else detections
        for row in rows:
            if set(row) != {"class_name", "confidence", "bbox_xyxy_px"}:
                raise ValueError("Detection schema forbids hidden state")
            b = row["bbox_xyxy_px"]
            if (
                row["class_name"] not in ("ball", "goal", "football")
                or not 0.45 <= row["confidence"] <= 1
                or len(b) != 4
                or not all(math.isfinite(v) for v in b)
                or not 0 <= b[0] < b[2] < cam.width_px
                or not 0 <= b[1] < b[3] < cam.height_px
            ):
                raise ValueError("Bad native-raster detection")
        result["detections"] = rows
        for row in sorted(rows, key=lambda r: -r["confidence"]):
            box = row["bbox_xyxy_px"]
            name = "ball" if row["class_name"] == "football" else row["class_name"]
            if result[name] and result[name].get("accepted"):
                continue
            if name == "ball" and detections is not None:
                v = estimate_box_sphere(
                    a,
                    box,
                    radius_m=0.035,
                    pose=pose,
                    ground=ground,
                    camera=cam,
                    capture_timestamp_s=stamp,
                    now_s=now_s,
                )
                result[name] = {
                    "accepted": v["accepted"],
                    "reason": v["reason"],
                    "geometry": v,
                }
            elif name == "ball":
                contour = extract_white_ball_contour(a, box)
                r = {"accepted": False, "reason": contour["reason"], "contour": contour}
                if contour["accepted"]:
                    points = np.asarray(
                        contour["contour_xy_px"], float
                    )  # Beta identity, no CW90
                    xy = cv2.undistortPoints(
                        points[:, None, :],
                        np.asarray(cam.K),
                        np.asarray(cam.distortion_k1_k2_p1_p2_k3),
                    )[:, 0, :]
                    rays = np.column_stack([xy, np.ones(len(xy))])
                    rays /= np.linalg.norm(rays, axis=1)[:, None]
                    v = estimate_grounded_direction(
                        rays,
                        radius_m=0.035,
                        pose=pose,
                        ground=ground,
                        capture_timestamp_s=stamp,
                        now_s=now_s,
                    )
                    r.update(accepted=v["accepted"], reason=v["reason"], geometry=v)
                result[name] = r
            else:
                pad = [
                    max(1.0, box[0] - 8),
                    max(1.0, box[1] - 8),
                    min(cam.width_px - 2.0, box[2] + 8),
                    min(cam.height_px - 2.0, box[3] + 8),
                ]
                gravity = np.asarray(pose.rotation_reference_from_raw).T @ (
                    -np.asarray(ground.normal)
                )
                parameterss = (
                    extract_goal_bases(a, pad, gravity, cam)
                    if detections is None
                    else extract_beta_goal_bases(a, box, gravity, cam)
                )
                v = (
                    project_goal_parameterss(
                        parameterss,
                        pose=pose,
                        ground=ground,
                        capture_timestamp_s=stamp,
                        now_s=now_s,
                        camera=cam,
                    )
                    if detections is None
                    else project_known_post_pair(
                        parameterss,
                        pose=pose,
                        ground=ground,
                        capture_timestamp_s=stamp,
                        now_s=now_s,
                        camera=cam,
                    )
                )
                result[name] = {
                    "accepted": v["accepted"],
                    "reason": v["reason"],
                    "geometry": v,
                    "parameterss": parameterss,
                }
        result["reason"] = "observed_parameterss_only"
        return result
