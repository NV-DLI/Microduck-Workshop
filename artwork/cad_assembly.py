"""Pinned Microduck CAD geometry, kinematics, materials and assembly helpers.
Use build_microduck.py for the complete web and editable Blender outputs.
Original visual STL geometry and the MJCF STAND pose are the authority.
"""

import json
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

try:
    import bpy  # noqa: F401
    from mathutils import Matrix, Vector

    IN_BLENDER = True
except ImportError:
    bpy = None
    IN_BLENDER = False

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "source")
ROBOT_XML = os.path.join(SRC, "robot_allcollisions.xml")
SCENE_XML = os.path.join(SRC, "scene_allcollisions.xml")
OUT = os.path.dirname(os.path.abspath(__file__))

KEYFRAME = "STAND"
WELD_GRID = 1e7  # vertex weld grid: 0.1 micrometre (STL is in metres)
SMOOTH_ANGLE_DEG = 30.0  # faces meeting at a sharper angle keep a hard edge

CAM_LOC = (0.50, -0.65, 0.35)
CAM_TARGET = (0.0, 0.0, 0.13)
ORTHO_ASSEMBLED = 0.40
ORTHO_EXPLODED = 0.90
RES = (1600, 1200)
SAMPLES = 32  # with a denoiser
SAMPLES_NO_DENOISER = 128  # adaptive; used when the Blender build has no denoiser
FRAME_ASSEMBLED = 1
FRAME_EXPLODED = 61


# --------------------------------------------------------------------------
# Small rigid-transform library (4x4 row-major nested lists, MJCF wxyz quats)
# --------------------------------------------------------------------------


def vec(s, n=3):
    v = [float(x) for x in s.split()]
    assert len(v) == n, s
    return v


def quat_norm(q):
    n = math.sqrt(sum(c * c for c in q))
    return [c / n for c in q]


def quat_to_rot(q):
    w, x, y, z = quat_norm(q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def axis_angle_rot(axis, a):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s, t = math.cos(a), math.sin(a), 1 - math.cos(a)
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def tf(R=None, p=(0.0, 0.0, 0.0)):
    R = R or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return [
        list(R[0]) + [p[0]],
        list(R[1]) + [p[1]],
        list(R[2]) + [p[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mul(A, B):
    return [
        [sum(A[i][k] * B[k][j] for k in range(4)) for j in range(4)] for i in range(4)
    ]


def apply(M, p):
    return [
        M[i][0] * p[0] + M[i][1] * p[1] + M[i][2] * p[2] + M[i][3] for i in range(3)
    ]


def apply_dir(M, d):
    return [M[i][0] * d[0] + M[i][1] * d[1] + M[i][2] * d[2] for i in range(3)]


def norm(v):
    n = math.sqrt(sum(c * c for c in v))
    return [c / n for c in v] if n > 1e-12 else [0.0, 0.0, 0.0]


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


def scale(a, s):
    return [c * s for c in a]


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def to_gltf(v):
    """Blender Z-up (x, y, z) -> glTF Y-up (x, z, -y)."""
    return [v[0], v[2], -v[1]]


# --------------------------------------------------------------------------
# MJCF parsing + forward kinematics
# --------------------------------------------------------------------------


def parse_xml_with_comments(path):
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    with open(path, "rb") as f:
        parser.feed(f.read())
    return parser.close()


def parse_keyframe(name):
    root = ET.parse(SCENE_XML).getroot()
    for key in root.iter("key"):
        if key.get("name") == name:
            return [float(x) for x in key.get("qpos").split()]
    raise RuntimeError("keyframe %s not found in %s" % (name, SCENE_XML))


def load_model():
    """Return (bodies, joints, geoms, materials, meshes) in world frame at STAND."""
    root = parse_xml_with_comments(ROBOT_XML)
    materials = {}
    meshes = {}
    for m in root.iter("material"):
        materials[m.get("name")] = [float(x) for x in m.get("rgba").split()]
    for m in root.iter("mesh"):
        f = m.get("file")
        meshes[m.get("name") or os.path.splitext(f)[0]] = f

    # Collect hinge joints in document (= qpos) order first.
    world = root.find("worldbody")
    hinge_order = [
        j.get("name") for j in world.iter("joint") if j.get("type", "hinge") == "hinge"
    ]
    qpos = parse_keyframe(KEYFRAME)
    n_free = 7
    assert len(qpos) == n_free + len(hinge_order), (len(qpos), len(hinge_order))
    q_of = dict(zip(hinge_order, qpos[n_free:]))
    free_pos, free_quat = qpos[0:3], qpos[3:7]

    bodies, joints, geoms = [], [], []

    def walk(elem, parent_M, parent_name, depth):
        for child in elem:
            if child.tag != "body":
                continue
            name = child.get("name")
            pos = vec(child.get("pos", "0 0 0"))
            quat = vec(child.get("quat", "1 0 0 0"), 4)
            free = child.find("freejoint") is not None
            if free:  # the freejoint qpos replaces the body's default placement
                M_rest = tf(quat_to_rot(free_quat), free_pos)
            else:
                M_rest = tf(quat_to_rot(quat), pos)
            M = mul(parent_M, M_rest)
            joint_info = None
            for j in child.findall("joint"):
                jname = j.get("name")
                axis = vec(j.get("axis", "0 0 1"))
                jpos = vec(j.get("pos", "0 0 0"))
                q = q_of[jname]
                # hinge about axis through jpos (body frame): T(jpos) R T(-jpos)
                J = mul(
                    mul(tf(None, jpos), tf(axis_angle_rot(axis, q))),
                    tf(None, [-c for c in jpos]),
                )
                anchor = apply(M, jpos)
                world_axis = norm(apply_dir(M, axis))
                M = mul(M, J)
                rng = [float(x) for x in j.get("range", "0 0").split()]
                joint_info = dict(
                    name=jname,
                    type="hinge",
                    parent_body=parent_name,
                    child_body=name,
                    axis_local=axis,
                    anchor_world=anchor,
                    axis_world=world_axis,
                    range_rad=rng,
                    stand_rad=q,
                    actuator_class=j.get("class"),
                )
                joints.append(joint_info)
            bodies.append(
                dict(
                    name=name,
                    parent=parent_name,
                    depth=depth,
                    pos=pos,
                    quat=quat,
                    free=free,
                    joint=joint_info["name"] if joint_info else None,
                    M=M,
                )
            )
            # Visual geoms in this body, with their "<!-- Part xxx -->" instance names.
            last_part = None
            for g in child:
                if g.tag is ET.Comment:
                    txt = (g.text or "").strip()
                    if txt.startswith("Part "):
                        last_part = txt[5:].strip()
                    continue
                if g.tag != "geom" or g.get("class") != "visual":
                    continue
                gpos = vec(g.get("pos", "0 0 0"))
                gquat = vec(g.get("quat", "1 0 0 0"), 4)
                G = mul(M, tf(quat_to_rot(gquat), gpos))
                geoms.append(
                    dict(
                        instance=last_part,
                        mesh=g.get("mesh"),
                        material=g.get("material"),
                        body=name,
                        pos=gpos,
                        quat=gquat,
                        M=G,
                    )
                )
            walk(child, M, name, depth + 1)

    walk(world, tf(), "world", 0)
    return dict(
        bodies=bodies,
        joints=joints,
        geoms=geoms,
        materials=materials,
        meshes=meshes,
        hinge_order=hinge_order,
        qpos=qpos,
    )


# --------------------------------------------------------------------------
# STL reading (binary with ASCII fallback) + vertex welding
# --------------------------------------------------------------------------


def read_stl_triangles(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack_from("<I", data, 80)[0]
        if len(data) == 84 + 50 * n:
            out = []
            for i in range(n):
                v = struct.unpack_from("<9f", data, 84 + 50 * i + 12)
                out.append((v[0:3], v[3:6], v[6:9]))
            return out
    out, cur = [], []
    for line in data.decode("ascii", "replace").splitlines():
        t = line.split()
        if t and t[0] == "vertex":
            cur.append(tuple(float(x) for x in t[1:4]))
            if len(cur) == 3:
                out.append(tuple(cur))
                cur = []
    return out


def weld(tris):
    """Merge coincident STL corners; drop degenerate and duplicate triangles."""
    index, verts, faces, seen = {}, [], [], set()
    n_degenerate = n_duplicate = 0
    for tri in tris:
        f = []
        for v in tri:
            k = (
                round(v[0] * WELD_GRID),
                round(v[1] * WELD_GRID),
                round(v[2] * WELD_GRID),
            )
            i = index.get(k)
            if i is None:
                i = index[k] = len(verts)
                verts.append((float(v[0]), float(v[1]), float(v[2])))
            f.append(i)
        if f[0] == f[1] or f[1] == f[2] or f[0] == f[2]:
            n_degenerate += 1
            continue
        key = tuple(sorted(f))
        if key in seen:
            n_duplicate += 1
            continue
        seen.add(key)
        faces.append(tuple(f))
    return (
        verts,
        faces,
        dict(
            stl_triangles=len(tris),
            degenerate_removed=n_degenerate,
            duplicate_removed=n_duplicate,
        ),
    )


def read_stl_numpy(path):
    """Fast path inside Blender (numpy is bundled). Same result as weld(read_stl_triangles())."""
    import numpy as np

    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack_from("<I", data, 80)[0] if len(data) >= 84 else -1
    if len(data) != 84 + 50 * n:
        return weld(read_stl_triangles(path))
    rec = np.dtype([("n", "<f4", 3), ("v", "<f4", 9), ("a", "<u2")])
    arr = np.frombuffer(data, dtype=rec, count=n, offset=84)
    corners = arr["v"].reshape(-1, 3).astype(np.float64)
    keys = np.round(corners * WELD_GRID).astype(np.int64)
    # first-occurrence ordering so vertex order matches the pure-python welder
    _, first, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    verts = corners[first[order]]
    tri = remap[inv].reshape(-1, 3)
    good = (
        (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])
    )
    n_deg = int((~good).sum())
    tri = tri[good]
    _, keep = np.unique(np.sort(tri, axis=1), axis=0, return_index=True)
    keep = np.sort(keep)
    n_dup = len(tri) - len(keep)
    tri = tri[keep]
    return (
        verts,
        tri,
        dict(
            stl_triangles=int(n), degenerate_removed=n_deg, duplicate_removed=int(n_dup)
        ),
    )


# --------------------------------------------------------------------------
# Descriptive catalogue (labels are ours; identifiers are the source's)
# --------------------------------------------------------------------------

GROUPS = {
    "torso": ("躯干", "Torso"),
    "left_leg": ("左腿", "Left leg"),
    "right_leg": ("右腿", "Right leg"),
    "neck": ("颈部", "Neck"),
    "head": ("头部与喙", "Head and beak"),
}

# source body -> (group, label_zh, label_en)
BODY_INFO = {
    "trunk_base": ("torso", "躯干", "Torso"),
    "yaw2roll": ("left_leg", "左髋偏航-滚转连杆", "Left hip yaw-to-roll link"),
    "hip_l": ("left_leg", "左髋滚转-俯仰连杆", "Left hip roll-to-pitch link"),
    "upper_leg_left": ("left_leg", "左大腿", "Left thigh"),
    "leg": ("left_leg", "左小腿", "Left shank"),
    "ankle_left": ("left_leg", "左脚", "Left foot"),
    "bearing_roll": ("right_leg", "右髋偏航-滚转连杆", "Right hip yaw-to-roll link"),
    "hip_l_2": ("right_leg", "右髋滚转-俯仰连杆", "Right hip roll-to-pitch link"),
    "upper_leg_right": ("right_leg", "右大腿", "Right thigh"),
    "leg_2": ("right_leg", "右小腿", "Right shank"),
    "ankle_right": ("right_leg", "右脚", "Right foot"),
    "neck": ("neck", "颈部连杆", "Neck link"),
    "neck_pitch": ("head", "头部俯仰支架", "Head-pitch bracket link"),
    "yaw_roll_motion": ("head", "头部偏航-滚转连杆", "Head yaw-to-roll link"),
    "jaw_soft": ("head", "头部（含喙）", "Head with beak"),
}

JOINT_LABELS = {
    "left_hip_yaw": ("左髋偏航", "left hip yaw"),
    "left_hip_roll": ("左髋滚转", "left hip roll"),
    "left_hip_pitch": ("左髋俯仰", "left hip pitch"),
    "left_knee": ("左膝", "left knee"),
    "left_ankle": ("左踝", "left ankle"),
    "neck_pitch": ("颈部俯仰", "neck pitch"),
    "head_pitch": ("头部俯仰", "head pitch"),
    "head_yaw": ("头部偏航", "head yaw"),
    "head_roll": ("头部滚转", "head roll"),
    "right_hip_yaw": ("右髋偏航", "right hip yaw"),
    "right_hip_roll": ("右髋滚转", "right hip roll"),
    "right_hip_pitch": ("右髋俯仰", "right hip pitch"),
    "right_knee": ("右膝", "right knee"),
    "right_ankle": ("右踝", "right ankle"),
}
BEAK = ("喙部开合", "beak open/close")

# source mesh -> (label_zh, label_en, description_zh, description_en, category, material)
MESH_INFO = {
    "trunk_base": (
        "躯干底板",
        "Trunk base plate",
        "躯干主承力板，承载两侧髋偏航舵机、电池支架与左右外壳。",
        "Main load-bearing torso plate carrying both hip-yaw servos, the battery support and the side shells.",
        "structure",
        "anodized_graphite",
    ),
    "right_shell": (
        "右侧躯干外壳",
        "Right torso shell",
        "躯干右半外壳，包覆髋偏航舵机与电池。",
        "Right half of the torso cover enclosing the hip-yaw servos and battery.",
        "shell",
        "shell_offwhite",
    ),
    "left_shell": (
        "左侧躯干外壳",
        "Left torso shell",
        "躯干左半外壳，与右壳沿中面对合。",
        "Left half of the torso cover; mates with the right shell on the mid-plane.",
        "shell",
        "shell_offwhite",
    ),
    "banana_pcb_locker": (
        "香蕉插头板锁扣",
        "Banana-connector PCB locker",
        "躯干后上方的锁扣件（源名 Banana_PCB_locker）；被锁定的电路板本身不在源 CAD 中。",
        "Retainer at the rear top of the torso (source name Banana_PCB_locker); the PCB it locks is not in the source CAD.",
        "structure",
        "plastic_grey",
    ),
    "power_support": (
        "电池支架",
        "Battery support bracket",
        "固定于躯干底板后部，夹持 NP-F 系列电池。",
        "Bracket at the rear of the trunk plate that cradles the NP-F series battery pack.",
        "structure",
        "anodized_champagne",
    ),
    "np_f970": (
        "NP-F 系列电池",
        "NP-F series battery pack",
        "源零件名 np_f970 的锂电池包，CAD 外形约 38.6×20.6×70.8 mm，位于躯干后部。",
        "Li-ion battery pack (source part np_f970); CAD envelope about 38.6 x 20.6 x 70.8 mm, at the back of the torso.",
        "power",
        "battery_black",
    ),
    "xl330": (
        "XL330 舵机",
        "Dynamixel XL330 servo",
        "ROBOTIS Dynamixel XL330 智能舵机，输出轴为零件局部 +X 方向。",
        "ROBOTIS Dynamixel XL330 smart servo; its output shaft lies on the part's local +X axis.",
        "actuator",
        "servo_dark",
    ),
    "seeed_bearing__configuration__22x16x4": (
        "薄壁球轴承 22×16×4",
        "Thin-section ball bearing 22x16x4",
        "Seeed 薄壁球轴承（源配置 22x16x4），为关节的非驱动侧提供支撑。",
        "Seeed thin-section ball bearing (source configuration 22x16x4) supporting the idle side of a joint.",
        "bearing",
        "bearing_steel",
    ),
    "seeed_bearing__configuration_default": (
        "小型球轴承",
        "Small ball bearing",
        "Seeed 球轴承（源默认配置，外径约 15 mm）。",
        "Seeed ball bearing (source default configuration, about 15 mm OD).",
        "bearing",
        "bearing_steel",
    ),
    "yaw2roll": (
        "髋偏航-滚转支架",
        "Hip yaw-to-roll bracket",
        "连接髋偏航输出与髋滚转舵机的支架。",
        "Bracket joining the hip-yaw output to the hip-roll servo.",
        "structure",
        "anodized_graphite",
    ),
    "bearing_roll": (
        "髋滚转轴承板",
        "Hip-roll bearing plate",
        "髋滚转关节非驱动侧的轴承安装板。",
        "Plate that seats the bearing on the idle side of the hip-roll joint.",
        "structure",
        "anodized_graphite",
    ),
    "hip_l": (
        "髋滚转-俯仰支架",
        "Hip roll-to-pitch bracket",
        "U 形支架，将髋滚转输出转接到髋俯仰轴。",
        "U-bracket transferring the hip-roll output onto the hip-pitch axis.",
        "structure",
        "anodized_champagne",
    ),
    "upper_leg_left": (
        "左大腿外壳",
        "Left thigh shell",
        "左大腿结构外壳，内含髋俯仰与膝关节舵机。",
        "Left thigh structural shell housing the hip-pitch and knee servos.",
        "shell",
        "shell_offwhite",
    ),
    "upper_leg_right": (
        "右大腿外壳",
        "Right thigh shell",
        "右大腿结构外壳，内含髋俯仰与膝关节舵机。",
        "Right thigh structural shell housing the hip-pitch and knee servos.",
        "shell",
        "shell_offwhite",
    ),
    "upper_leg_rigidity_plate": (
        "大腿加强板",
        "Thigh rigidity plate",
        "大腿内侧的加强板，与外壳夹持两颗舵机。",
        "Inner stiffening plate that clamps the two thigh servos against the shell.",
        "structure",
        "anodized_champagne",
    ),
    "leg": (
        "小腿支架",
        "Shank bracket",
        "小腿主结构，连接膝关节输出与踝关节舵机。",
        "Main shank member linking the knee output to the ankle servo.",
        "structure",
        "anodized_champagne",
    ),
    "ankle_left": (
        "左踝外壳",
        "Left ankle housing",
        "左脚上方的踝部外壳，连接踝关节舵机输出。",
        "Left ankle housing above the foot, driven by the ankle servo.",
        "foot",
        "orange_hard",
    ),
    "ankle_right": (
        "右踝外壳",
        "Right ankle housing",
        "右脚上方的踝部外壳，连接踝关节舵机输出。",
        "Right ankle housing above the foot, driven by the ankle servo.",
        "foot",
        "orange_hard",
    ),
    "foot_left": (
        "左鸭掌",
        "Left duck foot",
        "左脚掌主体，橙色鸭掌造型。",
        "Left foot body with the orange webbed duck-foot form.",
        "foot",
        "orange_hard",
    ),
    "foot_right": (
        "右鸭掌",
        "Right duck foot",
        "右脚掌主体，橙色鸭掌造型。",
        "Right foot body with the orange webbed duck-foot form.",
        "foot",
        "orange_hard",
    ),
    "sole_left": (
        "左脚底垫",
        "Left sole pad",
        "左脚接地底垫（MJCF 中亦用作左脚接触几何）。",
        "Left ground-contact sole pad (also the left foot contact geom in the MJCF).",
        "foot",
        "sole_rubber",
    ),
    "sole_right": (
        "右脚底垫",
        "Right sole pad",
        "右脚接地底垫（MJCF 中亦用作右脚接触几何）。",
        "Right ground-contact sole pad (also the right foot contact geom in the MJCF).",
        "foot",
        "sole_rubber",
    ),
    "neck": (
        "颈部侧板",
        "Neck side plate",
        "两块平行侧板之一，夹持颈部俯仰与头部俯仰两颗舵机。",
        "One of two parallel side plates holding the neck-pitch and head-pitch servos.",
        "structure",
        "anodized_champagne",
    ),
    "neck_pitch": (
        "头部俯仰支架",
        "Head-pitch bracket",
        "头部俯仰关节输出支架，上方承接头部偏航轴承。",
        "Output bracket of the head-pitch joint; seats the head-yaw bearing above.",
        "structure",
        "anodized_blue",
    ),
    "yaw_roll_motion": (
        "头部偏航-滚转支架",
        "Head yaw-to-roll bracket",
        "承载头部偏航舵机并提供头部滚转轴的支架。",
        "Bracket carrying the head-yaw servo and providing the head-roll axis.",
        "structure",
        "anodized_graphite",
    ),
    "motor_support": (
        "头部舵机支架",
        "Head motor support",
        "头部内部骨架，安装头部滚转舵机与喙部舵机。",
        "Internal head frame mounting the head-roll servo and the beak servo.",
        "structure",
        "anodized_graphite",
    ),
    "top_head_shell": (
        "头部上壳",
        "Upper head shell",
        "头顶外壳，白色鸭头造型。",
        "Upper head cover giving the off-white duck head form.",
        "shell",
        "shell_offwhite",
    ),
    "bottom_head_shell": (
        "头部下壳",
        "Lower head shell",
        "头部下半外壳，橙色，与上壳对合。",
        "Orange lower half of the head cover; mates with the upper shell.",
        "beak",
        "orange_hard",
    ),
    "jaw": (
        "下颌（下喙）",
        "Jaw (lower beak)",
        "由喙部舵机驱动的下喙刚性件；源 MJCF 将其并入 jaw_soft 刚体，无独立关节。",
        "Rigid lower beak driven by the beak servo; the source MJCF merges it into the jaw_soft body without a joint.",
        "beak",
        "orange_hard",
    ),
    "jaw_soft": (
        "下喙软垫",
        "Soft lower-beak pad",
        "下喙前端的柔性垫片。",
        "Flexible pad on the front of the lower beak.",
        "beak",
        "orange_soft",
    ),
    "soft_mouth_top": (
        "上喙软垫",
        "Soft upper-beak pad",
        "上喙前端的柔性垫片。",
        "Flexible pad on the front of the upper beak.",
        "beak",
        "orange_soft",
    ),
    "face_part": (
        "面部框架",
        "Face frame",
        "头部前方面板，定位相机镜头与眼圈。",
        "Front face panel that locates the camera lens and eye ring.",
        "structure",
        "anodized_champagne",
    ),
    "noenoeil": (
        "镜头眼圈",
        "Lens eye ring",
        "与 M12 镜头同轴的面部饰圈（源名 noenoeil，功能由位置推断）。",
        "Face ring concentric with the M12 lens (source name noenoeil; role inferred from its position).",
        "optics",
        "amber_gloss",
    ),
    "lens": (
        "M12 镜头",
        "M12 camera lens",
        "头部相机的 M12 螺纹镜头。",
        "M12-thread lens of the head camera.",
        "optics",
        "lens_black",
    ),
    "m12_lens_holder": (
        "M12 镜头座",
        "M12 lens holder",
        "安装 M12 镜头的镜头座。",
        "Mount that holds the M12 lens.",
        "optics",
        "plastic_black",
    ),
    "elec_rpi_robot_hat_pcb": (
        "树莓派扩展板",
        "Raspberry Pi robot HAT PCB",
        "叠装在树莓派后方的机器人扩展电路板。",
        "Robot HAT circuit board stacked behind the Raspberry Pi.",
        "electronics",
        "pcb_green",
    ),
    "pcb__raspberry_pi_zero_2_w": (
        "树莓派 Zero 2 W",
        "Raspberry Pi Zero 2 W",
        "头部主控计算板。",
        "Main compute board in the head.",
        "electronics",
        "pcb_green",
    ),
    "speaker": (
        "扬声器",
        "Speaker",
        "头部后方的扬声器；源 CAD 中仅为 12 个三角形的简化方块。",
        "Speaker at the back of the head; the source CAD models it as a simplified 12-triangle box.",
        "electronics",
        "speaker_grey",
    ),
}

# PBR presets (linear base colour). Source MJCF rgba is kept per part as source_rgba.
MATERIALS = {
    "shell_offwhite": dict(
        color=(0.78, 0.76, 0.71), metallic=0.0, roughness=0.42, coat=0.15
    ),
    "orange_hard": dict(
        color=(0.86, 0.20, 0.022), metallic=0.0, roughness=0.36, coat=0.2
    ),
    "orange_soft": dict(
        color=(0.84, 0.24, 0.035), metallic=0.0, roughness=0.62, coat=0.0
    ),
    "sole_rubber": dict(
        color=(0.80, 0.36, 0.035), metallic=0.0, roughness=0.78, coat=0.0
    ),
    "amber_gloss": dict(
        color=(0.88, 0.45, 0.04), metallic=0.0, roughness=0.28, coat=0.3
    ),
    "servo_dark": dict(
        color=(0.018, 0.018, 0.02), metallic=0.0, roughness=0.36, coat=0.1
    ),
    "anodized_graphite": dict(
        color=(0.10, 0.105, 0.11), metallic=1.0, roughness=0.38, coat=0.0
    ),
    "anodized_champagne": dict(
        color=(0.52, 0.42, 0.30), metallic=1.0, roughness=0.42, coat=0.0
    ),
    "anodized_blue": dict(
        color=(0.20, 0.27, 0.38), metallic=1.0, roughness=0.36, coat=0.0
    ),
    "bearing_steel": dict(
        color=(0.86, 0.86, 0.85), metallic=1.0, roughness=0.12, coat=0.0
    ),
    "pcb_green": dict(
        color=(0.02, 0.17, 0.06), metallic=0.0, roughness=0.32, coat=0.35
    ),
    "battery_black": dict(
        color=(0.03, 0.03, 0.033), metallic=0.0, roughness=0.55, coat=0.0
    ),
    "plastic_grey": dict(
        color=(0.32, 0.32, 0.32), metallic=0.0, roughness=0.5, coat=0.0
    ),
    "plastic_black": dict(
        color=(0.02, 0.02, 0.02), metallic=0.0, roughness=0.45, coat=0.0
    ),
    "lens_black": dict(
        color=(0.03, 0.03, 0.035), metallic=0.8, roughness=0.18, coat=0.5
    ),
    "speaker_grey": dict(
        color=(0.07, 0.07, 0.07), metallic=0.0, roughness=0.6, coat=0.0
    ),
}


# --------------------------------------------------------------------------
# Explosion design (Blender world metres, Z up)
# --------------------------------------------------------------------------
# 1) Each kinematic body gets a rigid offset. Feet stay on the floor; leg
#    segments step outward along the pitch axes (+/-Y) and upward; the torso
#    lifts; neck and head move up/forward following the neck's lean.
# 2) Inside a body, parts separate along mechanically meaningful directions:
#    servos slide along their output-shaft axis, bearings along their bore
#    axis, shells/plates/boards along hand-chosen assembly directions.
# 3) Nothing may end below the floor: a part whose exploded min Z < 0 is lifted.

BODY_OFFSET_LEFT = {
    "yaw2roll": (0.0, 0.05, 0.13),  # drops off the hip-yaw axis (Z) below the torso
    "hip_l": (0.02, 0.09, 0.11),  # slides forward off the hip-roll axis (X)
    "upper_leg_left": (0.0, 0.13, 0.085),  # steps outward along the hip-pitch axis (Y)
    "leg": (
        0.0,
        0.165,
        0.06,
    ),  # outward along the knee axis, lifted out of the ankle U-bracket
    "ankle_left": (
        0.0,
        0.20,
        0.0,
    ),  # outward along the ankle axis (Y); feet stay on the floor
}
BODY_OFFSET = {
    "trunk_base": (-0.065, 0.0, 0.145),
    "neck": (0.0, 0.0, 0.14),  # up/forward out of the torso slot (neck-pitch, Y)
    "neck_pitch": (0.035, 0.0, 0.15),
    "yaw_roll_motion": (0.035, 0.0, 0.185),  # lifts off the head-yaw axis (Z)
    "jaw_soft": (0.125, 0.0, 0.155),  # head slides forward off the head-roll axis (X)
}
_MIRROR = {
    "yaw2roll": "bearing_roll",
    "hip_l": "hip_l_2",
    "upper_leg_left": "upper_leg_right",
    "leg": "leg_2",
    "ankle_left": "ankle_right",
}
for _b, _o in BODY_OFFSET_LEFT.items():
    BODY_OFFSET[_b] = _o
    BODY_OFFSET[_MIRROR[_b]] = (_o[0], -_o[1], _o[2])

# Hand-chosen part directions (instance -> offset relative to its body).
# Right-side instances are mirrored automatically from their left twins.
PART_OFFSET = {
    # torso
    "trunk_base": (0.0, 0.0, 0.0),
    "left_shell": (0.0, 0.055, 0.0),
    "right_shell": (0.0, -0.055, 0.0),
    "power_support": (-0.045, 0.0, 0.0),
    "np_f970": (-0.09, 0.0, 0.0),
    "banana_pcb_locker": (-0.065, 0.0, 0.045),
    # neck
    "neck": (0.0, -0.035, 0.0),
    "neck_2": (0.0, 0.035, 0.0),
    # head: shells open vertically, beak parts drop forward, the camera/PCB
    # stack slides out along the optical axis (+X)
    "motor_support": (0.0, 0.0, 0.0),
    "top_head_shell": (0.0, 0.0, 0.04),
    "bottom_head_shell": (0.0, 0.0, -0.02),
    "jaw": (0.035, 0.0, -0.06),
    "soft_mouth_top": (0.07, 0.0, -0.035),
    "jaw_soft": (0.07, 0.0, -0.08),
    "elec_rpi_robot_hat_pcb": (0.025, 0.0, 0.0),
    "pcb__raspberry_pi_zero_2_w": (0.045, 0.0, 0.0),
    "m12_lens_holder": (0.068, 0.0, 0.0),
    "face_part": (0.092, 0.0, 0.0),
    "lens": (0.115, 0.0, 0.0),
    "noenoeil": (0.135, 0.0, 0.0),
    "speaker": (0.0, -0.065, 0.02),  # lifted clear of the head-roll servo, out the side
    # legs (left; right mirrored)
    "yaw2roll": (0.0, 0.0, 0.0),
    "bearing_roll": (0.025, 0.0, 0.0),
    "hip_l": (0.0, 0.0, 0.0),
    "upper_leg_left": (0.0, 0.055, 0.0),
    "upper_leg_rigidity_plate": (0.0, -0.012, 0.0),
    "leg": (0.0, -0.022, 0.0),
    "ankle_left": (0.0, 0.0, 0.03),
    "foot_left": (0.0, 0.0, 0.012),
    "sole_left": (0.0, 0.0, 0.0),
    # head brackets
    "neck_pitch": (0.0, 0.0, 0.0),
    "yaw_roll_motion": (0.0, 0.0, 0.0),
}
PART_MIRROR = {
    "right_shell": None,
    "left_shell": None,
    "neck": None,
    "neck_2": None,
    "bearing_roll_2": "bearing_roll",
    "yaw2roll_2": "yaw2roll",
    "hip_l_2": "hip_l",
    "upper_leg_right": "upper_leg_left",
    "upper_leg_rigidity_plate_2": "upper_leg_rigidity_plate",
    "leg_2": "leg",
    "ankle_right": "ankle_left",
    "foot_right": "foot_left",
    "sole_right": "sole_left",
}
SERVO_SLIDE = 0.025  # servo pulled out along its shaft axis
BEARING_SLIDE = 0.018  # bearing pushed out along its bore axis
BEARING_SLIDE_OVERRIDE = {
    "seeed_bearing__configuration__22x16x4_5": 0.035,  # clear the thigh plate
    "seeed_bearing__configuration__22x16x4_11": 0.035,
    "seeed_bearing__configuration__22x16x4_6": 0.015,
    "seeed_bearing__configuration__22x16x4_8": 0.008,
}
# Servos whose shaft-axis slide would run into a neighbour inside the same body:
# beak servo out sideways; neck, ankle, head-yaw and head-roll servos stay seated
# in their brackets while neighbouring plates/bodies move away instead.
SERVO_SLIDE_OVERRIDE = {
    "xl330_10": (0.0, 0.045, 0.0),
    "xl330_11": (0.0, 0.0, 0.0),
    "xl330_3": (-0.04, 0.0, 0.0),
    "xl330_12": (-0.04, 0.0, 0.0),
    "xl330_7": (0.0, 0.0, 0.0),
    "xl330_8": (0.0, 0.0, 0.0),
    "xl330_6": (0.0, 0.0, 0.0),
    "xl330_15": (0.0, 0.0, 0.0),
    "xl330_9": (0.0, 0.0, 0.0),
}

BODY_MAIN = {
    "trunk_base": ["trunk_base"],
    "yaw2roll": ["yaw2roll"],
    "hip_l": ["hip_l"],
    "upper_leg_left": ["upper_leg_left"],
    "leg": ["leg"],
    "ankle_left": ["ankle_left"],
    "bearing_roll": ["yaw2roll_2"],
    "hip_l_2": ["hip_l_2"],
    "upper_leg_right": ["upper_leg_right"],
    "leg_2": ["leg_2"],
    "ankle_right": ["ankle_right"],
    "neck": ["neck", "neck_2"],
    "neck_pitch": ["neck_pitch"],
    "yaw_roll_motion": ["yaw_roll_motion"],
    "jaw_soft": ["motor_support"],
}


# --------------------------------------------------------------------------
# Assembly layout: meshes, world transforms, labels, explosion, statistics
# --------------------------------------------------------------------------


def file_sha256(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _coaxial(p, a, o, b, tol=0.004):
    if abs(dot(a, b)) < 0.99:
        return False
    d = sub(p, o)
    perp = sub(d, scale(b, dot(d, b)))
    return math.sqrt(dot(perp, perp)) < tol


def coaxial_joint(model, origin, axis, tol=0.003):
    """Joint whose axis line passes through `origin` parallel to `axis` (or None)."""
    best = None
    for j in model["joints"]:
        if abs(dot(axis, j["axis_world"])) < 0.99:
            continue
        d = sub(origin, j["anchor_world"])
        perp = sub(d, scale(j["axis_world"], dot(d, j["axis_world"])))
        dist = math.sqrt(dot(perp, perp))
        if dist < tol and (best is None or dist < best[1]):
            best = (j, dist)
    return best[0] if best else None


def build_layout(load_mesh):
    """load_mesh(path) -> (verts, faces, stats). Returns the full assembly description."""
    model = load_model()
    meshes = {}
    for g in model["geoms"]:
        if g["mesh"] not in meshes:
            path = os.path.join(SRC, "assets", model["meshes"][g["mesh"]])
            v, f, st = load_mesh(path)
            st.update(
                vertices=len(v),
                triangles=len(f),
                file="assets/" + model["meshes"][g["mesh"]],
                sha256=file_sha256(path),
            )
            meshes[g["mesh"]] = dict(verts=v, faces=f, stats=st)

    def world_bounds(M, verts):
        mn, mx = [1e9] * 3, [-1e9] * 3
        for p in verts:
            w = apply(M, p)
            for k in range(3):
                if w[k] < mn[k]:
                    mn[k] = w[k]
                if w[k] > mx[k]:
                    mx[k] = w[k]
        return mn, mx

    # Ground the STAND pose: shift the whole robot so its lowest vertex touches Z = 0.
    for g in model["geoms"]:
        g["bounds"] = world_bounds(g["M"], meshes[g["mesh"]]["verts"])
    ground_dz = -min(g["bounds"][0][2] for g in model["geoms"])
    G = tf(None, (0.0, 0.0, ground_dz))
    for g in model["geoms"]:
        g["M"] = mul(G, g["M"])
        g["bounds"] = (
            [g["bounds"][0][0], g["bounds"][0][1], g["bounds"][0][2] + ground_dz],
            [g["bounds"][1][0], g["bounds"][1][1], g["bounds"][1][2] + ground_dz],
        )
        g["center"] = [(g["bounds"][0][k] + g["bounds"][1][k]) / 2 for k in range(3)]
    for b in model["bodies"]:
        b["M"] = mul(G, b["M"])
    for j in model["joints"]:
        j["anchor_world"] = add(j["anchor_world"], (0.0, 0.0, ground_dz))

    by_inst = {g["instance"]: g for g in model["geoms"]}
    body_center = {}
    for b in model["bodies"]:
        names = BODY_MAIN.get(b["name"]) or [
            g["instance"] for g in model["geoms"] if g["body"] == b["name"]
        ]
        cs = [by_inst[n]["center"] for n in names]
        body_center[b["name"]] = [sum(c[k] for c in cs) / len(cs) for k in range(3)]

    # Beak servo = the XL330 whose shaft is not coaxial with any MJCF hinge.
    beak_axis = None
    for g in model["geoms"]:
        if g["mesh"] == "xl330":
            o, a = apply(g["M"], (0.0, 0.0, 0.0)), norm(
                apply_dir(g["M"], (1.0, 0.0, 0.0))
            )
            if coaxial_joint(model, o, a) is None:
                beak_axis = (o, a)

    parts = []
    used_ids = set()
    for idx, g in enumerate(model["geoms"]):
        inst, mesh, body = g["instance"], g["mesh"], g["body"]
        info = MESH_INFO[mesh]
        group = BODY_INFO[body][0]
        lzh, len_, dzh, den = info[0], info[1], info[2], info[3]
        drives = None
        qual_zh = qual_en = ""
        origin = apply(g["M"], (0.0, 0.0, 0.0))
        if mesh == "xl330":
            shaft = norm(apply_dir(g["M"], (1.0, 0.0, 0.0)))
            j = coaxial_joint(model, origin, shaft)
            if j is not None:
                drives = j["name"]
                qual_zh, qual_en = JOINT_LABELS[drives]
                dzh += "本件驱动「%s」关节（%s）。" % (qual_zh, drives)
                den += " Drives joint %s (%s)." % (drives, qual_en)
            else:
                drives = "beak"
                qual_zh, qual_en = BEAK
                dzh += (
                    "本件为喙部开合舵机；源 MJCF 未建模喙部关节，下喙固定在头部刚体上。"
                )
                den += " Beak open/close servo; the source MJCF has no beak joint, so the jaw is fixed to the head body."
        elif mesh.startswith("seeed_bearing"):
            bore = norm(apply_dir(g["M"], (0.0, 0.0, 1.0)))
            j = coaxial_joint(model, apply(g["M"], (0.0, 0.0, 0.002)), bore, tol=0.004)
            if j is not None:
                qual_zh, qual_en = JOINT_LABELS[j["name"]]
                dzh += "位于「%s」关节轴线上。" % qual_zh
                den += " Sits on the %s joint axis." % qual_en
            elif beak_axis is not None and _coaxial(
                apply(g["M"], (0.0, 0.0, 0.002)), bore, *beak_axis
            ):
                qual_zh, qual_en = "喙部转轴", "beak pivot"
                dzh += "与喙部舵机输出轴同轴，为下喙转轴另一侧的支撑。"
                den += " Coaxial with the beak servo shaft; supports the far side of the jaw pivot."
        if qual_en.startswith(("left ", "right ")):
            qual_en = ("L " if qual_en.startswith("left") else "R ") + qual_en.split(
                " ", 1
            )[1]
        side_zh = {"left_leg": "左", "right_leg": "右"}.get(group, "")
        side_en = {"left_leg": "Left ", "right_leg": "Right "}.get(group, "")
        if qual_en:
            label_zh = "%s · %s" % (lzh, qual_zh)
            label_en = "%s - %s" % (len_, qual_en)
        elif side_en and not len_.startswith(("Left", "Right")):
            label_zh = side_zh + lzh
            label_en = side_en + len_[0].lower() + len_[1:]
        else:
            label_zh, label_en = lzh, len_
        pid = "%s/%s" % (body, inst)
        assert pid not in used_ids
        used_ids.add(pid)

        # ---- explosion vector -------------------------------------------
        off = list(BODY_OFFSET[body])
        if mesh == "xl330":
            if inst in SERVO_SLIDE_OVERRIDE:
                local = list(SERVO_SLIDE_OVERRIDE[inst])
            else:
                shaft = norm(apply_dir(g["M"], (1.0, 0.0, 0.0)))
                jj = model_joint(model, drives)
                s = (
                    1.0
                    if dot(shaft, sub(g["center"], jj["anchor_world"])) >= 0
                    else -1.0
                )
                local = scale(shaft, s * SERVO_SLIDE)
        elif mesh.startswith("seeed_bearing"):
            bore = norm(apply_dir(g["M"], (0.0, 0.0, 1.0)))
            s = 1.0 if dot(bore, sub(g["center"], body_center[body])) >= 0 else -1.0
            local = scale(bore, s * BEARING_SLIDE_OVERRIDE.get(inst, BEARING_SLIDE))
        elif inst in PART_OFFSET:
            local = list(PART_OFFSET[inst])
        elif PART_MIRROR.get(inst):
            o = PART_OFFSET[PART_MIRROR[inst]]
            local = [o[0], -o[1], o[2]]
        else:
            raise KeyError("no explode rule for " + inst)
        off = add(off, local)
        lifted = 0.0
        if g["bounds"][0][2] + off[2] < 0.0:
            lifted = -(g["bounds"][0][2] + off[2])
            off[2] += lifted

        st = meshes[mesh]["stats"]
        parts.append(
            dict(
                index=idx,
                id=pid,
                instance=inst,
                mesh=mesh,
                body=body,
                group=group,
                category=info[4],
                material=info[5],
                source_material=g["material"],
                source_rgba=model["materials"][g["material"]],
                label_zh=label_zh,
                label_en=label_en,
                description_zh=dzh,
                description_en=den,
                drives_joint=drives,
                qual_en=qual_en,
                qual_zh=qual_zh,
                M=g["M"],
                geom_pos=g["pos"],
                geom_quat_wxyz=g["quat"],
                bounds=g["bounds"],
                center=g["center"],
                explode=off,
                ground_lift=lifted,
                vertices=st["vertices"],
                triangles=st["triangles"],
            )
        )
    return dict(
        model=model,
        meshes=meshes,
        parts=parts,
        ground_dz=ground_dz,
        body_center=body_center,
    )


def model_joint(model, name):
    for j in model["joints"]:
        if j["name"] == name:
            return j
    raise KeyError(name)


def object_name(p):
    """Readable, unique Blender/glTF node name: short English role + source instance."""
    if p["mesh"] == "xl330":
        role = "Servo " + p["qual_en"]
    elif p["category"] == "bearing":
        role = ("Bearing " + p["qual_en"]) if p["qual_en"] else "Bearing"
    else:
        role = p["label_en"]
    name = "%s | %s" % (role, p["instance"])
    assert len(name) <= 63, name
    return name


def mat_to_quat_wxyz(M):
    m = M
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [
            0.25 * s,
            (m[2][1] - m[1][2]) / s,
            (m[0][2] - m[2][0]) / s,
            (m[1][0] - m[0][1]) / s,
        ]
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = [
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        ]
    if q[0] < 0:
        q = [-c for c in q]
    return q


def r6(v):
    return [round(float(c), 6) for c in v]


def camera_basis():
    d = norm(sub(CAM_TARGET, CAM_LOC))
    right = norm([d[1], -d[0], 0.0])  # d x Z
    up = [
        right[1] * d[2] - right[2] * d[1],
        right[2] * d[0] - right[0] * d[2],
        right[0] * d[1] - right[1] * d[0],
    ]
    return d, right, up


def fit_ortho(points, nominal, margin=1.04):
    """Smallest ortho scale >= nominal that keeps all points inside the 4:3 frame."""
    d, right, up = camera_basis()
    aspect = RES[1] / float(RES[0])
    need = 0.0
    for p in points:
        r = sub(p, CAM_TARGET)
        need = max(
            need, 2 * abs(dot(r, right)) * margin, 2 * abs(dot(r, up)) * margin / aspect
        )
    return max(nominal, need), need


def bbox_corners(mn, mx):
    return [
        [(mn, mx)[i][0], (mn, mx)[j][1], (mn, mx)[k][2]]
        for i in (0, 1)
        for j in (0, 1)
        for k in (0, 1)
    ]


def components_json(layout, extra=None):
    model = layout["model"]
    parts = []
    tot_v = tot_t = 0
    pts_a, pts_e = [], []
    for p in layout["parts"]:
        mn, mx = p["bounds"]
        e = p["explode"]
        pts_a += bbox_corners(mn, mx)
        pts_e += bbox_corners(add(mn, e), add(mx, e))
        tot_v += p["vertices"]
        tot_t += p["triangles"]
        parts.append(
            dict(
                id=p["id"],
                object_name=object_name(p),
                source_instance=p["instance"],
                source_mesh=p["mesh"],
                source_file=layout["meshes"][p["mesh"]]["stats"]["file"],
                source_body=p["body"],
                source_material=p["source_material"],
                source_rgba=p["source_rgba"],
                group=p["group"],
                group_zh=GROUPS[p["group"]][0],
                group_en=GROUPS[p["group"]][1],
                category=p["category"],
                render_material=p["material"],
                label_zh=p["label_zh"],
                label_en=p["label_en"],
                description_zh=p["description_zh"],
                description_en=p["description_en"],
                drives_joint=p["drives_joint"],
                mjcf_geom_pos=p["geom_pos"],
                mjcf_geom_quat_wxyz=p["geom_quat_wxyz"],
                world_position_blender=r6([p["M"][0][3], p["M"][1][3], p["M"][2][3]]),
                world_quat_wxyz_blender=r6(mat_to_quat_wxyz(p["M"])),
                explode_blender=r6(e),
                explode=r6(to_gltf(e)),
                explode_ground_lift=round(p["ground_lift"], 6),
                vertices=p["vertices"],
                triangles=p["triangles"],
                bounds_blender_zup=dict(min=r6(mn), max=r6(mx)),
                bounds_gltf_yup=dict(
                    min=r6([mn[0], mn[2], -mx[1]]), max=r6([mx[0], mx[2], -mn[1]])
                ),
            )
        )
    joints = []
    for j in model["joints"]:
        servo = [
            p["instance"] for p in layout["parts"] if p["drives_joint"] == j["name"]
        ]
        joints.append(
            dict(
                name=j["name"],
                label_zh=JOINT_LABELS[j["name"]][0],
                label_en=JOINT_LABELS[j["name"]][1],
                type="hinge",
                parent_body=j["parent_body"],
                child_body=j["child_body"],
                axis_local=j["axis_local"],
                axis_world_blender=r6(j["axis_world"]),
                anchor_world_blender=r6(j["anchor_world"]),
                range_rad=j["range_rad"],
                range_deg=[round(math.degrees(x), 3) for x in j["range_rad"]],
                stand_rad=j["stand_rad"],
                stand_deg=round(math.degrees(j["stand_rad"]), 4),
                actuator_class=j["actuator_class"],
                servo_instance=servo[0] if servo else None,
                policy_action_index=model["hinge_order"].index(j["name"]),
            )
        )
    beak = [p["instance"] for p in layout["parts"] if p["drives_joint"] == "beak"]
    mn = [min(p["bounds"][0][k] for p in layout["parts"]) for k in range(3)]
    mx = [max(p["bounds"][1][k] for p in layout["parts"]) for k in range(3)]
    emn = [
        min(p["bounds"][0][k] + p["explode"][k] for p in layout["parts"])
        for k in range(3)
    ]
    emx = [
        max(p["bounds"][1][k] + p["explode"][k] for p in layout["parts"])
        for k in range(3)
    ]
    ortho_a, need_a = fit_ortho(pts_a, ORTHO_ASSEMBLED)
    ortho_e, need_e = fit_ortho(pts_e, ORTHO_EXPLODED)
    src_files = {
        os.path.basename(ROBOT_XML): file_sha256(ROBOT_XML),
        os.path.basename(SCENE_XML): file_sha256(SCENE_XML),
    }
    for m in layout["meshes"].values():
        src_files[m["stats"]["file"]] = m["stats"]["sha256"]
    prov = {}
    try:
        with open(os.path.join(SRC, "provenance.json")) as f:
            pj = json.load(f)
        prov = {
            k: pj[k]
            for k in ("source", "commit", "models_path", "license_as_stated_upstream")
            if k in pj
        }
    except (OSError, ValueError):
        pass
    out = dict(
        title_zh="Microduck 鸭子机器人机械爆炸装配图",
        title_en="Microduck mechanical exploded assembly",
        generator="build.py (Blender CAD assembly)",
        source=dict(
            directory=SRC,
            robot_mjcf=os.path.basename(ROBOT_XML),
            scene_mjcf=os.path.basename(SCENE_XML),
            upstream=prov,
            onshape_document="https://cad.onshape.com/documents/804927696f06d877f3f1803e",
            license_zh="上游 README 声明 3D 模型采用 Creative Commons BY-SA-NC（未注明版本）；Apache-2.0 仅覆盖软件。公开或商业使用前请核对模型许可。",
            license_en="Upstream README: 3D model files are licensed under Creative Commons BY-SA-NC (version not stated). The Apache-2.0 LICENSE covers software only. Verify the model licence before public or commercial use.",
            notice_file="NOTICE.md",
            sha256=src_files,
        ),
        pose=dict(
            keyframe=KEYFRAME,
            qpos=model["qpos"],
            freejoint_qpos=model["qpos"][:7],
            hinge_order=model["hinge_order"],
            grounding_dz_m=round(layout["ground_dz"], 6),
            note="STAND keyframe from scene_allcollisions.xml; whole robot then translated by grounding_dz so the lowest vertex sits on Z=0.",
        ),
        conventions=dict(
            blender="metres, Z up, robot +X forward, +Y robot-left",
            gltf="metres, Y up; Blender (x, y, z) -> glTF (x, z, -y)",
            explode="per-part translation in glTF Y-up world metres; assembled = exported position, exploded = position + explode",
            mjcf_quaternions="w x y z",
        ),
        counts=dict(
            parts=len(parts),
            unique_meshes=len(layout["meshes"]),
            servos=sum(1 for p in parts if p["source_mesh"] == "xl330"),
            hinge_joints=len(joints),
            policy_actions=len(joints),
            beak_servos=len(beak),
            bearings=sum(1 for p in parts if p["category"] == "bearing"),
            vertices_total_instanced=tot_v,
            triangles_total_instanced=tot_t,
            triangles_unique_meshes=sum(
                m["stats"]["triangles"] for m in layout["meshes"].values()
            ),
        ),
        bounds=dict(
            assembled_blender=dict(min=r6(mn), max=r6(mx)),
            exploded_blender=dict(min=r6(emn), max=r6(emx)),
            assembled_gltf=dict(
                min=r6([mn[0], mn[2], -mx[1]]), max=r6([mx[0], mx[2], -mn[1]])
            ),
            exploded_gltf=dict(
                min=r6([emn[0], emn[2], -emx[1]]), max=r6([emx[0], emx[2], -emn[1]])
            ),
        ),
        camera=dict(
            location=CAM_LOC,
            target=CAM_TARGET,
            type="orthographic",
            resolution=list(RES),
            ortho_scale_assembled=round(ortho_a, 4),
            ortho_scale_exploded=round(ortho_e, 4),
            ortho_fit_required_assembled=round(need_a, 4),
            ortho_fit_required_exploded=round(need_e, 4),
            nominal=dict(assembled=ORTHO_ASSEMBLED, exploded=ORTHO_EXPLODED),
        ),
        groups={k: dict(label_zh=v[0], label_en=v[1]) for k, v in GROUPS.items()},
        bodies=[
            dict(
                name=b["name"],
                parent=b["parent"],
                label_zh=BODY_INFO[b["name"]][1],
                label_en=BODY_INFO[b["name"]][2],
                group=BODY_INFO[b["name"]][0],
                mjcf_pos=b["pos"],
                mjcf_quat_wxyz=b["quat"],
                joint=b["joint"],
                explode_body_blender=list(BODY_OFFSET[b["name"]]),
            )
            for b in model["bodies"]
        ],
        joints=joints,
        beak=dict(
            servo_instance=beak[0] if beak else None,
            joint=None,
            note_zh="源 MJCF 中喙部（jaw）随 jaw_soft 刚体固定，无关节；第 15 颗舵机为喙部舵机。",
            note_en="In the source MJCF the jaw is merged into the jaw_soft body with no joint; the 15th servo drives the beak.",
        ),
        unique_meshes={
            k: dict(
                file=m["stats"]["file"],
                vertices=m["stats"]["vertices"],
                triangles=m["stats"]["triangles"],
                stl_triangles=m["stats"]["stl_triangles"],
                degenerate_removed=m["stats"]["degenerate_removed"],
                duplicate_removed=m["stats"]["duplicate_removed"],
            )
            for k, m in sorted(layout["meshes"].items())
        },
        parts=parts,
    )
    if extra:
        out.update(extra)
    return out


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


# --------------------------------------------------------------------------
# GLB verification (pure python; reads the JSON chunk only)
# --------------------------------------------------------------------------


def verify_glb(path, layout):
    with open(path, "rb") as f:
        data = f.read()
    magic, version, length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, "not a GLB"
    clen, ctype = struct.unpack_from("<II", data, 12)
    gl = json.loads(data[20 : 20 + clen].decode("utf-8"))
    nodes = gl.get("nodes", [])
    by_name = {
        p_name: p for p_name, p in ((object_name(p), p) for p in layout["parts"])
    }
    mesh_nodes = [n for n in nodes if "mesh" in n]
    tri_count = 0
    for n in mesh_nodes:
        for prim in gl["meshes"][n["mesh"]]["primitives"]:
            tri_count += gl["accessors"][prim["indices"]]["count"] // 3
    issues = []
    max_err = 0.0
    for n in mesh_nodes:
        p = by_name.get(n.get("name"))
        if p is None:
            issues.append("unexpected node " + str(n.get("name")))
            continue
        ex = n.get("extras", {})
        for k in (
            "id",
            "label_zh",
            "label_en",
            "description_zh",
            "description_en",
            "group",
            "source_mesh",
            "source_body",
            "explode",
        ):
            if k not in ex:
                issues.append("%s missing extras.%s" % (n["name"], k))
        t = n.get("translation", [0.0, 0.0, 0.0])
        want = to_gltf([p["M"][0][3], p["M"][1][3], p["M"][2][3]])
        max_err = max(max_err, max(abs(t[i] - want[i]) for i in range(3)))
        if "explode" in ex:
            e = to_gltf(p["explode"])
            max_err_e = max(abs(ex["explode"][i] - e[i]) for i in range(3))
            if max_err_e > 1e-5:
                issues.append("%s explode mismatch" % n["name"])
    names = {n.get("name") for n in mesh_nodes}
    missing = [object_name(p) for p in layout["parts"] if object_name(p) not in names]
    non_mesh = [n.get("name") for n in nodes if "mesh" not in n]
    parented = [n.get("name") for n in nodes if n.get("children")]
    roots = gl.get("scenes", [{}])[gl.get("scene", 0)].get("nodes", [])
    return dict(
        glb_bytes=len(data),
        gltf_nodes=len(nodes),
        mesh_nodes=len(mesh_nodes),
        scene_root_nodes=len(roots),
        instanced_triangles=tri_count,
        expected_triangles=sum(p["triangles"] for p in layout["parts"]),
        gltf_meshes=len(gl.get("meshes", [])),
        gltf_materials=len(gl.get("materials", [])),
        missing_parts=missing,
        non_mesh_nodes=non_mesh,
        nodes_with_children=parented,
        animations=len(gl.get("animations", [])),
        cameras=len(gl.get("cameras", [])),
        max_translation_error_m=max_err,
        issues=issues[:20],
        ok=not (issues or missing or non_mesh or parented)
        and len(roots) == len(layout["parts"])
        and tri_count == sum(p["triangles"] for p in layout["parts"]),
    )


# --------------------------------------------------------------------------
# Blender scene
# --------------------------------------------------------------------------


def _set_input(node, names, value):
    for n in names:
        if n in node.inputs:
            node.inputs[n].default_value = value
            return True
    return False


def make_material(key, spec):
    color = tuple(spec["color"]) + (1.0,)
    m = bpy.data.materials.new("MD_" + key)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    _set_input(bsdf, ["Base Color"], color)
    _set_input(bsdf, ["Metallic"], spec["metallic"])
    _set_input(bsdf, ["Roughness"], spec["roughness"])
    _set_input(bsdf, ["Coat Weight", "Clearcoat"], spec["coat"])
    _set_input(bsdf, ["Coat Roughness", "Clearcoat Roughness"], 0.12)
    # Render-only micro edge rounding (Cycles bevel shader): no topology change.
    bevel = nt.nodes.new("ShaderNodeBevel")
    bevel.location = (bsdf.location.x - 250, bsdf.location.y - 400)
    bevel.inputs["Radius"].default_value = 0.00025
    bevel.samples = 6
    nt.links.new(bevel.outputs["Normal"], bsdf.inputs["Normal"])
    m.diffuse_color = color
    m.metallic = spec["metallic"]
    m.roughness = spec["roughness"]
    return m


def make_mesh(name, verts, faces):
    me = bpy.data.meshes.new("MESH_" + name)
    vl = verts.tolist() if hasattr(verts, "tolist") else list(verts)
    fl = faces.tolist() if hasattr(faces, "tolist") else [list(f) for f in faces]
    me.from_pydata(vl, [], fl)
    me.validate(clean_customdata=False)
    me.update(calc_edges=True)
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    ang = math.radians(SMOOTH_ANGLE_DEG)
    if hasattr(me, "use_auto_smooth"):  # Blender <= 4.0
        me.use_auto_smooth = True
        me.auto_smooth_angle = ang
    elif hasattr(me, "set_sharp_from_angle"):  # Blender >= 4.1
        me.set_sharp_from_angle(angle=ang)
    return me


def bake_weighted_normals(me, scene):
    """Face-area weighted custom normals (keeps large CAD flats flat next to fillets).
    Baked into a plain mesh so instances stay shared in the GLB."""
    tmp = bpy.data.objects.new("_wn_tmp", me)
    scene.collection.objects.link(tmp)
    mod = tmp.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
    mod.mode = "FACE_AREA"
    mod.weight = 50
    mod.keep_sharp = True
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    baked = bpy.data.meshes.new_from_object(
        tmp.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg
    )
    bpy.data.objects.remove(tmp)
    name = me.name
    bpy.data.meshes.remove(me)
    baked.name = name
    if hasattr(baked, "use_auto_smooth"):
        baked.use_auto_smooth = True
        baked.auto_smooth_angle = math.radians(SMOOTH_ANGLE_DEG)
    return baked


def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def add_area_light(
    name,
    col,
    azim_deg,
    elev_deg,
    dist,
    size,
    power,
    shape="DISK",
    color=(1, 1, 1),
    shadow=True,
):
    t = Vector(CAM_TARGET)
    a, e = math.radians(azim_deg), math.radians(elev_deg)
    loc = (
        t
        + Vector((math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)))
        * dist
    )
    ld = bpy.data.lights.new(name, "AREA")
    ld.shape = shape
    ld.size = size
    ld.energy = power
    ld.color = color
    if not shadow:  # fill/rim shape the parts but leave no extra haze on the floor
        if hasattr(ld, "cycles") and hasattr(ld.cycles, "cast_shadow"):
            ld.cycles.cast_shadow = False
        if hasattr(ld, "use_shadow"):
            ld.use_shadow = False
    ob = bpy.data.objects.new(name, ld)
    col.objects.link(ob)
    ob.location = loc
    look_at(ob, t)
    return ob


def setup_studio(scene, col, ortho_a, ortho_e):
    # Camera (orthographic, fixed pose from the brief)
    cd = bpy.data.cameras.new("MD_Camera")
    cd.type = "ORTHO"
    cd.ortho_scale = ortho_a
    cd.clip_start = 0.01
    cd.clip_end = 20.0
    cam = bpy.data.objects.new("MD_Camera", cd)
    col.objects.link(cam)
    cam.location = CAM_LOC
    look_at(cam, CAM_TARGET)
    scene.camera = cam
    cd.ortho_scale = ortho_a
    cd.keyframe_insert("ortho_scale", frame=FRAME_ASSEMBLED)
    cd.ortho_scale = ortho_e
    cd.keyframe_insert("ortho_scale", frame=FRAME_EXPLODED)

    # Soft product lighting (area lights, W): warm key + overhead softbox cast the
    # floor shadow; cool fill and rim only model the parts.
    add_area_light("MD_Key", col, -20, 52, 1.0, 0.8, 9.0, color=(1.0, 0.97, 0.93))
    add_area_light("MD_Top", col, -40, 86, 1.1, 1.4, 5.0)
    add_area_light(
        "MD_Fill", col, -105, 22, 1.2, 1.4, 4.0, color=(0.93, 0.96, 1.0), shadow=False
    )
    add_area_light("MD_Rim", col, 150, 35, 1.0, 0.5, 9.0, shadow=False)

    # Studio floor: shadow catcher, composited over a neutral near-white backdrop
    fm = bpy.data.meshes.new("MD_Floor")
    s = 4.0
    fm.from_pydata([(-s, -s, 0), (s, -s, 0), (s, s, 0), (-s, s, 0)], [], [(0, 1, 2, 3)])
    floor = bpy.data.objects.new("MD_Floor", fm)
    col.objects.link(floor)
    floor.location.z = -0.00005
    fmat = bpy.data.materials.new("MD_Floor_White")
    fmat.use_nodes = True
    fb = next(n for n in fmat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    _set_input(fb, ["Base Color"], (0.85, 0.85, 0.85, 1.0))
    _set_input(fb, ["Roughness"], 0.6)
    fm.materials.append(fmat)
    if hasattr(floor, "is_shadow_catcher"):
        floor.is_shadow_catcher = True
    else:
        floor.cycles.is_shadow_catcher = True

    world = bpy.data.worlds.new("MD_Studio")
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.92, 0.92, 0.93, 1.0)
    bg.inputs["Strength"].default_value = 0.2
    scene.world = world
    return cam


def setup_render(scene, samples):
    scene.render.engine = "CYCLES"
    cy = scene.cycles
    cy.device = "CPU"
    cy.samples = samples
    for attr, val in (
        ("use_adaptive_sampling", True),
        ("adaptive_threshold", 0.01),
        ("adaptive_min_samples", min(32, samples)),
        ("max_bounces", 8),
        ("diffuse_bounces", 3),
        ("glossy_bounces", 4),
        ("transmission_bounces", 4),
        ("caustics_reflective", False),
        ("caustics_refractive", False),
        ("blur_glossy", 0.5),
        ("sample_clamp_indirect", 4.0),
    ):
        if hasattr(cy, attr):
            setattr(cy, attr, val)
    # Denoise only where the build actually ships a denoiser (the Ubuntu arm64
    # Blender 4.0.2 runtime has none, so more samples are used instead).
    denoisers = [i.identifier for i in cy.bl_rna.properties["denoiser"].enum_items]
    cy.use_denoising = "OPENIMAGEDENOISE" in denoisers
    if cy.use_denoising:
        cy.denoiser = "OPENIMAGEDENOISE"
    scene.render.resolution_x, scene.render.resolution_y = RES
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    vs = scene.view_settings
    for vt in ("AgX", "Filmic", "Standard"):
        try:
            vs.view_transform = vt
            break
        except TypeError:
            continue
    for look in ("AgX - Medium High Contrast", "Medium High Contrast", "None"):
        try:
            vs.look = look
            break
        except TypeError:
            continue
    vs.exposure = 0.0
    vs.gamma = 1.0
    return denoisers


def _box_blur(a, r):
    """Separable box blur with edge clamping (numpy only)."""
    import numpy as np

    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float64)
        hi = [slice(None), slice(None)]
        lo = [slice(None), slice(None)]
        hi[axis] = slice(2 * r + 1, None)
        lo[axis] = slice(0, -2 * r - 1)
        a = ((c[tuple(hi)] - c[tuple(lo)]) / (2 * r + 1)).astype(np.float32)
    return a


def _load_pixels(path):
    import numpy as np

    img = bpy.data.images.load(path, check_existing=False)
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    bpy.data.images.remove(img)
    return px.reshape(h, w, 4), w, h


def render_still(scene, path, floor=None, bg=(0.955, 0.955, 0.95)):
    """Render with transparent film + shadow-catcher floor, then composite onto a
    flat near-white backdrop in display space. The (soft, low-frequency) floor
    shadow is smoothed only where a second coverage render shows pure background,
    so part pixels are never filtered (the arm64 runtime has no denoiser)."""
    import numpy as np

    tmp = path[:-4] + "_rgba_tmp.png"
    scene.render.filepath = tmp
    bpy.ops.render.render(write_still=True)
    status = "composited"
    try:
        px, w, h = _load_pixels(tmp)
        a = px[..., 3]
        if floor is not None:
            cy = scene.cycles
            keep = (cy.samples, cy.use_adaptive_sampling, floor.hide_render)
            mtmp = path[:-4] + "_mask_tmp.png"
            cy.samples, cy.use_adaptive_sampling, floor.hide_render = 8, False, True
            scene.render.filepath = mtmp
            bpy.ops.render.render(write_still=True)
            cy.samples, cy.use_adaptive_sampling, floor.hide_render = keep
            mpx, _, _ = _load_pixels(mtmp)
            os.remove(mtmp)
            cover = _box_blur(mpx[..., 3], 1)  # dilate edges by ~1 px
            domain = (cover < 0.002).astype(np.float32)  # floor-only pixels
            r = max(2, int(round(5 * scene.render.resolution_percentage / 100.0)))
            num = _box_blur(_box_blur(_box_blur(a * domain, r), r), r)
            den = _box_blur(_box_blur(_box_blur(domain, r), r), r)
            smooth = num / np.maximum(den, 1e-6)
            a = np.where(domain > 0.5, smooth, a)
            status = "composited+shadow-smoothed"
        a = a[..., None]
        rgb = px[..., :3] * a + np.asarray(bg, dtype=np.float32)[None, None, :] * (
            1.0 - a
        )
        out = np.concatenate([rgb, np.ones_like(a)], axis=-1).ravel()
        res = bpy.data.images.new("MD_composite", w, h, alpha=False)
        res.pixels.foreach_set(out)
        res.filepath_raw = path
        res.file_format = "PNG"
        res.save()
        bpy.data.images.remove(res)
        os.remove(tmp)
        return status
    except Exception as exc:  # keep the transparent render rather than nothing
        os.replace(tmp, path)
        return "transparent-fallback: %r" % (exc,)


def export_glb(path, objs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    want = dict(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_extras=True,
        export_yup=True,
        export_apply=False,
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        export_materials="EXPORT",
        export_normals=True,
        export_tangents=False,
        export_morph=False,
        export_skins=False,
        export_draco_mesh_compression_enable=False,
        will_save_settings=False,
    )
    props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
    kw = {k: v for k, v in want.items() if k in props}
    bpy.ops.export_scene.gltf(**kw)
    return sorted(set(want) - set(kw))


def blender_main(args):
    import time

    t0 = time.time()
    log = dict(blender_version=bpy.app.version_string, args=args, steps={})

    def step(name, t):
        log["steps"][name] = round(time.time() - t, 2)

    samples = SAMPLES
    if not (getattr(bpy.app.build_options, "openimagedenoise", False)):
        samples = SAMPLES_NO_DENOISER
    if "--samples" in args:
        samples = int(args[args.index("--samples") + 1])
    do_render = "--no-render" not in args
    test_render = "--test-render" in args

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.save_version = 0  # no .blend1 backups
    scene = bpy.context.scene
    scene.name = "Microduck"
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0

    t = time.time()
    layout = build_layout(read_stl_numpy)
    comp = components_json(layout)
    step("parse_fk_stl", t)

    t = time.time()
    mats = {k: make_material(k, v) for k, v in MATERIALS.items()}
    parts_col = bpy.data.collections.new("Microduck_Parts")
    scene.collection.children.link(parts_col)
    gcols = {}
    for g, (zh, en) in GROUPS.items():
        c = bpy.data.collections.new("MD_%s (%s)" % (en, zh))
        parts_col.children.link(c)
        gcols[g] = c
    studio_col = bpy.data.collections.new("Studio (not exported)")
    scene.collection.children.link(studio_col)

    mesh_data, normals_mode = {}, {}
    for name, m in layout["meshes"].items():
        me = make_mesh(name, m["verts"], m["faces"])
        try:
            me = bake_weighted_normals(me, scene)
            normals_mode[name] = "weighted_face_area+sharp>%gdeg" % SMOOTH_ANGLE_DEG
        except Exception as exc:
            normals_mode[name] = "smooth_by_angle_only (%r)" % (exc,)
        me.materials.clear()
        me.materials.append(mats[MESH_INFO[name][5]])
        m["stats"]["blender_vertices"] = len(me.vertices)
        m["stats"]["blender_triangles"] = len(me.polygons)
        mesh_data[name] = me
    # statistics follow the meshes actually built (validate() may drop bad faces)
    for p in layout["parts"]:
        st = layout["meshes"][p["mesh"]]["stats"]
        p["vertices"], p["triangles"] = st["blender_vertices"], st["blender_triangles"]
    for m in layout["meshes"].values():
        m["stats"]["vertices"], m["stats"]["triangles"] = (
            m["stats"]["blender_vertices"],
            m["stats"]["blender_triangles"],
        )
    comp = components_json(layout)
    step("meshes_materials", t)

    t = time.time()
    objs = []
    by_id = {c["id"]: c for c in comp["parts"]}
    for p in layout["parts"]:
        c = by_id[p["id"]]
        ob = bpy.data.objects.new(object_name(p), mesh_data[p["mesh"]])
        gcols[p["group"]].objects.link(ob)
        ob.matrix_world = Matrix(p["M"])
        for k in (
            "id",
            "label_zh",
            "label_en",
            "description_zh",
            "description_en",
            "group",
            "source_mesh",
            "source_body",
            "source_instance",
            "source_material",
            "category",
            "group_zh",
            "group_en",
        ):
            ob[k] = c[k]
        ob["source_file"] = c["source_file"]
        ob["source_rgba"] = [float(x) for x in c["source_rgba"]]
        ob["drives_joint"] = c["drives_joint"] or ""
        ob["explode"] = [float(x) for x in c["explode"]]  # glTF Y-up metres
        ob["explode_blender"] = [float(x) for x in c["explode_blender"]]
        ob["triangles"] = int(c["triangles"])
        ob["vertices"] = int(c["vertices"])
        loc = ob.location.copy()
        ob.keyframe_insert("location", frame=FRAME_ASSEMBLED)
        ob.location = loc + Vector(p["explode"])
        ob.keyframe_insert("location", frame=FRAME_EXPLODED)
        ob.location = loc
        objs.append(ob)
    scene.frame_start = FRAME_ASSEMBLED
    scene.frame_end = FRAME_EXPLODED
    scene.frame_set(FRAME_ASSEMBLED)
    cam = setup_studio(
        scene,
        studio_col,
        comp["camera"]["ortho_scale_assembled"],
        comp["camera"]["ortho_scale_exploded"],
    )
    log["denoisers_available"] = setup_render(scene, samples)
    scene["microduck_note"] = (
        "Frame %d assembled (STAND), frame %d exploded. Source: %s"
        % (FRAME_ASSEMBLED, FRAME_EXPLODED, ROBOT_XML)
    )
    step("objects_studio", t)

    t = time.time()
    glb = os.path.join(OUT, "microduck.glb")
    scene.frame_set(FRAME_ASSEMBLED)
    log["glb_unsupported_kwargs"] = export_glb(glb, objs)
    log["glb_check"] = verify_glb(glb, layout)
    step("export_glb", t)

    comp["blender"] = dict(
        version=bpy.app.version_string,
        normals=normals_mode,
        mesh_counts={
            k: dict(
                vertices=m["stats"]["blender_vertices"],
                triangles=m["stats"]["blender_triangles"],
            )
            for k, m in layout["meshes"].items()
        },
        glb_check=log["glb_check"],
    )
    write_json(os.path.join(OUT, "components.json"), comp)

    blend = os.path.join(OUT, "microduck.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
    log["renders"] = {}
    if do_render:
        rdir, prefix = OUT, ""
        floor = bpy.data.objects.get("MD_Floor")
        if test_render:  # quick look-dev check, never overwrites the deliverables
            rdir, prefix = os.path.join(OUT, "checks"), "test_"
            os.makedirs(rdir, exist_ok=True)
            scene.render.resolution_percentage = (
                int(args[args.index("--percent") + 1]) if "--percent" in args else 35
            )
        t = time.time()
        scene.frame_set(FRAME_ASSEMBLED)
        log["renders"][prefix + "assembled.png"] = render_still(
            scene, os.path.join(rdir, prefix + "assembled.png"), floor
        )
        step("render_assembled", t)
        t = time.time()
        scene.frame_set(FRAME_EXPLODED)
        log["renders"][prefix + "exploded.png"] = render_still(
            scene, os.path.join(rdir, prefix + "exploded.png"), floor
        )
        step("render_exploded", t)
        scene.frame_set(FRAME_ASSEMBLED)  # objects back in assembled position
        scene.render.resolution_percentage = 100
        bpy.ops.wm.save_as_mainfile(filepath=blend, compress=True)
    log["total_s"] = round(time.time() - t0, 2)
    log["render_settings"] = dict(
        engine=scene.render.engine,
        device=scene.cycles.device,
        samples=scene.cycles.samples,
        view_transform=scene.view_settings.view_transform,
        look=scene.view_settings.look,
        resolution=list(RES),
    )
    write_json(os.path.join(OUT, "build_log.json"), log)
    print("[microduck] done:", json.dumps(log, ensure_ascii=False)[:2000])


def offline_main():
    layout = build_layout(lambda p: weld(read_stl_triangles(p)))
    comp = components_json(
        layout,
        extra=dict(
            generated_offline="Written by `python3 build.py --offline` (CPython, no Blender). "
            "Running build.py inside Blender regenerates this file with Blender mesh counts."
        ),
    )
    write_json(os.path.join(OUT, "components.json"), comp)
    print(
        json.dumps(
            dict(
                counts=comp["counts"],
                camera=comp["camera"],
                bounds=comp["bounds"],
                grounding_dz=comp["pose"]["grounding_dz_m"],
            ),
            indent=1,
        )
    )


if __name__ == "__main__":
    if IN_BLENDER:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
        blender_main(argv)
    elif "--offline" in sys.argv:
        offline_main()
    else:
        print(__doc__)
