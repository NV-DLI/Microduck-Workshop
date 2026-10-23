"""Render recorded joint/root/ball poses without simulating or changing a policy.
blender -b artwork/microduck.blend --python artwork/render_recorded_poses.py -- RUN OUTPUT
The CAD uses effective output joint angles: servo angle + passive backlash angle.
Lighting and goal-frame appearance are presentation; physical results remain in RUN.
"""

import bpy

import sys

import json

import hashlib

import importlib.util
from pathlib import Path
from collections import defaultdict
from mathutils import Matrix, Vector

run = Path(sys.argv[sys.argv.index("--") + 1])
out = Path(sys.argv[sys.argv.index("--") + 2])
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location(
    "cad", Path(__file__).with_name("cad_assembly.py")
)
cad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cad)
manifest = json.loads((run / "manifest.json").read_text())
rows = [json.loads(x) for x in (run / "physics-trace.jsonl").read_text().splitlines()]
onset = next(r["kick_onset_s"] for r in rows if r.get("kick_onset_s"))
targets = [onset - 0.1, onset + 0.12, rows[-1]["timestamp_s"]]
scene = bpy.context.scene
scene.frame_set(1)
objects = {o["id"]: o for o in bpy.data.objects if o.type == "MESH" and "id" in o}
for ob in objects.values():
    ob.animation_data_clear()
scene.camera.data.animation_data_clear()
scene.camera.animation_data_clear()
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.render.engine = "CYCLES"
scene.cycles.samples = 512
scene.cycles.device = "CPU"
scene.cycles.use_adaptive_sampling = True
scene.cycles.adaptive_threshold = 0.008
scene.cycles.use_denoising = False
scene.render.threads_mode = "FIXED"
scene.render.threads = 12
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (
    0.30,
    0.36,
    0.44,
    1,
)
scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.5
for ob in bpy.data.objects:
    if ob.type == "LIGHT":
        ob.data.energy *= 8
        ob.data.size *= 2
        ob.location *= 2
for m in bpy.data.materials:
    if m.name.startswith("Studio matte"):
        m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
            0.055,
            0.10,
            0.075,
            1,
        )


def material(name, color, roughness):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bs = m.node_tree.nodes["Principled BSDF"]
    bs.inputs["Base Color"].default_value = (*color, 1)
    bs.inputs["Roughness"].default_value = roughness
    return m


white = material("Goal and ball", (0.83, 0.86, 0.81), 0.45)
bpy.ops.mesh.primitive_uv_sphere_add(
    segments=64, ring_count=32, radius=manifest["fixture"]["ball_radius_m"]
)
ball = bpy.context.object
ball.name = "Recorded football"
ball.data.materials.append(white)
for f in ball.data.polygons:
    f.use_smooth = True
fx = manifest["fixture"]["goal_plane_x_m"]
gw = manifest["fixture"]["goal_opening_width_m"]
gh = manifest["fixture"]["goal_opening_height_m"]
# Put visual posts just outside the recorded clear opening.
for y in [-gw / 2 - 0.0075, gw / 2 + 0.0075]:
    bpy.ops.mesh.primitive_cube_add(size=1, location=(fx, y, gh / 2))
    o = bpy.context.object
    o.dimensions = (0.015, 0.015, gh)
    o.data.materials.append(white)
bpy.ops.mesh.primitive_cube_add(size=1, location=(fx, 0, gh + 0.0075))
o = bpy.context.object
o.dimensions = (0.015, gw + 0.03, 0.015)
o.data.materials.append(white)
scene.camera.location = (1.35, -1.18, 0.63)
scene.camera.rotation_euler = (
    (Vector((0.55, 0, 0.16)) - scene.camera.location)
    .to_track_quat("-Z", "Y")
    .to_euler()
)
scene.camera.data.ortho_scale = 1.32
scene.camera.data.clip_start = 0.001
hinges = cad.load_model()["hinge_order"]
receipt = []
for index, target in enumerate(targets):
    r = min(rows, key=lambda r: abs(r["timestamp_s"] - target))
    q = dict(zip(manifest["joint_names"], r["joint_positions_absolute_rad"]))
    effective = {n: q[n] + q["passive_" + n + "_backlash"] for n in hinges}
    x, y, z, w = r["root_quaternion_xyzw"]
    qpos = r["root_position_world_m"] + [w, x, y, z] + [effective[n] for n in hinges]
    cad.parse_keyframe = lambda name: qpos
    model = cad.load_model()
    counts = defaultdict(int)
    for g in model["geoms"]:
        ident = f"{g['body']}--{g['mesh']}--{counts[g['body']]:02d}"
        counts[g["body"]] += 1
        objects[ident].matrix_world = Matrix(g["M"])
    ball.location = r["ball_position_world_m"]
    bpy.context.view_layer.update()
    scene.render.filepath = str(out / f"pose-{index+1}.png")
    bpy.ops.render.render(write_still=True)
    receipt.append(
        dict(
            frame=index + 1,
            timestamp_s=r["timestamp_s"],
            task_time_s=r["task_time_s"],
            root_position_world_m=r["root_position_world_m"],
            root_quaternion_xyzw=r["root_quaternion_xyzw"],
            effective_joint_positions_rad=effective,
            ball_position_world_m=r["ball_position_world_m"],
        )
    )
(out / "replay-receipt.json").write_text(
    json.dumps(
        dict(
            source_trace_sha256=hashlib.sha256(
                (run / "physics-trace.jsonl").read_bytes()
            ).hexdigest(),
            model_updates=0,
            physics_steps=0,
            frames=receipt,
            scope="Offline CAD rendering of recorded root, effective joint and ball poses; materials, lighting and visible goal posts are presentation, not another physical trial",
        ),
        indent=2,
    )
)
