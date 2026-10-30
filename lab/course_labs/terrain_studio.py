"""Studio appearance for the slope (terrain) lab: a plywood wedge on a white floor.

The geometry (floor + finite wedge: ramp, platform, sides and back resting on the floor) is the terrain
mesh built in terrain_check.ramp_mesh; this module only styles it: a plain white floor
binding, a plywood display skin on the wedge faces (a textured mesh 0.6 mm outside the
surface, no collision API) and a softer key light. The physics material of the terrain
(friction, restitution) and the robot are unchanged. The plywood texture is the base
colour of NVIDIA's Omniverse base material Materials/Base/Wood/Plywood.mdl from the
Isaac Sim asset library, rendered with UsdPreviewSurface (no new MDL shader compile).

Kept separate from lab/terrain_visuals.py, whose camera and recording helpers are
shared with the football scenes and must stay unchanged; this module is used only by
the terrain check (lab/course_labs/terrain_check.py) and the training viewer.
"""

import math
import os

STUDIO_FLOOR = (0.93, 0.93, 0.94)  # plain white studio floor
PLYWOOD_MDL = "Materials/Base/Wood/Plywood.mdl"
PLYWOOD_FILES = ("Plywood.mdl", "Plywood/Plywood_BaseColor.png", "Plywood/Plywood_ORM.png", "Plywood/Plywood_N.png")
# Persistent in the course container (host ~/.cache/microduck/ov), so it is fetched once.
MATERIAL_CACHE = "/root/.cache/ov/microduck-materials/Wood"


def plywood_mdl_path():
    """Local copy of the Plywood material and its textures, fetched once from the Isaac Sim
    asset library; falls back to the remote URL if the copy cannot be made. The source used
    is printed so a missing texture shows up in the log instead of a silent blank material."""
    import json
    import os
    import urllib.request

    from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

    remote_dir = f"{NVIDIA_NUCLEUS_DIR}/{PLYWOOD_MDL.rsplit('/', 1)[0]}"
    local = os.path.join(MATERIAL_CACHE, "Plywood.mdl")
    source = "local cache"
    try:
        for name in PLYWOOD_FILES:
            target = os.path.join(MATERIAL_CACHE, name)
            if not os.path.isfile(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with urllib.request.urlopen(f"{remote_dir}/{name}", timeout=30) as response:
                    data = response.read()
                with open(target + ".part", "wb") as stream:
                    stream.write(data)
                os.replace(target + ".part", target)
                source = "downloaded to local cache"
        path = local
    except Exception as error:  # appearance only: keep going with the remote material
        path, source = f"{NVIDIA_NUCLEUS_DIR}/{PLYWOOD_MDL}", f"remote (local copy failed: {error!r})"
    print("COURSE_MATERIAL " + json.dumps(dict(material="Plywood", path=path, source=source)), flush=True)
    return path


def apply_studio_floor(env, env_ids):
    """Bind a neutral light-grey visual material to the terrain meshes (no physics change)."""
    from pxr import Usd, UsdGeom, UsdShade, Sdf

    stage = env.sim.stage
    material = UsdShade.Material.Define(stage, "/World/CourseStudioFloor")
    shader = UsdShade.Shader.Define(stage, "/World/CourseStudioFloor/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(STUDIO_FLOOR)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    ground = stage.GetPrimAtPath(env.cfg.scene.terrain.prim_path)
    for prim in Usd.PrimRange(ground):
        if prim.IsA(UsdGeom.Mesh):
            # All-purpose (visual) binding. Isaac Lab binds the ground's physics material with
            # materialPurpose="physics", which USD resolves before any all-purpose binding, so
            # friction and restitution stay as they were (the football floor style does the same).
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )


WOOD_TILE_M = 0.6  # one texture repeat every 0.6 m of board


def plywood_texture():
    """Base-colour texture of the plywood (local cached copy when available)."""
    import os

    plywood_mdl_path()  # fetches the material folder once into the local cache
    local = os.path.join(MATERIAL_CACHE, "Plywood/Plywood_BaseColor.png")
    if os.path.isfile(local):
        return local
    from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

    return f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wood/Plywood/Plywood_BaseColor.png"


def wood_material(stage, path):
    """UsdPreviewSurface reading the plywood texture through the mesh's 'st' coordinates.
    Uses the preview-surface shaders the scene already compiles, so no new MDL shader
    has to be compiled the first time the window opens."""
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, path + "/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.55)
    reader = UsdShade.Shader.Define(stage, path + "/St")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture = UsdShade.Shader.Define(stage, path + "/BaseColor")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(plywood_texture())
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material


def _terrain_points_world(stage, terrain_path):
    """World-space vertices of the terrain meshes under terrain_path (the meshes Newton
    collides with), via each prim's local-to-world transform."""
    import numpy as np
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache()
    chunks = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(terrain_path)):
        if prim.IsA(UsdGeom.Mesh):
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
            matrix = np.array(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
            chunks.append((np.c_[points, np.ones(len(points))] @ matrix)[:, :3])  # USD: row vectors
    return np.concatenate(chunks) if chunks else np.zeros((0, 3))


def wedge_box(points, wedge, slope_deg, tolerance=1e-4):
    """Corners of the wedge found in terrain vertices: x0 (ramp foot), x1 (ramp top),
    x2 (platform end), y0/y1 (sides), floor z and platform height. The wedge top is the
    highest part of the terrain (Isaac Lab's generator also adds a border whose boxes reach
    below the floor, so the lowest vertex is not the floor); its four corners are the
    highest vertices. The two ramp-foot vertices lie on the side lines just before the top,
    below it, and give the floor height. Duplicate vertices are merged first. Dimensions are
    checked against WEDGE so that a mismatch is reported instead of a skin drawn elsewhere."""
    import numpy as np

    expected_h = wedge["ramp_length"] * math.tan(math.radians(abs(slope_deg)))
    points = np.unique(np.round(points, 6), axis=0)
    top = points[points[:, 2] > points[:, 2].max() - 1e-4]
    if len(top) != 4:
        raise ValueError(f"expected 4 wedge top vertices, found {len(top)}")
    x1, x2 = float(top[:, 0].min()), float(top[:, 0].max())
    y0, y1 = float(top[:, 1].min()), float(top[:, 1].max())
    top_z = float(top[:, 2].mean())
    at_side = (np.abs(points[:, 1] - y0) < 1e-5) | (np.abs(points[:, 1] - y1) < 1e-5)
    before_top = (points[:, 0] < x1 - 1e-3) & (points[:, 0] > x1 - 2 * wedge["ramp_length"])
    foot = points[at_side & before_top & (points[:, 2] < top_z - 1e-3)]
    if len(foot) != 2 or np.ptp(foot[:, 0]) > 1e-5 or np.ptp(foot[:, 2]) > 1e-5:
        raise ValueError(f"expected 2 ramp-foot vertices, found {len(foot)}")
    floor = float(foot[:, 2].mean())
    box = dict(x0=float(foot[:, 0].mean()), x1=x1, x2=x2, y0=y0, y1=y1, z=floor, h=top_z - floor)
    measured = dict(ramp_length=x1 - box["x0"], platform_length=x2 - x1, half_width=(y1 - y0) / 2, height=box["h"])
    wanted = dict(ramp_length=wedge["ramp_length"], platform_length=wedge["platform_length"],
                  half_width=wedge["half_width"], height=expected_h)
    off = {k: measured[k] - wanted[k] for k in wanted if abs(measured[k] - wanted[k]) > tolerance}
    if off:
        raise ValueError(f"wedge dimensions differ from WEDGE: {off}")
    return box


SKIN_PATH = "/World/CourseWoodWedge"
SKIN_GAP = 0.0006  # the skin sits 0.6 mm outside the physical surface


def author_wood_wedge(env, env_ids, slope_deg, wedge):
    """Plywood skin for the finite wedge of the slope test: sloping top, platform, back and
    both sides as one textured display mesh, 0.6 mm outside the terrain surface and without
    any collision API. Its corners are taken from the world-space vertices of the terrain
    mesh itself (the mesh Newton collides with), and it is authored in world coordinates
    under /World, so no environment or terrain offset is applied twice. All environments
    share this one wedge. The terrain mesh underneath does all the contact physics."""
    import json

    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    stage = env.sim.stage
    if wedge["ramp_length"] * math.tan(math.radians(abs(slope_deg))) < 1e-3:
        return
    try:
        b = wedge_box(_terrain_points_world(stage, env.cfg.scene.terrain.prim_path), wedge, slope_deg)
    except ValueError as error:  # appearance only: leave the plain wedge rather than a misplaced skin
        print("COURSE_WOOD_WEDGE " + json.dumps(dict(status="skipped", reason=str(error))), flush=True)
        return
    e = SKIN_GAP
    x0, x1, x2, y0, y1, z, h = (b[k] for k in ("x0", "x1", "x2", "y0", "y1", "z", "h"))
    width = y1 - y0
    ramp = math.hypot(x1 - x0, h)
    plat = x2 - x1
    T = WOOD_TILE_M
    # (corner points, texture coordinates) per face, already pushed 0.6 mm outwards.
    faces = [
        ([(x0, y0, z + e), (x1, y0, z + h + e), (x1, y1, z + h + e), (x0, y1, z + e)],
         [(0, 0), (ramp / T, 0), (ramp / T, width / T), (0, width / T)]),
        ([(x1, y0, z + h + e), (x2, y0, z + h + e), (x2, y1, z + h + e), (x1, y1, z + h + e)],
         [(ramp / T, 0), ((ramp + plat) / T, 0), ((ramp + plat) / T, width / T), (ramp / T, width / T)]),
        ([(x2 + e, y0, z), (x2 + e, y1, z), (x2 + e, y1, z + h), (x2 + e, y0, z + h)],
         [(0, 0), (width / T, 0), (width / T, h / T), (0, h / T)]),
        ([(x0, y0 - e, z), (x2, y0 - e, z), (x2, y0 - e, z + h), (x1, y0 - e, z + h)],
         [(0, 0), ((x2 - x0) / T, 0), ((x2 - x0) / T, h / T), ((x1 - x0) / T, h / T)]),
        ([(x0, y1 + e, z), (x1, y1 + e, z + h), (x2, y1 + e, z + h), (x2, y1 + e, z)],
         [(0, 0), ((x1 - x0) / T, h / T), ((x2 - x0) / T, h / T), ((x2 - x0) / T, 0)]),
    ]
    points, uvs, counts, indices = [], [], [], []
    for corners, st in faces:
        base = len(points)
        points += corners
        uvs += st
        counts.append(len(corners))
        indices += [base + i for i in range(len(corners))]
    # World coordinates -> the parent's frame (identity for /World, but do not assume it).
    parent = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(SKIN_PATH.rsplit("/", 1)[0]))
    to_parent = np.array(parent.GetInverse(), dtype=np.float64)
    local = (np.c_[np.array(points), np.ones(len(points))] @ to_parent)[:, :3]
    mesh = UsdGeom.Mesh.Define(stage, SKIN_PATH)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in local]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, local.min(0))), Gf.Vec3f(*map(float, local.max(0)))]))
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                            UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray([Gf.Vec2f(*u) for u in uvs]))
    material = wood_material(stage, "/World/CourseWood")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    print("COURSE_WOOD_WEDGE " + json.dumps(dict(status="authored", path=SKIN_PATH, wedge_world=b)), flush=True)


def wood_alignment(env, manager):
    """Proof that the skin sits on the physical wedge: every skin vertex (world, from USD)
    against the nearest vertex of the wedge in the finalized Newton model (shape geometry
    with its scale and world transform). Expected distance: the 0.6 mm gap for every vertex."""
    import numpy as np
    from pxr import UsdGeom

    stage = env.sim.stage
    prim = stage.GetPrimAtPath(SKIN_PATH)
    if not prim.IsValid():
        return dict(status="no skin")
    skin = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=np.float64)
    matrix = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=np.float64)
    skin = (np.c_[skin, np.ones(len(skin))] @ matrix)[:, :3]
    from physics_audit import _model

    model = _model(manager)
    sources = list(getattr(model, "shape_source", []) or [])
    xforms = model.shape_transform.numpy()
    scales = model.shape_scale.numpy()
    bodies = model.shape_body.numpy()
    labels = list(getattr(model, "shape_label", []) or [])
    best = None
    for i, source in enumerate(sources):
        vertices = getattr(source, "vertices", None)
        if vertices is None or bodies[i] >= 0:  # static meshes only (the terrain)
            continue
        v = np.asarray(vertices.numpy() if hasattr(vertices, "numpy") else vertices, dtype=np.float64) * scales[i]
        p, q = xforms[i][:3], xforms[i][3:]  # warp transform: position, quaternion (x, y, z, w)
        qv, w = q[:3], q[3]
        world = v + 2 * np.cross(qv, np.cross(qv, v) + w * v) + p
        top = world[world[:, 2] > world[:, 2].max() - 1e-4]  # wedge top = highest terrain vertices
        if len(top) == 0 or world[:, 2].max() - np.median(world[:, 2]) < 1e-3:
            continue
        lo, hi = top.min(0) - 2.0, top.max(0) + 0.01
        near = world[np.all((world[:, :2] >= lo[:2]) & (world[:, :2] <= hi[:2]), axis=1)]
        distance = np.linalg.norm(skin[:, None, :] - near[None, :, :], axis=2).min(1)
        best = dict(status="checked", shape=labels[i] if i < len(labels) else i,
                    skin_vertices=len(skin), gap_expected_m=SKIN_GAP,
                    nearest_physics_vertex_m=dict(min=float(distance.min()), max=float(distance.max())),
                    aligned=bool(np.all(np.abs(distance - SKIN_GAP) < 2e-5)),
                    physics_wedge_top_world=dict(min=top.min(0).tolist(), max=top.max(0).tolist()),
                    skin_world=dict(min=skin.min(0).tolist(), max=skin.max(0).tolist()))
    return best or dict(status="no static mesh with a wedge in the Newton model")


def apply_wood_terrain(env, env_ids):
    """Plywood look for generated training terrain: planar 'st' coordinates on the terrain
    meshes (a render-only primvar) and an all-purpose visual binding. The physics material
    stays bound with materialPurpose='physics'; geometry and collision are unchanged."""
    import numpy as np
    from pxr import Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = env.sim.stage
    material = wood_material(stage, "/World/CourseWood")
    ground = stage.GetPrimAtPath(env.cfg.scene.terrain.prim_path)
    for prim in Usd.PrimRange(ground):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        st = points[:, :2] / WOOD_TILE_M
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(st))
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
        )


def configure_scene(cfg):
    import isaaclab.sim as sim_utils
    from isaaclab.managers import EventTermCfg

    from scene_style import configure_lighting

    configure_lighting(cfg)
    # Studio look for this lab only (the football scenes keep their own floor).
    cfg.events.course_surface_style = EventTermCfg(func=apply_studio_floor, mode="prestartup")
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=STUDIO_FLOOR, roughness=0.35, metallic=0.0
    )
    cfg.scene.key_light.spawn.angle = 20.0  # wider light source: softer shadows
    cfg.scene.sky_light.spawn.intensity = 450.0
    angle = cfg.scene.terrain.terrain_generator.sub_terrains["ramp"].slope_deg
    # Plywood skin on the wedge (textured display mesh, no collision API), placed from the
    # world-space vertices of the generated terrain mesh.
    from terrain_check import WEDGE

    cfg.events.course_wood_wedge = EventTermCfg(
        func=author_wood_wedge, mode="prestartup",
        params=dict(slope_deg=angle, wedge=dict(WEDGE)),
    )
