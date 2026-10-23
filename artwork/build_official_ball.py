"""Port the official Microduck browser ball to a metric USD visual asset.

Color construction follows Pollen Robotics ball-visual.js, revision
ad9d41a763f30dc770817c33d92dcfec0205db1e, Apache-2.0. The browser's 10 cm diameter
and 30 g mass are preserved. Analytic collision is used, not mesh collision.
"""

from pathlib import Path
import math
import numpy as np
from PIL import Image
import newton_usd_schemas
import mujoco_usd_converter
from pxr import UsdGeom, UsdPhysics, UsdShade, Sdf
from build_scene_assets import OUT, stage, material, bind


def ball_texture():
    width, height = 2048, 1024
    verts = [[0, 0, 1], [0, 0, -1]]
    lat = math.atan(0.5)
    for i in range(5):
        for deg, z in [(i * 72, math.sin(lat)), (i * 72 + 36, -math.sin(lat))]:
            a = math.radians(deg)
            verts.append([math.cos(lat) * math.cos(a), math.cos(lat) * math.sin(a), z])
    verts = np.asarray(verts)
    sites = list(verts)
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                if all(verts[a] @ verts[b] > 0.3 for a, b in [(i, j), (i, k), (j, k)]):
                    v = verts[i] + verts[j] + verts[k]
                    sites.append(v / np.linalg.norm(v))
    assert len(sites) == 32
    sites = np.asarray(sites)
    lon = (np.arange(width) + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    out = np.empty((height, width, 3), dtype=np.uint8)
    for row, l in enumerate(lat):
        direction = np.stack(
            [
                np.cos(l) * np.cos(lon),
                np.cos(l) * np.sin(lon),
                np.full(width, np.sin(l)),
            ],
            axis=1,
        )
        dots = direction @ sites.T
        winner = np.argmax(dots, axis=1)
        top = np.partition(dots, -2, axis=1)[:, -2:]
        gap = (
            np.arccos(np.clip(top[:, 0], -1, 1)) - np.arccos(np.clip(top[:, 1], -1, 1))
        ) / 2
        black = (winner < 12) * np.clip(
            (gap - math.radians(1.6)) / math.radians(0.35), 0, 1
        )
        stitch = np.clip(1 - gap / (math.radians(1.6) * 0.6), 0, 1) * (1 - black)
        base = np.array([233, 231, 224]) + stitch[:, None] * (
            np.array([200, 197, 188]) - np.array([233, 231, 224])
        )
        out[row] = np.clip(
            base + black[:, None] * (np.array([23, 23, 29]) - base), 0, 255
        )
    Image.fromarray(out).save(OUT / "official_ball_color.png")


def build():
    ball_texture()
    s = stage("official_ball.usda")
    root = s.GetDefaultPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(0.03)
    collider = UsdGeom.Sphere.Define(s, "/Asset/Collision")
    collider.CreateRadiusAttr(0.05)
    collider.CreateVisibilityAttr("invisible")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    collider.GetPrim().ApplyAPI("NewtonCollisionAPI")
    collider.GetPrim().ApplyAPI("MjcCollisionAPI")
    collider.GetPrim().CreateAttribute(
        "newton:contactGap", Sdf.ValueTypeNames.Float
    ).Set(0.0)
    collider.GetPrim().CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(6)
    collider.GetPrim().CreateAttribute(
        "mjc:solref", Sdf.ValueTypeNames.DoubleArray
    ).Set([0.03, 0.4])
    contact = UsdShade.Material.Define(s, "/Asset/OfficialContact")
    physics = UsdPhysics.MaterialAPI.Apply(contact.GetPrim())
    physics.CreateStaticFrictionAttr(0.4)
    physics.CreateDynamicFrictionAttr(0.4)
    contact.GetPrim().ApplyAPI("NewtonMaterialAPI")
    # The same schema mapping as mujoco-usd-converter 0.6.1 for this geom.
    for name, value in {
        "rollingFriction": 0.003,
        "torsionalFriction": 0.01,
        "contactDamping": 66.666664,
        "contactStiffness": 6944.4443,
    }.items():
        contact.GetPrim().CreateAttribute(
            "newton:" + name, Sdf.ValueTypeNames.Float
        ).Set(value)
    UsdShade.MaterialBindingAPI.Apply(collider.GetPrim()).Bind(
        contact, materialPurpose="physics"
    )
    m = material(
        s, "OfficialSoccerBall", (0.91, 0.90, 0.88), 0.55, 0, "official_ball_color.png"
    )
    mesh = UsdGeom.Mesh.Define(s, "/Asset/Visual")
    points = []
    uv = []
    normals = []
    faces = []
    nx, ny = 64, 48
    for row in range(ny + 1):
        lat = np.pi / 2 - row / ny * np.pi
        for col in range(nx + 1):
            lon = col / nx * 2 * np.pi - np.pi
            n = [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
            points.append((np.array(n) * 0.05).tolist())
            normals.append(n)
            uv.append((col / nx, 1 - row / ny))
    for row in range(ny):
        for col in range(nx):
            i = row * (nx + 1) + col
            faces.extend([i, i + nx + 1, i + nx + 2, i + 1])
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4] * (nx * ny))
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation("vertex")
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
    ).Set(uv)
    bind(mesh, m)
    s.GetRootLayer().Save()


if __name__ == "__main__":
    build()
