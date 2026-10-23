"""Build measured-size scene geometry. Product reference is recorded in assets/README.
Run with a Python environment providing pxr, numpy and Pillow.
"""

from pathlib import Path
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf

OUT = Path(__file__).resolve().parents[1] / "lab/assets"
OUT.mkdir(exist_ok=True)


def stage(name):
    p = OUT / name
    s = Usd.Stage.CreateNew(str(p))
    root = UsdGeom.Xform.Define(s, "/Asset")
    s.SetDefaultPrim(root.GetPrim())
    UsdGeom.SetStageMetersPerUnit(s, 1.0)
    UsdGeom.SetStageUpAxis(s, "Z")
    return s


def material(s, name, color, roughness=0.7, metallic=0, texture=None):
    m = UsdShade.Material.Define(s, "/Asset/Materials/" + name)
    bs = UsdShade.Shader.Define(s, m.GetPath().AppendChild("Surface"))
    bs.CreateIdAttr("UsdPreviewSurface")
    bs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    bs.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    bs.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if texture:
        uv = UsdShade.Shader.Define(s, m.GetPath().AppendChild("UV"))
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        tex = UsdShade.Shader.Define(s, m.GetPath().AppendChild("Color"))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv.ConnectableAPI(), "result"
        )
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        bs.GetInput("diffuseColor").ConnectToSource(tex.ConnectableAPI(), "rgb")
    m.CreateSurfaceOutput().ConnectToSource(bs.ConnectableAPI(), "surface")
    return m


def bind(g, m):
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m)


def rod(s, name, a, b, r, m, collision=True):
    a = np.array(a)
    b = np.array(b)
    v = b - a
    length = float(np.linalg.norm(v))
    g = UsdGeom.Capsule.Define(s, "/Asset/" + name)
    g.CreateRadiusAttr(r)
    g.CreateHeightAttr(max(0.0001, length - 2 * r))
    g.CreateAxisAttr("Z")
    xf = UsdGeom.Xformable(g)
    xf.AddTranslateOp().Set(Gf.Vec3d(*((a + b) / 2).tolist()))
    rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(v / length).tolist()))
    q = rot.GetQuat()
    xf.AddOrientOp().Set(Gf.Quatf(q.GetReal(), Gf.Vec3f(q.GetImaginary())))
    bind(g, m)
    if collision:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    return g


def make_goal():
    s = stage("steel_goal.usda")
    white = material(s, "PowderCoatedSteel", (0.82, 0.85, 0.86), 0.3, 0.22)
    steel = material(s, "WovenSteel", (0.64, 0.68, 0.7), 0.24, 0.7)
    # Outer W/D/H .40/.20/.25 m. Frame diameter is an explicit approximation.
    yl, yr = -0.197, 0.197
    z = 0.247
    back = 0.194
    edges = [
        ((0, yl, 0.003), (0, yl, z)),
        ((0, yr, 0.003), (0, yr, z)),
        ((0, yl, z), (0, yr, z)),
        ((0, yl, 0.003), (back, yl, 0.003)),
        ((0, yr, 0.003), (back, yr, 0.003)),
        ((back, yl, 0.003), (back, yr, 0.003)),
        ((back, yl, 0.003), (back, yl, z)),
        ((back, yr, 0.003), (back, yr, z)),
        ((back, yl, z), (back, yr, z)),
        ((0, yl, z), (back, yl, z)),
        ((0, yr, z), (back, yr, z)),
    ]
    for i, (a, b) in enumerate(edges):
        rod(s, "Frame_" + str(i), a, b, 0.003, white)
    i = 0

    def wire(a, b):
        nonlocal i
        rod(s, "Net_" + str(i), a, b, 0.0004, steel)
        i += 1

    for y in np.arange(yl + 0.006, yr, 0.012):
        wire((back, y, 0.003), (back, y, z))
        wire((0, y, z), (back, y, z))
    for zz in np.arange(0.009, z, 0.012):
        wire((back, yl, zz), (back, yr, zz))
        for y in [yl, yr]:
            wire((0, y, zz), (back, y, zz))
    for x in np.arange(0.006, back, 0.012):
        wire((x, yl, z), (x, yr, z))
        for y in [yl, yr]:
            wire((x, y, 0.003), (x, y, z))
    s.GetRootLayer().Save()
    return i


def make_mat(style):
    # Original vector drawing inspired by the four user-provided product images.
    # No retailer photograph or room background is redistributed as a texture.
    W, H = 2560, 1600
    palette = {
        "A": ((72, 119, 47), (123, 165, 42), 0.19),
        "B": ((66, 116, 29), (66, 116, 29), 0.28),
        "C": ((9, 108, 47), (10, 133, 54), 0.22),
        "D": ((101, 156, 61), (143, 188, 94), 0.12),
    }
    c1, c2, rad = palette[style]
    im = Image.new("RGB", (W, H), c1)
    d = ImageDraw.Draw(im)
    for j in range(12):
        if j % 2:
            d.rectangle((round(j * W / 12), 0, round((j + 1) * W / 12), H), fill=c2)

    def xy(x, y):
        return round((x + 0.8) * W / 1.6), round((0.5 - y) * H)

    white = (235, 239, 222)
    line = 7

    def poly(points):
        d.line([xy(*p) for p in points], fill=white, width=line, joint="curve")

    def circle(x, y, r):
        a = xy(x - r, y + r)
        b = xy(x + r, y - r)
        d.ellipse((*a, *b), outline=white, width=line)

    poly([(-0.76, -0.46), (0.76, -0.46), (0.76, 0.46), (-0.76, 0.46), (-0.76, -0.46)])
    poly([(0, -0.46), (0, 0.46)])
    circle(0, 0, rad)
    for sign in [-1, 1]:
        x = sign * 0.76
        poly([(x, -0.29), (sign * 0.52, -0.29), (sign * 0.52, 0.29), (x, 0.29)])
        poly([(x, -0.15), (sign * 0.67, -0.15), (sign * 0.67, 0.15), (x, 0.15)])
        cx, cy = xy(sign * 0.58, 0)
        d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), fill=white)
        for y in [-0.46, 0.46]:
            circle(x, y, 0.048)
    rng = np.random.default_rng(103)
    a = np.array(im, dtype=float)
    noise = rng.normal(0, 1.3, (H, W, 1))
    weave = (np.indices((H, W)).sum(axis=0) % 3 == 0)[..., None] * 1.5
    a = np.clip(a + noise - weave, 0, 255).astype("uint8")
    Image.fromarray(a).save(OUT / f"pitch_{style}.png")
    s = stage(f"pitch_{style}.usda")
    m = material(s, "PrintedShortPile", (0.2, 0.4, 0.1), 0.94, 0, f"pitch_{style}.png")
    rubber = material(s, "RubberBinding", (0.10, 0.13, 0.09), 0.94)
    # Rounded top polygon with exact metric UVs.
    pts = []
    for cx, cy, start in [
        (0.785, 0.485, 0),
        (-0.785, 0.485, 90),
        (-0.785, -0.485, 180),
        (0.785, -0.485, 270),
    ]:
        for ang in np.linspace(start, start + 90, 13):
            t = math.radians(ang)
            pts.append((cx + 0.015 * math.cos(t), cy + 0.015 * math.sin(t), 0.00005))
    g = UsdGeom.Mesh.Define(s, "/Asset/Print")
    g.CreatePointsAttr(pts)
    g.CreateFaceVertexCountsAttr([len(pts)])
    g.CreateFaceVertexIndicesAttr(list(range(len(pts))))
    g.CreateSubdivisionSchemeAttr("none")
    uv = UsdGeom.PrimvarsAPI(g).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
    )
    uv.Set([((p[0] + 0.8) / 1.6, (p[1] + 0.5)) for p in pts])
    bind(g, m)
    for i, a in enumerate(pts):
        b = pts[(i + 1) % len(pts)]
        rod(
            s,
            "StitchedEdge_" + str(i),
            (a[0], a[1], -0.0008),
            (b[0], b[1], -0.0008),
            0.001,
            rubber,
            False,
        )
    body = UsdGeom.Cube.Define(s, "/Asset/Backing")
    body.CreateSizeAttr(1)
    xf = UsdGeom.Xformable(body)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.002))
    xf.AddScaleOp().Set(Gf.Vec3f(1.57, 0.97, 0.004))
    bind(body, rubber)
    s.GetRootLayer().Save()


if __name__ == "__main__":
    print("Net wires:", make_goal())
    for style in "ABCD":
        make_mat(style)
