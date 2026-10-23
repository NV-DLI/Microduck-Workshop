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
    top = 0.060  # Shallow roof and sloping back, following the product side profile.
    edges = [
        ((0, yl, 0.003), (0, yl, z)),
        ((0, yr, 0.003), (0, yr, z)),
        ((0, yl, z), (0, yr, z)),
        ((0, yl, 0.003), (back, yl, 0.003)),
        ((0, yr, 0.003), (back, yr, 0.003)),
        ((back, yl, 0.003), (back, yr, 0.003)),
        ((back, yl, 0.003), (top, yl, z)),
        ((back, yr, 0.003), (top, yr, z)),
        ((top, yl, z), (top, yr, z)),
        ((0, yl, z), (top, yl, z)),
        ((0, yr, z), (top, yr, z)),
    ]
    for i, (a, b) in enumerate(edges):
        rod(s, "Frame_" + str(i), a, b, 0.003, white)
    i = 0

    def wire(a, b):
        nonlocal i
        rod(s, "Net_" + str(i), a, b, 0.0004, steel)
        i += 1

    def diamond_panel(p00, p10, p11, p01, nu, nv):
        corners = list(map(np.asarray, [p00, p10, p11, p01]))

        def at(u, v):
            a, b, c, d = corners
            return (1 - u) * (1 - v) * a + u * (1 - v) * b + u * v * c + (1 - u) * v * d

        for sign in [-1, 1]:
            for k in range(-nv, nu + nv + 1):
                candidates = []
                for u in [0.0, 1.0]:
                    v = (k - nu * u) / (sign * nv)
                    if 0 <= v <= 1:
                        candidates.append((u, v))
                for v in [0.0, 1.0]:
                    u = (k - sign * nv * v) / nu
                    if 0 <= u <= 1:
                        candidates.append((u, v))
                unique = list(dict.fromkeys(candidates))
                if len(unique) == 2:
                    wire(at(*unique[0]), at(*unique[1]))

    diamond_panel(
        (back, yl, 0.003), (back, yr, 0.003), (top, yr, z), (top, yl, z), 33, 21
    )
    diamond_panel((0, yl, z), (0, yr, z), (top, yr, z), (top, yl, z), 33, 5)
    for y in [yl, yr]:
        diamond_panel((0, y, 0.003), (back, y, 0.003), (top, y, z), (0, y, z), 16, 21)
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
    if style in "CD":
        a = floor_texture(style, W, H)
    Image.fromarray(a).save(OUT / f"pitch_{style}.png")
    s = stage(f"pitch_{style}.usda")
    roughness = {"A": 0.94, "B": 0.94, "C": 0.28, "D": 0.40}[style]
    m = material(
        s, "GroundSurface", (0.2, 0.4, 0.1), roughness, 0, f"pitch_{style}.png"
    )
    rubber = material(
        s,
        "Edge",
        (
            (0.10, 0.13, 0.09)
            if style in "AB"
            else ((0.68, 0.67, 0.64) if style == "C" else (0.18, 0.085, 0.035))
        ),
        roughness,
    )
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
    for i, a in enumerate(pts if style in "AB" else []):
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


def floor_texture(style, width, height):
    """Original deterministic marble and walnut patterns, in metric proportions."""
    rng = np.random.default_rng(814)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x, y = xx / width * 1.6, yy / height
    if style == "C":
        wave = (
            x * 4.2
            + y * 1.7
            + 0.18 * np.sin(y * 12 + x * 3)
            + 0.08 * np.sin(x * 31 - y * 21)
        )
        vein = np.exp(-((np.sin(wave * 6.0) / 0.055) ** 2))
        hair = np.exp(-((np.sin((wave + 0.04 * np.sin(y * 41)) * 19) / 0.035) ** 2))
        cloudy = 4 * np.sin(x * 11 + y * 14) * np.sin(x * 5 - y * 3)
        value = cloudy - 31 * vein - 8 * hair + rng.normal(0, 0.65, (height, width))
        a = np.array([224, 224, 218])[None, None, :] + value[:, :, None]
        seams = (np.mod(x, 0.4) < 0.0015) | (np.mod(y, 0.5) < 0.0015)
        a[seams] = [190, 190, 186]
    else:
        row = np.floor(y / 0.125).astype(int)
        col = np.floor((x + 0.25 * (row % 2)) / 0.6).astype(int)
        local = np.mod(y, 0.125) / 0.125
        grain = np.sin(local * 100 + 6 * np.sin(x * 8) + 2 * np.sin(x * 29 + row))
        fine = np.sin(local * 310 + 9 * np.sin(x * 5 + row))
        variation = 7 * np.sin(row * 8.3 + col * 3.7)
        grain_value = (
            4 * grain + 1.6 * fine + variation + rng.normal(0, 0.7, (height, width))
        )
        a = np.array([121, 76, 43])[None, None, :] + grain_value[:, :, None] * [
            1,
            0.78,
            0.52,
        ]
        seams = (np.mod(y, 0.125) < 0.001) | (np.mod(x + 0.25 * (row % 2), 0.6) < 0.001)
        a[seams] = [68, 42, 24]
    return np.clip(a, 0, 255).astype("uint8")


if __name__ == "__main__":
    print("Net wires:", make_goal())
    for style in "ABCD":
        make_mat(style)
