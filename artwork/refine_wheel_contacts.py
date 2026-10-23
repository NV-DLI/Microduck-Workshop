"""Fit analytic wheel contacts while preserving official visuals and rigid bodies."""

from pathlib import Path
import argparse, hashlib, json, shutil
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--audit", type=Path, required=True)
a = p.parse_args()
if a.output.exists() or a.audit.exists():
    raise FileExistsError("Retain earlier assets and audits")
shutil.copy2(a.source, a.output)
stage = Usd.Stage.Open(str(a.output))
rows = []
for old in list(stage.Traverse()):
    if old.GetName() != "tire_1" or not old.HasAPI(UsdPhysics.CollisionAPI):
        continue
    points = UsdGeom.Mesh(old).GetPointsAttr().Get()
    # The converter retains STL coordinates beneath a compensating local Xform.
    matrix = UsdGeom.Xformable(old).GetLocalTransformation()
    transformed = [matrix.Transform(Gf.Vec3d(*point)) for point in points]
    radial = [(v[0] ** 2 + v[1] ** 2) ** 0.5 for v in transformed]
    zmin = min(v[2] for v in transformed)
    zmax = max(v[2] for v in transformed)
    if max(abs(zmin + 0.0038), abs(zmax - 0.0038), abs(max(radial) - 0.015)) > 1e-6:
        raise ValueError("Unexpected official wheel geometry")
    UsdPhysics.CollisionAPI(old).GetCollisionEnabledAttr().Set(False)
    cyl = UsdGeom.Cylinder.Define(
        stage, old.GetPath().GetParentPath().AppendChild("analytic_tire_contact")
    )
    cyl.CreateAxisAttr("Z")
    cyl.CreateHeightAttr(0.0076)
    cyl.CreateRadiusAttr(0.015)
    cyl.CreatePurposeAttr("guide")
    UsdPhysics.CollisionAPI.Apply(cyl.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(cyl.GetPrim()).CreateMassAttr(0.0)
    for attr in old.GetAttributes():
        if (
            attr.GetName().startswith(("mjc:", "newton:"))
            and attr.HasAuthoredValueOpinion()
        ):
            cyl.GetPrim().CreateAttribute(attr.GetName(), attr.GetTypeName()).Set(
                attr.Get()
            )
    for rel in old.GetRelationships():
        if rel.GetName().startswith("material:binding"):
            UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim())
            cyl.GetPrim().CreateRelationship(rel.GetName()).SetTargets(rel.GetTargets())
    rows.append(
        dict(
            body=str(old.GetPath().GetParentPath()),
            radius_m=0.015,
            width_m=0.0076,
            axis="Z",
            z_bounds_m=[zmin, zmax],
            max_vertex_radius_m=max(radial),
            original_collision=str(old.GetPath()),
        )
    )
if len(rows) != 4:
    raise ValueError("Expected four passive wheels")
stage.GetRootLayer().Save()
a.audit.write_text(
    json.dumps(
        dict(
            source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),
            output_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
            scope="Only contact representation changes; official visual meshes, body mass/inertia, joints and motors retained. Analytic circular tread replaces faceted collision mesh.",
            wheels=rows,
        ),
        indent=2,
    )
)
