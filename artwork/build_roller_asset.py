"""Preserve the official passive-wheel mechanics; split motor/play DOFs for BAM."""

from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
import shutil, json, hashlib
import argparse

parser = argparse.ArgumentParser(
    description="Add native BAM backlash bodies to the converted official roller asset"
)
parser.add_argument(
    "--converted",
    type=Path,
    required=True,
    help="Output of mujoco-usd-converter for robot_allcollisions_rollers_backlash.xml",
)
parser.add_argument("--audit", type=Path, required=True)
a = parser.parse_args()
R = Path(__file__).resolve().parents[1]
out = R / "lab/assets/rollers"
out.mkdir(exist_ok=True)
p = out / "microduck_rollers.usdc"
shutil.copy2(a.converted, p)
s = Usd.Stage.Open(str(p))
old = Usd.Stage.Open(str(R / "lab/assets/microduck_contact.usda"))
colors = {
    prim.GetName(): prim.GetAttribute("inputs:diffuseColor").Get()
    for prim in old.TraverseAll()
    if prim.GetAttribute("inputs:diffuseColor").HasAuthoredValue()
}
bodies = [p for p in s.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
cache = UsdGeom.XformCache()
world = {p.GetPath(): cache.GetLocalToWorldTransform(p) for p in bodies}
for p in bodies:
    xf = UsdGeom.Xformable(p)
    xf.ClearXformOpOrder()
    op = xf.AddTransformOp()
    op.Set(world[p.GetPath()])
    xf.SetXformOpOrder([op], resetXformStack=True)
# Split the coincident servo/play pair exactly as the upstream walking USD.
for play in [
    UsdPhysics.RevoluteJoint(p)
    for p in s.Traverse()
    if p.IsA(UsdPhysics.RevoluteJoint) and p.GetName().endswith("_backlash")
]:
    servo_name = play.GetPrim().GetName()[len("passive_") : -len("_backlash")]
    servo = UsdPhysics.RevoluteJoint(
        s.GetPrimAtPath(play.GetPath().GetParentPath().AppendChild(servo_name))
    )
    child = play.GetBody1Rel().GetTargets()[0]
    parent = play.GetBody0Rel().GetTargets()[0]
    dummy_path = parent.AppendChild(play.GetPrim().GetName() + "_dummy")
    dummy = UsdGeom.Xform.Define(s, dummy_path)
    op = dummy.AddTransformOp()
    op.Set(world[child])
    dummy.SetXformOpOrder([op], resetXformStack=True)
    UsdPhysics.RigidBodyAPI.Apply(dummy.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(dummy.GetPrim())
    mass.CreateMassAttr(1e-6)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-9))
    servo.GetBody1Rel().SetTargets([dummy_path])
    servo.GetLocalPos1Attr().Set(Gf.Vec3f(0))
    servo.GetLocalRot1Attr().Set(Gf.Quatf(1))
    play.GetBody0Rel().SetTargets([dummy_path])
    play.GetLocalPos0Attr().Set(Gf.Vec3f(0))
    play.GetLocalRot0Attr().Set(Gf.Quatf(1))
    play.GetLocalPos1Attr().Set(Gf.Vec3f(0))
    play.GetLocalRot1Attr().Set(Gf.Quatf(1))
for prim in s.TraverseAll():
    if prim.GetName() in colors:
        prim.CreateAttribute("inputs:diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            colors[prim.GetName()]
        )
        prim.CreateAttribute("inputs:roughness", Sdf.ValueTypeNames.Float).Set(0.5)
s.GetRootLayer().Save()
report = {
    "bodies": len(bodies) + 14,
    "extra_dummy_mass_kg": 14e-6,
    "revolute_joints": [
        str(p.GetPath()) for p in s.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)
    ],
    "source": "community/isaaclab-newton/assets/mjcf/robot_allcollisions_rollers_backlash.xml",
    "converter": "mujoco-usd-converter0.6.1",
    "sha256": (
        hashlib.sha256(p.read_bytes()).hexdigest()
        if isinstance(p, Path)
        else hashlib.sha256((out / "microduck_rollers.usdc").read_bytes()).hexdigest()
    ),
}
a.audit.write_text(json.dumps(report, indent=2))
print(report)
