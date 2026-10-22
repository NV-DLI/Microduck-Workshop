"""Read-only USD collision extraction. Imports USD only, never starts Isaac/GPU."""
from pathlib import Path
import argparse,hashlib,json
from pxr import Usd,UsdGeom,Gf
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--backlash',action='store_true');args=parser.parse_args()
asset='microduck_walk_backlash.usd' if args.backlash else 'microduck_walk.usd'
output_name='sole-collision-geometry-backlash.json' if args.backlash else 'sole-collision-geometry.json'
report_name='collision-extraction-backlash.json' if args.backlash else 'collision-extraction.json'
usd=P/'sources'/asset
s=Usd.Stage.Open(str(usd));cache=UsdGeom.XformCache();out={}
for p in Usd.PrimRange(s.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
 path=str(p.GetPath())
 if p.GetTypeName()!='Mesh' or p.GetAttribute('physics:collisionEnabled').Get() is not True:continue
 side=next((v for v in ['left','right'] if '/'+v+'_foot_collision/' in path),None)
 if side is None:continue
 ankle=p.GetParent().GetParent();R=cache.GetLocalToWorldTransform(p)*cache.GetLocalToWorldTransform(ankle).GetInverse()
 vertices=[list(R.Transform(Gf.Vec3d(*v))) for v in UsdGeom.Mesh(p).GetPointsAttr().Get()]
 site=s.GetPrimAtPath(str(ankle.GetPath())+'/'+side+'_foot');base=s.GetPrimAtPath('/microduck/Geometry/trunk_base')
 t=cache.GetLocalToWorldTransform(site)*cache.GetLocalToWorldTransform(base).GetInverse()
 if side+'_foot' in out:raise ValueError('Duplicate sole mesh')
 out[side+'_foot']={'mesh_path':path,'ankle_path':str(ankle.GetPath()),'approximation':p.GetAttribute('physics:approximation').Get(),'vertices_ankle_m':vertices,'vertex_count':len(vertices),'USD_zero_joint_site_body_m':list(t.Transform(Gf.Vec3d(0,0,0)))}
assert set(out)=={'left_foot','right_foot'}
(P/'sources'/output_name).write_text(json.dumps(out,separators=(',',':'))+'\n')
report={'source_url':'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Robots/PollenRobotics/MicroDuck/'+asset,'usd_sha256':hashlib.sha256(usd.read_bytes()).hexdigest(),'output_sha256':hashlib.sha256((P/'sources'/output_name).read_bytes()).hexdigest(),'vertex_counts':{k:v['vertex_count'] for k,v in out.items()},'physics_run':False,'gpu_used':False,'transform':'USD mesh local -> ankle rigid-body local, instance proxies traversed','scope':'Geometry from current official walk USD; geometry-only extraction, not physical validation'}
(P/'sources'/report_name).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
