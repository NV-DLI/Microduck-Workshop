"""Rebuild the editable component assembly and web assets from pinned CAD.
Blender 4.0+: blender --background --python artwork/build_microduck.py -- --output artwork/rebuilt
All computations and renders use the CPU; no simulator or learned policy is involved.
"""
import bpy,sys,json,argparse,importlib.util,hashlib
from pathlib import Path
from collections import defaultdict
from mathutils import Vector
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=HERE/'source');ap.add_argument('--output',type=Path,required=True);ap.add_argument('--samples',type=int,default=192);ap.add_argument('--no-render',action='store_true');args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);D=args.output.resolve();D.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('cad_assembly',HERE/'cad_assembly.py');cad=importlib.util.module_from_spec(spec);spec.loader.exec_module(cad);cad.SRC=str(args.source.resolve());cad.ROBOT_XML=str(args.source.resolve()/'robot_allcollisions.xml');cad.SCENE_XML=str(args.source.resolve()/'scene_allcollisions.xml');cad.OUT=str(D);cad.blender_main(['--no-render'])
raw=json.loads((D/'components.json').read_text());objs={o['id']:o for o in bpy.data.objects if o.type=='MESH' and 'id' in o};scene=bpy.context.scene;scene.frame_set(1)
counts=defaultdict(int);parts=[];objects=[];colls={}
notes={
'pcb__raspberry_pi_zero_2_w':('原始 CAD 中的 Raspberry Pi Zero 2 W 计算板外形，不代表运行课程的工作站。','Raspberry Pi Zero 2 W compute-board envelope in the source CAD, not the workstation running this course.'),
'elec_rpi_robot_hat_pcb':('原始 CAD 的机器人扩展电路板，用于理解电子组件的安装空间。','Robot expansion circuit board from the source CAD, showing the electronics mounting envelope.'),
'np_f970':('原始 CAD 的 NP-F970 电池组件；其位置会影响整机重心。','NP-F970 battery component in the source CAD; its position affects the center of mass.'),
'speaker':('原始 CAD 用简化外形表示扬声器安装空间。','The source CAD uses a simplified envelope for the speaker mounting space.')}
for rp in raw['parts']:
 body=rp['source_body'];mesh=rp['source_mesh'];ident=f'{body}--{mesh}--{counts[body]:02d}';counts[body]+=1
 group={'left_leg':'left-leg','right_leg':'right-leg','neck':'head'}.get(rp['group'],rp['group']);ob=objs[rp['id']];ob.animation_data_clear();ob.name=ident
 p={k:rp[k] for k in ('label_zh','label_en','description_zh','description_en','explode','vertices','triangles')};p.update(id=ident,group=group,source_body=body,source_mesh=mesh+'.stl')
 if rp.get('drives_joint'):p['drives_joint']=rp['drives_joint']
 if mesh in notes:p['description_zh'],p['description_en']=notes[mesh]
 for k in list(ob.keys()):del ob[k]
 for k,v in p.items():ob[k]=v
 if group not in colls:
  colls[group]=bpy.data.collections.new(group);scene.collection.children.link(colls[group])
 for c in list(ob.users_collection):c.objects.unlink(ob)
 colls[group].objects.link(ob);parts.append(p);objects.append(ob)
# Keep exactly the source component geometry in the web model.
for ob in list(bpy.data.objects):
 if ob not in objects:bpy.data.objects.remove(ob,do_unlink=True)
bpy.ops.object.select_all(action='DESELECT')
for ob in objects:ob.select_set(True)
bpy.context.view_layer.objects.active=objects[0]
bpy.ops.export_scene.gltf(filepath=str(D/'microduck.glb'),export_format='GLB',use_selection=True,export_extras=True,export_yup=True,export_animations=False,export_apply=True,export_copyright='Pollen Robotics; geometry adaptations CC BY-NC-SA 4.0')
def y_up(v):return [v[0],v[2],-v[1]]
joints=[{'id':j['name'],'label_zh':j['label_zh'],'label_en':j['label_en'],'axis':y_up(j['axis_world_blender']),'pivot':y_up(j['anchor_world_blender']),'stand_radians':j['stand_rad'],'range_radians':j['range_rad'],'policy_action_index':j['policy_action_index']} for j in raw['joints']]
bounds=raw['bounds']['assembled_blender'];c=raw['counts']
meta={'schema':'microduck.components.v1','generator':'Blender CAD assembly','coordinate_system':'glTF Y-up; meters; source robot +X forward','source':{'repository':'https://github.com/pollen-robotics/microduck_rl','revision':'2b581c641406a48346e696212930ea881c222c52','license':'CC BY-NC-SA 4.0','license_url':'https://creativecommons.org/licenses/by-nc-sa/4.0/','license_declaration':'https://github.com/pollen-robotics/microduck_rl/blob/develop/LICENSE-MODELS.md','xml_sha256':hashlib.sha256((args.source/'robot_allcollisions.xml').read_bytes()).hexdigest(),'pose':'STAND','variant_note_zh':'原始 CAD 展示 Raspberry Pi Zero 2 W、Robot HAT 和 NP-F970 外形；不代表新版本电子器件的工程图。','variant_note_en':'The source CAD depicts Raspberry Pi Zero 2 W, Robot HAT and NP-F970 envelopes; it is not an engineering drawing of newer electronics.'},'stats':{'components':len(parts),'unique_meshes':c['unique_meshes'],'physical_servos':c['servos'],'simulated_joints':len(joints),'policy_actions':14,'beak_joint_in_source':False,'triangles':c['triangles_total_instanced'],'vertices':c['vertices_total_instanced'],'dimensions_m_xyz':[round(bounds['max'][i]-bounds['min'][i],6) for i in range(3)]},'parts':parts,'joints':joints}
assert len(parts)==70 and len(joints)==14 and sum(p.get('drives_joint')=='beak' for p in parts)==1
(D/'components.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
# Reproducible neutral studio. Lighting is presentation, not physical simulation.
scene.world=bpy.data.worlds.new('Studio world');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.72,.76,.80,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.35
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=args.samples;scene.cycles.use_denoising=False;scene.render.threads_mode='FIXED';scene.render.threads=12;scene.render.resolution_x=1600;scene.render.resolution_y=1200;scene.render.resolution_percentage=100;scene.render.film_transparent=False;scene.view_settings.view_transform='AgX';scene.view_settings.look='AgX - Medium High Contrast';scene.view_settings.exposure=0;scene.view_settings.gamma=1
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.001));floor=bpy.context.object;floor.name='Studio floor';m=bpy.data.materials.new('Studio matte');m.use_nodes=True;m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value=(.82,.84,.86,1);m.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value=.8;floor.data.materials.append(m)
def light(name,pos,power,size):
 d=bpy.data.lights.new(name,'AREA');d.energy=power;d.shape='DISK';d.size=size;o=bpy.data.objects.new(name,d);scene.collection.objects.link(o);o.location=pos;o.rotation_euler=(Vector((0,0,.14))-o.location).to_track_quat('-Z','Y').to_euler()
light('Key softbox',(.35,-.45,.65),7,.5);light('Fill softbox',(.2,.5,.3),3,.4);light('Rim softbox',(-.45,.15,.6),10,.35)
cd=bpy.data.cameras.new('Assembly camera');cam=bpy.data.objects.new('Assembly camera',cd);scene.collection.objects.link(cam);scene.camera=cam;cam.location=(.5,-.65,.35);cam.rotation_euler=(Vector((0,0,.13))-cam.location).to_track_quat('-Z','Y').to_euler();cd.type='ORTHO';cd.ortho_scale=.43;cd.clip_start=.001
for ob in objects:
 ob.keyframe_insert(data_path='location',frame=1);e=ob['explode'];ob.location+=Vector((e[0],-e[2],e[1]));ob.keyframe_insert(data_path='location',frame=61)
cd.ortho_scale=.43;cd.keyframe_insert(data_path='ortho_scale',frame=1);cd.ortho_scale=.9;cd.keyframe_insert(data_path='ortho_scale',frame=61);scene.frame_start=1;scene.frame_end=61;scene.render.fps=12;scene.frame_set(1)
scene['geometry_author']='Pollen Robotics';scene['geometry_license']='CC BY-NC-SA 4.0';scene['geometry_license_url']='https://creativecommons.org/licenses/by-nc-sa/4.0/';bpy.ops.wm.save_as_mainfile(filepath=str(D/'microduck.blend'),compress=True)
if not args.no_render:
 scene.render.filepath=str(D/'assembled.png');bpy.ops.render.render(write_still=True);scene.frame_set(61);scene.render.filepath=str(D/'exploded.png');bpy.ops.render.render(write_still=True);scene.frame_set(1);scene['geometry_author']='Pollen Robotics';scene['geometry_license']='CC BY-NC-SA 4.0';scene['geometry_license_url']='https://creativecommons.org/licenses/by-nc-sa/4.0/';bpy.ops.wm.save_as_mainfile(filepath=str(D/'microduck.blend'),compress=True)
print('FINAL_ASSEMBLY_COMPLETE',json.dumps(meta['stats']))
