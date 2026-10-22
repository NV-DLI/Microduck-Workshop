"""Render a five-second mechanical assembly illustration using CPU Cycles.
blender --background artwork/microduck.blend --python artwork/render_animation.py -- --output artwork/animation-frames
This authored movement is an anatomy illustration, not a physics rollout.
"""
import bpy,sys,argparse,math,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--samples',type=int,default=24);args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);D=args.output;D.mkdir(parents=True,exist_ok=True)
s=bpy.context.scene;s.render.engine='CYCLES';s.cycles.device='CPU';s.cycles.samples=args.samples;s.cycles.use_denoising=False;s.cycles.max_bounces=6;s.render.threads_mode='FIXED';s.render.threads=12;s.render.resolution_x=720;s.render.resolution_y=540;s.render.resolution_percentage=100;s.render.image_settings.file_format='PNG';s.render.image_settings.color_mode='RGB';s.render.fps=10
for i in range(51):
 frame=61-i*1.2;integer=math.floor(frame);s.frame_set(integer,subframe=frame-integer);s.render.filepath=str(D/f'frame-{i:03d}.png');bpy.ops.render.render(write_still=True)
(D/'animation.json').write_text(json.dumps({'kind':'mechanical-assembly-illustration','physics_rollout':False,'frames':51,'fps':10,'duration_seconds':5.1,'resolution':[720,540],'renderer':'Blender Cycles CPU','samples':args.samples,'direction':'exploded to assembled'},indent=2)+'\n')
