"""Render the preserved CAD assembly with a darker studio and higher resolution.
Usage: blender -b artwork/microduck.blend --python artwork/render_studio.py -- OUTPUT.png
Presentation-only: does not execute physics or train a model.
"""

import bpy

import sys
from pathlib import Path

s = bpy.context.scene
s.frame_set(1)
s.render.resolution_x = 2560
s.render.resolution_y = 1920
s.render.resolution_percentage = 100
s.render.engine = "CYCLES"
s.cycles.device = "CPU"
s.cycles.samples = 128
s.cycles.use_adaptive_sampling = True
s.cycles.adaptive_threshold = 0.025
s.cycles.use_denoising = False
s.render.threads_mode = "FIXED"
s.render.threads = 12
s.world.node_tree.nodes["Background"].inputs[0].default_value = (0.28, 0.34, 0.42, 1)
s.world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
for m in bpy.data.materials:
    if m.name.startswith("Studio matte"):
        m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
            0.055,
            0.075,
            0.10,
            1,
        )
        m.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.7
s.view_settings.look = "AgX - Medium High Contrast"
s.view_settings.exposure = 0.1
s.render.image_settings.file_format = "PNG"
s.render.filepath = sys.argv[sys.argv.index("--") + 1]
bpy.ops.render.render(write_still=True)
