"""Package rendered mechanical illustration frames with a stable colour palette."""

from pathlib import Path
import argparse
import json
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("frames", type=Path)
ap.add_argument("output", type=Path)
a = ap.parse_args()
paths = sorted(a.frames.glob("frame-*.png"))
assert len(paths) == 51
frames = [Image.open(p).convert("RGB") for p in paths]
assert all(f.size == (720, 540) for f in frames)
sheet = Image.new("RGB", (9 * 120, 6 * 90), (230, 230, 230))
for i, f in enumerate(frames):
    sheet.paste(f.resize((120, 90)), ((i % 9) * 120, (i // 9) * 90))
palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
a.output.parent.mkdir(parents=True, exist_ok=True)
quantized[0].save(
    a.output,
    save_all=True,
    append_images=quantized[1:],
    duration=100,
    loop=0,
    optimize=False,
    disposal=2,
    comment=b"Mechanical anatomy illustration. Authored Blender assembly motion, not a physics or policy rollout.",
)
with Image.open(a.output) as im:
    duration = 0
    for i in range(im.n_frames):
        im.seek(i)
        duration += im.info.get("duration", 0)
    assert im.n_frames == 51 and duration == 5100
    print(
        json.dumps(
            {
                "path": str(a.output),
                "frames": im.n_frames,
                "duration_ms": duration,
                "size": list(im.size),
                "bytes": a.output.stat().st_size,
            }
        )
    )
