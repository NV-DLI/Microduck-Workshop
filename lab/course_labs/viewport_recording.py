"""Record the live Kit viewport without advancing physics during capture."""

import subprocess


def start_recording(path, size, fps):
    width, height = size
    return subprocess.Popen(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
         "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
         "-threads", "2", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(path)],
        stdin=subprocess.PIPE,
    )


def capture(env, process, output, index, size):
    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
    from PIL import Image
    import torch

    robot = env.scene["robot"]
    before = robot.data.root_pos_w.clone()
    path = output / f"capture-{index:06d}.png"
    pending = capture_viewport_to_file(get_active_viewport(), str(path))
    for _ in range(40):
        env.sim.render()
        if path.exists() and path.stat().st_size:
            break
    else:
        raise RuntimeError("Native viewport capture timed out")
    if not torch.equal(before, robot.data.root_pos_w):
        raise RuntimeError("Viewport capture advanced physics")
    with Image.open(path) as image:
        if image.size != size:
            raise RuntimeError(f"Viewport size {image.size}, expected {size}")
        frame = image.convert("RGB").tobytes()
    process.stdin.write(frame)
    path.unlink()
    return frame
