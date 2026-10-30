"""Run a course PPO training entry with the Isaac Sim window open.

The training entries launch physics-only (no window) for speed. This wrapper
changes only the launch options so the Kit viewer shows the parallel
environments while they train; task, rewards, PPO settings and outputs are those
of the wrapped entry. Rendering the viewer makes each iteration slower.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import runpy
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--entry", type=Path, required=True)
# A raised camera looking across a corner of the environment grid, so many ducks
# are visible at once instead of only the first environment.
parser.add_argument("--view-eye", type=float, nargs=3, default=(-3.5, -3.5, 2.2))
parser.add_argument("--view-lookat", type=float, nargs=3, default=(1.5, 1.5, 0.0))
known, remaining = parser.parse_known_args()

import isaaclab.app as app  # noqa: E402
from kit_settings import kit_args  # noqa: E402

original_launch = app.launch_simulation


def course_floor_look(cfg):
    """Visual look of the training ground (appearance only; geometry, collision, physics
    material, task and rewards are unchanged):
    - slope training (generated terrain): the plywood of the Section 3.3 wedge, through a
      render-only texture coordinate primvar and an all-purpose visual binding;
    - flat-ground training (a plane): the plain white studio floor instead of the stock
      checker texture, through the same kind of visual binding.
    Terrain debug markers (environment origins) are turned off in both cases."""
    terrain = getattr(getattr(cfg, "scene", None), "terrain", None)
    if terrain is None:
        return
    import isaaclab.sim as sim_utils
    from isaaclab.managers import EventTermCfg

    from terrain_studio import STUDIO_FLOOR, apply_studio_floor, apply_wood_terrain

    terrain.debug_vis = False
    kind = getattr(terrain, "terrain_type", None)
    if kind == "generator":
        cfg.events.course_wood_terrain = EventTermCfg(func=apply_wood_terrain, mode="prestartup")
    elif kind == "plane":
        terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=STUDIO_FLOOR, roughness=0.35, metallic=0.0)
        cfg.events.course_white_floor = EventTermCfg(func=apply_studio_floor, mode="prestartup")


@contextmanager
def launch_with_window(cfg, options):
    options = dict(options or {}, visualizer=["kit"], require_kit=True, kit_args=kit_args())
    if cfg is not None:
        course_floor_look(cfg)
    with original_launch(cfg, options) as handle:
        if cfg is not None:
            from isaaclab_visualizers.kit import KitVisualizerCfg

            cfg.sim.visualizer_cfgs = [
                KitVisualizerCfg(eye=tuple(known.view_eye), lookat=tuple(known.view_lookat))
            ]
        yield handle


app.launch_simulation = launch_with_window
sys.argv = [str(known.entry), *remaining]
# The training entries import Isaac Lab tasks (and with them USD) at module level.
# With a window, Kit has to load its own USD libraries first, so start Kit here; the
# entry's own launch_simulation then reuses this Kit instead of starting another.
with original_launch(None, {"device": "cuda:0", "require_kit": True, "visualizer": ["kit"], "kit_args": kit_args()}):
    runpy.run_path(str(known.entry), run_name="__main__")
