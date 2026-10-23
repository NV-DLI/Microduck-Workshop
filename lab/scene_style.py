"""Shared physical assets and lighting for the native Isaac/RTX teaching labs."""

from pathlib import Path
import math

ASSETS = Path(__file__).parent / "assets"


def apply_surface_style(env, env_ids):
    """Replace the stock checker texture on the visible room floor."""
    from pxr import Usd, UsdGeom, UsdShade, Sdf

    stage = env.sim.stage
    material = UsdShade.Material.Define(stage, "/World/CourseFloorMaterial")
    shader = UsdShade.Shader.Define(stage, "/World/CourseFloorMaterial/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        (0.34, 0.42, 0.48)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    ground = stage.GetPrimAtPath(env.cfg.scene.terrain.prim_path)
    for prim in Usd.PrimRange(ground):
        if prim.IsA(UsdGeom.Mesh):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )


def configure_lighting(cfg):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.managers import EventTermCfg

    cfg.events.course_surface_style = EventTermCfg(
        func=apply_surface_style, mode="prestartup"
    )

    cfg.scene.sky_light.spawn.intensity = 350.0
    cfg.scene.sky_light.spawn.color = (0.84, 0.90, 1.0)
    cfg.scene.key_light = AssetBaseCfg(
        prim_path="/World/CourseKey",
        spawn=sim_utils.DistantLightCfg(
            intensity=700.0, angle=8.0, color=(1.0, 0.94, 0.85)
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8))
        ),
    )
    cfg.scene.robot.spawn.usd_path = str(ASSETS / "microduck_contact.usda")
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.34, 0.42, 0.48), roughness=0.8, metallic=0.0
    )


def configure_football(cfg, *, style="A", goal_x=0.68):
    """A rigid, flat mat and secured steel goal; no fabric/foam deformation model.

    The existing Newton plane is the mat's contact surface at z=0. The mat's
    thickness is visual only and its underside is below that plane. Thus this
    version models a flush installation, not a raised obstacle at the edge.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    if style not in ("A", "B", "C", "D"):
        raise ValueError("Mat style must be A, B, C, or D")
    configure_lighting(cfg)
    cfg.sim.physics.solver_cfg.nconmax = 300
    cfg.sim.physics.solver_cfg.njmax = 1500
    for name in ("goal_left", "goal_right", "goal_bar"):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    cfg.scene.pitch = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PitchVisual",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / f"pitch_{style}.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.1, 0.0, 0.0)),
    )
    cfg.scene.goal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Goal",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "steel_goal.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(goal_x, 0.0, 0.0)),
    )
    return {
        "profile": "indoor-steel-goal-v1",
        "mat_style": style,
        "mat_dimensions_m": [1.6, 1.0, 0.004],
        "mat_center_xy_m": [0.1, 0.0],
        "mat_edge": "flush contact plane",
        "goal_outer_width_depth_height_m": [0.4, 0.2, 0.25],
        "goal_frame_diameter_m": 0.006,
        "goal_wire_diameter_m": 0.0008,
        "goal_wire_spacing_m": 0.012,
        "goal_plane_x_m": goal_x,
        "goal_opening_width_m": 0.388,
        "goal_opening_height_m": 0.244,
        "goal_fixed": True,
        "unmeasured": [
            "mat friction",
            "mat compression",
            "goal wire/frame diameters",
            "goal mass",
        ],
        "rendering": "Isaac RTX, shared daylight, official CAD",
    }
