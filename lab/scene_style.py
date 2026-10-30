"""Shared physical assets and lighting for the native Isaac/RTX teaching labs."""

from pathlib import Path
import math

ASSETS = Path(__file__).parent / "assets"

LIGHTING_PROFILES = {
    "daylight": {
        "sky_intensity": 350.0,
        "sky_color": (0.84, 0.90, 1.0),
        "key_intensity": 700.0,
        "key_color": (1.0, 0.94, 0.85),
    },
    "warm": {
        "sky_intensity": 300.0,
        "sky_color": (1.0, 0.86, 0.70),
        "key_intensity": 650.0,
        "key_color": (1.0, 0.80, 0.60),
    },
    "cool-dim": {
        "sky_intensity": 220.0,
        "sky_color": (0.72, 0.84, 1.0),
        "key_intensity": 450.0,
        "key_color": (0.86, 0.92, 1.0),
    },
}


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


def configure_lighting(cfg, profile="daylight"):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg
    from isaaclab.managers import EventTermCfg

    cfg.events.course_surface_style = EventTermCfg(
        func=apply_surface_style, mode="prestartup"
    )

    lighting = LIGHTING_PROFILES[profile]
    cfg.scene.sky_light.spawn.intensity = lighting["sky_intensity"]
    cfg.scene.sky_light.spawn.color = lighting["sky_color"]
    cfg.scene.key_light = AssetBaseCfg(
        prim_path="/World/CourseKey",
        spawn=sim_utils.DistantLightCfg(
            intensity=lighting["key_intensity"],
            angle=8.0,
            color=lighting["key_color"],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            rot=(0.0, math.sin(math.pi / 8), 0.0, math.cos(math.pi / 8))
        ),
    )
    cfg.scene.robot.spawn.usd_path = str(ASSETS / "microduck_contact.usda")
    cfg.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.34, 0.42, 0.48), roughness=0.8, metallic=0.0
    )


def configure_football(cfg, *, style="A", goal_x=0.68, lighting="daylight"):
    """A rigid, flat mat and secured steel goal; no fabric/foam deformation model.

    The existing Newton plane is the mat's contact surface at z=0. The mat's
    thickness is visual only and its underside is below that plane. Thus this
    version models a flush installation, not a raised obstacle at the edge.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    if style not in ("A", "B", "C", "D"):
        raise ValueError("Mat style must be A, B, C, or D")
    configure_lighting(cfg, lighting)
    # Official browser sphere: diameter 10 cm, mass 30 g.
    cfg.scene.ball.spawn = sim_utils.UsdFileCfg(
        usd_path=str(ASSETS / "official_ball.usda"),
    )
    cfg.scene.ball.init_state.pos = (*cfg.scene.ball.init_state.pos[:2], 0.05)
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
    profile = {
        "profile": "indoor-trapezoid-official-ball-v2",
        "mat_style": style,
        "surface_name": {
            "A": "striped football",
            "B": "plain football",
            "C": "marble",
            "D": "walnut planks",
        }[style],
        "surface_physics": "rigid flush plane; shared uncalibrated contact parameters",
        "ball_visual": "official Microduck simulator, original 10 cm diameter",
        "ball_diameter_m": 0.10,
        "ball_mass_kg": 0.03,
        "ball_contact_source": "official browser MuJoCo geom; USD mapping from mujoco-usd-converter0.6.1",
        "ball_friction_slide_spin_roll": [0.4, 0.01, 0.003],
        "ball_condim": 6,
        "ball_solref": [0.03, 0.4],
        "cross_engine_contact_equivalence_measured": False,
        "mat_dimensions_m": [1.6, 1.0, 0.004],
        "mat_center_xy_m": [0.1, 0.0],
        "mat_edge": "flush contact plane",
        "goal_outer_width_depth_height_m": [0.4, 0.2, 0.25],
        "goal_top_depth_m": 0.066,
        "goal_side_profile": "trapezoid, sloping back; inferred from product photograph",
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
    if lighting != "daylight":
        profile["lighting"] = lighting
        profile["lighting_parameters"] = LIGHTING_PROFILES[lighting].copy()
        profile["rendering"] = f"Isaac RTX, shared {lighting} lighting, official CAD"
    return profile
