"""Terminate fallen training episodes without constraining normal head motion."""


def body_ground_contact(env, threshold=0.5):
    """Only ground contact counts; robot self-contact and foot contact do not."""
    import torch

    data = env.scene.sensors["fall_contact"].data.normal_force_matrix_w_history
    if data is None:
        raise RuntimeError("Fall sensor has no ground-filtered contact matrix")
    forces = getattr(data, "torch", data)
    return torch.linalg.vector_norm(forces, dim=-1).flatten(1).amax(dim=1) > threshold


def apply_training_safety(cfg):
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.sensors import ContactSensorCfg
    from isaaclab.envs.mdp import root_height_below_minimum

    solver = cfg.sim.physics.solver_cfg
    solver.nconmax = max(solver.nconmax or 0, 64)
    solver.njmax = max(solver.njmax, 512)
    # These rigid bodies contain the trunk and head, never either foot. Filter
    # ground shapes rather than bodies because the terrain is a static collider.
    cfg.scene.fall_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base(/neck(/neck_pitch_1(/yaw_roll_motion(/jaw_soft)?)?)?)?",
        filter_shape_prim_expr=[cfg.scene.terrain.prim_path + "/.*"],
        history_length=cfg.decimation,
        update_period=cfg.sim.dt,
    )
    cfg.terminations.body_ground_contact = TerminationTermCfg(
        func=body_ground_contact, params={"threshold": 0.5}, time_out=False,
    )
    if cfg.scene.terrain.terrain_type == "plane":
        # Plane origin is known; never use an absolute-world-height threshold on
        # hills. Terrain episodes instead use the ground-filtered contact test.
        cfg.terminations.collapsed_body = TerminationTermCfg(
            func=root_height_below_minimum,
            params={"minimum_height": 0.065}, time_out=False,
        )
    return cfg
