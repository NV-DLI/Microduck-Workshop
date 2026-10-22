"""Evidence for the new native-BAM Newton runtime; no global runtime patching."""
from importlib.metadata import version,PackageNotFoundError


def backend_evidence(sim):
    manager=sim.backend_manager
    name=type(manager).__module__+'.'+type(manager).__qualname__
    if getattr(manager,'backend_name',None)!='newton':raise RuntimeError(f'Newton required; got {name}')
    solver=manager._solver
    versions={}
    for key,package in [('newton_version','newton'),('mujoco_version','mujoco'),('warp_version','warp-lang'),('mujoco_warp_version','mujoco-warp'),('isaacsim_version','isaacsim')]:
        try:versions[key]=version(package)
        except PackageNotFoundError:versions[key]=None
    return {**versions,'physics_manager':name,'solver':type(solver).__module__+'.'+type(solver).__qualname__,
        'device':str(manager.get_device()),'physics_dt':sim.get_physics_dt(),
        'gravity_m_s2':list(manager._gravity_vector),
        'use_mujoco_cpu':getattr(sim.cfg.physics.solver_cfg,'use_mujoco_cpu',None),
        'cuda_graph_requested':sim.cfg.physics.use_cuda_graph,'cuda_graph_active':manager._graph is not None,
        'cuda_graph_pending':manager._graph_capture_pending,'backend_handles_decimation':manager.handles_decimation(),
        'native_newton_actuators':sim.cfg.use_newton_actuators,
        'isaaclab_commit':'5088b326c7fc81caa7899395b0176ad08b89e9e1'}
