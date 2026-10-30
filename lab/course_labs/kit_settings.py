"""Kit settings shared by every course entry that opens an Isaac Sim window.

Isaac Lab's windowed experience reads body transforms from Fabric in the RTX
render delegate (/rtx/hydra/readTransformsFromFabricInRenderDelegate = true),
which is how Newton's simulated poses reach the viewport, and RTX reports that this
and geometry streaming are enabled together. Verified so far in this Isaac Sim
build: the documented streaming switches (/persistent/UJITSO/geometry, session
/UJITSO/geometry) alone do not change the streaming state; /UJITSO/enabled=false
removes the message but makes PhysX cooking and Hydra report missing UJITSO
services (errors), so it is not used; an empty /UJITSO/geometryTypes is accepted but
leaves streaming active. The renderer baseline that was reviewed is kept (no UJITSO
setting); Fabric transform reading is never changed.

MICRODUCK_USD_DIAGNOSTICS=1 is for diagnosis only. It unmutes USD diagnostics,
prints the effective render settings, brackets the Kit viewer's set-up calls with
markers, lists Newton body labels that are not valid USD paths and scans the live
stage for empty path values, so that a USD warning in the log can be attributed.
MICRODUCK_RENDER_PROBE=<name> (with diagnostics) adds one bounded probe set.
"""
import json
import os

# Viewport grid off. Isaac Lab's experience defaults it off, but Kit keeps a per-viewport
# persistent value (guide/grid/visible) in the user config that overrides the default once
# set; these command-line values win at every start. Only the editor overlay is affected,
# not scene geometry, pitch markings or any camera image used by a policy.
GUI_KIT_SETTINGS = (
    "--/persistent/app/viewport/Viewport/Viewport0/guide/grid/visible=false",
    "--/app/viewport/grid/enabled=false",
)
# Bounded render probes (diagnosis only): MICRODUCK_RENDER_PROBE=<name>.
RENDER_PROBES = {
    "ujitso-off": ("--/UJITSO/enabled=false", "--/UJITSO/geometry=false"),
    "geometry-types-empty": ("--/UJITSO/geometry=false", "--/UJITSO/geometryTypes="),
    "ujitso-off-all": ("--/UJITSO/enabled=false", "--/UJITSO/geometry=false", "--/UJITSO/textures=false",
                       "--/UJITSO/materials=false", "--/UJITSO/materialUsdToHlsl=false"),
}
WATCHED_SETTINGS = (
    "/persistent/app/viewport/Viewport/Viewport0/guide/grid/visible",
    "/app/viewport/grid/enabled",
    "/UJITSO/enabled",
    "/UJITSO/geometry",
    "/UJITSO/geometryTypes",
    "/UJITSO/textures",
    "/UJITSO/materials",
    "/persistent/UJITSO/geometry",
    "/rtx/hydra/readTransformsFromFabricInRenderDelegate",
    "/rtx-transient/hydra/geometrystreaming/active",
    "/persistent/app/usd/muteUsdDiagnostics",
)


def diagnostics():
    return os.environ.get("MICRODUCK_USD_DIAGNOSTICS") == "1"


def kit_args(existing=""):
    """`kit_args` value for launch_simulation options of a windowed run."""
    args = existing.split() if existing else []
    args += [a for a in GUI_KIT_SETTINGS if a not in args]
    if diagnostics():
        args.append("--/persistent/app/usd/muteUsdDiagnostics=false")
        probe = os.environ.get("MICRODUCK_RENDER_PROBE")
        if probe:
            if probe not in RENDER_PROBES:
                raise SystemExit(f"Unknown MICRODUCK_RENDER_PROBE {probe!r}; choose from {sorted(RENDER_PROBES)}")
            args += [a for a in RENDER_PROBES[probe] if a not in args]
    return " ".join(args)


def log_settings(label):
    """Diagnosis: print the effective values of the watched Kit settings."""
    if not diagnostics():
        return
    import carb.settings

    settings = carb.settings.get_settings()
    values = {key: settings.get(key) for key in WATCHED_SETTINGS}
    print("COURSE_KIT_SETTINGS " + json.dumps(dict(at=label, **values)), flush=True)


def trace_kit_viewer(calls_per_method=3):
    """Diagnosis: print enter/exit markers around the Kit viewer's own methods."""
    if not diagnostics():
        return
    import functools
    import inspect
    from isaaclab_visualizers.kit import kit_visualizer

    cls = kit_visualizer.KitVisualizer
    for name, method in list(vars(cls).items()):
        if name.startswith("__") or not inspect.isfunction(method):
            continue

        def wrap(method=method, name=name):
            seen = [0]

            @functools.wraps(method)
            def traced(*args, **kwargs):
                seen[0] += 1
                if seen[0] > calls_per_method:
                    return method(*args, **kwargs)
                print(f"COURSE_USD_TRACE enter KitVisualizer.{name}", flush=True)
                try:
                    return method(*args, **kwargs)
                finally:
                    print(f"COURSE_USD_TRACE exit KitVisualizer.{name}", flush=True)

            return traced

        setattr(cls, name, wrap())


def trace_fabric_paths():
    """Diagnosis: around Fabric's rigid-body binding in the Kit viewer, list Newton body
    labels that are not valid USD paths, scan the live stage for empty path values and
    print a marker between each step of the binding (same steps and order as
    isaaclab_physx FabricBackend.bind_transforms), so an 'Ill-formed SdfPath <>'
    warning can be placed inside one step."""
    if not diagnostics():
        return
    from isaaclab_physx.renderers import fabric

    original_init = fabric.FabricBackend.__init__

    def init(self, cfg):
        print("COURSE_FABRIC_STEP enter FabricBackend.__init__", flush=True)
        original_init(self, cfg)
        print("COURSE_FABRIC_STEP exit FabricBackend.__init__", flush=True)

    def step(name):
        print("COURSE_FABRIC_STEP " + name, flush=True)

    def bind_transforms(self, provider):
        if self.transforms is not None:
            return
        print("COURSE_EMPTY_PATH_SCAN " + json.dumps(_scan_stage_for_empty_paths()), flush=True)
        paths = list(provider.backend.transform_paths)
        bad = [(i, p) for i, p in enumerate(paths) if not p or not str(p).startswith("/")]
        print("COURSE_FABRIC_PATHS " + json.dumps(dict(
            count=len(paths), invalid=[dict(index=i, label=p,
                                            previous=paths[i - 1] if i else None,
                                            next=paths[i + 1] if i + 1 < len(paths) else None)
                                       for i, p in bad[:10]])), flush=True)
        if fabric.SceneDataFormat.FabricMatrix44 in provider.backend.native_transform_formats:
            self.transforms = fabric.SceneDataFormat.FabricMatrix44()
            step("native Fabric transforms, nothing to bind")
            return
        usdrt = fabric.usdrt
        stage = self.stage
        step("before SynchronizeToFabric")
        stage.SynchronizeToFabric()
        step("after SynchronizeToFabric")
        for path in provider.backend.transform_paths:
            prim = stage.GetPrimAtPath(path)
            if prim:
                usdrt.Rt.Xformable(prim).SetWorldXformFromUsd()
            else:
                prim = stage.DefinePrim(path, "Xform")
                usdrt.Rt.Xformable(prim).CreateFabricHierarchyWorldMatrixAttr()
            prim.AddAppliedSchema("PhysicsRigidBodyAPI")
        step("after body prim loop")
        self.hierarchy.update_world_xforms()
        step("after update_world_xforms")
        for index, path in enumerate(provider.backend.transform_paths):
            prim = stage.GetPrimAtPath(path)
            prim.CreateAttribute("isaaclab:transformIndex", usdrt.Sdf.ValueTypeNames.Int, custom=True).Set(index)
            self.hierarchy.set_reset_xform_stack(prim.GetPath().fabricPath, True)
        step("after transform index loop")
        attrs = [
            (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.Int, "isaaclab:transformIndex", usdrt.Usd.Access.Read),
            (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:localMatrix", usdrt.Usd.Access.Read),
        ]
        self._selection = stage.SelectPrims(require_attrs=attrs, device=self.device)
        self._write_selection = stage.SelectPrims(
            require_attrs=[*attrs[:-1], (*attrs[-1][:2], usdrt.Usd.Access.ReadWrite)], device=self.device
        )
        step("after SelectPrims")
        self._scales = fabric.wp.empty(provider.transform_count, dtype=fabric.wp.vec3f, device=self.device)
        self.transforms = fabric.SceneDataFormat.FabricMatrix44()
        step("bind_transforms done")

    fabric.FabricBackend.__init__ = init
    fabric.FabricBackend.bind_transforms = bind_transforms


def _scan_stage_for_empty_paths(limit=15):
    """Relationships with empty targets and empty path-like string attributes on the live stage."""
    import omni.usd
    from pxr import Sdf

    stage = omni.usd.get_context().get_stage()
    found, prims = [], 0
    for prim in stage.TraverseAll():
        prims += 1
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if any(t.isEmpty for t in targets):
                found.append(dict(kind="relationship", path=str(rel.GetPath()), targets=[str(t) for t in targets]))
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in (Sdf.ValueTypeNames.String, Sdf.ValueTypeNames.Token):
                continue
            name = attr.GetName().lower()
            if "path" in name or "prim" in name or "target" in name or "camera" in name:
                value = attr.Get()
                if value is not None and str(value).strip() == "":
                    found.append(dict(kind="attribute", path=str(attr.GetPath())))
        if len(found) >= limit:
            break
    return dict(prims_scanned=prims, findings=found)
