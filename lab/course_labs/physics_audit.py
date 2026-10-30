"""Physics-contract snapshot of a built Newton model (diagnosis only).

MICRODUCK_PHYSICS_AUDIT=<file.json> makes a course check write, after the scene is
built, every per-shape contact parameter of the finalized Newton model, the body
labels, and a geometry hash per shape (mesh vertices and indices, so a changed ramp
mesh is detected; shape_scale/transform alone would not show it). Comparing two
snapshots with compare_physics_audit.py shows whether a change touched the physics
or only the picture. Nothing is modified.
"""
import hashlib
import json
import os

# Per-shape fields that decide contact physics (all must be present to compare).
PHYSICS_FIELDS = (
    "shape_body", "shape_type", "shape_flags", "shape_collision_group", "shape_world",
    "shape_scale", "shape_transform", "shape_is_solid", "shape_margin", "shape_gap",
    "shape_material_mu", "shape_material_restitution", "shape_material_mu_torsional",
    "shape_material_mu_rolling", "shape_material_ke", "shape_material_kd",
    "shape_material_kf", "shape_material_ka", "shape_material_kh",
)
# Appearance-only fields, recorded for the visual-difference report.
VISUAL_FIELDS = ("shape_color", "shape_opacity")


def _model(manager):
    if hasattr(manager, "get_model"):
        model = manager.get_model()
        if model is not None:
            return model
    return getattr(manager, "model", None)


def _array(value):
    value = value.numpy() if hasattr(value, "numpy") else value
    return value.tolist() if hasattr(value, "tolist") else value


def _geometry_hash(source):
    """SHA-256 of a shape's geometry source (mesh vertices + indices); None for primitives."""
    if source is None:
        return None
    digest = hashlib.sha256(type(source).__name__.encode())
    for name in ("vertices", "indices", "data", "heights"):
        value = getattr(source, name, None)
        if value is None:
            continue
        value = value.numpy() if hasattr(value, "numpy") else value
        try:
            import numpy as np

            array = np.ascontiguousarray(np.asarray(value))
            digest.update(name.encode() + str(array.dtype).encode() + str(array.shape).encode())
            digest.update(array.tobytes())
        except Exception:
            digest.update(name.encode() + repr(value).encode())
    return digest.hexdigest()


def write_snapshot(manager, label):
    path = os.environ.get("MICRODUCK_PHYSICS_AUDIT")
    if not path:
        return None
    model = _model(manager)
    if model is None:
        raise RuntimeError("physics audit: no Newton model available")
    labels = list(getattr(model, "shape_label", []) or [])
    fields, missing = {}, []
    for name in PHYSICS_FIELDS + VISUAL_FIELDS:
        value = getattr(model, name, None)
        if value is None:
            missing.append(name)
            continue
        fields[name] = _array(value)
    sources = list(getattr(model, "shape_source", []) or [])
    fields["geometry_sha256"] = [_geometry_hash(s) for s in sources] if sources else []
    body_labels = list(getattr(model, "body_label", []) or [])
    snapshot = dict(
        label=label,
        shape_count=int(getattr(model, "shape_count", len(labels))),
        body_count=int(getattr(model, "body_count", len(body_labels))),
        body_label=body_labels,
        shape_contact_pair_count=int(getattr(model, "shape_contact_pair_count", -1) or -1),
        shape_label=labels,
        physics_fields=list(PHYSICS_FIELDS),
        missing_fields=missing,
        fields=fields,
    )
    snapshot["sha256"] = hashlib.sha256(json.dumps(
        {k: snapshot[k] for k in ("shape_label", "body_label", "fields")}, sort_keys=True).encode()).hexdigest()
    with open(path, "w") as stream:
        json.dump(snapshot, stream)
    print("COURSE_PHYSICS_AUDIT " + json.dumps(dict(label=label, path=path, shape_count=snapshot["shape_count"],
                                                     body_count=snapshot["body_count"], missing=missing,
                                                     sha256=snapshot["sha256"])), flush=True)
    return snapshot
