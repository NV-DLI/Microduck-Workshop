"""Narrow USD-import compatibility for the course's pinned Newton/BAM runtime.

Isaac Lab e527032 adds an explicit permission for trusted Torch-backed actuator
assets. Newton 71eeba16 predates that keyword and already loads those assets.
The course retains this Newton revision for its validated fixed policies; this
adapter changes only that import keyword, never contacts, actuators or dynamics.
It supports the course Newton route, not arbitrary Isaac Lab physics backends.
"""
from functools import wraps
from importlib.metadata import distribution
from importlib.abc import MetaPathFinder, Loader
from importlib.machinery import PathFinder
import sys
import inspect
import json

NEWTON_COMMIT = "71eeba16a717d4dcf0b9dd7056a7b6f863054dc3"


def install():
    """Validate provenance, then install one idempotent, signature-checked adapter."""
    metadata = distribution("newton")
    source = json.loads(metadata.read_text("direct_url.json") or "{}")
    if source.get("vcs_info", {}).get("commit_id") != NEWTON_COMMIT:
        raise RuntimeError("The course requires Newton " + NEWTON_COMMIT)
    if "newton" in sys.modules:
        _patch(sys.modules["newton"])
    elif not any(getattr(finder, "_microduck_newton_hook", False) for finder in sys.meta_path):
        # Do not import Newton here: it imports USD. GUI entries must let Kit
        # initialize its USD libraries first, using their existing launch order.
        sys.meta_path.insert(0, _NewtonFinder())


def _patch(newton):
    original = newton.ModelBuilder.add_usd
    if getattr(original, "_microduck_import_permission", False):
        return
    signature = inspect.signature(original)
    if "allow_unsafe_torch_checkpoints" in signature.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    ):
        raise RuntimeError("Unexpected Newton USD importer signature; review the course adapter")

    @wraps(original)
    def add_usd(self, *args, **kwargs):
        if "allow_unsafe_torch_checkpoints" in kwargs:
            if kwargs.pop("allow_unsafe_torch_checkpoints") is not True:
                raise ValueError("This pinned importer cannot enforce a restricted Torch-asset import")
        return original(self, *args, **kwargs)

    add_usd._microduck_import_permission = True
    newton.ModelBuilder.add_usd = add_usd


class _AfterImport(Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch(module)


class _NewtonFinder(MetaPathFinder):
    _microduck_newton_hook = True

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "newton":
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            raise ImportError("The pinned Newton package is missing")
        spec.loader = _AfterImport(spec.loader)
        sys.meta_path.remove(self)
        return spec
