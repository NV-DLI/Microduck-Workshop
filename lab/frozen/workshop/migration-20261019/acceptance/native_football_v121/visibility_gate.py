"""Install the independent image-only border veto in this new process only."""
from pathlib import Path
import importlib.util
import sys

D = Path(__file__).parent
M = D.parents[1]
CANDIDATE = M / "sensors/experimental/ball_image_border_v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install():
    import perception
    guard = load("football_v62_public_border_guard", CANDIDATE / "guard.py")
    adapter = load("football_v62_public_border_adapter", CANDIDATE / "adapter.py")
    previous = perception.estimate_box_sphere
    perception.estimate_box_sphere = adapter.wrap_estimator(
        previous, border_checker=guard.check_observed_ball_border)
    return previous
