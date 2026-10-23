"""Run the preserved training recipe with the current full-contact robot asset.

A new run records this scene change separately. Existing checkpoints and historical
training results retain their original contact configuration.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

p = argparse.ArgumentParser(add_help=False)
p.add_argument("--runtime", type=Path, required=True)
a, remaining = p.parse_known_args()
source = a.runtime / "gait/student_train.py"
spec = importlib.util.spec_from_file_location("course_walk_training", source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original_resolve = module.resolve_task_config
asset = Path(__file__).parent / "assets/microduck_contact.usda"


def resolve(*args, **kwargs):
    cfg, agent = original_resolve(*args, **kwargs)
    cfg.scene.robot.spawn.usd_path = str(asset.resolve())
    output = Path(remaining[remaining.index("--output") + 1])
    (output / "scene-contract.json").write_text(
        json.dumps(
            {
                "robot": "official full-contact backlash model",
                "scene_adapter_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                "robot_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "historical_training_results_reused": False,
                "rendering": "physics-only training; view resulting policies in the shared RTX walking lab",
            },
            indent=2,
        )
    )
    return cfg, agent


module.resolve_task_config = resolve
sys.argv = [str(source), *remaining]
module.main()
