"""Train walking with the selected official straight-walking recipe.

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
    from course_labs.training_recipe import apply_official_walking_recipe

    apply_official_walking_recipe(cfg, agent)
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
                "training_recipe_sha256": hashlib.sha256(
                    (Path(__file__).parent / "course_labs/training_recipe.py").read_bytes()
                ).hexdigest(),
                "training_safety_sha256": hashlib.sha256(
                    (Path(__file__).parent / "course_labs/training_safety.py").read_bytes()
                ).hexdigest(),
                "historical_training_results_reused": False,
                "recipe": "official-straight",
                "rendering": "physics-only training; view resulting policies in the shared RTX walking lab",
            },
            indent=2,
        )
    )
    return cfg, agent


module.resolve_task_config = resolve


class CourseRunner(module.OnPolicyRunner):
    """Continue the curriculum as well as PPO when resuming a saved run."""

    def save(self, path, infos=None):
        infos = dict(infos or {})
        infos["course_environment_steps"] = int(self.env.unwrapped.common_step_counter)
        super().save(path, infos)

    def load(self, path, *args, **kwargs):
        infos = super().load(path, *args, **kwargs)
        # RSL-RL saves the zero-based index of the completed iteration. Legacy
        # course checkpoints did not save the environment's curriculum clock.
        steps = (infos or {}).get("course_environment_steps")
        if steps is None:
            if Path(path).name == "initial.pt":
                steps = 0
            else:
                steps = (self.current_learning_iteration + 1) * self.cfg["num_steps_per_env"]
        self.current_learning_iteration = int(steps) // self.cfg["num_steps_per_env"]
        self.env.unwrapped.common_step_counter = int(steps)
        # Episode reset applies the saved schedule before sampling commands.
        self.env.reset()
        output = Path(remaining[remaining.index("--output") + 1])
        base = self.env.unwrapped
        # Record scalar values from live manager configs. Their stateful reward
        # functions hold simulator references and cannot be serialized as YAML.
        (output / "resume-state.json").write_text(json.dumps({
            "environment_steps": int(steps),
            "next_learning_iteration": self.current_learning_iteration,
            "recipe": "official-straight",
            "physics_episodes_restarted": True,
            "head_pose_bias_weight": base.reward_manager.get_term_cfg("head_pose_bias").weight,
            "head_pose_ranges": base.command_manager.get_term("head_pose").cfg.ranges,
            "standing_share": base.command_manager.get_term("base_velocity").cfg.rel_standing_envs,
        }, indent=2))
        return infos


module.OnPolicyRunner = CourseRunner
sys.argv = [str(source), *remaining]
module.main()
