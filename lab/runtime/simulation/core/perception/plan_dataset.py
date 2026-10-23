#!/usr/bin/env python3
"""Freeze scene-disjoint Beta collection requests, without manufacturing images."""
import argparse, hashlib, json, random
from pathlib import Path


def make_plan(seed=972001, train_scenes=64, val_scenes=16, test_scenes=16):
    rows = []
    counter = 0
    for split, count in [
        ("train", train_scenes),
        ("val", val_scenes),
        ("test", test_scenes),
    ]:
        for scene in range(count):
            scene_seed = seed + counter * 101
            counter += 1
            for view, hfov in enumerate([60, 70, 80, 90]):
                view_seed = scene_seed * 17 + view
                r = random.Random(view_seed)
                distortion = [
                    [-0.08, 0.01, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.04, -0.005, 0.001, -0.001, 0.0],
                ][(counter + view) % 3]
                rows.append(
                    {
                        "id": f"{split}-{scene_seed}-v{view}",
                        "split": split,
                        "scene_seed": scene_seed,
                        "view_seed": view_seed,
                        "view_index": view,
                        "view_mode": ["far", "mid", "near", "search"][
                            (counter + view) % 4
                        ],
                        "horizontal_fov_deg": hfov,
                        "distortion": distortion,
                        "light_intensity": r.uniform(900, 2400),
                        "root_yaw_rad": r.uniform(-0.18, 0.18),
                        "root_roll_rad": r.uniform(-0.025, 0.025),
                        "root_pitch_rad": r.uniform(-0.025, 0.025),
                        "head_yaw_jitter_rad": r.uniform(-0.12, 0.12),
                        "head_pitch_jitter_rad": r.uniform(-0.10, 0.10),
                    }
                )
    return {
        "schema": "microduck.beta-detector-collection-plan.v1",
        "runtime_id": "newton-bam-20261019",
        "hardware_profile_id": "beta-gc2093-l8cx-v1",
        "seed": seed,
        "raster": [960, 540],
        "classes": ["ball", "goal"],
        "renderer": "actual Isaac RTX head camera",
        "split_unit": "scene_seed: every pose/lens view of a scene stays in one split",
        "optics_scope": "hypothetical Beta lens sensitivity, not hardware calibration",
        "rows": rows,
        "actual_images": 0,
        "generated_templates_are_not_data": True,
    }


def validate_plan(p):
    if (
        p["runtime_id"] != "newton-bam-20261019"
        or p["hardware_profile_id"] != "beta-gc2093-l8cx-v1"
    ):
        raise ValueError("Wrong runtime/profile")
    scenes = {}
    ids = set()
    views = set()
    for row in p["rows"]:
        if row["id"] in ids or row["view_seed"] in views:
            raise ValueError("Duplicate view")
        ids.add(row["id"])
        views.add(row["view_seed"])
        old = scenes.setdefault(row["scene_seed"], row["split"])
        if old != row["split"]:
            raise ValueError("Scene crosses splits")
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Unknown split")
    return {
        "views": len(ids),
        "scenes": len(scenes),
        "split_scene_counts": {
            s: sum(v == s for v in scenes.values()) for s in ("train", "val", "test")
        },
    }


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--seed", type=int, default=972001)
    a.add_argument("--train-scenes", type=int, default=64)
    a.add_argument("--val-scenes", type=int, default=16)
    a.add_argument("--test-scenes", type=int, default=16)
    args = a.parse_args()
    p = make_plan(args.seed, args.train_scenes, args.val_scenes, args.test_scenes)
    summary = validate_plan(p)
    if args.output.exists():
        raise FileExistsError("Preserve existing frozen plan")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(p, indent=2) + "\n")
    print(
        json.dumps(
            {
                **summary,
                "plan_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "images_created": 0,
            }
        )
    )
