#!/usr/bin/env python3
"""Match the learned gait to delivered BODY50 input, preserving native BAM."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
G = HERE.parents[1]
M = G.parent
sys.path[:0] = [str(HERE), str(G), str(M / "evaluation/body50_training")]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def atomic(p, value):
    temp = p.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(p)


def checked_hook_summary(hook):
    value = hook.summary()
    if value["actual_task_external_wrench_peak"] != 0.0:
        raise RuntimeError(
            "Actual external support detected during a task; refuse checkpoint evaluation"
        )
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--resume", required=True, type=Path)
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--iterations", type=int, default=1200)
    p.add_argument("--seed", type=int, default=970301)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument(
        "--eager-sensor",
        action="store_true",
        help="Use the validated eager oracle instead of compiled sensor updates",
    )
    p.add_argument(
        "--diagnostic-reset-after-step",
        type=int,
        help="Small technical run only: reset every fourth environment once during learning",
    )
    a = p.parse_args()
    a.output, a.resume = a.output.resolve(), a.resume.resolve()
    if a.diagnostic_reset_after_step is not None and (
        a.num_envs > 32 or a.iterations > 20 or a.diagnostic_reset_after_step < 1
    ):
        p.error("Injected reset is restricted to an explicitly small technical run")
    a.output.mkdir(parents=True, exist_ok=False)
    import torch
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab.app import launch_simulation
    from isaaclab.managers import TerminationTermCfg
    from isaaclab.utils import to_dict
    from isaaclab.utils.io import dump_yaml
    from isaaclab_tasks.utils import resolve_task_config
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from stop_gaze_recipe import configure, configure_agent
    from head_reference_curriculum import FinalHeadResidualAction
    from environment_hook import Body50TrainingHook, startup_failed
    from ppo_mask import install

    parent = json.loads((a.resume.parent / "training-ledger.json").read_text())
    saved = torch.load(a.resume, map_location="cpu", weights_only=False)
    initial = torch.load(
        a.resume.parent / "initial.pt", map_location="cpu", weights_only=False
    )
    stage_updates = int(saved["iter"]) - int(initial["iter"]) + 1
    if (
        parent.get("recipe") != "physical-stop-gaze-head-reference-exploration-v8"
        or stage_updates < 300
    ):
        raise ValueError(
            "Use a v8 checkpoint trained through the final head-reference mapping"
        )
    parent_updates = int(parent["parent_completed_updates"]) + stage_updates
    cfg, agent = resolve_task_config(
        "IsaacContrib-Velocity-Flat-MicroDuck",
        "rsl_rl_cfg_entry_point",
        overrides=["presets=backlash"],
    )
    configure(cfg)
    configure_agent(agent)
    cfg.actions.joint_pos.class_type = FinalHeadResidualAction
    cfg.terminations.beta_startup_failed = TerminationTermCfg(func=startup_failed)
    cfg.scene.num_envs = a.num_envs
    cfg.seed = agent.seed = a.seed
    cfg.commands.base_velocity.debug_vis = False
    cfg.sim.device = agent.device = "cuda:0"
    cfg.log_dir = str(a.output)
    agent.save_interval = a.save_interval
    agent.logger = "tensorboard"
    agent.algorithm.normalize_advantage_per_mini_batch = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    source_files = [
        HERE / n
        for n in (
            "train.py",
            "environment_hook.py",
            "startup_fixture.py",
            "ppo_mask.py",
            "compiled_sensor.py",
        )
    ]
    source_files += [
        M / "evaluation/body50_training" / n
        for n in ("physics_producer.py", "vector_body_imu.py")
    ]
    snapshot = a.output / "source_snapshot/body_imu"
    snapshot.mkdir(parents=True)
    for source in source_files:
        shutil.copy2(source, snapshot / source.name)
    for name in [
        "stop_gaze_recipe.py",
        "head_reference_curriculum.py",
        "head_reference.py",
        "gaze_recipe.py",
        "gaze-command-table.json",
        "exploration_curriculum.py",
        "command_curriculum.py",
        "cycle_tracking.py",
        "bounded_tracking.py",
        "tracking_rewards.py",
        "bounded_actions.py",
        "bounded-action-limits.json",
    ]:
        shutil.copy2(G / name, a.output / name)
    ledger = {
        "schema": "microduck.training-ledger.v1",
        "run_id": a.output.name,
        "runtime_id": "newton-bam-20261019",
        "hardware_profile_id": "beta-gc2093-l8cx-v1",
        "skill": "walking",
        "recipe": "body50-stop-gaze-v9",
        "initialization": "continuation",
        "parent_checkpoint": str(a.resume),
        "parent_checkpoint_sha256": sha(a.resume),
        "parent_completed_updates": parent_updates,
        "env_count": a.num_envs,
        "rollout_steps": agent.num_steps_per_env,
        "iterations_planned": a.iterations,
        "seed": a.seed,
        "control_hz": 50,
        "physics_hz": 200,
        "shared_gpu": True,
        "evaluation_data_used_for_training": False,
        "state": "starting",
        "source_sha256": {str(f.resolve()): sha(f) for f in source_files},
        "sensor_contract": "Actual virtual BODY50 tau2/median3/25read gate, same-read encoders; upstream additional mounting/noise/delay DR explicitly retained",
        "initialization_fixture": "Physical spring/damper wrench capped12N/.6Nm; .5s supported stand and1s withdrawal after actual sensor readiness",
        "loss_mask": "Only ready AND task-active transitions enter PPO; initialization/release excluded",
        "raw_history_preserved": True,
        "state_projection": False,
        "compiled_sensor_requested": not a.eager_sensor,
        "prewarm_gate": "Each world completing physical startup is accumulated across time; current task occupancy is not startup coverage. At least90% must have completed once within8s; all masks remain active during PPO.",
        "diagnostic_reset_after_step": a.diagnostic_reset_after_step,
        "action_mapping": {
            "head_reference_blend": 1.0,
            "head_residual_limit_rad": 0.15,
            "target_rate_rad_s": 8.0,
        },
        "native_sensor_hardware_fit_claimed": False,
    }
    atomic(a.output / "training-ledger.json", ledger)
    started = time.monotonic()
    with launch_simulation(cfg, {"device": "cuda:0", "visualizer": []}):
        base = gym.make("IsaacContrib-Velocity-Flat-MicroDuck", cfg=cfg).unwrapped
        hook = Body50TrainingHook(base, seed=a.seed, extra_observation_dr=True)
        compiled = None
        if not a.eager_sensor:
            from compiled_sensor import install as compile_sensor

            compiled = compile_sensor(hook.producer)
        env = RslRlVecEnvWrapper(base, clip_actions=agent.clip_actions)
        dump_yaml(str(a.output / "env.yaml"), cfg)
        dump_yaml(str(a.output / "agent.yaml"), agent)
        runner = OnPolicyRunner(
            env, to_dict(agent), log_dir=str(a.output), device="cuda:0"
        )
        runner.load(str(a.resume))
        masking = install(runner.alg, base)
        runner.save(str(a.output / "initial.pt"))
        ledger["initial_checkpoint_sha256"] = sha(a.output / "initial.pt")
        start_iteration = int(runner.current_learning_iteration)
        # Real prewarming, with no reward update or optimizer update. At every
        # reset during training the same fixture runs with the same masks.
        ever_task = torch.zeros(a.num_envs, dtype=torch.bool, device=base.device)
        with torch.inference_mode():
            for _ in range(400):
                obs = env.get_observations()
                actions = torch.zeros((a.num_envs, 14), device=base.device)
                ready = base.ready_mask
                if ready.any():
                    actions[ready] = runner.alg.actor(obs[ready])
                env.step(actions)
                ever_task |= base.task_active_mask
                if bool(ever_task.all()):
                    break
        ledger["prewarm"] = checked_hook_summary(hook)
        ledger["prewarm"]["environments_completed_startup_once"] = int(ever_task.sum())
        ledger["prewarm"]["current_task_occupancy_is_not_startup_coverage"] = True
        ledger["prewarm_wall_seconds"] = time.monotonic() - started
        if float(ever_task.float().mean()) < 0.9:
            ledger.update(
                state="failed",
                error="Fewer than90% of original training worlds completed physical startup at least once within8s",
            )
            atomic(a.output / "training-ledger.json", ledger)
            env.close()
            raise RuntimeError(ledger["error"])
        ledger.update(state="training", started_at_unix=time.time())
        atomic(a.output / "training-ledger.json", ledger)
        training_start = time.monotonic()
        if a.diagnostic_reset_after_step is not None:
            original_step = base.step
            call_count = 0

            def diagnostic_step(action):
                nonlocal call_count
                result = original_step(action)
                call_count += 1
                if call_count == a.diagnostic_reset_after_step:
                    ids = torch.arange(0, a.num_envs, 4, device=base.device)
                    obs, reward, terminated, timeouts, extras = result
                    base._reset_idx(ids)
                    terminated = terminated.clone()
                    terminated[ids] = True
                    obs = base.observation_manager.compute(update_history=True)
                    ledger["injected_reset"] = {
                        "training_control_step": call_count,
                        "env_ids": ids.cpu().tolist(),
                        "kind": "Explicit development reset, not a fall or task success",
                        "ready_mask_after_reset": base.ready_mask.cpu().tolist(),
                    }
                    return obs, reward, terminated, timeouts, extras
                return result

            base.step = diagnostic_step
        original_save = runner.save

        def save(path, infos=None):
            original_save(path, infos)
            counts = masking.report()
            ledger.update(
                latest_checkpoint=str(path),
                latest_checkpoint_sha256=sha(path),
                rollout_iterations_this_run=int(runner.current_learning_iteration)
                - start_iteration
                + 1,
                iterations_this_run=counts["rollouts_with_optimizer_updates"],
                iterations_cumulative=parent_updates
                + counts["rollouts_with_optimizer_updates"],
                training_wall_seconds=time.monotonic() - training_start,
                optimizer_mask=counts,
                actual_sensor_fixture=checked_hook_summary(hook),
            )
            ledger["compiled_sensor"] = compiled.report() if compiled else None
            atomic(a.output / "training-ledger.json", ledger)

        runner.save = save
        try:
            runner.learn(
                num_learning_iterations=a.iterations, init_at_random_ep_len=False
            )
            ledger.update(
                state="completed",
                training_wall_seconds=time.monotonic() - training_start,
                optimizer_mask=masking.report(),
                actual_sensor_fixture=checked_hook_summary(hook),
            )
            ledger["compiled_sensor"] = compiled.report() if compiled else None
        except BaseException as error:
            ledger.update(
                state=(
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                ),
                error=repr(error),
                training_wall_seconds=time.monotonic() - training_start,
                optimizer_mask=masking.report(),
                actual_sensor_fixture=hook.summary(),
            )
            raise
        finally:
            ledger["sources_unchanged"] = all(
                sha(p) == h for p, h in ledger["source_sha256"].items()
            )
            atomic(a.output / "training-ledger.json", ledger)
            env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if "--output" in sys.argv:
            destination = (
                Path(sys.argv[sys.argv.index("--output") + 1]) / "training-ledger.json"
            )
            if destination.exists():
                value = json.loads(destination.read_text())
                value.update(
                    state=(
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed"
                    ),
                    error=repr(error),
                )
                atomic(destination, value)
        raise
