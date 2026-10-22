#!/usr/bin/env python3
"""Development test of the BODY50 gait using the trained final head-reference mapping.

Every case starts neutral, slews a real head command at 0.3 rad/s, and remains
unsupported. All physics ticks are recorded at 200 Hz. No video stabilization.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

M = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def T(value):
    return getattr(value, 'torch', value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--recipe-dir', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=978801)
    p.add_argument('--envs-per-case', type=int, default=4)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(a.recipe_dir.resolve()))
    import numpy as np
    import torch
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab.app import launch_simulation
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.io import dump_yaml
    from head_reference_curriculum_v7 import FinalHeadResidualAction
    from bounded_actions import load_limits, JOINT_NAMES
    from batch_policy import BatchPolicy

    export_report = json.loads(a.model.with_suffix('.json').read_text())
    checkpoint = Path(export_report['checkpoint'])
    if checkpoint.parent.resolve() != a.recipe_dir.resolve() or sha(checkpoint) != export_report['checkpoint_sha256']:
        raise ValueError('Export must belong to the frozen recipe directory')
    initial = torch.load(a.recipe_dir/'initial.pt', map_location='cpu', weights_only=False)
    updates = int(export_report['checkpoint_iteration']) - int(initial['iter']) + 1
    ledger = json.loads((a.recipe_dir/'training-ledger.json').read_text())
    if ledger.get('recipe') != 'body50-target-authority-v11' or ledger['action_mapping']['head_reference_blend'] != 1.:
        raise ValueError('Frozen BODY50 training recipe with final head adapter required')
    body_source = a.recipe_dir/'source_snapshot/body50_v9'
    body_sources = {str((body_source/Path(path).name).resolve()):h for path,h in ledger['source_sha256'].items()}
    if any(sha(path)!=h for path,h in body_sources.items()):
        raise ValueError('BODY50 frozen source changed')
    sys.path.insert(0,str(body_source))
    from environment_hook import Body50TrainingHook, T as body_tensor
    from compiled_sensor import install as compile_sensor
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    table_path = M / 'gait/gaze-command-table-v6.json'
    table = json.loads(table_path.read_text())
    rows = table['rows']
    cases = []
    for vx in (0., .06):
        for yaw, pitch in ((0., 0.), (0., .7), (0., 1.4), (-.4, .7), (.4, .7)):
            match = next(r for r in rows if r['gaze_body_yaw_pitch_rad'] == [yaw, pitch])
            cases.append({'name': f'vx{vx:g}_yaw{yaw:g}_pitch{pitch:g}', 'velocity': [vx, 0., 0.],
                          'gaze_body_yaw_pitch_rad': [yaw, pitch],
                          'head_delta_target_rad': match['head_pose_native_offsets_rad']})
    n = len(cases) * a.envs_per_case
    cfg = parse_env_cfg('IsaacContrib-Velocity-Flat-MicroDuck', device='cuda:0', num_envs=n,
                        overrides=['presets=backlash'])
    cfg.seed = a.seed
    cfg.sim.dt = .005
    cfg.decimation = 4
    cfg.sim.render_interval = 4
    cfg.curriculum = None
    cfg.actions.joint_pos.class_type = FinalHeadResidualAction
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.resampling_time_range = (1e8, 1e8)
    cfg.commands.head_pose.ranges = tuple((0., 0.) for _ in range(4))
    cfg.commands.body_pose.ranges = tuple((0., 0.) for _ in range(6))
    for name in ('foot_friction', 'encoder_bias', 'mass_inertia', 'randomize_com',
                 'randomize_head_com', 'randomize_joint_friction', 'randomize_armature', 'push_robot'):
        setattr(cfg.events, name, None)
    cfg.scene.robot.actuators['servos'].vin_range = None
    cfg.scene.robot.actuators['servos'].vin_drop_gain_range = None
    cfg.scene.robot.actuators['servos'].min_delay = 4
    cfg.scene.robot.actuators['servos'].max_delay = 4
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.imu.params.update(max_angle_deg=0, min_lag=0, max_lag=0)
    cfg.scene.robot.spawn.usd_path = json.loads((M/'skills/experiments/built_com/generated/seed-974001/manifest.json').read_text())['source_usd']
    dump_yaml(str(a.output / 'environment.yaml'), cfg)
    sources = [Path(__file__), table_path, a.recipe_dir / 'head_reference_v7.py', a.recipe_dir / 'head_reference_curriculum_v7.py', M / 'gait/batch_policy.py',
               a.recipe_dir / 'bounded_actions.py', a.recipe_dir / 'bounded-action-limits.json',
               a.recipe_dir / 'precision_recipe_v3.py', a.recipe_dir / 'precision_recipe.py']
    sources += [Path(path) for path in body_sources]
    sources += [a.recipe_dir/name for name in ('precision_recipe_v2.py','precision_recipe_v4.py','precision_recipe_v5.py')]
    original_identity = {str(path.resolve()): sha(path) for path in sources}
    archive = a.output/'source_snapshot'
    archive.mkdir()
    identity = {}
    for path in sources:
        dest = archive/path.name
        if dest.exists() and sha(dest)!=sha(path):
            raise ValueError('Source snapshot basename collision')
        dest.write_bytes(path.read_bytes())
        identity[str(dest.resolve())] = sha(dest)
    report = {'schema': 'microduck.trained-head-reference-gaze-development.v1', 'status': 'starting',
              'runtime_id': 'newton-bam-20261019', 'hardware_profile_id': 'beta-gc2093-l8cx-v1',
              'model': str(a.model.resolve()), 'model_sha256': sha(a.model),
              'source_sha256': identity, 'configuration_sha256': sha(a.output / 'environment.yaml'),
              'robot_sha256': sha(cfg.scene.robot.spawn.usd_path), 'seed': a.seed,
              'condition': 'nominal', 'cases': cases, 'action_architecture': 'head reference plus0.15rad*tanh(raw)', 'head_residual_limit_rad': .15, 'actor_was_trained_with_this_adapter': True, 'reference_blend': 1., 'envs_per_case': a.envs_per_case,
              'qualification_claimed': False, 'automatic_resets': False, 'external_support': False,
              'state_projection': False, 'physics_hz': 200, 'policy_hz': 50, 'duration_s': 20,
              'head_command_schedule': 'Neutral 0–3s; then approach each fixed target at0.3rad/s per joint',
              'sensor_scope': 'Actual virtual BODY50 tau2/median3/25-read gate with same-read quantized encoders; physical sensor and lens parameters are not hardware-calibrated',
              'external_support_scope': 'Task trace only; explicit physical calibration/withdrawal fixture precedes the task',
              'inference_precision':'FP32, TF32 disabled',
              'camera_measurement': 'Actual rigid jaw_soft angular velocity; identical at optical site, no filtering'}
    started = time.monotonic()
    state, controls, bam_trace = [], [], []
    actor_rows, sensor_rows, wrench_rows, inference_rows = [], [], [], []
    sensor_fields = ('gyro_body_rad_s','gravity_body','encoder_q_rad','encoder_qd_rad_s','read_timestamp_s','encoder_read_timestamp_s','orientation_valid')
    try:
        with launch_simulation(cfg, {'device': 'cuda:0', 'visualizer': []}):
            env = gym.make('IsaacContrib-Velocity-Flat-MicroDuck', cfg=cfg).unwrapped
            manager = env.sim.backend_manager
            manager.set_decimation(1)
            env._physics_handles_decimation = False
            if manager.backend_name != 'newton' or not env.sim.cfg.use_newton_actuators:
                raise RuntimeError('Native Newton BAM required')
            hook = Body50TrainingHook(env,seed=a.seed,extra_observation_dr=False)
            compiled = compile_sensor(hook.producer)
            env.reset()
            robot = env.scene['robot']
            ids, names = robot.find_joints(JOINT_NAMES, preserve_order=True)
            pids, _ = robot.find_joints(['passive_' + x + '_backlash' for x in JOINT_NAMES], preserve_order=True)
            head_ids, head_names = robot.find_bodies(['jaw_soft'], preserve_order=True)
            if list(names) != JOINT_NAMES or robot.num_joints != 28 or len(head_ids) != 1:
                raise ValueError('Expected native28joint robot and one physical camera body')
            term = env.action_manager.get_term('joint_pos')
            model = BatchPolicy(a.model, report['model_sha256'], env.device)
            warmup_origin = env.sim.get_physics_step_count()
            with torch.inference_mode():
                for _ in range(400):
                    cmd = env.command_manager.get_term('base_velocity')
                    cmd.is_standing_env[:] = False
                    cmd.vel_command_b[:] = 0
                    env.command_manager.get_term('head_pose').command[:] = 0
                    env.command_manager.get_term('body_pose').command[:] = 0
                    obs = env.observation_manager.compute(update_history=True)['policy']
                    raw = torch.zeros((n,14),device=env.device)
                    ready = env.ready_mask
                    if ready.any(): raw[ready] = model(obs[ready])
                    env.action_manager.process_action(raw)
                    for _ in range(4):
                        env._sim_step_counter += 1
                        env.action_manager.apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(.005)
                    env.common_step_counter += 1
                    if bool(env.task_active_mask.all()): break
            report['physical_initialization'] = {**hook.summary(),'physics_ticks':env.sim.get_physics_step_count()-warmup_origin,'automatic_resets':0}
            if not bool(env.task_active_mask.all()):
                raise RuntimeError('Original cohort did not all finish physical startup within8s; no survivor selection')
            if float(hook.task_wrench_peak)!=0.: raise RuntimeError('Task assistance detected')
            initial_previous_raw = hook.previous_raw.cpu().numpy().copy()
            native_home14 = T(robot.data.default_joint_pos)[:,ids].cpu().numpy().copy()
            report['parity'] = model.parity(env.observation_manager.compute()['policy'][env.ready_mask], tolerance=1e-5)
            drives = [x for x in manager.get_model().actuators if type(x.drive).__name__ == 'DriveBam']
            if len(drives) != 1: raise ValueError('Expected one native BAM actuator batch')
            drive = drives[0].drive
            report['native_bam'] = {'class': type(drive).__module__ + '.' + type(drive).__name__,
                'parameters': {key: getattr(drive, key).numpy().tolist() for key in drive.PER_DOF_PARAMS
                               if getattr(drive, key, None) is not None}}
            limits = T(robot.data.joint_pos_limits).clone()
            output_limits = np.asarray(load_limits()['physical_output_limits_rad'])
            desired = torch.tensor([c['head_delta_target_rad'] for c in cases for _ in range(a.envs_per_case)], device=env.device)
            velocity = torch.tensor([c['velocity'] for c in cases for _ in range(a.envs_per_case)], device=env.device)
            head = torch.zeros_like(desired)
            fell = torch.zeros(n, dtype=torch.bool, device=env.device)
            origin = env.sim.get_physics_step_count()
            with torch.inference_mode():
                for step in range(1000):
                    cmd = env.command_manager.get_term('base_velocity')
                    cmd.is_standing_env[:] = False
                    cmd.vel_command_b[:] = velocity if step >= 150 else 0
                    if step >= 150: head += (desired-head).clamp(-.006, .006)
                    env.command_manager.get_term('head_pose').command[:] = head
                    env.command_manager.get_term('body_pose').command[:] = 0
                    observation = env.observation_manager.compute(update_history=True)['policy']
                    actor_rows.append(observation.cpu().numpy().copy())
                    packet = hook.producer.observe()
                    sensor_rows.append({key:packet[key].cpu().numpy().copy() for key in sensor_fields})
                    ready = env.ready_mask.clone()
                    inference_rows.append(ready.cpu().numpy().copy())
                    raw = torch.zeros((n,14),device=env.device)
                    if ready.any(): raw[ready] = model(observation[ready])
                    env.action_manager.process_action(raw)
                    if not torch.allclose(env.action_manager.action, raw):
                        raise ValueError('Raw action-history contract changed')
                    controls.append(torch.cat([raw, term.processed_actions-term.encoder_bias,
                                               cmd.vel_command_b, head], dim=1).cpu().numpy())
                    for _ in range(4):
                        env._sim_step_counter += 1
                        env.action_manager.apply_action()
                        env.scene.write_data_to_sim()
                        env.sim.step(render=False)
                        env.scene.update(.005)
                        d = robot.data
                        fell |= (T(d.projected_gravity_b)[:, 2] > -.6) | (T(d.root_pos_w)[:, 2] < .065)
                        state.append(torch.cat([T(d.root_pos_w), T(d.root_lin_vel_b), T(d.root_ang_vel_b),
                            T(d.joint_pos), T(d.body_ang_vel_w)[:, head_ids[0]], fell[:, None].float()], dim=1).cpu().numpy())
                        wrench_rows.append(body_tensor(robot.data._sim_bind_body_external_wrench).reshape(n,-1).abs().amax(-1).cpu().numpy().copy())
                        bam_trace.append(np.stack([getattr(drive, key).numpy().copy() for key in ('motor_torque', 'effective_vin', '_delayed_target')]))
                    env.episode_length_buf += 1
                    env.common_step_counter += 1
                    hook.task_steps.add_(1)
            report['body50_task_readback'] = hook.summary()
            report['compiled_sensor'] = compiled.report()
            if float(hook.task_wrench_peak)!=0.: raise RuntimeError('Actual task support detected')
            if env.sim.get_physics_step_count()-origin != 4000: raise ValueError('Missing real physics samples')
            data = np.stack(state)
            control = np.stack(controls)
            sensor_arrays = {'body50_'+key:np.stack([row[key] for row in sensor_rows]) for key in sensor_fields}
            np.savez_compressed(a.output/'raw-trace.npz', state=data, control=control, bam=np.stack(bam_trace),
                                servo_ids=ids, passive_ids=pids, servo_limits=limits.cpu().numpy(),
                                output_limits=output_limits, time_s=np.arange(1,4001)/200,
                                actor_observation=np.stack(actor_rows), initial_previous_raw=initial_previous_raw, native_home14=native_home14, external_wrench_peak=np.stack(wrench_rows), actor_inferred_mask=np.stack(inference_rows), **sensor_arrays)
            q = data[:, :, 9:37]
            effective = q[:, :, ids] + q[:, :, pids]
            lim = limits.cpu().numpy()[:, ids]
            servo_violation = np.maximum(lim[None, :, :, 0]-q[:, :, ids], q[:, :, ids]-lim[None, :, :, 1]).clip(0).max(axis=(0,2))
            physical_violation = np.maximum(output_limits[None,None,:,0]-effective, effective-output_limits[None,None,:,1]).clip(0).max(axis=(0,2))
            hidx = [JOINT_NAMES.index(x) for x in table['head_joint_order']]
            desired_absolute = desired.cpu().numpy() + native_home14[:,hidx]
            report['episodes'] = []
            for i in range(n):
                omega = np.linalg.norm(data[1600:,i,37:40], axis=1)
                error = effective[1600:,i][:,hidx] - desired_absolute[i]
                report['episodes'].append({'case': cases[i//a.envs_per_case]['name'], 'episode_index': i,
                    'sensor_unready_control_steps': int((~np.stack(inference_rows)[:,i]).sum()),
                    'uninterrupted_sensor_control': bool(np.stack(inference_rows)[:,i].all()),
                    'fell': bool(data[-1,i,40]), 'first_fall_s': (float((np.flatnonzero(data[:,i,40])[0]+1)/200) if data[-1,i,40] else None),
                    'servo_peak_violation_rad': float(servo_violation[i]), 'physical_output_peak_violation_rad': float(physical_violation[i]),
                    'head_tracking_rmse_rad_after8s': float(np.sqrt(np.mean(error**2))),
                    'camera_angular_rms_rad_s_after8s': float(np.sqrt(np.mean(omega**2))),
                    'camera_angular_p95_rad_s_after8s': float(np.percentile(omega,95)),
                    'body_velocity_mean_after8s': data[1600:,i,3:6].mean(axis=0).tolist(),
                    'root_displacement_m_after3s': (data[-1,i,:3]-data[599,i,:3]).tolist()})
            report.update(status='recorded', trace_sha256=sha(a.output/'raw-trace.npz'),
                state_fields={'root_position':[0,3],'body_linear_velocity':[3,6],'body_angular_velocity':[6,9],
                              'joint_position':[9,37],'world_camera_angular_velocity':[37,40],'fell':40},
                control_fields={'raw_action':[0,14],'target':[14,28],'velocity_command':[28,31],'head_delta_command':[31,35]})
            env.close()
        report['sources_unchanged'] = original_identity == {str(path.resolve()): sha(path) for path in sources} and all(sha(path)==h for path,h in identity.items())
        if not report['sources_unchanged']: raise ValueError('Probe source changed')
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        if state: np.savez_compressed(a.output/'partial-trace.npz', state=np.stack(state), control=np.stack(controls))
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'status':report['status'],'output':str(a.output)}),flush=True)


if __name__ == '__main__':
    main()
