#!/usr/bin/env python3
"""Collect walking demonstrations and evaluate GR00T with the same physical task."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

from run import LOCAL, ROOT, config, environment, fresh

sys.path.insert(0, str(ROOT / 'lab'))
from learning.walking_data import score


def scene_command(c, output, seed, seconds, server=None):
    runtime = Path(c['materialized'])
    detector = LOCAL / 'scene-detector-deployment.json'
    if not detector.is_file():
        raise ValueError('Prepare the football scene and detector before collecting data')
    entry = 'walking_policy_scene.py' if server else 'football_scene.py'
    args = [c['python'], str(ROOT / 'lab' / entry), '--runtime', str(runtime),
            '--style', 'A', '--model', str(runtime / 'models/reference/velstand.onnx'),
            '--output', str(output), '--development', '--seconds', str(seconds),
            '--seed', str(seed), '--render-subframes', '4',
            '--body-fusion-tau-s', '2', '--body-added-delay-s', '0', '--policy-imu', 'body50',
            '--detector-deployment', str(detector), '--detector-backend', 'torch',
            '--detector-device', 'cuda:0', '--shared-imu-readback', '--packed-physics-readback',
            '--root-x', '-.55', '--root-y', '.04', '--ball-xy', '.02', '0',
            '--recovery-approval', str(runtime / 'evaluation/football/recovery-approval.json')]
    if server:
        args += ['--walking-policy-server', server]
    return args


def learning_environment():
    e = os.environ.copy()
    e['PYTHONPATH'] = str(ROOT / 'lab')
    e['TOKENIZERS_PARALLELISM'] = 'false'
    return e


def run_module(python, module, args):
    subprocess.run([str(python), '-m', 'learning.' + module, *map(str, args)],
                   cwd=ROOT, env=learning_environment(), check=True)


def episodes(a, server=None):
    c = config()
    output = fresh(a.output)
    output.mkdir(parents=True)
    seeds = list(range(a.seed, a.seed + a.episodes))
    if server:
        lineage = json.loads((a.checkpoint / 'training-provenance.json').read_text())
        if set(seeds) & {r['seed'] for r in lineage}:
            raise ValueError('Physical evaluation seeds must be separate from demonstrations')
    (output / 'plan.json').write_text(json.dumps(dict(
        method='groot_walking' if server else 'sensor_based_traditional_walking',
        seeds=seeds, seconds=a.seconds, style='A', split=a.split,
        root_xy_m=[-.55, .04], ball_xy_m=[.02, 0.],
        ball_diameter_m=.10, ball_mass_kg=.03,
        checkpoint=str(a.checkpoint.resolve()) if server else None,
        action_execution_steps=getattr(a, 'action_execution_steps', None)), indent=2))
    attempts = []
    for seed in seeds:
        destination = output / f'episode-{seed}'
        cmd = scene_command(c, destination, seed, a.seconds, server)
        with (output / f'episode-{seed}.log').open('x') as log:
            code = subprocess.call(cmd, cwd=ROOT, env=environment(c),
                                   stdout=log, stderr=subprocess.STDOUT)
        record = dict(seed=seed, path=str(destination.resolve()), split=a.split, exit_code=code)
        if code == 0:
            record['score'] = score(destination)
        attempts.append(record)
        (output / 'attempts.json').write_text(json.dumps(attempts, indent=2))
        accepted = [r for r in attempts if r.get('score', {}).get('passed')]
        (output / 'accepted.json').write_text(json.dumps(accepted, indent=2))
        print(f'Walking episode {seed}: {record.get("score", {}).get("passed", False)}; '
              f'{len(accepted)}/{len(attempts)} complete tasks', flush=True)
        if code:
            raise RuntimeError('Simulator failed; original log and outputs are retained')


def rollout(a):
    # One model server and one simulator constitute this evaluation job. Both
    # are owned here; existing desktop, ROS, and inference services are untouched.
    if a.output.exists():
        raise FileExistsError(a.output)
    contract = json.loads((a.checkpoint / 'control-contract.json').read_text())
    if contract.get('locomotion') != 'walking':
        raise ValueError('Select a walking checkpoint')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = a.output.with_name(a.output.name + '-inference')
    base.mkdir(parents=True, exist_ok=False)
    execution_args = (['--action-execution-steps', str(a.action_execution_steps)]
                      if a.action_execution_steps is not None else [])
    with (base / 'server.log').open('x') as log:
        process = subprocess.Popen([str(a.python), '-m', 'learning.server',
            '--checkpoint', str(a.checkpoint.resolve()), '--port', str(port),
            '--log', str(base / 'requests.jsonl'), *execution_args], cwd=ROOT,
            env=learning_environment(), stdout=log, stderr=subprocess.STDOUT)
        url = f'http://127.0.0.1:{port}'
        try:
            for _ in range(240):
                if process.poll() is not None:
                    raise RuntimeError('Inference server failed; inspect its saved log')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=2) as response:
                        health = json.load(response)
                    if health.get('ready'):
                        (base / 'identity.json').write_text(json.dumps(health, indent=2))
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('Inference server did not become ready')
            episodes(a, url)
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    for name in ('collect', 'rollout'):
        c = sub.add_parser(name)
        c.add_argument('--output', type=Path, required=True)
        c.add_argument('--seed', type=int, required=True)
        c.add_argument('--episodes', type=int, default=10)
        c.add_argument('--seconds', type=float, default=60.)
        if name == 'collect':
            c.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
        else:
            c.set_defaults(split='test')
            c.add_argument('--python', type=Path, required=True)
            c.add_argument('--checkpoint', type=Path, required=True)
            c.add_argument('--action-execution-steps', type=int,
                           help='Optional actions per replan; defaults to checkpoint')
    for name in ('export', 'initialize', 'train', 'evaluate'):
        c = sub.add_parser(name)
        c.add_argument('--python', type=Path, required=True)
        c.add_argument('--output', type=Path, required=True)
        if name == 'export':
            c.add_argument('--manifest', type=Path, required=True)
            c.add_argument('--align-kick-onset', action='store_true',
                           help='Keep each demonstrated kick initiation on the sampling grid')
        else:
            c.add_argument('--dataset', type=Path, required=True)
        if name == 'initialize':
            c.add_argument('--full-camera-frame', action='store_true',
                           help='Retain the complete camera image in the saved GR00T processors')
            c.add_argument('--action-execution-steps', type=int,
                           help='Save the default number of actions before replanning (1–16)')
        if name in ('train', 'evaluate'):
            c.add_argument('--checkpoint', type=Path, required=True)
        if name == 'train':
            c.add_argument('--validation-dataset', type=Path, required=True)
            c.add_argument('--initialization', choices=['official-base', 'checkpoint'], default='official-base')
            c.add_argument('--training-scope', choices=['projectors', 'action-head'], default='projectors',
                           help='Train the robot adapters, or additionally fine-tune the action model')
            c.add_argument('--steps', type=int, default=3000)
            c.add_argument('--batch-size', type=int, default=16)
            c.add_argument('--seed', type=int, required=True)
        if name == 'evaluate':
            c.add_argument('--samples', type=int, default=128)
    a = p.parse_args()
    if a.command in ('collect', 'rollout'):
        if a.episodes < 1 or not 0 < a.seconds <= 180:
            p.error('Use positive episode count and a duration of at most 180 seconds')
        episodes(a) if a.command == 'collect' else rollout(a)
    elif a.command == 'export':
        run_module(a.python, 'walking_data', ['--manifest', a.manifest, '--output', a.output,
                   *(['--align-kick-onset'] if a.align_kick_onset else [])])
    elif a.command == 'initialize':
        run_module(a.python, 'initialize', ['--dataset', a.dataset, '--output', a.output,
                   *(['--full-camera-frame'] if a.full_camera_frame else []),
                   *(['--action-execution-steps', a.action_execution_steps]
                     if a.action_execution_steps is not None else [])])
    elif a.command == 'evaluate':
        run_module(a.python, 'evaluate', ['--checkpoint', a.checkpoint, '--dataset', a.dataset,
                                        '--output', a.output, '--samples', a.samples])
    else:
        run_module(a.python, 'train_cached', ['--checkpoint', a.checkpoint, '--dataset', a.dataset,
            '--validation-dataset', a.validation_dataset, '--output', a.output,
            '--initialization', a.initialization, '--steps', a.steps, '--batch-size', a.batch_size,
            '--seed', a.seed, '--balanced-action-groups', '--training-scope', a.training_scope,
            '--head-dropout', 'disabled', '--learning-rate', '0.0001'])


if __name__ == '__main__':
    main()
