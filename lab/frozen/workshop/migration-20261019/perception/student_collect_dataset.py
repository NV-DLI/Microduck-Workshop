#!/usr/bin/env python3
"""Collect a student's complete new RTX dataset, optionally train on that data.

All four lens assumptions share scene-disjoint train/validation/test splits.
This is static-pose data collection: truth labels never enter robot control.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
M = ROOT.parent
L = M / 'runtime/IsaacLab-5088b326c7fc81caa7899395b0176ad08b89e9e1'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_identity():
    paths = [ROOT / name for name in (
        'student_collect_dataset.py', 'plan_dataset.py', 'collect_isaac.py',
        'assemble_dataset.py', 'teacher_labels.py', 'train_detector.py',
        'validate_export.py', 'student_entry.py',
    )]
    paths += [M / folder / name for folder, name in (
        ('football', 'isaac_execute.py'), ('football', 'head_adapter.py'),
        ('sensors', 'isaac_beta_sensor_rig.py'), ('sensors', 'beta_sensor_profile.py'),
        ('sensors', 'beta_head_imu.py'), ('sensors', 'beta_mounts.py'),
        ('sensors', 'capture_clock.py'),
    )]
    return {str(path.relative_to(M)): digest(path) for path in paths}


def previews(dataset, manifest, job):
    """Show actual captured pixels with their teacher-only boxes for label QA."""
    from PIL import Image, ImageDraw
    artifacts = []
    # Deterministic coverage of all splits and four lens batches, rather than
    # choosing the most flattering frames after inspecting detection scores.
    selected = []
    for split, count in (('train', 12), ('val', 4), ('test', 4)):
        candidates = sorted((r for r in manifest['images'] if r['split'] == split), key=lambda r: r['id'])
        selected.extend(candidates[round(i * (len(candidates) - 1) / (count - 1))] for i in range(count))
    for index, row in enumerate(selected):
        image = Image.open(dataset / row['image']).convert('RGB')
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for line in (dataset / row['label']).read_text().splitlines():
            cls, x, y, w, h = map(float, line.split())
            bounds = ((x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height)
            color = '#fb923c' if cls == 0 else '#22c55e'
            draw.rectangle(bounds, outline=color, width=3)
            draw.text((bounds[0]+4, max(2, bounds[1]-14)), 'ball' if cls == 0 else 'goal', fill=color, stroke_width=1, stroke_fill='#111827')
        filename = f'capture-label-{index:02d}.png'
        target = job / filename
        if target.exists():
            raise FileExistsError(target)
        image.save(target)
        artifacts.append({'file': filename, 'kind': 'image', 'sha256': digest(target),
                          'label': {'zh': f"{row['split']} · {row['horizontal_fov_deg']}° · {row['id']}",
                                    'en': f"{row['split']} · {row['horizontal_fov_deg']}° · {row['id']}"}})
    return artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=978501)
    parser.add_argument('--train-epochs', type=int, choices=(0, 2, 120), default=0)
    args = parser.parse_args()
    result = args.output.resolve()
    job = result.parent
    if result.exists():
        raise FileExistsError('Keep each student attempt; use a fresh output')
    job.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    report = {'schema': 'microduck.student-new-dataset.v1', 'status': 'running',
              'runtime_id': 'newton-bam-20261019', 'hardware_profile_id': 'beta-gc2093-l8cx-v1',
              'source_sha256': identity, 'seed': args.seed, 'stage': 'plan',
              'qualification_claimed': False, 'physical_episode': False,
              'robot_task_success_claimed': False, 'student_capture_performed': False,
              'training_performed': False, 'commands': [], 'artifacts': [],
              'metrics': {}, 'summary': {},
              'metricLabels': {
                  'images': {'zh': '本次新拍图像数', 'en': 'New images captured'},
                  'scenes': {'zh': '独立场景数', 'en': 'Independent scenes'},
                  'trainImages': {'zh': '训练图像数', 'en': 'Training images'},
                  'valImages': {'zh': '验证图像数', 'en': 'Validation images'},
                  'testImages': {'zh': '测试图像数', 'en': 'Test images'},
                  'epochs': {'zh': '本次训练轮数', 'en': 'Training epochs'},
                  'optimizerUpdates': {'zh': '参数更新次数', 'en': 'Optimizer updates'},
                  'mAP50': {'zh': '测试集 mAP50', 'en': 'Test mAP50'},
                  'mAP50to95': {'zh': '测试集 mAP50–95', 'en': 'Test mAP50–95'},
              }}
    started = time.monotonic()
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1')
    env['PATH'] = str(M/'runtime/tools/bin') + os.pathsep + env.get('PATH', '')
    python = str(L/'.venv/bin/python')
    detector_python = str(ROOT/'.venv/bin/python')

    def save():
        report['wall_seconds'] = time.monotonic() - started
        tmp = result.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        tmp.replace(result)

    def run(stage, argv):
        if source_identity() != identity:
            raise RuntimeError('A source changed during this attempt; retain and rerun with frozen sources')
        report['stage'] = stage
        report['commands'].append(argv)
        save()
        print(json.dumps({'stage': stage, 'output': str(result)}), flush=True)
        subprocess.run(argv, check=True, env=env, cwd=job)

    try:
        plan = job/'collection-plan.json'
        run('plan', [python, str(ROOT/'plan_dataset.py'), '--output', str(plan), '--seed', str(args.seed)])
        chunks = []
        for hfov in (60, 70, 80, 90):
            destination = job/f'capture-fov-{hfov}'
            run(f'capture-{hfov}', [python, str(ROOT/'collect_isaac.py'), '--plan', str(plan),
                '--output', str(destination), '--hfov', str(hfov), '--render-subframes', '3', '--device', 'cuda:0'])
            chunks.append(destination)
        dataset = job/'captured-dataset'
        run('assemble', [python, str(ROOT/'assemble_dataset.py'), '--plan', str(plan),
                        '--chunks', *map(str, chunks), '--output', str(dataset)])
        manifest = json.loads((dataset/'manifest.json').read_text())
        counts = Counter(row['split'] for row in manifest['images'])
        scenes = len({row['scene_seed'] for row in manifest['images']})
        if len(manifest['images']) != 384 or counts != {'train':256, 'val':64, 'test':64} or scenes != 96:
            raise ValueError('The complete student dataset contract was not satisfied')
        report.update(student_capture_performed=True, dataset=str(dataset),
                      dataset_manifest_sha256=digest(dataset/'manifest.json'),
                      dataset_source='new real RTX capture in this student attempt',
                      optics_scope='60/70/80/90 degree hypotheses; not a measured hardware calibration',
                      label_scope='teacher-only geometry/depth labels; never controller input',
                      metrics={'images':384, 'scenes':96, 'trainImages':256, 'valImages':64, 'testImages':64})
        report['artifacts'] = previews(dataset, manifest, job)
        target = job/'dataset-manifest.json'
        shutil.copy2(dataset/'manifest.json', target)
        report['artifacts'].append({'file':target.name, 'kind':'json', 'sha256':digest(target),
            'label':{'zh':'本次新采数据与划分清单', 'en':'New capture and split manifest'}})
        report['summary'] = {'zh':'已重新拍摄384张实际RTX图像，按96个场景分组；页面展示20张带标注样本。',
                             'en':'Captured384 new actual RTX images across96 disjoint scenes, with20 labeled samples shown here.'}
        if args.train_epochs:
            training = job/'detector-training'
            argv = [detector_python, str(ROOT/'train_detector.py'), '--dataset', str(dataset),
                    '--output', str(training), '--epochs', str(args.train_epochs), '--batch', '16',
                    '--seed', str(args.seed+1), '--device', '0']
            if args.train_epochs == 2:
                argv.append('--smoke')
            run('train-own-data', argv)
            parity = job/'export-parity'
            argv = [detector_python, str(ROOT/'validate_export.py'), '--deployment', str(training/'deployment.json'),
                    '--dataset', str(dataset), '--output', str(parity), '--images', '8']
            if args.train_epochs == 2:
                argv.append('--allow-smoke')
            run('export-parity', argv)
            from student_entry import detector_artifacts
            report['artifacts'].extend(detector_artifacts(training, parity, job))
            tr = json.loads((training/'training-report.json').read_text())
            report.update(training_performed=True, student_weights_trained_from='random initialization on this attempt’s new captures',
                          smoke=args.train_epochs==2, training_report=str(training/'training-report.json'),
                          training_report_sha256=digest(training/'training-report.json'),
                          deployment=str(training/'deployment.json'), deployment_sha256=digest(training/'deployment.json'))
            report['metrics'].update(epochs=tr['epochs_completed'], optimizerUpdates=tr['optimizer_updates'],
                mAP50=tr['heldout_test_metrics'].get('metrics/mAP50(B)'), mAP50to95=tr['heldout_test_metrics'].get('metrics/mAP50-95(B)'))
            report['summary'] = {
                'zh': ('已用本次384张新采图像完成2轮随机短训与导出检查；这是流程验证，尚不代表检测准确。' if args.train_epochs==2 else
                       '已用本次384张新采图像从随机权重训练，完成独立测试与导出检查；检测成绩不等于射门成功率。'),
                'en': ('Completed2random-weight epochs and export checks on your384new captures: pipeline validation only.' if args.train_epochs==2 else
                       'Trained random weights on your384new captures, evaluated held-out scenes and checked export. Detection scores are not football success rates.')}
        if source_identity() != identity:
            raise RuntimeError('Sources changed during this attempt; no acceptance claim')
        report.update(status='passed', stage='complete', sources_unchanged=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error), sources_unchanged=source_identity()==identity)
        raise
    finally:
        save()
        print(json.dumps({'status':report['status'], 'output':str(result), 'qualification_claimed':False}), flush=True)


if __name__ == '__main__':
    main()
