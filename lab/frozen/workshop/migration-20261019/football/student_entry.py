#!/usr/bin/env python3
"""Owned student run of the current sensor-only navigation development bundle.

This runs a fresh Newton episode, not a teacher recording. It deliberately does
not enable an unvalidated kick or grant whole-task qualification. The live
consumer is restricted to this job's fixed football-runtime directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

F=Path(__file__).resolve().parent
M=F.parent
C=M.parents[1]
O=C/'outputs/migration-20261019'
L=M/'runtime/IsaacLab-5088b326c7fc81caa7899395b0176ad08b89e9e1'
MODEL=O/'gait-precision-v5-1900/policy.onnx'
MODEL_SHA='3abbf37f2ebc25aaa2cfdf0a0244ff7b9edcce9ba0dd017836bf0aaa0b4cf684'
RECIPE=O/'gait-precision-v5-from1800-1024x1000-seed970201'
DETECTOR=O/'beta-detector/train-random-001/deployment.json'
DETECTOR_SHA='c40b2e743d08e9ab370e0b2d4d4817066eb1e54c104f699895eb9770a25cc99f'
METRICS={
    'taskSeconds':('本次无支撑任务时间（秒）','Unsupported task duration (s)'),
    'cameraFrames':('本次同步画面对数','Paired frames in this run'),
    'fell':('本次是否跌倒','Fall observed'),
    'footBallContact':('实际踢球接触','Actual kick contact'),
    'wholeBallGoal':('整个球通过球门','Whole ball passed the goal'),
    'taskSuccess':('本次完整射门成功','Whole-task success'),
    'activeLimitPeakRad':('主动关节越界峰值（弧度）','Active joint-limit excess (rad)'),
    'outputLimitPeakRad':('实体输出越界峰值（弧度）','Physical output-limit excess (rad)'),
    'elapsedWallSeconds':('本次实际用时（秒）','Elapsed wall time (s)'),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,value):
    temporary=path.with_name('.'+path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def build_command(destination,seconds=20):
    if seconds not in (5,20):raise ValueError('Only declared5s smoke or20s development episodes')
    return [str(L/'.venv/bin/python'),str(F/'isaac_execute.py'),'--development',
        '--model',str(MODEL),'--trained-recipe-dir',str(RECIPE),'--output',str(destination),
        '--seconds',str(seconds),'--seed','978701','--layout','empty','--root-y','0','--ball-xy','0','0',
        '--policy-imu','body50','--body-added-delay-s','0','--navigation-ground','tof-virtual-floor',
        '--detector-deployment',str(DETECTOR),'--detector-backend','torch','--detector-device','cuda:0',
        '--render-subframes','1','--shared-imu-readback','--packed-physics-readback','--body-fusion-tau-s','2']


def collect_artifacts(runtime,job):
    artifacts=[]
    items=[('manifest.json','navigation-report.json','本次物理与失败记录','Current physics and failure record','json'),
           ('first-person.mp4','first-person.mp4','本次第一视角','This run: first person','video'),
           ('overhead.mp4','overhead.mp4','本次全局视角','This run: overhead','video'),
           ('paired-views.mp4','paired-views.mp4','本次同步双视角','This run: paired views','video')]
    for source,filename,zh,en,kind in items:
        if not(runtime/source).is_file():continue
        target=job/filename
        if target.exists():raise FileExistsError('Retain each previous attempt')
        shutil.copy2(runtime/source,target)
        artifacts.append({'file':filename,'label':{'zh':zh,'en':en},'kind':kind,'sha256':sha(target)})
    return artifacts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='Explicit5s live transport check; no task success claim')
    args=parser.parse_args();result=args.output.resolve();job=result.parent
    if result.exists()or(job/'football-runtime').exists():raise FileExistsError('Use a new owned student job')
    if sha(MODEL)!=MODEL_SHA or sha(DETECTOR)!=DETECTOR_SHA:raise ValueError('Frozen development model bundle changed')
    job.mkdir(parents=True,exist_ok=True);runtime=job/'football-runtime';started=time.monotonic()
    report={'schema':'microduck.student-navigation-development.v1','runtime_id':'newton-bam-20261019',
        'hardware_profile_id':'beta-gc2093-l8cx-v1','status':'running','physical_episode':True,
        'qualification_claimed':False,'teacher_video_replayed':False,'student_training_performed':False,
        'gait_model_sha256':MODEL_SHA,'detector_deployment_sha256':DETECTOR_SHA,
        'sensorOutput':'football-runtime','kick_enabled':False,'smoke':args.smoke,
        'virtual_body_fusion':{'tau_s':2.,'hardware_calibrated':False,'scope':'Explicit development filter sensitivity; not a measured ST SFLP setting'},
        'metricLabels':{k:{'zh':v[0],'en':v[1]}for k,v in METRICS.items()},
        'summary':{'zh':'当前学生会话正在运行真实传感器导航开发回合；尚未开放踢球，结果不作为正式任务验收。',
                   'en':'This student session is running a fresh sensor-only navigation development episode. Kick remains disabled; this is not formal task qualification.'}}
    env=dict(os.environ);env['OMNI_KIT_ACCEPT_EULA']='YES';env['LD_PRELOAD']='/lib/aarch64-linux-gnu/libgomp.so.1'
    env['PATH']=str(M/'runtime/tools/bin')+os.pathsep+env.get('PATH','')
    argv=build_command(runtime,5 if args.smoke else 20);report['command']=argv;atomic(result,report)
    try:
        subprocess.run(argv,check=True,env=env,cwd=job)
        manifest=json.loads((runtime/'manifest.json').read_text())
        if manifest['source_changed_during_run']:raise ValueError('Runtime sources changed during the student episode')
        if manifest['model_sha256']!=MODEL_SHA or manifest['qualification_claimed']is not False:raise ValueError('Runtime identity or claim changed')
        scorer=manifest['scorer'];report['status']='completed';report['terminal_reason']=manifest['terminal_reason']
        report['metrics']={'taskSeconds':manifest['task_seconds'],'cameraFrames':manifest['camera_frames'],
            'fell':scorer['fell'],'footBallContact':scorer['real_foot_ball_contact'],
            'wholeBallGoal':scorer['whole_ball_crossed_inside_goal'],'taskSuccess':scorer['physical_episode_success'],
            'activeLimitPeakRad':scorer['servo_coordinate_peak_violation_rad'],
            'outputLimitPeakRad':scorer['physical_output_peak_violation_rad']}
        report['artifacts']=collect_artifacts(runtime,job)
        report['summary']={
            'zh':('本次真实导航开发回合发生跌倒，失败记录和同步双视角已保留。'if scorer['fell']else'本次真实导航开发回合运行结束，同步双视角和传感器记录已保留。')+'当前踢球未开放，本次不代表完整射门通过。',
            'en':('This fresh navigation episode fell; its failure and paired views are retained. 'if scorer['fell']else'This fresh navigation episode ended; paired views and sensor records are retained. ')+'Kick is disabled and this does not pass the full football task.'}
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        report.setdefault('metrics',{})['elapsedWallSeconds']=time.monotonic()-started
        atomic(result,report)
        print(json.dumps({'status':report['status'],'output':str(result),'qualification_claimed':False}),flush=True)


if __name__=='__main__':main()
