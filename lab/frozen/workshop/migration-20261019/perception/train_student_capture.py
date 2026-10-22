#!/usr/bin/env python3
"""Train new weights on the latest completed capture in this student workspace."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
M=ROOT.parent
sys.path.insert(0,str(M/'bridge'))
from runtime_identity import STUDENT,RUNTIME_ID,PROFILE_ID


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_identity():
    names=['train_student_capture.py','train_detector.py','validate_export.py','student_entry.py']
    return {str((ROOT/name).relative_to(M)):sha(ROOT/name) for name in names}


def prepare_own_dataset(source, output):
    from train_detector import validate_dataset
    source=Path(source);output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh destination')
    manifest=validate_dataset(source)
    if manifest.get('development_only') is not False or len(manifest['images'])!=384:
        raise ValueError('Expected the complete student collection')
    output.mkdir(parents=True)
    for row in manifest['images']:
        for kind in ('image','label'):
            target=output/row[kind];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/row[kind],target)
    if (source/'collection-plan.json').is_file():
        shutil.copy2(source/'collection-plan.json',output/'collection-plan.json')
    (output/'dataset.yaml').write_text('path: '+json.dumps(str(output))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ball\n  1: goal\n')
    manifest['dataset_yaml_sha256']=sha(output/'dataset.yaml')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validate_dataset(output)
    return manifest


def completed_capture(job_id=None):
    jobs=STUDENT/'jobs'
    if job_id is not None and not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}',job_id):
        raise ValueError('Invalid student capture identifier')
    folders=[jobs/job_id]if job_id else list(jobs.glob('*'))
    valid=[]
    for folder in folders:
        try:
            if folder.is_symlink()or folder.resolve().parent!=jobs.resolve():continue
            owner=json.loads((folder/'supervisor.json').read_text())
            if owner.get('action')!='capture-sensors'or owner.get('state')!='completed'or owner.get('exitCode')!=0 or owner.get('processExitCode')!=0:continue
            if owner.get('runtimeId')!=RUNTIME_ID or owner.get('profileId')!=PROFILE_ID:continue
            report_path=folder/'course-result.json';report=json.loads(report_path.read_text())
            if report.get('schema')!='microduck.student-new-dataset.v1'or report.get('status')!='passed'or report.get('student_capture_performed')is not True or report.get('sources_unchanged')is not True:continue
            if report.get('runtime_id')!=RUNTIME_ID or report.get('hardware_profile_id')!=PROFILE_ID:continue
            dataset=folder/'captured-dataset'
            if dataset.is_symlink()or dataset.resolve().parent!=folder.resolve()or Path(report['dataset']).resolve()!=dataset.resolve():continue
            manifest=dataset/'manifest.json'
            if sha(manifest)!=report['dataset_manifest_sha256']:continue
            data=json.loads(manifest.read_text())
            if data.get('development_only')is not False or len(data['images'])!=384:continue
            valid.append((owner.get('finishedAt',0),{'job_id':folder.name,'dataset':str(dataset.resolve()),
                         'dataset_manifest_sha256':sha(manifest),'capture_report_sha256':sha(report_path)}))
        except(OSError,KeyError,ValueError,TypeError):continue
    if not valid:
        raise ValueError('先完成本版384张学生采集；不会借用教师数据。 / Complete your 384-image capture first; teacher data is never substituted.')
    return max(valid,key=lambda item:item[0])[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-job');p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=978601)
    a=p.parse_args();result=a.output.resolve();job=result.parent
    if result.exists():raise FileExistsError('Preserve each training attempt')
    job.mkdir(parents=True,exist_ok=True);started=time.monotonic();identity=source_identity();report={'schema':'microduck.train-own-capture.v1','status':'running',
        'runtime_id':RUNTIME_ID,'hardware_profile_id':PROFILE_ID,'qualification_claimed':False,'physical_episode':False,
        'robot_task_success_claimed':False,'student_capture_performed':False,'training_performed':False,
        'commands':[],'summary':{},'metrics':{},'artifacts':[],'source_sha256':identity}
    env=dict(os.environ,OMNI_KIT_ACCEPT_EULA='YES',LD_PRELOAD='/lib/aarch64-linux-gnu/libgomp.so.1')
    python=str(ROOT/'.venv/bin/python')
    def save():
        tmp=result.with_suffix('.tmp');tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(result)
    def run(argv):
        if source_identity()!=identity:raise ValueError('Training source changed during this attempt')
        report['commands'].append(argv);save();subprocess.run(argv,check=True,cwd=job,env=env)
    try:
        source=completed_capture(a.source_job);dataset=job/'own-captured-dataset';training=job/'detector-training';parity=job/'export-parity'
        report.update(source_capture=source,initialization='random weights; no existing detector weights',source_kind='own completed current-release student capture')
        manifest=prepare_own_dataset(source['dataset'],dataset)
        manifest['student_dataset_provenance']={'mode':'own-earlier-student-RTX-capture','source_capture':source,
            'student_capture_performed_in_this_training_job':False,'student_capture_performed_in_source_job':True,
            'source_detector_weights_copied':False}
        (dataset/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (dataset/'student-preparation.json').write_text(json.dumps(manifest['student_dataset_provenance'],indent=2)+'\n')
        argv=[python,str(ROOT/'train_detector.py'),'--dataset',str(dataset),'--output',str(training),'--epochs','120','--batch','16','--seed',str(a.seed),'--device','0']
        if a.smoke:argv.append('--smoke')
        run(argv)
        argv=[python,str(ROOT/'validate_export.py'),'--deployment',str(training/'deployment.json'),'--dataset',str(dataset),'--output',str(parity),'--images','8']
        if a.smoke:argv.append('--allow-smoke')
        run(argv)
        from student_entry import detector_artifacts,METRIC_LABELS
        tr=json.loads((training/'training-report.json').read_text());report.update(status='passed',training_performed=True,
            smoke=a.smoke,training_report=str(training/'training-report.json'),deployment=str(training/'deployment.json'),
            training_report_sha256=sha(training/'training-report.json'),deployment_sha256=sha(training/'deployment.json'),
            artifacts=detector_artifacts(training,parity,job),metricLabels={**METRIC_LABELS,'sourceCapture':{'zh':'数据采集任务','en':'Source capture job'}},
            metrics={'sourceCapture':source['job_id'],'epochs':tr['epochs_completed'],'optimizerUpdates':tr['optimizer_updates'],
                     'trainingWallSeconds':tr['training_wall_seconds'],'mAP50':tr['heldout_test_metrics'].get('metrics/mAP50(B)'),
                     'mAP50to95':tr['heldout_test_metrics'].get('metrics/mAP50-95(B)')},
            summary={'zh':('已用自己采集的图像完成2轮流程检查，未声明检测准确。'if a.smoke else'已用自己采集的图像训练新权重，独立测试并检查导出；检测成绩不是射门成功率。'),
                     'en':('Completed a two-epoch pipeline check on your own images; detector accuracy is not established.'if a.smoke else'Trained new weights on your own captures, evaluated held-out scenes and checked export. Detection scores are not goal-success rates.')})
        if sha(Path(source['dataset'])/'manifest.json')!=source['dataset_manifest_sha256']:raise ValueError('Source dataset changed during training')
        report['sources_unchanged']=source_identity()==identity
        if not report['sources_unchanged']:raise ValueError('Training source changed during this attempt')
    except BaseException as error:report.update(status='failed',error=repr(error));raise
    finally:
        report['wall_seconds']=time.monotonic()-started;save();print(json.dumps({'status':report['status'],'output':str(result)}),flush=True)


if __name__=='__main__':main()
