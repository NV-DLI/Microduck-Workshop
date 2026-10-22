#!/usr/bin/env python3
"""Compare the trained PyTorch and exported ONNX detector on held-out RGB.

This is a CPU-only deployment equivalence check, not a new detection-accuracy
estimate or an RK3566 benchmark. Both paths use fixed square640 preprocessing.
All tested images, differences and wall latencies are retained, including failures.
"""
import argparse,hashlib,json,os,platform,time
from pathlib import Path
if __name__=='__main__':os.environ['CUDA_VISIBLE_DEVICES']=''  # Importable comparison helpers must not hide another caller's CUDA devices.
os.environ.setdefault('YOLO_AUTOINSTALL','false')

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def iou(a,b):
    overlap=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=max(0.,a[2]-a[0])*max(0.,a[3]-a[1])+max(0.,b[2]-b[0])*max(0.,b[3]-b[1])-overlap
    return overlap/union if union else 0.

def compare(a,b):
    """Class-constrained greedy matching, retaining unmatched detections."""
    unmatched=set(range(len(b)));matches=[]
    for row in sorted(a,key=lambda x:-x['confidence']):
        parameterss=[j for j in unmatched if b[j]['class_id']==row['class_id']]
        best=max(parameterss,key=lambda j:iou(row['box'],b[j]['box']))if parameterss else None
        if best is None:matches.append({'matched':False,'torch':row});continue
        unmatched.remove(best);other=b[best];overlap=iou(row['box'],other['box']);delta=abs(row['confidence']-other['confidence'])
        matches.append({'matched':True,'iou':overlap,'confidence_delta':delta,'passed':overlap>=.98 and delta<=.001,'torch':row,'onnx':other})
    return {'passed':not unmatched and all(m.get('passed',False)for m in matches),'torch_count':len(a),'onnx_count':len(b),'pairs':matches,'unmatched_onnx':[b[j]for j in sorted(unmatched)]}

def boxes(result):
    return [{'class_id':int(c),'confidence':float(s),'box':b}for b,s,c in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist())]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deployment',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--images',type=int,default=16);p.add_argument('--allow-smoke',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve every validation attempt')
    if a.images<8:raise ValueError('At least8 actual test images required')
    from train_detector import validate_dataset,deployment_scope
    dataset=validate_dataset(a.dataset);deployment=json.loads(a.deployment.read_text())
    if deployment['schema']!='microduck.beta-detector-deployment.v1'or deployment['runtime_id']!=dataset['runtime_id']or deployment['hardware_profile_id']!=dataset['hardware_profile_id']:raise ValueError('Deployment/dataset mismatch')
    if digest(a.dataset/'manifest.json')!=deployment['training_dataset_manifest_sha256']:raise ValueError('Use the held-out split of the exact training dataset')
    if deployment['smoke_only']and not a.allow_smoke:raise ValueError('Smoke model requires explicit development override')
    for name in ('weights','onnx'):
        if digest(deployment[name])!=deployment[name+'_sha256']:raise ValueError('Model hash changed: '+name)
    parameterss=sorted((r for r in dataset['images']if r['split']=='test'),key=lambda r:r['id'])
    if len(parameterss)<a.images:raise ValueError('Too few actual held-out test images')
    rows=[parameterss[i*(len(parameterss)-1)//(a.images-1)]for i in range(a.images)]
    a.output.mkdir(parents=True);os.environ['YOLO_CONFIG_DIR']=str(a.output/'settings')
    import cv2,numpy as np,torch,onnxruntime
    from ultralytics import YOLO
    report={'schema':'microduck.beta-detector-export-validation.v1','runtime_id':dataset['runtime_id'],'hardware_profile_id':dataset['hardware_profile_id'],'deployment_sha256':digest(a.deployment),'dataset_manifest_sha256':digest(a.dataset/'manifest.json'),'smoke_only':deployment['smoke_only'],'status':'running','passed':False,'device':'cpu','host':platform.node(),'architecture':platform.machine(),'torch':torch.__version__,'onnxruntime':onnxruntime.__version__,'confidence_threshold':.45,'matching_thresholds':{'min_iou':.98,'max_confidence_difference':.001},'rows':[],'latency_scope':'This Spark CPU process, including preprocessing and postprocessing; not robot NPU throughput','deployment_scope':deployment_scope(),'qualification_claimed':False}
    models={'torch':YOLO(deployment['weights']),'onnx':YOLO(deployment['onnx'],task='detect')}
    try:
        first=cv2.imread(str(a.dataset/rows[0]['image']))
        for model in models.values():
            for _ in range(2):model.predict(first,imgsz=640,rect=False,conf=.45,device='cpu',verbose=False)
        times={k:[]for k in models}
        for row in rows:
            image=cv2.imread(str(a.dataset/row['image']))
            if image is None or image.shape!=(540,960,3):raise ValueError('Actual960x540 image required')
            outputs={};elapsed={}
            for key,model in models.items():
                start=time.perf_counter();outputs[key]=boxes(model.predict(image,imgsz=640,rect=False,conf=.45,device='cpu',verbose=False)[0]);elapsed[key]=(time.perf_counter()-start)*1000;times[key].append(elapsed[key])
            report['rows'].append({'id':row['id'],'scene_seed':row['scene_seed'],'image_sha256':row['image_sha256'],'wall_latency_ms':elapsed,**compare(outputs['torch'],outputs['onnx'])})
        report.update(status='completed',passed=all(r['passed']for r in report['rows']),latency_ms={k:{'p50':float(np.median(v)),'p95':float(np.quantile(v,.95)),'samples':len(v)}for k,v in times.items()},cuda_initialized=torch.cuda.is_initialized())
        if report['cuda_initialized']:raise RuntimeError('CPU equivalence check unexpectedly initialized CUDA')
    except BaseException as error:report.update(status='failed',error=repr(error),passed=False);raise
    finally:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':report['passed'],'frames':len(report['rows']),'latency_ms':report.get('latency_ms'),'report':str(a.output/'report.json')}))
    if not report['passed']:raise SystemExit(2)

if __name__=='__main__':main()
