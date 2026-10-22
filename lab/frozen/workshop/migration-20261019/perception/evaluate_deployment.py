#!/usr/bin/env python3
"""Independent full test split audit at the actual controller threshold .45.

No weight/threshold selection occurs here. This is simulated detection evidence,
not a robot football success rate or a physical camera/RK3566 validation.
"""
import argparse,hashlib,json,math,os,platform,time
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['YOLO_AUTOINSTALL']='false'
from validate_export import iou

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def wilson(success,n):
 if not n:return None
 z=1.959963984540054;p=success/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;r=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
 return [c-r,c+r]
def score(detections,labels):
 out={name:{'tp':0,'fp':0,'fn':0,'matched_iou':[]}for name in ('ball','goal')}
 for name in out:
  truth=[r['bbox_xyxy_px']for r in labels if r['class_name']==name];unmatched=set(range(len(truth)))
  for d in sorted((r for r in detections if r['class_name']==name),key=lambda r:-r['confidence']):
   j=max(unmatched,key=lambda j:iou(d['bbox_xyxy_px'],truth[j]))if unmatched else None
   overlap=0 if j is None else iou(d['bbox_xyxy_px'],truth[j])
   if overlap>=.5:unmatched.remove(j);out[name]['tp']+=1;out[name]['matched_iou'].append(overlap)
   else:out[name]['fp']+=1
  out[name]['fn']=len(unmatched)
 return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deployment',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Preserve independent test attempts')
 from train_detector import validate_dataset
 from detector import BetaDetector
 import cv2,numpy as np,torch
 dataset=validate_dataset(a.dataset);deployment=json.loads(a.deployment.read_text())
 if deployment['training_dataset_manifest_sha256']!=sha(a.dataset/'manifest.json'):raise ValueError('Exact held-out training dataset required')
 model=BetaDetector(a.deployment,backend='onnx');rows=sorted((r for r in dataset['images']if r['split']=='test'),key=lambda r:r['id'])
 if len(rows)<32:raise ValueError('Full test split expected')
 a.output.mkdir(parents=True);report={'schema':'microduck.beta-detector-independent-test.v1','runtime_id':dataset['runtime_id'],'hardware_profile_id':dataset['hardware_profile_id'],'deployment_sha256':sha(a.deployment),'dataset_manifest_sha256':sha(a.dataset/'manifest.json'),'weights_sha256':deployment['weights_sha256'],'onnx_sha256':deployment['onnx_sha256'],'status':'running','backend':'onnx','device':'cpu','host':platform.node(),'confidence_threshold':.45,'match_iou_threshold':.5,'rows':[],'threshold_or_weights_selected_on_test':False,'qualification_claimed':False,'scope':'Held-out real RTX images only, all planned test scenes; no real-world camera or RK3566 claim','denominator':len(rows)}
 total={n:{'tp':0,'fp':0,'fn':0,'matched_iou':[]}for n in ('ball','goal')};times=[]
 try:
  for row in rows:
   bgr=cv2.imread(str(a.dataset/row['image']))
   if bgr is None or bgr.shape!=(540,960,3):raise ValueError('Actual RGB raster mismatch')
   start=time.perf_counter();detections=model.detect(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));latency=(time.perf_counter()-start)*1000;times.append(latency);labels=[]
   for line in (a.dataset/row['label']).read_text().splitlines():
    c,x,y,w,h=map(float,line.split());labels.append({'class_name':('ball','goal')[int(c)],'bbox_xyxy_px':[(x-w/2)*960,(y-h/2)*540,(x+w/2)*960,(y+h/2)*540]})
   counts=score(detections,labels)
   for name in total:
    for key in ('tp','fp','fn'):total[name][key]+=counts[name][key]
    total[name]['matched_iou']+=counts[name]['matched_iou']
   report['rows'].append({'id':row['id'],'scene_seed':row['scene_seed'],'image_sha256':row['image_sha256'],'detections':detections,'ground_truth_test_labels':labels,'counts':counts,'wall_latency_ms':latency})
  for name,c in total.items():
   p=c['tp']+c['fp'];r=c['tp']+c['fn'];c.update(precision=c['tp']/p if p else None,recall=c['tp']/r if r else None,precision_wilson95=wilson(c['tp'],p),recall_wilson95=wilson(c['tp'],r),mean_matched_iou=float(np.mean(c['matched_iou']))if c['matched_iou']else None)
  report.update(status='completed',class_metrics=total,all_test_images_retained=len(report['rows'])==len(rows),wall_latency_ms={'p50':float(np.median(times[1:])),'p95':float(np.quantile(times[1:],.95)),'cold_first':times[0],'samples':len(times)-1},cuda_initialized=torch.cuda.is_initialized())
  if report['cuda_initialized']:raise RuntimeError('CPU test unexpectedly initialized CUDA')
 except BaseException as error:report.update(status='failed',error=repr(error));raise
 finally:(a.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({'status':report['status'],'frames':len(rows),'class_metrics':total,'latency_ms':report['wall_latency_ms']},indent=2))
if __name__=='__main__':main()
