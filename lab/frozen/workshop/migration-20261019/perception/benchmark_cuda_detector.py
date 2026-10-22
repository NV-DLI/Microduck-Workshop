#!/usr/bin/env python3
"""Spark CUDA detector equivalence/latency, explicitly not RK3566 performance."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import cv2,numpy as np,torch
from detector import BetaDetector
from train_detector import validate_dataset
from validate_export import compare

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normalized(rows):return [{'class_id':('ball','goal').index(r['class_name']),'confidence':r['confidence'],'box':r['bbox_xyxy_px']}for r in rows]
def main():
 p=argparse.ArgumentParser();p.add_argument('--deployment',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Preserve every timing/equivalence trial')
 dataset=validate_dataset(a.dataset);d=json.loads(a.deployment.read_text())
 if sha(a.dataset/'manifest.json')!=d['training_dataset_manifest_sha256']:raise ValueError('Dataset hash mismatch')
 a.output.mkdir(parents=True);models={'cuda_torch':BetaDetector(a.deployment,backend='torch',device='cuda:0'),'cpu_onnx':BetaDetector(a.deployment,backend='onnx',device='cpu')};allrows=sorted((r for r in dataset['images']if r['split']=='test'),key=lambda r:r['id']);rows=[allrows[i*(len(allrows)-1)//15]for i in range(16)];report={'schema':'microduck.beta-detector-cuda-comparison.v1','status':'running','passed':False,'deployment_sha256':sha(a.deployment),'weights_sha256':d['weights_sha256'],'onnx_sha256':d['onnx_sha256'],'host':platform.node(),'gpu':torch.cuda.get_device_name(0),'shared_load':True,'scope':'Spark classroom shared-load wall inference, not RK3566 NPU or camera-to-action latency','rows':[]}
 try:
  first=cv2.cvtColor(cv2.imread(str(a.dataset/rows[0]['image'])),cv2.COLOR_BGR2RGB)
  for model in models.values():
   for _ in range(3):model.detect(first)
  for row in rows:
   rgb=cv2.cvtColor(cv2.imread(str(a.dataset/row['image'])),cv2.COLOR_BGR2RGB);result={};times={}
   for key,model in models.items():
    torch.cuda.synchronize();start=time.perf_counter();result[key]=normalized(model.detect(rgb));torch.cuda.synchronize();times[key]=(time.perf_counter()-start)*1000
   report['rows'].append({'id':row['id'],'image_sha256':row['image_sha256'],'wall_latency_ms':times,**compare(result['cuda_torch'],result['cpu_onnx'])})
  report.update(status='completed',passed=all(r['passed']for r in report['rows']),wall_latency_ms={k:{'p50':float(np.median([r['wall_latency_ms'][k]for r in report['rows']])),'p95':float(np.quantile([r['wall_latency_ms'][k]for r in report['rows']],.95))}for k in models})
 except BaseException as error:report.update(status='failed',error=repr(error));raise
 finally:(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:v for k,v in report.items()if k!='rows'},indent=2))
 if not report['passed']:raise SystemExit(2)
if __name__=='__main__':main()
