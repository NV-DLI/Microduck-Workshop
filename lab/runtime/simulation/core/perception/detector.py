"""Production-facing detector: RGB in, original-pixel boxes out. No labels API."""
import hashlib,json
from pathlib import Path
import numpy as np

class BetaDetector:
    def __init__(self,deployment,*,device='cpu',allow_smoke=False,backend='onnx'):
        d=json.loads(Path(deployment).read_text())
        if d['schema']!='microduck.beta-detector-deployment.v1'or d['runtime_id']!='newton-bam-20261019'or d['hardware_profile_id']!='beta-gc2093-l8cx-v1':raise ValueError('Wrong detector runtime/profile')
        if d.get('smoke_only')and not allow_smoke:raise ValueError('Smoke detector is not a deployment model')
        weights=Path(d['weights'])
        if hashlib.sha256(weights.read_bytes()).hexdigest()!=d['weights_sha256']:raise ValueError('Detector checkpoint hash mismatch')
        if backend not in ('onnx','torch'):raise ValueError('Unknown detector backend')
        artifact=Path(d['onnx'])if backend=='onnx'else weights
        if backend=='onnx'and hashlib.sha256(artifact.read_bytes()).hexdigest()!=d['onnx_sha256']:raise ValueError('ONNX hash mismatch')
        from ultralytics import YOLO
        self.model=YOLO(str(artifact),task='detect');self.device=device;self.manifest=d;self.backend=backend
        if self.model.names!={0:'ball',1:'goal'}:raise ValueError('Class order must be ball,goal')
    def detect(self,rgb):
        a=np.asarray(rgb)
        if a.dtype!=np.uint8 or a.shape!=(540,960,3):raise ValueError('Actual Beta RGB960x540 required')
        # Ultralytics numpy source expects BGR, but public sensor contract is RGB.
        result=self.model.predict(source=a[:,:,::-1].copy(),imgsz=640,rect=False,conf=.45,device=self.device,verbose=False)[0]
        rows=[]
        for box,confidence,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()):
            box[0]=max(0.,min(959.,box[0]));box[2]=max(0.,min(959.,box[2]));box[1]=max(0.,min(539.,box[1]));box[3]=max(0.,min(539.,box[3]))
            if box[2]>box[0]and box[3]>box[1]:rows.append({'class_name':self.model.names[int(cls)],'confidence':float(confidence),'bbox_xyxy_px':box})
        return rows
