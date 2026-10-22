#!/usr/bin/env python3
"""Build an isolated detector overlay; never write the source Isaac environment.

--install resolves detector packages inside the new environment while sharing
the source CUDA torch/torchvision read-only via a .pth path. Versions are pinned
to the source runtime; the bootstrap does not import torch or allocate a GPU.
"""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-python',type=Path,required=True);p.add_argument('--venv',type=Path,required=True);p.add_argument('--install',action='store_true');a=p.parse_args()
    probe="import sys,sysconfig,json,importlib.metadata as m;print(json.dumps({'python':sys.executable,'site':sysconfig.get_path('purelib'),'versions':{k:m.version(k) for k in ['torch','torchvision','numpy']}}))"
    source=json.loads(subprocess.check_output([str(a.source_python),'-c',probe],text=True))
    if a.venv.exists():raise FileExistsError('Use a new isolated detector environment')
    subprocess.run([str(a.source_python),'-m','venv',str(a.venv)],check=True)
    python=a.venv/'bin/python';site=Path(subprocess.check_output([str(python),'-c',"import sysconfig;print(sysconfig.get_path('purelib'))"],text=True).strip())
    (site/'microduck-readonly-torch-runtime.pth').write_text(source['site']+'\n')
    constraints=a.venv/'torch-runtime-constraints.txt';constraints.write_text(''.join(k+'=='+v+'\n'for k,v in source['versions'].items()))
    report={'schema':'microduck.beta-detector-environment.v1','source_runtime':source,'source_environment_modified':False,'source_packages_read_only':True,'overlay':str(a.venv.resolve()),'ultralytics_version':'8.4.182','installed':False,'GPU_imported':False}
    if a.install:
        # Deliberately omit the broad[export] extra: its ARM64 torch constraint
        # could replace the pinned Spark stack. Explicit ONNX packages suffice.
        subprocess.run([str(python),'-m','pip','install','--constraint',str(constraints),'ultralytics==8.4.182','onnx>=1.20','onnxruntime>=1.20','opencv-python==5.0.0.93'],check=True)
        report['installed']=True
        (a.venv/'requirements-resolved.txt').write_text(subprocess.check_output([str(python),'-m','pip','freeze'],text=True))
    (a.venv/'environment-lineage.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
