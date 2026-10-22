"""Fixed long student baseline; no claim that the budget learns to walk."""
from pathlib import Path
import argparse,hashlib,json,runpy,sys
HERE=Path(__file__).resolve().parent

def plan(output):
 manifest=HERE/'student-baseline-v1.json';data=json.loads(manifest.read_text())
 if data['schema']!='microduck.student-baseline.v1' or data['recipe']!={'num_envs':1024,'iterations':2500,'seed':970311}:raise RuntimeError('Frozen student baseline recipe changed')
 for source in data['sourceFiles']:
  path=(HERE.parent/source['path']).resolve()
  if not path.is_relative_to(HERE.parent.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']:raise RuntimeError('Frozen baseline dependency changed')
 return data,[str(HERE/'student_train.py'),'--output',str(Path(output).resolve()),'--num-envs','1024','--iterations','2500','--seed','970311']

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--inspect',action='store_true');args=parser.parse_args()
 data,argv=plan(args.output)
 if args.inspect:print(json.dumps({'recipe':data,'trainer_argv':argv,'executionPerformed':False},indent=2));return
 # Reuse the frozen, actually smoke-tested trainer/export path. Never resume teacher weights.
 sys.argv=argv
 runpy.run_path(argv[0],run_name='__main__')
if __name__=='__main__':main()
