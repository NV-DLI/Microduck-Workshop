#!/usr/bin/env python3
"""Join complete, hashed RTX chunks; reject missing views and split leakage."""
import argparse,hashlib,json,shutil
from collections import Counter
from pathlib import Path
from plan_dataset import validate_plan

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def inspect(plan_path,chunks,*,development=False):
    plan=json.loads(plan_path.read_text());validate_plan(plan);expected={r['id']:r for r in plan['rows']};seen={};hashes=set();counts=Counter();sources=[]
    for folder in chunks:
        manifest=folder/'manifest.json';m=json.loads(manifest.read_text())
        if m['schema']!='microduck.beta-detector-rtx-chunk.v1' or m['status']!='complete' or m['plan_sha256']!=digest(plan_path):raise ValueError('Chunk incomplete or from another plan')
        if m['runtime_id']!=plan['runtime_id']or m['hardware_profile_id']!=plan['hardware_profile_id']:raise ValueError('Mixed runtime/profile')
        sources.append({'path':str(manifest.resolve()),'sha256':digest(manifest)})
        for row in m['images']:
            key=row['id']
            if key in seen or key not in expected:raise ValueError('Duplicate or unexpected image id')
            request=expected[key]
            if not row.get('actual_rtx')or any(row[k]!=request[k]for k in ('split','scene_seed','view_seed')):raise ValueError('Image/split identity mismatch')
            for kind in ('image','label','teacher'):
                path=(folder/row[kind]).resolve()
                if not path.is_relative_to(folder.resolve())or not path.is_file()or digest(path)!=row[kind+'_sha256']:raise ValueError('Missing file or fake '+kind+' hash')
            if row['image_sha256']in hashes:raise ValueError('Duplicate image bytes: possible frozen RTX camera or split leakage')
            hashes.add(row['image_sha256'])
            for line in (folder/row['label']).read_text().splitlines():
                v=list(map(float,line.split()))
                if len(v)!=5 or v[0]not in (0.,1.)or any(not 0<=x<=1 for x in v[1:])or v[3]<=0 or v[4]<=0:raise ValueError('Invalid YOLO label')
                counts[(row['split'],int(v[0]))]+=1
            seen[key]=(folder,row)
    if set(seen)!=set(expected):raise ValueError(f'Missing planned views: {len(set(expected)-set(seen))}; failed/incomplete frames cannot disappear')
    if not development:
        if len(seen)<300:raise ValueError('Full collection requires at least300 real frames')
        for split,minimum in [('train',20),('val',8),('test',8)]:
            for cls in (0,1):
                if counts[(split,cls)]<minimum:raise ValueError('Insufficient per-class examples in '+split)
    return plan,seen,sources,{f'{s}/{c}':n for(s,c),n in counts.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--chunks',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--development',action='store_true');a=p.parse_args()
    plan,rows,sources,counts=inspect(a.plan,a.chunks,development=a.development)
    if a.output.exists():raise FileExistsError('Preserve assembled dataset')
    a.output.mkdir(parents=True);records=[]
    for key,(folder,row)in rows.items():
        out=dict(row)
        for kind,target in [('image',f'images/{row["split"]}/{key}.png'),('label',f'labels/{row["split"]}/{key}.txt'),('teacher',f'teacher/{key}.json')]:
            path=a.output/target;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(folder/row[kind],path);out[kind]=target
        records.append(out)
    shutil.copy2(a.plan,a.output/'collection-plan.json')
    yaml='path: '+json.dumps(str(a.output.resolve()))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: ball\n  1: goal\n'
    (a.output/'dataset.yaml').write_text(yaml)
    manifest={'schema':'microduck.beta-detector-dataset.v1','runtime_id':plan['runtime_id'],'hardware_profile_id':plan['hardware_profile_id'],'development_only':a.development,'source':'actual_newton_RTX_beta_native_landscape','raster':plan['raster'],'plan_sha256':digest(a.plan),'chunks':sources,'images':records,'class_counts':counts,'split_unit':plan['split_unit'],'truth_boundary':'teacher geometry/depth labels only; no labels accepted by deployed detector/controller','dataset_yaml_sha256':digest(a.output/'dataset.yaml')}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'images':len(records),'class_counts':counts,'development_only':a.development,'manifest':str(a.output/'manifest.json')}))

if __name__=='__main__':main()
