"""A small visual imitation baseline for the version-3 walking interface.

Frozen ImageNet ResNet18 spatial features, a four-frame GRU, proprioception,
and a 16x8 action head. No detector, object poses, expert phase, or ToF inputs.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

try:
    from .visual_validation import contract_check, verify_cache, verify_sampling, budget_check
except ImportError:
    from visual_validation import contract_check, verify_cache, verify_sampling, budget_check

IMAGE_KEYS = tuple(f'observation.images.frame{i:02d}' for i in range(4))

def read(path):
    return json.loads(Path(path).read_text())

def save(path, obj):
    with Path(path).open('x') as stream:
        json.dump(obj, stream, indent=2)

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

class SpatialEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.body = nn.Sequential(*list(model.children())[:-2])
        self.register_buffer('mean', torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std', torch.tensor([.229,.224,.225])[None,:,None,None])
        self.requires_grad_(False)
        self.eval()

    def forward(self, images):
        # Preserve the full field of view: no center crop that could remove ball/goal.
        x = F.interpolate(images, size=(192,320), mode='bilinear', align_corners=False, antialias=True)
        return F.adaptive_avg_pool2d(self.body((x-self.mean)/self.std), (3,5)).flatten(1)

class VisualPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(7680), nn.Linear(7680,256), nn.GELU())
        self.temporal = nn.GRU(256,256,batch_first=True)
        self.proprio = nn.Sequential(nn.Linear(127,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.head = nn.Sequential(nn.Linear(384,256),nn.GELU(),nn.Linear(256,16*8))

    def forward(self, images, states):
        _, hidden = self.temporal(self.project(images))
        return self.head(torch.cat([hidden[-1],self.proprio(states)],-1)).reshape(-1,16,8)

def validate_dataset(path, split):
    contract = read(path/'control-contract.json')
    if contract.get('version')!=3 or len(contract['state_names'])!=127:
        raise ValueError('Requires causal version-3 walking history data, not roller/58D data')
    contract_check(contract)
    records = read(path/'provenance.json')
    if read(path/'split.json')['split']!=split or any(r['split']!=split for r in records):
        raise ValueError('Episode split differs')
    if len({r['seed']for r in records})!=len(records):
        raise ValueError('Repeated episode seed')
    return contract, records

def cache(a):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    if a.output.exists(): raise FileExistsError(a.output)
    contract, records = validate_dataset(a.dataset, a.split)
    a.output.mkdir(parents=True)
    started=time.monotonic();torch.manual_seed(a.seed);torch.set_num_threads(4)
    encoder=SpatialEncoder(pretrained=a.encoder=='imagenet').to(a.device).eval()
    torch.save(encoder.state_dict(), a.output/'encoder.pt')
    data=LeRobotDataset('local/'+a.dataset.name,root=a.dataset,
         delta_timestamps={'action':[i/12.5 for i in range(16)]})
    n=len(data)
    feats=np.lib.format.open_memmap(a.output/'features.npy',mode='w+',dtype=np.float16,shape=(n,4,7680))
    states=np.zeros((n,127),np.float32);actions=np.zeros((n,16,8),np.float32)
    valid=np.zeros((n,16),bool);episodes=np.zeros(n,np.int64)
    # Cache exact repeated image content, never future observations or test data.
    memo={}
    with torch.inference_mode():
        for i in range(n):
            row=data[i];images=torch.stack([row[k]for k in IMAGE_KEYS]);keys=[];missing=[]
            for j,img in enumerate(images):
                key=hashlib.sha256(img.numpy().tobytes()).hexdigest();keys.append(key)
                if key not in memo:missing.append(j)
            if missing:
                values=encoder(images[missing].to(a.device)).float().cpu().numpy().astype(np.float16)
                for j,value in zip(missing,values):memo[keys[j]]=value
            feats[i]=np.stack([memo[key]for key in keys])
            states[i]=row['observation.state'].numpy();actions[i]=row['action'].numpy()
            valid[i]=~row['action_is_pad'].numpy();episodes[i]=int(row['episode_index'])
            if i%200==0:print(json.dumps(dict(cached=i,total=n)),flush=True)
    feats.flush()
    np.savez(a.output/'targets.npz',state=states,action=actions,valid=valid,episode=episodes)
    save(a.output/'control-contract.json',contract);save(a.output/'provenance.json',records)
    save(a.output/'completion.json',dict(complete=True,dataset=str(a.dataset.resolve()),split=a.split,
        rows=n,episodes=len(records),encoder=a.encoder,encoder_seed=a.seed,encoder_sha256=sha(a.output/'encoder.pt'),
        cache_seconds=time.monotonic()-started,image_size=[192,320],spatial_grid=[3,5],
        source_provenance_sha256=sha(a.dataset/'provenance.json'),source_code_sha256=sha(__file__)))

    save(a.output/'content-receipt.json',dict(sha256={name:sha(a.output/name)for name in ['features.npy','targets.npz','encoder.pt','control-contract.json','provenance.json','completion.json']}))

def train(a):
    if a.output.exists():raise FileExistsError(a.output)
    budget_check(a.steps,a.batch_size,a.learning_rate)
    verify_cache(a.train);verify_cache(a.validation)
    tr=read(a.train/'completion.json');va=read(a.validation/'completion.json')
    if not tr['complete']or not va['complete']or tr['split']!='train'or va['split']!='validation':raise ValueError('Complete separate caches required')
    # Encoders must be exactly equal (also for randomly initialized frozen control).
    if tr['encoder_sha256']!=va['encoder_sha256']:raise ValueError('Different image encoders')
    records=read(a.train/'provenance.json');vrecords=read(a.validation/'provenance.json')
    if {r['seed']for r in records}&{r['seed']for r in vrecords} or {r['source']for r in records}&{r['source']for r in vrecords}:raise ValueError('Train/validation episode leakage')
    if read(a.train/'control-contract.json')!=read(a.validation/'control-contract.json'):raise ValueError('Interface differs')
    order=np.random.default_rng(a.subset_seed).permutation(len(records))
    count=len(records)if a.episodes==0 else a.episodes
    if not 1<=count<=len(records):raise ValueError('Requested more demonstrations than available')
    selected=sorted(order[:count].tolist())
    target=np.load(a.train/'targets.npz');vtarget=np.load(a.validation/'targets.npz')
    ix=np.flatnonzero(np.isin(target['episode'],selected));vx=np.arange(len(vtarget['episode']))
    state=target['state'][ix];action=target['action'][ix]
    sm=state.mean(0);ss=state.std(0).clip(.01)
    # Only real action targets contribute to normalization, never padding or validation.
    real=action[target['valid'][ix]];am=real.mean(0);ascale=real.std(0).clip(.03)
    def tensors(path,t,ids):
        return (torch.from_numpy(np.asarray(np.load(path/'features.npy',mmap_mode='r')[ids],dtype=np.float32)),
                torch.from_numpy((t['state'][ids]-sm)/ss),torch.from_numpy((t['action'][ids]-am)/ascale),
                torch.from_numpy(t['valid'][ids]))
    x,s,y,mask=tensors(a.train,target,ix);vdata=tensors(a.validation,vtarget,vx)
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4)
    model=VisualPolicy().to(a.device);optim=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=1e-4)
    a.output.mkdir(parents=True);start=time.monotonic()
    if a.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    metrics=(a.output/'metrics.jsonl').open('x',buffering=1)
    # Same declared phase weights as GR00T can be supplied; weights never become input.
    weights=torch.ones(len(ix))
    if a.sampling:
        sampling=read(a.sampling)
        raw=verify_sampling(sampling,records,target['episode'])
        if len(raw)!=len(target['episode']):raise ValueError('Sampling rows differ')
        if sampling['training_seeds']!=[r['seed']for r in records]:raise ValueError('Sampling lineage differs')
        weights=torch.tensor(raw,dtype=torch.float)[ix]
    for update in range(1,a.steps+1):
        ids=torch.multinomial(weights,a.batch_size,replacement=True)
        pred=model(x[ids].to(a.device),s[ids].to(a.device));m=mask[ids].to(a.device)
        loss=((pred-y[ids].to(a.device)).square()*m[...,None]).sum()/(m.sum()*8)
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        optim.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
        if update%50==0 or update==1:
            metrics.write(json.dumps(dict(update=update,loss=float(loss),elapsed_s=time.monotonic()-start))+'\n')
    model.eval();absolute=torch.zeros(8);valid_count=0
    with torch.inference_mode():
        for starti in range(0,len(vx),64):
            xx,ssv,yy,mm=[z[starti:starti+64].to(a.device)for z in vdata]
            error=(model(xx,ssv)-yy).abs()*torch.tensor(ascale,device=a.device)
            absolute+=(error*mm[...,None]).sum((0,1)).cpu();valid_count+=int(mm.sum())
    checkpoint=dict(model=model.state_dict(),state_mean=sm.tolist(),state_std=ss.tolist(),action_mean=am.tolist(),action_std=ascale.tolist())
    torch.save(checkpoint,a.output/'policy.pt')
    save(a.output/'control-contract.json',read(a.train/'control-contract.json'))
    save(a.output/'training-provenance.json',[records[i]for i in selected]);save(a.output/'validation-provenance.json',vrecords)
    save(a.output/'completion.json',dict(complete=True,architecture='frozen_resnet18_spatial_gru',trainable_parameters=sum(p.numel()for p in model.parameters()),
        train_cache=str(a.train.resolve()),validation_cache=str(a.validation.resolve()),encoder_sha256=tr['encoder_sha256'],
        train_episodes=count,train_rows=len(ix),validation_episodes=len(vrecords),validation_mae=(absolute/valid_count).tolist(),
        seed=a.seed,subset_seed=a.subset_seed,steps=a.steps,batch_size=a.batch_size,learning_rate=a.learning_rate,
        sampling_manifest=str(a.sampling)if a.sampling else None,sampling_sha256=sha(a.sampling)if a.sampling else None,
        training_seconds=time.monotonic()-start,feature_cache_seconds=tr['cache_seconds']+va['cache_seconds'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated()if a.device.startswith('cuda')else None,
        source_code_sha256=sha(__file__),prediction_horizon=16,execution_steps=4,
        task='fixed walking football instruction; language omitted',physical_success=None))
    print(json.dumps(read(a.output/'completion.json')),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('cache');c.add_argument('--dataset',type=Path,required=True);c.add_argument('--split',choices=['train','validation'],required=True)
    c.add_argument('--encoder',choices=['imagenet','random'],default='imagenet')
    t=sub.add_parser('train');t.add_argument('--train',type=Path,required=True);t.add_argument('--validation',type=Path,required=True)
    t.add_argument('--episodes',type=int,default=0);t.add_argument('--subset-seed',type=int,default=2026);t.add_argument('--steps',type=int,default=2000)
    t.add_argument('--batch-size',type=int,default=16);t.add_argument('--learning-rate',type=float,default=.0003);t.add_argument('--sampling',type=Path)
    for parser in [c,t]:
        parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',default='cuda');parser.add_argument('--seed',type=int,default=42)
    a=p.parse_args();cache(a)if a.command=='cache'else train(a)
if __name__=='__main__':main()
