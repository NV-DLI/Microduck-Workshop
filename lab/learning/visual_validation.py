"""Verify cached input identity, causal interface, and episode-aligned sampling."""
import hashlib,json,math
from pathlib import Path
import numpy as np
try:
    from .walking_visual_history import specification
except ImportError:
    from walking_visual_history import specification

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()

def contract_check(c):
    names=[n.removeprefix('q_relative/')for n in c.get('state_names',[])[6:20]]
    if c!=specification(names):raise ValueError('Exact version-3 walking control contract required')

def verify_cache(path):
    path=Path(path);info=read(path/'completion.json')
    if info.get('complete')is not True:raise ValueError('Incomplete cache')
    contract_check(read(path/'control-contract.json'))
    if sha(path/'encoder.pt')!=info['encoder_sha256']:raise ValueError('Encoder file changed')
    receipt=read(path/'content-receipt.json')
    expected={'features.npy','targets.npz','encoder.pt','control-contract.json','provenance.json','completion.json'}
    if set(receipt['sha256'])!=expected:raise ValueError('Incomplete cache receipt')
    for name,digest in receipt['sha256'].items():
        if sha(path/name)!=digest:raise ValueError('Cache file changed: '+name)
    records=read(path/'provenance.json');t=np.load(path/'targets.npz');x=np.load(path/'features.npy',mmap_mode='r')
    n=info['rows']
    if x.shape!=(n,4,7680)or t['state'].shape!=(n,127)or t['action'].shape!=(n,16,8)or t['valid'].shape!=(n,16):raise ValueError('Cache dimensions differ')
    if t['valid'].dtype!=np.bool_ or not t['valid'][:,0].all():raise ValueError('Invalid padding mask')
    if any(not np.isfinite(t[k]).all()for k in ['state','action'])or not np.isfinite(x).all():raise ValueError('Nonfinite cache')
    expected_episode=np.concatenate([np.repeat(i,r['frames'])for i,r in enumerate(records)])
    if not np.array_equal(t['episode'],expected_episode):raise ValueError('Episode provenance and rows differ')
    if len(records)!=info['episodes']or any(r['split']!=info['split']for r in records):raise ValueError('Cache split differs')
    return info

def verify_sampling(sampling, records, episodes):
    if sampling['training_seeds']!=[r['seed']for r in records]:raise ValueError('Sampling seed lineage differs')
    rows=sampling['rows']
    if len(rows)!=len(episodes):raise ValueError('Sampling row count differs')
    counters={}
    for i,(row,e)in enumerate(zip(rows,episodes)):
        e=int(e);frame=counters.get(e,0);counters[e]=frame+1
        if (row['index'],row['episode_index'],row['frame_index'],row['seed'])!=(i,e,frame,records[e]['seed']):raise ValueError('Sampling row identity differs')
        if not math.isfinite(row['weight'])or row['weight']<=0:raise ValueError('Invalid sampling weight')
    return [r['weight']for r in rows]

def budget_check(steps,batch_size,lr):
    if type(steps)is not int or steps<1 or type(batch_size)is not int or batch_size<1 or not math.isfinite(lr)or lr<=0:raise ValueError('Positive finite training budget required')
