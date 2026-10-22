"""CPU-only, fail-closed control protocol shared by bridge and Kit receiver."""
from __future__ import annotations
import hashlib,json,math,time,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
RUNTIME_ID='newton-bam-20261019'
PROFILE_ID='beta-gc2093-l8cx-v1'
SIGNAL_PORT=49218
MEDIA_PORT=48018
DEADMAN=.35
KEYS=frozenset(('KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','ArrowUp','ArrowDown','ArrowLeft','ArrowRight'))
DEFAULT_CAMERA={'yaw':-125.,'pitch':28.,'radius':2.0}

def model_registry():
    data=json.loads((HERE/'models.json').read_text())
    if data.get('runtimeId')!=RUNTIME_ID or data.get('profileId')!=PROFILE_ID:raise ValueError('Model registry identity mismatch')
    return data['models']

def resolve_model(model_id):
    if model_id.startswith('student-export-'):
        sys.path.insert(0,str(HERE.parent/'bridge'))
        from student_artifacts import exported
        source=exported(model_id.removeprefix('student-export-'))
        return {'id':model_id,'resolvedPath':source['path'],'sha256':source['sha256'],'observations':61,'actions':14,'actionId':'play-student-walk','sourceTrainingJob':source['sourceTraining']['jobId'],'sourceExportJob':source['exportJobId'],'qualification':'chain-smoke-only' if source['sourceTraining']['smokeOnly'] else 'not-evaluated','actionAdapter':'official-native-scale1-no-wrapper-clip','label':{'zh':'学生模型 · 未验收','en':'Student model · unqualified'}}
    model=model_registry().get(model_id)
    if not model:raise ValueError('This policy is not registered in the new runtime')
    path=(HERE.parent/model['path']).resolve()
    if not path.is_relative_to(HERE.parent.resolve()):raise ValueError('Policy path escaped this release')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=model['sha256']:raise ValueError('Policy digest changed')
    if model.get('observations')!=61 or model.get('actions')!=14:raise ValueError('Policy shape mismatch')
    return dict(model,id=model_id,resolvedPath=str(path))

def validate_input(data,previous=None,now=None):
    """Only the bridge supplies timestamps; browser timestamps are never trusted."""
    now=time.monotonic() if now is None else now
    if not isinstance(data,dict) or set(data)-{'seq','keys','reset','paused','camera'}:raise ValueError('Unsupported control fields')
    seq=data.get('seq');keys=data.get('keys')
    if type(seq) is not int or not 0<=seq<2**53:raise ValueError('Invalid control sequence')
    if previous is not None and seq<=previous.get('seq',-1):raise ValueError('Stale control sequence')
    if not isinstance(keys,list) or len(keys)>10 or len(set(keys))!=len(keys) or any(not isinstance(k,str) or k not in KEYS for k in keys):raise ValueError('Invalid held keys')
    for flag in ('reset','paused'):
        if flag in data and type(data[flag]) is not bool:raise ValueError('Control flags must be booleans')
    camera=data.get('camera', (previous or {}).get('camera',DEFAULT_CAMERA))
    if not isinstance(camera,dict) or set(camera)!=set(DEFAULT_CAMERA):raise ValueError('Invalid camera orbit')
    limits={'yaw':(-360.,360.),'pitch':(8.,75.),'radius':(.65,8.)}
    for key,(low,high) in limits.items():
        if type(camera[key]) not in (int,float) or not math.isfinite(camera[key]) or not low<=camera[key]<=high:raise ValueError('Camera value outside its bounds')
    reset_count=(previous or {}).get('resetCount',0)+int(data.get('reset',False))
    return {'schema':'microduck.live-control.v1','seq':seq,'keys':keys,'receivedMonotonic':now,'expiresMonotonic':now+DEADMAN,
            'resetCount':reset_count,'paused':data.get('paused',(previous or {}).get('paused',False)),'camera':dict(camera)}

def motion(packet,now=None):
    """No fresh input means zero command; this is not a physical anti-drift claim."""
    now=time.monotonic() if now is None else now
    try:
        if packet.get('schema')!='microduck.live-control.v1' or not packet['receivedMonotonic']<=now<=packet['expiresMonotonic'] or packet.get('paused'):return [0.,0.,0.]
        held=set(packet['keys'])
        if not held<=KEYS:return [0.,0.,0.]
    except (TypeError,KeyError):return [0.,0.,0.]
    forward=int(bool(held&{'KeyW','ArrowUp'}))-int(bool(held&{'KeyS','ArrowDown'}))
    lateral=int('KeyA' in held)-int('KeyD' in held)
    turn=int(bool(held&{'KeyQ','ArrowLeft'}))-int(bool(held&{'KeyE','ArrowRight'}))
    return [.4*forward if forward>0 else .2*forward,.2*lateral,.7*turn]

def read_packet(path):
    try:
        data=json.loads(Path(path).read_text())
        return data if isinstance(data,dict) else {}
    except (OSError,ValueError):return {}

def atomic(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False)+'\n');temporary.replace(path)
