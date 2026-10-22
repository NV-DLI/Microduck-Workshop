"""CPU-only provenance, timing and video helpers for independent policy replay."""
from pathlib import Path
import hashlib,json,math,os,shutil,subprocess,time
import numpy as np

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_run(checkpoint):
    checkpoint=Path(checkpoint).resolve();run=checkpoint.parent
    ledger=json.loads((run/'training-ledger.json').read_text())
    for name in ('agent','env'):
        if sha(run/f'{name}.yaml')!=ledger[f'{name}_yaml_sha256']:
            raise ValueError(f'Frozen {name} configuration SHA mismatch')
    snapshot=run/'source_snapshot'
    if not snapshot.is_dir():raise ValueError('A frozen source_snapshot and ledger are required')
    checked={}
    for name,expected in ledger['source_sha256'].items():
        actual=sha(snapshot/name)
        if actual!=expected:raise ValueError(f'Frozen source SHA mismatch: {name}')
        checked[name]=actual
    checkpoints=[json.loads(line) for line in (run/'checkpoint_times.jsonl').read_text().splitlines() if line.strip()]
    actual=sha(checkpoint)
    records=[r for r in checkpoints if r.get('checkpoint_sha256')==actual]
    if not records:raise ValueError('Checkpoint SHA is absent from the measured training ledger')
    record=records[-1]
    return {'checkpoint':str(checkpoint),'checkpoint_sha256':actual,'run':str(run),'snapshot':str(snapshot),
        'source_sha256':checked,'skill':ledger['skill'],'reward_version':ledger['reward_version'],
        'checkpoint_record':record,'training_env_count':ledger['env_count'],
        'agent_yaml_sha256':sha(run/'agent.yaml'),'env_yaml_sha256':sha(run/'env.yaml'),
        'runtime':json.loads((run/'runtime.json').read_text())}

def find_ffmpeg(explicit=None):
    parameterss=[explicit,os.environ.get('MICRODUCK_FFMPEG'),shutil.which('ffmpeg'),
        '/usr/bin/ffmpeg',
        '/workspace/reference/work/isaac_tools/ffmpeg']
    for value in parameterss:
        if value and Path(value).is_file() and os.access(value,os.X_OK):
            path=str(Path(value).resolve())
            result=subprocess.run([path,'-version'],text=True,capture_output=True,check=True)
            return path,result.stdout.splitlines()[0]
    raise FileNotFoundError('Set --ffmpeg or MICRODUCK_FFMPEG to an executable ffmpeg')

def camera_pose(origins,horizontal_fov,aspect):
    origins=np.asarray(origins);low=origins.min(0);high=origins.max(0)
    center=(low+high)/2;center[2]=.17
    radius=float(np.linalg.norm((high-low)/2+np.array([.20,.20,.20])))
    vfov=2*math.atan(math.tan(math.radians(horizontal_fov)/2)/aspect)
    distance=radius/math.sin(vfov/2)*1.12
    direction=np.array([1.,-1.,1.35]);direction/=np.linalg.norm(direction)
    return center+direction*distance,center

class FrameSchedule:
    """Select real 200 Hz physical samples; never interpolate robot motion."""
    def __init__(self,fps,physics_dt):self.fps=fps;self.dt=physics_dt;self.tick=0;self.frame=0
    def advance(self,dt):
        if abs(dt-self.dt)>1e-10:raise RuntimeError('Expected one actual physics tick per scene update')
        self.tick+=1
        if self.tick*self.dt+1e-10<(self.frame+1)/self.fps:return False
        self.frame+=1;return True
    @property
    def sim_seconds(self):return self.tick*self.dt

class RealtimePacer:
    def __init__(self):self.origin=time.monotonic();self.rebases=0;self.lateness=[]
    def pace(self,sim_seconds):
        delay=self.origin+sim_seconds-time.monotonic()
        self.lateness.append(max(0.,-delay))
        if delay>0:time.sleep(delay)
        elif delay<-.25:
            # A shader/window stall must not cause a fast-forward catch-up.
            self.origin=time.monotonic()-sim_seconds;self.rebases+=1
    def reset_clock(self,sim_seconds):self.origin=time.monotonic()-sim_seconds

class VideoWriter:
    def __init__(self,path,ffmpeg,width,height,fps,labels):
        from PIL import ImageFont
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.width,self.height,self.fps=width,height,fps;self.labels=labels;self.frames=0
        self.log=self.path.with_suffix('.ffmpeg.log').open('wb')
        self.process=subprocess.Popen([ffmpeg,'-hide_banner','-loglevel','error','-n',
            '-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
            '-framerate',str(fps),'-i','-','-an','-c:v','libx264','-preset','veryfast',
            '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(self.path)],
            stdin=subprocess.PIPE,stderr=self.log)
        try:self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',max(14,width//64))
        except OSError:self.font=ImageFont.load_default()
    def write(self,rgb,sim_seconds):
        from PIL import Image,ImageDraw
        if rgb.shape!=(self.height,self.width,3):raise RuntimeError(f'Unexpected RGB shape {rgb.shape}')
        image=Image.fromarray(np.asarray(rgb,dtype=np.uint8));draw=ImageDraw.Draw(image)
        bar=max(50,self.width//24);draw.rectangle((0,0,self.width,bar),fill=(12,18,30))
        draw.text((14,6),self.labels[0],font=self.font,fill=(255,182,74))
        draw.text((14,bar//2),self.labels[1]+f' | sim {sim_seconds:.2f}s',font=self.font,fill=(228,237,247))
        self.process.stdin.write(np.asarray(image).tobytes());self.frames+=1
    def finish(self):
        if self.process.stdin and not self.process.stdin.closed:self.process.stdin.close()
        code=self.process.wait(timeout=60);self.log.close()
        if code:raise RuntimeError(f'Video encoder exited {code}; see {self.path.with_suffix(".ffmpeg.log")}')
        return {'file':str(self.path.resolve()),'frames':self.frames,'fps':self.fps,
            'duration_seconds':self.frames/self.fps,'sha256':sha(self.path)}
