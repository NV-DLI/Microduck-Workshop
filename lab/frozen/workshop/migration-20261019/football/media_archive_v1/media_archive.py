"""Lossless bounded evidence queue; video encoding happens after simulation.

The worker never imports Isaac, Torch, a detector or a controller. No latest-only
queue is used. PNG compression changes file bytes, never captured pixels.
"""
from __future__ import annotations
import hashlib, io, json, math, multiprocessing as mp, os, queue, re
import signal, subprocess, time, traceback
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

FFMPEG=Path(__file__).resolve().parents[2]/'runtime/tools/bin/ffmpeg'
FPS=15

def start_ticks(pid):
    try:return int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19])
    except(FileNotFoundError,ProcessLookupError):return None

def cleanup_owned_encoders(output):
    """Only exact recorded child PID/startTicks; never scan or signal other work."""
    path=Path(output)/'owned-encoders.json'
    if not path.exists():return []
    actions=[]
    for child in json.loads(path.read_text())['children']:
        pid=child['pid'];ticks=child['startTicks']
        if ticks is not None and start_ticks(pid)==ticks:
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:continue
            until=time.monotonic()+2
            while start_ticks(pid)==ticks and time.monotonic()<until:time.sleep(.02)
            if start_ticks(pid)==ticks:
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass
            actions.append({'pid':pid,'startTicks':ticks,'action':'terminated exact owned encoder after worker cleanup'})
    return actions

class ArchiveError(RuntimeError): pass

def sha_bytes(value): return hashlib.sha256(value).hexdigest()
def sha(path): return sha_bytes(Path(path).read_bytes())
def dump(value): return json.dumps(value,ensure_ascii=False,allow_nan=False)

def atomic_bytes(path,data,*,replace=False):
    path=Path(path)
    if path.exists() and not replace: raise FileExistsError(path)
    tmp=path.with_name(path.name+'.writing')
    with tmp.open('xb') as stream:
        stream.write(data);stream.flush()
    os.replace(tmp,path)

def atomic_json(path,value):
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode(),replace=True)

def png_bytes(rgb,level):
    output=io.BytesIO();Image.fromarray(rgb).save(output,format='PNG',compress_level=level)
    return output.getvalue()

def annotated(rgb,overhead,t,state,phase,policy):
    first=Image.fromarray(rgb.copy());third=Image.fromarray(overhead.copy())
    ImageDraw.Draw(first).text((12,12),f'HEAD CAMERA  t={t:.3f}s  '+state,fill='yellow',stroke_width=1,stroke_fill='black')
    ImageDraw.Draw(third).text((12,12),f'OVERHEAD t={t:.3f}s {phase} / {policy}',fill='yellow',stroke_width=1,stroke_fill='black')
    first=np.asarray(first);third=np.asarray(third)
    return first,third,np.concatenate([first,third],axis=1)

class Video:
    """Pinned original encoder options. Own every child and close on all errors."""
    def __init__(self,path,width,height,*,ffmpeg=FFMPEG):
        self.path=Path(path);self.count=0;self.closed=False
        self.log=self.path.with_suffix('.ffmpeg.log').open('x')
        try:
            self.proc=subprocess.Popen([str(ffmpeg),'-nostdin','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}','-framerate','15','-i','pipe:0','-an','-c:v','libx264','-threads','2','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=self.log)
        except BaseException:
            self.log.close();raise
        identity={'pid':self.proc.pid,'startTicks':start_ticks(self.proc.pid),'output':str(self.path)}
        registry=self.path.parent/'owned-encoders.json'
        try:
            known=json.loads(registry.read_text())if registry.exists()else {'children':[]}
            known['children'].append(identity);atomic_json(registry,known)
        except BaseException:
            if self.proc.poll()is None:self.proc.kill()
            self.proc.wait(timeout=5)
            if self.proc.stdin is not None:
                try:self.proc.stdin.close()
                except BaseException:pass
            self.log.close();raise
    def add(self,rgb):
        if self.closed: raise ArchiveError('Encoder already closed')
        self.proc.stdin.write(rgb.tobytes());self.count+=1
    def close(self,*,abort=False):
        if self.closed: return self.result
        error=None
        def reap():
            if self.proc.poll()is None:self.proc.terminate()
            try:self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:self.proc.kill();self.proc.wait(timeout=5)
        try:
            # Kill a hung reader BEFORE BufferedWriter.close attempts a flush.
            if abort:reap()
            if self.proc.stdin is not None:
                try: self.proc.stdin.close()
                except BrokenPipeError as exc: error=repr(exc)
                self.proc.stdin=None
            try: self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                reap();error='encoder timeout'
        except BaseException as exc:
            error=repr(exc);reap()
            if self.proc.stdin is not None:
                try:self.proc.stdin.close()
                except BaseException:pass
                self.proc.stdin=None
        finally:
            self.log.close();self.closed=True
        code=self.proc.poll()
        self.result={'path':self.path.name,'frames':self.count,'nominal_fps':FPS,'exit_code':code,'sha256':sha(self.path)if self.path.exists()else None,'pid':self.proc.pid,'child_exited':self.proc.poll()is not None}
        if error:self.result['error']=error
        if code or error: raise ArchiveError('Encoder failed: '+dump(self.result))
        return self.result

def encode_after_episode(output,records,*,ffmpeg=FFMPEG):
    """Read original control decisions, never invent an unrecorded decision."""
    output=Path(output);controls={}
    path=output/'control.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            row=json.loads(line);t=row['timestamp_s']
            if t in controls:raise ArchiveError('Duplicate control timestamp')
            controls[t]=row
    videos={};result={};missing=[];success=False
    try:
        if not records:return {'videos':{},'missingControlTimestamps':[],'frameCount':0}
        first=np.asarray(Image.open(output/records[0]['row']['rgb']).convert('RGB'))
        height,width=first.shape[:2]
        for name,file,w in [('first_person','first-person.mp4',width),('overhead','overhead.mp4',width),('paired','paired-views.mp4',width*2)]:
            videos[name]=Video(output/file,w,height,ffmpeg=ffmpeg)
        with (output/'media-frame-ledger.jsonl').open('x') as ledger:
            for packet in records:
                row=packet['row'];t=row['timestamp_s'];control=controls.get(t)
                if control is None: missing.append(t)
                state=control['state']if control is not None else 'NO_RECORDED_CONTROL'
                rgb=np.asarray(Image.open(output/row['rgb']).convert('RGB'))
                third=np.asarray(Image.open(output/row['overhead']).convert('RGB'))
                first,overhead,paired=annotated(rgb,third,t,state,packet['phase'],packet['policy'])
                for name,pixels in [('first_person',first),('overhead',overhead),('paired',paired)]:videos[name].add(pixels)
                ledger.write(dump({'captureOrdinal':packet['ordinal'],'timestamp_s':t,'physics_step':row['physics_step'],'render_generation':row['render_generation'],'sourceRGBPixelSHA256':sha_bytes(rgb.tobytes()),'sourceOverheadPixelSHA256':sha_bytes(third.tobytes()),'firstPersonAnnotatedPixelSHA256':sha_bytes(first.tobytes()),'overheadAnnotatedPixelSHA256':sha_bytes(overhead.tobytes()),'pairedAnnotatedPixelSHA256':sha_bytes(paired.tobytes()),'state':state,'controlRecorded':control is not None,'fps':FPS})+'\n')
            ledger.flush();os.fsync(ledger.fileno())
        for name,video in videos.items():result[name]=video.close()
        success=True
        return {'videos':result,'missingControlTimestamps':missing,'frameCount':len(records),'durationSeconds':len(records)/FPS}
    finally:
        for name,video in videos.items():
            if not video.closed:
                try:video.close(abort=not success)
                except BaseException:pass
        atomic_json(output/'video-children.json',{'children':[{'name':name,'pid':video.proc.pid,'exitCode':video.proc.poll(),'exited':video.proc.poll()is not None}for name,video in videos.items()]})

def _worker(channel,output,level,no_video,preview_period,delay,fail_after,ffmpeg):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    signal.signal(signal.SIGTERM,lambda *args:(_ for _ in ()).throw(KeyboardInterrupt('archive worker termination')))
    output=Path(output);records=[];expected=None;started=time.monotonic();last_preview=-math.inf
    report={'schema':'microduck.football-media-archive.v1','pid':os.getpid(),'status':'running','committedCaptures':0,'qualificationClaimed':False,'compressionLevel':level,'losslessPixels':True,'dropPolicy':'none; bounded blocking backpressure','videos':{},'timingsSeconds':{k:0. for k in ['pngEncode','archiveWrite','preview','postEpisodeVideo']}}
    stream=None
    try:
        stream=(output/'sensor-inputs.jsonl').open('x')
        with (output/'media-commits.jsonl').open('x') as journal:
            while True:
                packet=channel.get()
                if packet['kind']=='finish':expected=packet['expected'];break
                if packet['ordinal']!=len(records):raise ArchiveError('Non-contiguous evidence queue')
                if delay:time.sleep(delay)
                if fail_after is not None and len(records)>=fail_after:raise OSError('Injected archive write failure')
                row=packet['row'];start=time.perf_counter()
                if sha_bytes(packet['rgb'].tobytes())!=row['rgb_pixel_sha256']or sha_bytes(packet['overhead'].tobytes())!=row['overhead_pixel_sha256']or sha_bytes(packet['depth'].tobytes())!=row['tof_source_float32_sha256']:raise ArchiveError('Queued sensor payload identity mismatch')
                rgb=png_bytes(packet['rgb'],level);overhead=png_bytes(packet['overhead'],level)
                report['timingsSeconds']['pngEncode']+=time.perf_counter()-start;start=time.perf_counter()
                atomic_bytes(output/row['rgb'],rgb);atomic_bytes(output/row['overhead'],overhead)
                depth=io.BytesIO();np.save(depth,packet['depth'],allow_pickle=False);atomic_bytes(output/row['tof_range'],depth.getvalue())
                row.update(rgb_file_sha256=sha_bytes(rgb),overhead_file_sha256=sha_bytes(overhead),tof_file_sha256=sha_bytes(depth.getvalue()))
                row.setdefault('published_unix_s',time.time())
                stream.write(dump(row)+'\n');stream.flush()
                record={'ordinal':packet['ordinal'],'row':row,'phase':packet['phase'],'policy':packet['policy']};records.append(record)
                journal.write(dump({'captureOrdinal':packet['ordinal'],'timestamp_s':row['timestamp_s'],'physics_step':row['physics_step'],'rgb_pixel_sha256':row['rgb_pixel_sha256'],'rgb_file_sha256':row['rgb_file_sha256'],'overhead_file_sha256':row['overhead_file_sha256'],'archiveCommittedUnixSeconds':time.time()})+'\n');journal.flush()
                report['committedCaptures']=len(records);report['timingsSeconds']['archiveWrite']+=time.perf_counter()-start
                if time.monotonic()-last_preview>=preview_period:
                    start=time.perf_counter()
                    # Human preview only. Final videos use the recorded same-frame control.
                    _,_,paired=annotated(packet['rgb'],packet['overhead'],row['timestamp_s'],'RECORDING',packet['phase'],packet['policy'])
                    atomic_bytes(output/'latest-paired.png',png_bytes(paired,level),replace=True)
                    last_preview=time.monotonic();report['timingsSeconds']['preview']+=time.perf_counter()-start
                atomic_json(output/'media-progress.json',report)
            stream.flush();os.fsync(stream.fileno());journal.flush();os.fsync(journal.fileno())
        stream.close();stream=None
        if expected!=len(records):raise ArchiveError('Submitted/committed count mismatch')
        if not no_video:
            start=time.perf_counter();encoded=encode_after_episode(output,records,ffmpeg=ffmpeg)
            report.update(encoded);report['timingsSeconds']['postEpisodeVideo']=time.perf_counter()-start
            if encoded['missingControlTimestamps']:raise ArchiveError('Captured frames without recorded control; episode retained')
        else:
            control_path=output/'control.jsonl'
            control_times=[json.loads(line)['timestamp_s']for line in control_path.read_text().splitlines()]if control_path.exists()else []
            report['missingControlTimestamps']=[p['row']['timestamp_s']for p in records if p['row']['timestamp_s']not in control_times]
            if report['missingControlTimestamps']:raise ArchiveError('Captured frames without recorded control; episode retained')
        report.update(status='complete',expectedCaptures=expected,evidenceComplete=True)
    except BaseException as error:
        report.update(status='failed',expectedCaptures=expected,evidenceComplete=False,error=repr(error),traceback=traceback.format_exc())
    finally:
        if stream is not None:stream.close()
        report['workerWallSeconds']=time.monotonic()-started
        atomic_json(output/'media-archive-report.json',report)

class ArchiveSink:
    def __init__(self,output,*,capacity=16,compression=1,no_video=False,preview_period=1.,close_timeout=300.,_test_delay=0.,_test_fail_after=None,_ffmpeg=FFMPEG):
        if type(capacity)is not int or not 1<=capacity<=64:raise ValueError('Bounded1..64 capture queue required')
        if compression not in (1,6):raise ValueError('Reviewed PNG compression level1 or6 required')
        self.output=Path(output);(self.output/'frames').mkdir(exist_ok=True)
        if (self.output/'media-archive-report.json').exists() or (self.output/'sensor-inputs.jsonl').exists():raise FileExistsError('Retain old artifacts')
        self.count=0;self.last_time=-math.inf;self.closed=False;self.wait_seconds=0.;self.full_waits=0;self.capacity=capacity;self.close_timeout=close_timeout
        self.result={'status':'failed','evidenceComplete':False,'committedCaptures':None,'error':'Archive has not completed'}
        self.names=set();ctx=mp.get_context('spawn');self.channel=ctx.Queue(maxsize=capacity)
        self.process=ctx.Process(target=_worker,args=(self.channel,str(self.output),compression,no_video,preview_period,_test_delay,_test_fail_after,str(_ffmpeg)),daemon=False)
        self.process.start()
    def _put(self,packet):
        started=time.monotonic()
        while True:
            if not self.process.is_alive():raise ArchiveError('Archive worker exited; all partial artifacts retained')
            if time.monotonic()-started>self.close_timeout:raise ArchiveError('Archive queue backpressure timeout; retain incomplete episode')
            try:self.channel.put(packet,timeout=.05);break
            except queue.Full:self.full_waits+=1
        self.wait_seconds+=time.monotonic()-started
    def capture(self,row,rgb,overhead,depth,*,phase,policy):
        if self.closed:raise ArchiveError('Archive already closed')
        # Snapshot before control inference: later producer mutations cannot change evidence.
        row=json.loads(dump(row));t=row['timestamp_s']
        if not isinstance(t,(int,float))or not math.isfinite(t)or t<=self.last_time:raise ValueError('Strict advancing capture timestamp required')
        for key,suffix in [('rgb','rgb.png'),('overhead','overhead.png'),('tof_range','tof-range.npy')]:
            if not re.fullmatch(r'frames/[0-9]{6}-'+re.escape(suffix),row[key]):raise ValueError('Owned flat frame path required')
            if row[key]in self.names:raise ValueError('Duplicate frame destination')
        rgb=np.array(rgb,copy=True,order='C');overhead=np.array(overhead,copy=True,order='C');depth=np.array(depth,copy=True,order='C')
        if rgb.dtype!=np.uint8 or rgb.shape!=(540,960,3)or overhead.dtype!=np.uint8 or overhead.shape!=rgb.shape:raise ValueError('Original960x540 RGB pair required')
        if depth.shape!=(32,32)or depth.dtype!=np.float32:raise ValueError('Original32x32float32 ToF source required')
        if sha_bytes(rgb.tobytes())!=row['rgb_pixel_sha256']:raise ValueError('Captured RGB pixels do not match source identity')
        row['overhead_pixel_sha256']=sha_bytes(overhead.tobytes());row['tof_source_float32_sha256']=sha_bytes(depth.tobytes())
        packet={'kind':'capture','ordinal':self.count,'row':row,'rgb':rgb,'overhead':overhead,'depth':depth,'phase':str(phase),'policy':str(policy)}
        self._put(packet);self.count+=1;self.last_time=t;self.names.update(row[k]for k in ('rgb','overhead','tof_range'))
    def close(self):
        if self.closed:
            if not self.result.get('evidenceComplete'):raise ArchiveError(dump(self.result))
            return self.result
        self.closed=True;error=None
        try:
            self._put({'kind':'finish','expected':self.count})
            self.process.join(timeout=self.close_timeout)
            if self.process.is_alive():raise ArchiveError('Archive drain timeout')
        except BaseException as exc:
            error=repr(exc)
        finally:
            if self.process.is_alive():
                self.process.terminate();self.process.join(timeout=25)
                if self.process.is_alive():self.process.kill();self.process.join(timeout=5)
            self.channel.cancel_join_thread();self.channel.close()
        path=self.output/'media-archive-report.json'
        try:
            loaded=json.loads(path.read_text())
            if not isinstance(loaded,dict):raise ValueError('Worker report must be an object')
            self.result=loaded
        except BaseException as exc:
            self.result={'status':'failed','evidenceComplete':False,'committedCaptures':None,'error':'Worker report unreadable; preserve incomplete episode','reportReadError':repr(exc)}
        finally:
            try:self.result['encoderCleanupFallback']=cleanup_owned_encoders(self.output)
            except BaseException as exc:self.result.update(encoderCleanupError=repr(exc),evidenceComplete=False,status='failed')
        self.result.update(submittedCaptures=self.count,queueCapacity=self.capacity,queueFullWaits=self.full_waits,queuePutWallSeconds=self.wait_seconds,workerExitCode=self.process.exitcode,workerExited=not self.process.is_alive())
        if error:self.result.update(parentError=error,evidenceComplete=False,status='failed')
        if self.result.get('committedCaptures')!=self.count:self.result.update(evidenceComplete=False,status='failed')
        try:atomic_json(path,self.result)
        except BaseException as exc:self.result.update(reportPersistenceError=repr(exc),evidenceComplete=False,status='failed')
        if not self.result.get('evidenceComplete'):raise ArchiveError('Media evidence incomplete: '+str(path))
        return self.result
