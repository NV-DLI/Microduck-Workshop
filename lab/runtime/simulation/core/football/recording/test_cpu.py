"""Actual completed RGB/ToF shadow and adversarial archive lifecycle checks; no GPU."""
import ast,copy,hashlib,json,os,re,subprocess,sys,tempfile,time,unittest
from unittest import mock
from pathlib import Path
import numpy as np
from PIL import Image
from media_archive import ArchiveSink,ArchiveError,annotated,FFMPEG,sha,sha_bytes

D=Path(__file__).resolve().parent;M=D.parents[1]
O=M.parents[1]/'configuration/core'
R=O/'football-v39-small-forward40s'
OUT=Path(os.environ.get('MEDIA_CPU_OUTPUT',str(O/('football-media-v1-cpu-'+str(time.time_ns())))))
SENSORS=[json.loads(l)for l in(R/'sensor-inputs.jsonl').read_text().splitlines()]
CONTROLS={r['timestamp_s']:r for r in map(json.loads,(R/'control.jsonl').read_text().splitlines())}
PHYSICS={r['timestamp_s']:r for r in map(json.loads,(R/'physics-trace.jsonl').read_text().splitlines())}
CHECKS=[]

def packet(index):
    row=copy.deepcopy(SENSORS[index]);t=row['timestamp_s']
    return row,np.asarray(Image.open(R/row['rgb']).convert('RGB')).copy(),np.asarray(Image.open(R/row['overhead']).convert('RGB')).copy(),np.load(R/row['tof_range'],allow_pickle=False),PHYSICS[t]['startup_phase'],CONTROLS[t]['active_policy']

def submit(sink,index):
    row,rgb,overhead,depth,phase,policy=packet(index)
    sink.capture(row,rgb,overhead,depth,phase=phase,policy=policy)
    return row

def controls(path,indices):
    path.mkdir(parents=True)
    (path/'control.jsonl').write_text(''.join(json.dumps(CONTROLS[SENSORS[i]['timestamp_s']],ensure_ascii=False)+'\n'for i in indices))

def decode(path):
    result=subprocess.run([str(FFMPEG),'-hide_banner','-i',str(path),'-map','0:v:0','-vsync','0','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=30)
    metadata=result.stderr.decode();fps=float(re.search(r' ([0-9.]+) fps',metadata).group(1))
    width,height=map(int,re.search(r', (\d+)x(\d+)[ ,]',metadata).groups())
    pixels=np.frombuffer(result.stdout,np.uint8).reshape((-1,height,width,3))
    return fps,list(pixels)

class MediaTests(unittest.TestCase):
    def tearDown(self):
        CHECKS.append({'name':self.id().split('.')[-1],'finished':True})

    def test_01_actual18_frame_lossless_and_original_realtime_video(self):
        indices=list(range(35,53));path=OUT/'shadow18';controls(path,indices)
        sink=ArchiveSink(path,capacity=2)
        try:
            for i in indices:submit(sink,i)
        finally:result=sink.close()
        self.assertTrue(result['evidenceComplete']);self.assertEqual(result['committedCaptures'],18);self.assertTrue(result['workerExited'])
        saved=[json.loads(l)for l in(path/'sensor-inputs.jsonl').read_text().splitlines()]
        for index,row in zip(indices,saved):
            original=SENSORS[index]
            new_identity=('rgb_file_sha256','overhead_file_sha256','overhead_pixel_sha256','tof_source_float32_sha256','tof_file_sha256')
            self.assertEqual({k:v for k,v in row.items()if k not in new_identity},{k:v for k,v in original.items()if k not in new_identity})
            for kind in ('rgb','overhead'):
                self.assertTrue(np.array_equal(np.asarray(Image.open(path/row[kind])),np.asarray(Image.open(R/original[kind]))))
                self.assertEqual(sha(path/row[kind]),row[kind+'_file_sha256'])
            self.assertTrue(np.array_equal(np.load(path/row['tof_range']),np.load(R/original['tof_range']),equal_nan=True))
            self.assertEqual(sha(path/row['tof_range']),row['tof_file_sha256'])
            self.assertEqual(sha_bytes(np.load(path/row['tof_range']).tobytes()),row['tof_source_float32_sha256'])
            self.assertEqual(sha_bytes(np.asarray(Image.open(path/row['overhead'])).tobytes()),row['overhead_pixel_sha256'])
        # Execute only the original Video class AST, never the Isaac program/imports.
        original=R/'source-snapshot/football/isaac_execute.py';tree=ast.parse(original.read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Video')
        ns={'subprocess':subprocess,'sha':sha};exec(compile(ast.Module(body=[node],type_ignores=[]),str(original),'exec'),ns)
        baseline=OUT/'original-realtime18';baseline.mkdir();old_path=os.environ.get('PATH','');os.environ['PATH']=str(FFMPEG.parent)+os.pathsep+old_path
        encoders={};base_report={}
        try:
            for name,file,w in [('first_person','first-person.mp4',960),('overhead','overhead.mp4',960),('paired','paired-views.mp4',1920)]:encoders[name]=ns['Video'](baseline/file,w,540)
            for i in indices:
                row,rgb,third,depth,phase,policy=packet(i);t=row['timestamp_s']
                # Exact original label and append order from the frozen executor.
                first,overhead,paired=annotated(rgb,third,t,CONTROLS[t]['state'],phase,policy)
                encoders['first_person'].add(first);encoders['overhead'].add(overhead);encoders['paired'].add(paired)
            for name,encoder in encoders.items():base_report[name]=encoder.close()
        finally:
            os.environ['PATH']=old_path
            for encoder in encoders.values():
                if encoder.proc.poll()is None:encoder.proc.terminate();encoder.proc.wait(timeout=5)
        comparisons={}
        for name,record in result['videos'].items():
            left=path/record['path'];right=baseline/record['path'];lfps,left_frames=decode(left);rfps,right_frames=decode(right)
            self.assertEqual(lfps,15);self.assertEqual(rfps,15);self.assertEqual(len(left_frames),18);self.assertEqual(len(right_frames),18)
            self.assertTrue(all(np.array_equal(x,y)for x,y in zip(left_frames,right_frames)))
            comparisons[name]={'frames':18,'fps':lfps,'durationSeconds':18/lfps,'decodedFramesExact':True,'mp4BytesExact':sha(left)==sha(right),'parametersSHA256':sha(left),'originalRealtimeSHA256':sha(right)}
        ledger=[json.loads(l)for l in(path/'media-frame-ledger.jsonl').read_text().splitlines()]
        self.assertEqual([x['timestamp_s']for x in ledger],[SENSORS[i]['timestamp_s']for i in indices])
        self.assertTrue(all(x['controlRecorded']for x in ledger))
        (OUT/'capture-verification.json').write_text(json.dumps({'scope':'CPU same-frame replay, not a fresh physics run or end-to-end speed claim','captureIndices':indices,'captureTimeRange':[SENSORS[35]['timestamp_s'],SENSORS[52]['timestamp_s']],'sensorRowsExactExceptNewPNGFileSHA':True,'decodedRGBToFPixelsExact':True,'videoComparison':comparisons,'result':result},indent=2)+'\n')

    def test_02_saturated_queue_blocks_without_loss(self):
        indices=list(range(60,68));path=OUT/'backpressure8';controls(path,indices)
        sink=ArchiveSink(path,capacity=1,no_video=True,_test_delay=.12)
        try:
            for i in indices:submit(sink,i)
        finally:result=sink.close()
        self.assertEqual(result['submittedCaptures'],8);self.assertEqual(result['committedCaptures'],8);self.assertGreater(result['queueFullWaits'],0);self.assertTrue(result['evidenceComplete'])
        saved=[json.loads(l)['timestamp_s']for l in(path/'sensor-inputs.jsonl').read_text().splitlines()]
        self.assertEqual(saved,[SENSORS[i]['timestamp_s']for i in indices])

    def test_03_producer_reuse_does_not_mutate_queued_pixels_or_packets(self):
        path=OUT/'immutable1';controls(path,[35]);sink=ArchiveSink(path,capacity=1,no_video=True,_test_delay=.1)
        row,rgb,third,depth,phase,policy=packet(35)
        try:
            sink.capture(row,rgb,third,depth,phase=phase,policy=policy)
            rgb[:]=0;third[:]=255;depth[:]=0;row['timestamp_s']=999;row['tof_frame'].clear()
        finally:result=sink.close()
        stored=json.loads((path/'sensor-inputs.jsonl').read_text());self.assertEqual(stored['timestamp_s'],SENSORS[35]['timestamp_s']);self.assertEqual(stored['tof_frame'],SENSORS[35]['tof_frame'])
        self.assertEqual(sha_bytes(np.asarray(Image.open(path/stored['rgb'])).tobytes()),SENSORS[35]['rgb_pixel_sha256'])
        self.assertTrue(np.array_equal(np.load(path/stored['tof_range']),np.load(R/SENSORS[35]['tof_range']),equal_nan=True))

    def test_04_write_failure_retains_committed_prefix_and_exits(self):
        path=OUT/'write-error';controls(path,list(range(35,43)));sink=ArchiveSink(path,capacity=1,no_video=True,_test_fail_after=2)
        try:
            for i in range(35,43):submit(sink,i)
        except ArchiveError:pass
        with self.assertRaises(ArchiveError):sink.close()
        report=json.loads((path/'media-archive-report.json').read_text());self.assertFalse(report['evidenceComplete']);self.assertTrue(report['workerExited']);self.assertEqual(report['committedCaptures'],2)
        self.assertGreaterEqual(report['submittedCaptures'],2);self.assertEqual(len((path/'sensor-inputs.jsonl').read_text().splitlines()),2)
        self.assertTrue((path/'frames'/Path(SENSORS[35]['rgb']).name).is_file())

    def test_05_encoder_failure_is_not_a_success(self):
        path=OUT/'encoder-error';controls(path,[35,36]);sink=ArchiveSink(path,capacity=1,_ffmpeg='/nonexistent-owned-test-ffmpeg')
        try:
            submit(sink,35);submit(sink,36)
        finally:
            with self.assertRaises(ArchiveError):sink.close()
        report=json.loads((path/'media-archive-report.json').read_text());self.assertFalse(report['evidenceComplete']);self.assertTrue(report['workerExited']);self.assertEqual(report['committedCaptures'],2)

    def test_06_missing_control_and_existing_outputs_are_explicit(self):
        path=OUT/'missing-control';controls(path,[]);sink=ArchiveSink(path,capacity=1)
        try:submit(sink,35)
        finally:
            with self.assertRaises(ArchiveError):sink.close()
        report=json.loads((path/'media-archive-report.json').read_text());self.assertEqual(report['missingControlTimestamps'],[SENSORS[35]['timestamp_s']]);self.assertTrue(report['workerExited']);self.assertFalse(report['evidenceComplete'])
        with self.assertRaises(FileExistsError):ArchiveSink(path)

    def test_07_parameters_inverse_patch_matches_entire_original(self):
        data=json.loads((D/'executor-contract.json').read_text());text=(D/'executor.py').read_text()
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),data['parametersSha256'])
        for pair in reversed(data['exactReplacements']):
            self.assertEqual(text.count(pair['new']),1);text=text.replace(pair['new'],pair['old'])
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),data['baseSha256'])
        self.assertEqual(text,Path(data['base']).read_text())
        compile((D/'executor.py').read_text(),str(D/'executor.py'),'exec')

    def test_08_worker_termination_retains_explicit_failure(self):
        path=OUT/'worker-exit';controls(path,[35]);sink=ArchiveSink(path,capacity=1,no_video=True)
        submit(sink,35)
        until=time.monotonic()+10
        while not (path/'media-progress.json').exists()and time.monotonic()<until:time.sleep(.02)
        self.assertTrue((path/'media-progress.json').exists())
        sink.process.terminate();sink.process.join(timeout=5)
        with self.assertRaises(ArchiveError):sink.close()
        report=json.loads((path/'media-archive-report.json').read_text());self.assertFalse(report['evidenceComplete']);self.assertTrue(report['workerExited']);self.assertEqual(report['committedCaptures'],1)

    def test_09_bounded_drain_timeout_cleans_worker(self):
        path=OUT/'drain-timeout';controls(path,[35]);sink=ArchiveSink(path,capacity=2,no_video=True,_test_delay=2,close_timeout=.2)
        submit(sink,35);start=time.monotonic()
        with self.assertRaises(ArchiveError):sink.close()
        self.assertLess(time.monotonic()-start,8);self.assertFalse(sink.process.is_alive());self.assertFalse(sink.result['evidenceComplete'])

    def test_10_started_encoder_processes_all_reaped_after_pipe_error(self):
        path=OUT/'encoder-pipe-error';controls(path,[35]);sink=ArchiveSink(path,capacity=1,_ffmpeg='/bin/false')
        try:submit(sink,35)
        finally:
            with self.assertRaises(ArchiveError):sink.close()
        children=json.loads((path/'video-children.json').read_text())['children']
        self.assertEqual(len(children),3);self.assertTrue(all(x['exited']for x in children));self.assertTrue(sink.result['workerExited']);self.assertFalse(sink.result['evidenceComplete'])

    def test_11_hung_encoder_is_terminated_before_pipe_flush(self):
        path=OUT/'hung-encoder';controls(path,[35])
        fake=path/'hung-encoder';fake.write_text('#!/usr/bin/python3\nimport time\ntime.sleep(60)\n');fake.chmod(0o700)
        sink=ArchiveSink(path,capacity=1,_ffmpeg=fake,close_timeout=2)
        submit(sink,35);start=time.monotonic()
        with self.assertRaises(ArchiveError):sink.close()
        self.assertLess(time.monotonic()-start,12);self.assertFalse(sink.process.is_alive())
        children=json.loads((path/'video-children.json').read_text())['children']
        self.assertEqual(len(children),3);self.assertTrue(all(x['exited']for x in children));self.assertFalse(sink.result['evidenceComplete'])

    def test_12_corrupt_worker_report_still_closes_and_has_failure_result(self):
        path=OUT/'corrupt-report';controls(path,[35]);sink=ArchiveSink(path,capacity=1,no_video=True)
        submit(sink,35);sink._put({'kind':'finish','expected':1});sink.process.join(timeout=10)
        self.assertFalse(sink.process.is_alive());(path/'media-archive-report.json').write_text('{broken')
        with self.assertRaises(ArchiveError):sink.close()
        self.assertFalse(sink.result['evidenceComplete']);self.assertIn('reportReadError',sink.result);self.assertIn('encoderCleanupFallback',sink.result);self.assertTrue(sink.result['workerExited'])

    def test_13_parent_report_disk_error_preserves_in_memory_result(self):
        path=OUT/'parent-report-error';controls(path,[35]);sink=ArchiveSink(path,capacity=1,no_video=True)
        submit(sink,35)
        with mock.patch('media_archive.atomic_json',side_effect=OSError('Injected parent report disk full')):
            with self.assertRaises(ArchiveError):sink.close()
        self.assertFalse(sink.result['evidenceComplete']);self.assertIn('reportPersistenceError',sink.result);self.assertTrue(sink.result['workerExited'])

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(MediaTests)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (OUT/'report.json').write_text(json.dumps({'schema':'microduck.football-media-cpu-shadow.v1','testsRun':result.testsRun,'passed':result.wasSuccessful(),'failures':[(str(x),y)for x,y in result.failures],'errors':[(str(x),y)for x,y in result.errors],'checks':CHECKS,'scope':'CPU-only original retained RGB/ToF replay. NoGPU/Kit/YOLO run, no model/controller/success qualification or executor speed claim.','sourceSHA256':{str(p):sha(p)for p in [Path(__file__),D/'media_archive.py',D/'executor.py',D/'execute.py']}},ensure_ascii=False,indent=2)+'\n')
    print(str(OUT));sys.exit(0 if result.wasSuccessful() else 1)
