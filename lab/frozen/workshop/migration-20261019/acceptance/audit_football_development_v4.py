"""Read-only run summary and gravity-level memory audit, separate from execution."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import sys
import numpy as np

M=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(M/'acceptance'),str(M/'football'),str(M/'sensors')]
from public_ball_read_adapter_v2 import parent
from audit_v66_memory_extension import rotation


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args();assert not a.output.exists();D=a.run
    launchers=list(D.parent.glob(D.name+'-native-*-launcher.json'));assert len(launchers)==1
    launcher=launchers[0];argv=json.loads(launcher.read_text())['argv']
    assert Path(argv[argv.index('--output')+1]).resolve()==D.resolve()
    planned_seconds=float(argv[argv.index('--seconds')+1]);assert 0<planned_seconds<=180
    m=json.loads((D/'manifest.json').read_text())
    assert m.get('terminal_reason')
    controls=[json.loads(l)for l in (D/'control.jsonl').open()]
    policies=[json.loads(l)for l in (D/'policy-trace.jsonl').open()]
    public={round(r['timestamp_s'],6):r for r in map(json.loads,(D/'public-memory-readback.jsonl').open())}
    wanted={}
    for r in policies:
        f=r.get('public_neutral_fine_step',{});h=r.get('public_handoff',{})
        # Stationary-ball transport is only used before the fixed kick starts.
        # Ball motion during K10 is physical outcome, not a tracking error.
        onset=h.get('kick_onset_s')
        if onset is not None and r['timestamp_s']>onset+1e-9:continue
        if f.get('state') not in ('IDLE',None) or h.get('state') not in ('APPROACH',None):
            p=public[round(r['timestamp_s'],6)]
            state=p['handoff']['memory'];packet=p['readback'].get('actual_packet')
            if state['memory_available'] and packet:
                wanted[round(r['timestamp_s'],6)]={'age':state['ball_age_s'],'point':state['point_level_m'],
                    'gravity':packet['gravity_body'],'phase':f.get('state'),'attempt':f.get('attempts')}
    errors=collections.defaultdict(list);first_contact=None;peak_contact=0.;count=0
    for r in map(json.loads,(D/'physics-trace.jsonl').open()):
        count+=1;force=float(np.linalg.norm(r['raw_paired_foot_ball_forces_n'],axis=-1).max())
        if r['score_enabled'] and force>.001:
            if first_contact is None:first_contact={'t':r['timestamp_s'],'force_n':force,'kick_onset_s':r.get('kick_onset_s')}
            peak_contact=max(peak_contact,force)
        sample=wanted.get(round(r['timestamp_s'],6))
        if sample:
            body=rotation(r['root_quaternion_xyzw']).T@(np.array(r['ball_position_world_m'])-r['root_position_world_m'])
            truth=parent.memory_module.level_rotation(sample['gravity'])@body
            error=float(np.linalg.norm(np.array(sample['point'][:2])-truth[:2]))
            key=f"attempt{sample['attempt']}:"+('0-4.5s'if sample['age']<=4.5 else '4.5-6s')
            errors[key].append(error)
    s=m['scorer'];entries=[]
    for r in controls:
        if r.get('public_fine_entry',{}).get('trigger'):
            f=r['public_neutral_fine_step'];d=r.get('final_approach',{})
            entries.append({'t':r['timestamp_s'],'reason':r['public_fine_entry']['reason'],
                'point':f['memory']['point_level_m'],'goal_heading':f['goal_heading_rad'],
                'measured_quiet_at_entry':d.get('measured_quiet'),'gyro_rms_at_entry':d.get('gyro_3D_RMS_rad_s'),
                'final_stride':f.get('final_stride',f.get('final_stride_candidate',False)),'motion_kind':f.get('motion_kind','STEP')})
    result={'schema':'microduck.football-development-audit.v4','run':str(D),'planned_seconds':planned_seconds,'budget_source_launcher':str(launcher),'budget_source_sha256':hashlib.sha256(launcher.read_bytes()).hexdigest(),
        'actual_task_seconds':m['task_seconds'],'terminal_reason':m['terminal_reason'],
        'fine_entries':entries,'kick_policy_ticks':sum(bool(r.get('kick_requested'))for r in policies),
        'whole_ball_goal':s['whole_ball_crossed_inside_goal'],'kick_contact':s['real_foot_ball_contact'],
        'pre_kick_contact':s['contact']['pre_kick_ball_contact_observed'],'fell':s['fell'],
        'first_contact':first_contact,'peak_contact_force_n':peak_contact,
        'memory_audit':{k:{'samples':len(v),'max_xy_error_m':max(v),'under10mm':max(v)<=.01}for k,v in errors.items()},
        'camera_frames':m['camera_frames'],'policy_updates':m['policy_updates'],'physics_samples':count,
        'sources_unchanged':not m['source_changed_during_run'] and not m['native_reference_contract']['source_changed_during_run'],
        'original_task_scorer':s,'qualification_claimed':False,'audit_only':True,'formal_test_started':False,
        'memory_audit_scope':'Before or at actual K10 onset only (later stationary-ball memory is not an action input). Actual root and ball truth projected into the contemporaneous public gravity-level frame. Audit only; no truth authorizes movement.',
        'source_sha256':{str(D/n):hashlib.sha256((D/n).read_bytes()).hexdigest()for n in ['manifest.json','control.jsonl','policy-trace.jsonl','physics-trace.jsonl','public-memory-readback.jsonl']}}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k]for k in ['terminal_reason','fine_entries','memory_audit','whole_ball_goal','sources_unchanged']},indent=2))


if __name__=='__main__':main()
