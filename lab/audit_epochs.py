"""Bind navigation restarts to public capture, policy and delivered-read records."""
from pathlib import Path
import bisect
import json
import math


def validate(run):
    run=Path(run)
    manifest=json.loads((run/'manifest.json').read_text())
    events=manifest.get('navigation_retries')
    if not isinstance(events,list):
        raise ValueError('Declared navigation retry records required')
    restarts=[]
    for index,event in enumerate(events):
        if event.get('epoch')!=len(restarts):
            raise ValueError('Retry epoch must follow the preceding restart')
        start=event.get('timestamp_s');deadline=event.get('retry_at_s')
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (start,deadline)):
            raise ValueError('Finite retry pause timestamps required')
        if abs(deadline-start-1.)>1e-8:
            raise ValueError('Navigation restart requires the recorded one-second pause')
        restart=event.get('restarted_at_s')
        if restart is None:
            if index!=len(events)-1:
                raise ValueError('Only the last retry may be unfinished')
            continue
        if not isinstance(restart,(int,float)) or not math.isfinite(restart) or not 0<=restart-deadline<=.01000001:
            raise ValueError('Restart must occur at the deadline on a physics tick')
        if event.get('next_epoch')!=len(restarts)+1 or (restarts and restart<=restarts[-1]):
            raise ValueError('Navigation restart clock and epoch must increase')
        restarts.append(restart)
    counts={}
    for name in ['control.jsonl','policy-trace.jsonl','public-memory-readback.jsonl']:
        count=0;previous=None;seen=set()
        for line in (run/name).open():
            row=json.loads(line);t=row.get('timestamp_s');epoch=row.get('navigation_epoch')
            if not isinstance(t,(int,float)) or not math.isfinite(t) or (previous is not None and t<=previous):
                raise ValueError('Public trace timestamps must strictly increase: '+name)
            expected=bisect.bisect_left(restarts,t-1e-8)
            if type(epoch) is not int or epoch!=expected:
                raise ValueError('Navigation epoch does not match restart boundaries: '+name+' at '+str(t))
            previous=t;seen.add(epoch);count+=1
        if not count or seen!=set(range(len(restarts)+1)):
            raise ValueError('Each navigation epoch must appear in every public stream: '+name)
        counts[name]=count
    return dict(restart_times_s=restarts,navigation_epochs=len(restarts)+1,public_row_counts=counts,
                scope='All public rows retained; navigation restarts reset only the replay adapter, never physics or scorer.')
