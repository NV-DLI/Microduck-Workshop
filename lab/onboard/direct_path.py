"""Straight public-observation path for the surveyed empty pitch; no ROS service.

This is the shortest planar line to the existing ball-front approach target.
The map helper only checks current ToF/known-field clearance. It does not search
for a route or authorize a kick. The same outer walking controller follows it.
"""
from pathlib import Path
import hashlib
import json
import math
import time
from onboard_geometry.map_contract import snapshot, check_path


class DirectPath:
    def __init__(self, output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False)
        self.count=0;self.wall=0.
        self.log=(self.output/'queries.jsonl').open('w',buffering=1)

    def query(self, case, provenance):
        start=time.monotonic()
        if provenance.get('layout')!='empty':
            raise ValueError('Direct approach is restricted to the empty pitch')
        snap=snapshot(case,provenance)
        a,b=snap['pose'][:2],snap['goal']
        distance=math.dist(a,b)
        n=max(1,math.ceil(distance/.02))
        points=[[a[0]+(b[0]-a[0])*k/n,a[1]+(b[1]-a[1])*k/n] for k in range(n+1)]
        gate=check_path(points,snap)
        result=dict(planner='direct_straight_line',path_xy=points if gate['accepted'] else [],
                    envelope=gate,external_service_started=False,query_s=case['query_s'],
                    path_length_m=distance,online_hidden_state=False)
        self.log.write(json.dumps(dict(sequence=self.count,case=case,provenance=provenance,response=result),allow_nan=False)+'\n')
        self.count+=1;self.wall+=time.monotonic()-start
        return result

    def close(self):
        self.log.close()
        result=dict(planner='direct_straight_line',external_service_started=False,queries=self.count,
                    query_wall_seconds=self.wall,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        (self.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        return result

