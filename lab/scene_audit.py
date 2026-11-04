"""Run the retained physical, sensor and recovery checks for the course scene."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--scene-runtime',type=Path,required=True)
parser.add_argument('--script',required=True)
args,remaining=parser.parse_known_args()
allowed={
    'audit_media_terminal_cadence.py','audit_football.py',
    'audit_sole_memory_actual.py','audit_floor_projection_actual.py',
    'audit_retreat_recovery.py',
}
if args.script not in allowed:
    raise ValueError('Unknown course scene audit')
runtime=args.scene_runtime.resolve()
lab=Path(__file__).resolve().parent
sys.path[:0]=[str(lab),str(runtime/'evaluation/football'),str(runtime/'evaluation'),
              str(runtime/'football'),str(runtime/'sensors')]
file=runtime/'evaluation'/args.script
sys.argv=[str(file),*remaining]
if args.script in {'audit_sole_memory_actual.py','audit_retreat_recovery.py'}:
    # Replay the course's 6.5 s public memory with the original independent
    # odometry, packet digests, visual anchors and invalidation checks.
    # Recovery entry itself keeps its stricter 4.5 s anchor-age restriction.
    from audit_epochs import validate
    epoch_report=validate(Path(remaining[0]))
    source=file.read_text()
    changed=source
    if args.script=='audit_sole_memory_actual.py':
        replacements=[('    adapter = SameReadMemoryAdapter(leg, first["joint_names"])',
                       '    adapter = SameReadMemoryAdapter(leg, first["joint_names"])\n    adapter.memory.maximum_age_s = 6.5\n    navigation_epoch = 0'),
                      ('    for row in reads:\n        t = row["timestamp_s"]',
                       '    for row in reads:\n        t = row["timestamp_s"]\n        if row["navigation_epoch"] != navigation_epoch:\n            adapter = SameReadMemoryAdapter(leg, first["joint_names"])\n            adapter.memory.maximum_age_s = 6.5\n            navigation_epoch = row["navigation_epoch"]')]
    else:
        replacements=[
            ('reserve_s=30.0,','reserve_s=5.0,'),
            ('    task_limit = min(180.0, float(scope.get("task_limit_s", 180.0)))','    declared_task_limit = scope.get("task_limit_s")\n    task_limit = 180.0 if declared_task_limit is None else min(180.0, float(declared_task_limit))'),
            ('    prev = {}\n    for c in ctrl:\n        fa = c.get("final_approach") or {}','    prev = {}\n    previous_navigation_epoch = 0\n    previous_capture_s = -1.\n    for c in ctrl:\n        if c["navigation_epoch"] != previous_navigation_epoch:\n            restart_s = course_epochs["restart_times_s"][previous_navigation_epoch]\n            audit.check("counter_reset_bound_to_navigation_restart",\n                        c["navigation_epoch"] == previous_navigation_epoch + 1\n                        and previous_capture_s <= restart_s < c["timestamp_s"],\n                        t=c["timestamp_s"], restart_s=restart_s)\n            prev = {}\n            previous_navigation_epoch = c["navigation_epoch"]\n        previous_capture_s = c["timestamp_s"]\n        fa = c.get("final_approach") or {}'),
            ('        self.adapter = SameReadMemoryAdapter(leg, first["joint_names"])',
             '        self.adapter = SameReadMemoryAdapter(leg, first["joint_names"])\n        self.adapter.memory.maximum_age_s = 6.5\n        self.navigation_epoch = 0\n        self.leg = leg\n        self.joint_names = first["joint_names"]'),
            ('            row = self.reads[self.index]\n            self.index += 1',
             '            row = self.reads[self.index]\n            self.index += 1\n            if row["navigation_epoch"] != self.navigation_epoch:\n                from ball_status_adapter import SameReadMemoryAdapter\n                self.adapter = SameReadMemoryAdapter(self.leg, self.joint_names)\n                self.adapter.memory.maximum_age_s = 6.5\n                self.navigation_epoch = row["navigation_epoch"]\n                self.epoch += 1\n                self.invalidations.append("navigation_supervisor_restart")\n                original = self.adapter.memory.invalidate\n                def counted(reason, _o=original):\n                    self.epoch += 1\n                    self.invalidations.append(reason)\n                    return _o(reason)\n                self.adapter.memory.invalidate = counted'),
            ('finite(m.get("ball_age_s")) and 0 <= m["ball_age_s"] <= LIMITS["ttl_s"],',
             'finite(m.get("ball_age_s")) and 0 <= m["ball_age_s"] <= 6.5,'),
        ]
    for old,new in replacements:
        if changed.count(old)!=1:
            raise RuntimeError('Unexpected public-memory-audit source version')
        changed=changed.replace(old,new)
    output=Path(remaining[remaining.index('--output')+1])
    with output.with_suffix('.profile.json').open('x') as handle:
        json.dump(dict(navigation_epoch_validation=epoch_report,memory_maximum_age_s=6.5,reserve_s=5.,recovery_entry_maximum_age_s=4.5,
                       epoch_audit_sha256=hashlib.sha256(Path(validate.__code__.co_filename).read_bytes()).hexdigest(),
                       source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       executed_sha256=hashlib.sha256(changed.encode()).hexdigest()),handle,indent=2)
    exec(compile(changed,str(file)+'<course-public-memory>','exec'),{'__name__':'__main__','__file__':str(file),'course_epochs':epoch_report})
else:
    runpy.run_path(str(file),run_name='__main__')
