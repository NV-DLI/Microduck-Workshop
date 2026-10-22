"""Measured PPO lineage and every-iteration accounting; no simulated-time estimates."""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path,data):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temp.replace(path)


def append(path,data):
    with Path(path).open('a') as file:file.write(json.dumps(data,allow_nan=False)+'\n')


def actor_sha(actor):
    """Canonical weights plus observation-normalizer hash, independent of zip metadata."""
    import torch
    digest=hashlib.sha256()
    for name,tensor in sorted(actor.state_dict().items()):
        value=tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name,str(value.dtype),list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def parent_work(checkpoint):
    if checkpoint is None:
        return dict(iterations=0,wall_seconds=0.,compute_seconds=0.,time_source='no parent',ancestry=[])
    checkpoint=Path(checkpoint).resolve();digest=sha(checkpoint)
    record_path=checkpoint.parent/'checkpoint_times.jsonl'
    record=None
    if record_path.exists():
        for candidate in reversed([json.loads(x) for x in record_path.read_text().splitlines() if x.strip()]):
            if candidate['checkpoint']==checkpoint.name:
                if candidate.get('checkpoint_sha256') and candidate['checkpoint_sha256']!=digest:
                    raise ValueError('Resume checkpoint differs from its measured ledger SHA')
                record=candidate;break
    ledger_path=checkpoint.parent/'training-ledger.json'
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    ancestors=ledger.get('parent_ancestry',[])
    ancestors=ancestors+[{'checkpoint':str(checkpoint),'sha256':digest,'ledger':str(ledger_path) if ledger else None}]
    if record:
        return dict(iterations=record['cumulative_iterations'],wall_seconds=record.get('cumulative_wall_seconds'),
            compute_seconds=record.get('cumulative_rollout_and_update_seconds'),time_source='parent measured checkpoint',ancestry=ancestors)
    # A checkpoint carries an exact optimizer index even when old timing is
    # unavailable. Never invent proportional hours for a legacy ancestor.
    import torch
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    return dict(iterations=int(saved['iter'])+1,wall_seconds=None,compute_seconds=None,
        time_source='checkpoint optimizer iteration; historical wall time unavailable',ancestry=ancestors)


class TrainingAccounting:
    def __init__(self,runner,run,parent,start,num_envs,rollout_steps,*,metadata=None,initial_checkpoint=None,readback=None):
        self.runner=runner;self.run=Path(run);self.parent=parent_work(parent);self.start=start
        self.start_iteration=int(runner.current_learning_iteration)
        self.num_envs=num_envs;self.rollout_steps=rollout_steps
        self.compute_seconds=0.;self.completed=0;self.readback=readback
        self.ledger={'schema':'microduck.training-ledger.v2','run_id':self.run.name,
            'runtime_id':'newton-bam-20261019','hardware_profile_id':'beta-gc2093-l8cx-v1',
            'initialization':'continuation' if parent else 'random','parent_ancestry':self.parent['ancestry'],
            'parent_run_ids':[Path(item['checkpoint']).parent.name for item in self.parent['ancestry']],
            'parent_checkpoint':str(Path(parent).resolve()) if parent else None,
            'parent_checkpoint_sha256':sha(parent) if parent else None,
            'initial_checkpoint':str(initial_checkpoint) if initial_checkpoint else None,
            'initial_checkpoint_sha256':sha(initial_checkpoint) if initial_checkpoint else None,
            'initial_actor_state_sha256':actor_sha(runner.alg.actor),
            'started_at_unix':time.time(),'state':'training','env_count':num_envs,
            'rollout_steps':rollout_steps,'control_hz':50,'physics_hz':200,
            'evaluation_data_used_for_training':False,'export_consistency':{'state':'not_exported'},
            **(metadata or {}),**self.summary()}
        self.persist()
        if initial_checkpoint:
            append(self.run/'checkpoint_times.jsonl',dict(checkpoint=Path(initial_checkpoint).name,
                checkpoint_sha256=sha(initial_checkpoint),actor_state_sha256=self.ledger['initial_actor_state_sha256'],
                checkpoint_role='initial snapshot before this run learns',**self.summary()))
        save=runner.save;log=runner.logger.log
        def timed_log(*args,**kwargs):
            self.compute_seconds+=float(kwargs['collect_time'])+float(kwargs['learn_time'])
            self.completed+=1
            current_hash=actor_sha(runner.alg.actor)
            record={**self.summary(),'optimizer_iteration_zero_based':int(kwargs['it']),
                'collection_seconds':float(kwargs['collect_time']),'update_seconds':float(kwargs['learn_time']),
                'actor_state_sha256':current_hash,'reward_version':self.ledger.get('reward_version'),
                'losses':{k:float(v) for k,v in kwargs.get('loss_dict',{}).items()}}
            append(self.run/'iteration_times.jsonl',record)
            self.ledger.update(record,state='training');self.persist()
            return log(*args,**kwargs)
        def timed_save(path,*args,**kwargs):
            result=save(path,*args,**kwargs)
            record=dict(checkpoint=Path(path).name,checkpoint_sha256=sha(path),actor_state_sha256=actor_sha(runner.alg.actor),**self.summary())
            if self.readback:record['actual_physics_parameters']=self.readback('checkpoint '+Path(path).name)
            append(self.run/'checkpoint_times.jsonl',record)
            self.ledger.update(latest_checkpoint=str(Path(path).resolve()),latest_checkpoint_sha256=record['checkpoint_sha256'],
                latest_checkpoint_completed_iterations=self.completed,latest_actor_state_sha256=record['actor_state_sha256'],**self.summary())
            self.persist();return result
        runner.logger.log=timed_log;runner.save=timed_save

    def persist(self):atomic(self.run/'training-ledger.json',self.ledger)

    def summary(self):
        wall=time.monotonic()-self.start
        return dict(stage_completed_iterations=self.completed,stage_wall_seconds=wall,
            iterations_this_run=self.completed,iterations_cumulative=self.parent['iterations']+self.completed,
            environment_steps=self.completed*self.num_envs*self.rollout_steps,training_wall_seconds=wall,
            stage_rollout_and_update_seconds=self.compute_seconds,initial_optimizer_iteration=self.start_iteration,
            cumulative_iterations=self.parent['iterations']+self.completed,
            cumulative_wall_seconds=None if self.parent['wall_seconds'] is None else self.parent['wall_seconds']+wall,
            cumulative_rollout_and_update_seconds=None if self.parent['compute_seconds'] is None else self.parent['compute_seconds']+self.compute_seconds,
            parent_time_source=self.parent['time_source'],stage_transitions=self.completed*self.num_envs*self.rollout_steps,
            num_envs=self.num_envs,rollout_steps=self.rollout_steps,
            wall_clock_scope='active process elapsed including save/record/export overhead; no queue time',
            compute_clock_scope='sum of measured PPO rollout + update times; may include recording/shared GPU waits')

    def finish(self,*,error=None,export=None):
        self.ledger.update(self.summary())
        if export is not None:self.ledger['export_consistency']=export
        if not error and self.ledger.get('latest_checkpoint_sha256'):
            self.ledger['selected_checkpoint_sha256']=self.ledger['latest_checkpoint_sha256']
            self.ledger['selection_scope']='final training candidate; independent development selection and held-out task qualification still required'
        self.ledger.update(state=('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed') if error else 'completed',
            ended_at_unix=time.time(),error=repr(error) if error else None)
        self.persist()
