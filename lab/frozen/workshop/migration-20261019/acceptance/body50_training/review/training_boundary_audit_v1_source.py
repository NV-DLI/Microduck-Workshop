"""Independent source/evidence and physical-fixture boundary review, no GPU."""
import hashlib, importlib.util, json
from pathlib import Path
import torch

M=Path(__file__).resolve().parents[3]
G=M/'gait/experiments/body50_v9'
O=M.parents[1]/'outputs/migration-20261019'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run():
    checks=[]
    def check(name,value):
        checks.append({'name':name,'passed':bool(value)})
    spec=importlib.util.spec_from_file_location('reviewed_fixture',G/'startup_fixture.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    fixture=module.StartupFixture(3,'cpu')
    pose=torch.zeros(3,7);pose[:,2]=.125;pose[:,6]=1.
    fixture.reset(None,reference_pose_xyzw=pose,actual_mass_kg=torch.ones(3))
    position=pose[:,:3].clone();q=pose[:,3:];zero=torch.zeros(3,3)
    for tick in range(450):
        t=torch.tensor([tick*.005]*3)
        ready=torch.tensor([tick>=130,tick>=150,False])
        force,torque=fixture.update(t,position=position,quaternion=q,linear_velocity=zero,angular_velocity=zero,orientation_valid=ready)
    check('ready_worlds_complete_physical_release',fixture.task_active[:2].all())
    check('unready_world_never_enters_task',not fixture.task_active[2])
    # A large subsequent physical displacement cannot re-enable support.
    position[:,0]=10.
    force,torque=fixture.update(torch.tensor([2.5]*3),position=position,quaternion=q,linear_velocity=zero+5,angular_velocity=zero+5,orientation_valid=torch.tensor([True,True,False]))
    check('task_fixture_force_exactly_zero_after_displacement',torch.count_nonzero(force[:2])==0 and torch.count_nonzero(torque[:2])==0)
    fixture.reset(torch.tensor([0]),reference_pose_xyzw=pose,actual_mass_kg=torch.ones(3))
    check('asynchronous_reset_does_not_reenable_peer_support',not fixture.task_active[0] and fixture.task_active[1])
    check('fixture_has_no_state_write_calls',not any(x in (G/'startup_fixture.py').read_text() for x in ('write_root','write_joint','set_root_state')))
    ledger_path=O/'gait-body50-v9-smoke16x3-b/training-ledger.json'
    ledger=json.loads(ledger_path.read_text())
    check('real_smoke_prewarms_all16',ledger['prewarm']['ready_environments']==16 and ledger['prewarm']['task_environments']==16)
    check('real_smoke_has_zero_actual_task_wrench',ledger['actual_sensor_fixture']['actual_task_external_wrench_peak']==0.)
    check('real_smoke_excludes_startup_samples',ledger['prewarm']['initialization_env_steps_excluded']>0)
    check('real_smoke_optimizer_uses1152_task_samples',ledger['optimizer_mask']['task_transitions']==1152 and ledger['optimizer_mask']['optimizer_steps']==60)
    check('real_smoke_sources_unchanged',ledger['sources_unchanged'])
    sources=[G/n for n in ('startup_fixture.py','environment_hook.py','ppo_mask.py','train.py')]
    report={'schema':'microduck.body50-training-independent-review.v1','checks':checks,
      'passed':all(c['passed']for c in checks),'source_sha256':{str(p):sha(p)for p in sources},
      'evidence':{'path':str(ledger_path),'sha256':sha(ledger_path)},
      'manual_boundary_review':[
        'World pose/velocity enters virtual sensor synthesis and temporary physical fixture only; actor first48 fields are replaced by delivered BODY50, quantized encoders and raw history.',
        'All4 real200Hz substeps are required by the scene-update guard. Readiness is computed from actual synthesized accelerometer/gyro initialization, never root quaternion injection.',
        'Action-time ready AND task-active masks select PPO transitions; release may run actor but cannot contribute task loss. Invalid transitions terminate GAE continuation.',
        'Task masks are taken before the control step, so a partial release-to-task control interval is excluded. The actual external-force buffer is monitored separately from requested force.'
      ],
      'open_review_request':'At reviewed source version task wrench peak is logged but not asserted by train.py. Require zero at checkpoint/finalization; parent acknowledged and will add after current smoke exits.',
      'limits':'CPU fixture review and completed16env smoke; no long-training behavior or task success claim.'}
    path=O/'body50-training-boundary-independent-review-v1.json'
    assert not path.exists();path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'checks':len(checks),'report':str(path)}))
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':run()
