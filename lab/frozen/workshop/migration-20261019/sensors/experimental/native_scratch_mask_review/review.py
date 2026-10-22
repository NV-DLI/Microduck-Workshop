"""Read-only CPU mask/GAE review of completed native-from-scratch experiment."""
from pathlib import Path
import hashlib,json,sys,unittest
import torch
M=Path('@COURSE@/workshop/migration-20261019');O=M.parents[1]/'outputs/migration-20261019';B=M/'gait/experiments/body50_v9';D=M/'gait/experiments/native_from_scratch_v1';R=O/'native-from-scratch-v1-random256x100-s980122'
sys.path.insert(0,str(B))
from test_ppo_mask import make,observations,step,TestMaskedPPO
from ppo_mask import install
class TerminalReview(unittest.TestCase):
 def test_true_fall_is_kept_before_reset_and_no_cross_gap_bootstrap(self):
  alg,env=make(n=2,steps=4);audit=install(alg,env)
  with torch.no_grad():
   for p in alg.critic.parameters():p.zero_()
   alg.critic.mlp[-1].bias.fill_(2.)
  env.ready_mask[:]=True;env.task_active_mask[:]=True
  step(alg,observations([1.,2.]),[-20.,1.],[True,False])
  env.ready_mask[0]=False;env.task_active_mask[0]=False
  step(alg,observations([float('nan'),3.]),[float('nan'),2.])
  step(alg,observations([float('nan'),4.]),[float('nan'),3.])
  env.ready_mask[:]=True;env.task_active_mask[:]=True
  step(alg,observations([5.,5.]),[4.,4.])
  with torch.inference_mode():alg.compute_returns(observations([6.,6.]))
  self.assertEqual(float(alg.storage.rewards[0,0]),-20.)
  self.assertTrue(bool(alg.storage.dones[0,0]))
  self.assertEqual(float(alg.storage.returns[0,0]),-20.)
  self.assertEqual(float(alg.storage.returns[1,0]),0.)
  self.assertEqual(float(alg.storage.returns[2,0]),0.)
  self.assertAlmostEqual(float(alg.storage.returns[3,0]),5.8,places=5)
  self.assertTrue(bool(audit.valid[:,1].all()))
  self.assertEqual(int(audit.valid.sum()),6)
  alg.update();self.assertEqual(audit.counts['task_transitions'],6)
  self.assertEqual(int(alg.actor.obs_normalizer.count),6)
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(TestMaskedPPO),unittest.defaultTestLoader.loadTestsFromTestCase(TerminalReview)])
result=unittest.TextTestRunner(verbosity=2).run(suite)
x=json.loads((R/'training-ledger.json').read_text());p=x['prewarm'];f=x['actual_sensor_fixture'];a=x['optimizer_mask'];valid=a['task_transitions'];total=x['stage']['envs']*x['rollout_steps']*a['rollouts_seen']
checks={'task_counter_equals_action_time_PPO_rows':f['task_env_steps']-p['task_env_steps']==valid,'initialization_counter_equals_placeholder_rows':f['initialization_env_steps_excluded']-p['initialization_env_steps_excluded']==a['startup_placeholder_action_rows'],'all_rows_accounted':valid+a['startup_placeholder_action_rows']==total,'task_only_ready_equals_valid':a['ready_inference_rows']==a['ready_normalizer_rows']==valid,'no_hidden_drop_or_release_rows':a['release_excluded_transitions']==a['task_active_without_ready_dropped']==0,'all_inactive_have_no_optimizer':a['rollouts_with_optimizer_updates']+a['all_task_inactive_rollouts']==a['rollouts_seen'] and a['optimizer_steps']==a['rollouts_with_optimizer_updates']*20,'no_startup_failure':f['startup_failures']==0,'all_actual_task_external_force_zero':f['actual_task_external_wrench_peak']==0}
files=[Path(__file__),B/'ppo_mask.py',B/'test_ppo_mask.py',B/'environment_hook.py',D/'recipe.py',D/'train.py',R/'training-ledger.json',R/'independent-audit.json']
report={'schema':'microduck.native-scratch-mask-independent.v2','scope':'Read-only source/ledger plus actual installed CPU RSL PPO regression; not a replay of unsaved GPU minibatches','all_checked_passed':result.wasSuccessful()and all(checks.values()),'cpu_tests':result.testsRun,'checks':checks,'actual_training':{'valid_rows':valid,'all_rows':total,'valid_fraction':valid/total,'rollouts':a['rollouts_seen'],'optimizer_updates':a['rollouts_with_optimizer_updates'],'all_inactive_rollouts':a['all_task_inactive_rollouts'],'optimizer_steps':a['optimizer_steps'],'startup_failed':f['startup_failures']},'interpretation':['The low valid fraction is fully reconciled by real initialization exclusion, with no missing task rows in recorded aggregate counters.','Task-only readiness excludes all initialization and release rows from actor, critic, normalizer, loss; fixture internal sensor readiness is unchanged.','A real action-time task fall retains its penalty/done even after automatic environment reset. GAE stops at done/invalid gaps, and independent active worlds remain trainable.','A 0.5s task followed by about2.5s preparation would yield about16.7% eligible samples; this is an explanatory duty-cycle calculation, not an exact measured reset-duration distribution.','Seven all-inactive rollouts legitimately skip optimization; 100 collection rollouts are93 optimizer-bearing PPO updates.','This review does not establish why the policy falls, diagnose KL drift, or qualify walking success.'],'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in files}}
out=O/'native-from-scratch-v1-mask-independent-review-v2.json';out.write_text(json.dumps(report,indent=2)+'\n');print(out);assert report['all_checked_passed']
