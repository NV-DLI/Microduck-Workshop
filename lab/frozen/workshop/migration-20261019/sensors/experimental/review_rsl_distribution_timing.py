"""CPU oracle for clamped Gaussian rollout/update and normalizer update timing."""
import hashlib
import inspect
import json
from pathlib import Path
import torch
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.modules.normalization import EmpiricalNormalization
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.models.mlp_model import MLPModel

torch.manual_seed(975192)
d=GaussianDistribution(14,init_std=1.,std_range=(.03,.10),learn_std=True)
mu=torch.randn(1024,14)*.01
d.update(mu);sample=d.sample();stored=tuple(x.detach().clone()for x in d.params);oldlog=d.log_prob(sample).detach().clone()
d.update(mu.clone());newlog=d.log_prob(sample);kl=d.kl_divergence(stored,d.params)
checks={'rollout_stored_std_actual_clamped':bool((stored[1]==.10).all()),
    'raw_parameter_still1':bool((d.std_param==1.).all()),
    'recompute_std_same':torch.equal(stored[1],d.params[1]),
    'same_mean_logprob_exact':torch.equal(oldlog,newlog),
    'same_distribution_KL_exact0':not bool(kl.any())}
newlog.mean().backward();checks['out_of_range_raw_std_gradient0']=not bool(d.std_param.grad.any())
delta=torch.full_like(mu,.01);d.update(mu+delta);actual=d.kl_divergence(stored,d.params)
expected=(delta.square()/(2*.1**2)).sum(-1)
checks['mean_change_KL_correct']=bool(torch.allclose(actual,expected,atol=1e-6,rtol=0))
norm=EmpiricalNormalization(61);norm.train();obs=torch.randn(128,61)
before={k:v.clone()for k,v in norm.state_dict().items()}
for _ in range(3):norm(obs)
checks['forward_normalizer_readonly']=all(torch.equal(v,norm.state_dict()[k])for k,v in before.items())
norm.update(obs);checks['explicit_update_changes_count']=int(norm.count)==128
source=inspect.getsource(PPO.update)
checks['normalization_after_all_optimizer_steps']=source.rfind('self.actor.update_normalization(obs)')>source.rfind('self.optimizer.step()')
assert all(checks.values()),checks
M=Path(__file__).resolve().parents[2];O=M.parents[1]/'outputs/migration-20261019';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
paths=[Path(__file__),Path(inspect.getfile(GaussianDistribution)),Path(inspect.getfile(EmpiricalNormalization)),Path(inspect.getfile(PPO)),Path(inspect.getfile(MLPModel)),M/'gait/experiments/body50_v9/ppo_mask.py']
report={'schema':'microduck.rsl-clamped-distribution-normalization-timing.v1','passed':True,'checks':checks,
        'evidence':{str(p):sha(p)for p in paths},'effective_std':float(stored[1][0,0]),'raw_std_parameter':float(d.std_param[0].detach()),
        'mean_shift_0p01_all14_KL':float(actual.mean()),
        'interpretation':'There is no unclamped-rollout versus clamped-update mismatch. All stored params/log-probs use the Normal with clampedstd; recomputation is identical before optimizer changes. Forward normalization is readonly; stock PPO updates statistics only after every minibatch optimizer step for the rollout. Normalizer movement between rollouts can affect behavior but is not a within-rollout old/new normalization mismatch in this path.',
        'important_limits':'This oracle does not attribute the measured D2 learning-rate collapse. Decompose actual mean change/normalizer change/10leg versus4unused head KL using its recorded checkpoints. A raw std parameter1 outside maximum.10 has zero clamp gradient and therefore stays effectively fixed unless separately reset; physical residual noise nearzero is approximately.15*.10rad.'}
out=O/'rsl-gaussian-normalizer-timing-independent-review-v1.json'
if out.exists():raise FileExistsError(out)
out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(checks))
